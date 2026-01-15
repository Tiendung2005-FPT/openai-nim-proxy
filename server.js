// server.js - Multi-Provider API Proxy (NVIDIA NIM + ElectronHub)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// =============================================================================
// PROVIDER CONFIGURATIONS
// =============================================================================
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const EHUB_API_BASE = 'https://api.electronhub.ai/v1';
const EHUB_API_KEY = process.env.EHUB_API_KEY || 'ek-uJfTKZb87hFO3fhUVm0UVw6ses446HozYvHlHtAJgfq4G6lmJo';

// =============================================================================
// FEATURE TOGGLES
// =============================================================================
const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = false; // Set to false so it's ONLY enabled via tag

// =============================================================================
// MODEL MAPPINGS
// =============================================================================
const NVIDIA_MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'kimi-k2': 'moonshotai/kimi-k2-instruct-0905',
  'deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'deepseek-v3.1-terminus': 'deepseek-ai/deepseek-v3.1-terminus',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2'
};

const EHUB_MODEL_MAPPING = {
  'deepseek-v3.1-terminus':'deepseek-v3.1-terminus:free',
  'deepseek-v3.2':'deepseek-v3.2:free',
  'deepseek-r1-0528':'deepseek-r1-0528:free',
  'gemini-2.5-flash':'gemini-2.5-flash:free',
  'gemini-2.5-pro':'gemini-2.5-pro:free',
  'minimax-m2':'minimax-m2:free',
  'kimi-k2':'kimi-k2-instruct-0905:free',
  'kimi-k2-thinking':'kimi-k2-thinking:free'
};

// =============================================================================
// HELPERS
// =============================================================================
function detectProvider(path) {
  if (path.startsWith('/nvidia/')) return 'nvidia';
  if (path.startsWith('/ehub/')) return 'ehub';
  return 'nvidia';
}

/**
 * Scans messages for <ENABLETHINKING> tag.
 * Returns { shouldThink: boolean, cleanedMessages: Array }
 */
function processThinkingTag(messages) {
  let shouldThink = false;
  if (!messages || !Array.isArray(messages)) return { shouldThink, cleanedMessages: messages };

  const cleanedMessages = messages.map(msg => {
    if (msg.content && typeof msg.content === 'string' && msg.content.includes('<ENABLETHINKING>')) {
      shouldThink = true;
      // Remove the tag so the model doesn't see it as part of the prompt
      return { ...msg, content: msg.content.replace(/<ENABLETHINKING>/g, '').trim() };
    }
    return msg;
  });

  return { shouldThink, cleanedMessages };
}

// =============================================================================
// NVIDIA NIM - CHAT COMPLETIONS
// =============================================================================
async function handleNvidiaCompletion(req, res) {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  // 1. Check for the tag and clean it out
  const { shouldThink: tagDetected, cleanedMessages } = processThinkingTag(messages);

  let nimModel = NVIDIA_MODEL_MAPPING[model];
  if (!nimModel && typeof model === 'string') {
    try {
      const probeResp = await axios.post(`${NIM_API_BASE}/chat/completions`, {
        model: model,
        messages: [{ role: 'user', content: 'test' }],
        max_tokens: 1
      }, {
        headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: status => status < 500
      });
      if (probeResp.status >= 200 && probeResp.status < 300) nimModel = model;
    } catch (err) {}
  }

  if (!nimModel) {
    const modelLower = (model || '').toString().toLowerCase();
    if (modelLower.includes('gpt-4') || modelLower.includes('405b')) nimModel = 'meta/llama-3.1-405b-instruct';
    else if (modelLower.includes('70b')) nimModel = 'meta/llama-3.1-70b-instruct';
    else nimModel = 'meta/llama-3.1-8b-instruct';
  }

  // 2. Determine thinking mode strictly based on Tag or Global Toggle
  const shouldEnableThinking = ENABLE_THINKING_MODE || tagDetected;

  const nimRequest = {
    model: nimModel,
    messages: cleanedMessages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.6,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
    ...(shouldEnableThinking ? { chat_template_kwargs: { thinking: true } } : {}),
    stream: !!stream
  };

  try {
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      validateStatus: status => status < 500
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      const streamData = response.data;
      let buffer = '';
      let reasoningStarted = false;

      streamData.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(line => {
          if (!line || !line.startsWith('data: ')) return;
          const payload = line.slice(6);
          if (payload === '[DONE]') { res.write(`data: [DONE]\n\n`); return; }

          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;
              if (SHOW_REASONING) {
                let combined = '';
                if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
                else if (reasoning) { combined = reasoning; }
                if (content && reasoningStarted) { combined += '\n</think>\n\n' + content; reasoningStarted = false; }
                else if (content) { combined += content; }
                if (combined) { parsed.choices[0].delta.content = combined; delete parsed.choices[0].delta.reasoning_content; }
              } else {
                delete parsed.choices[0].delta.reasoning_content;
              }
            }
            res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          } catch (err) { res.write(`data: ${line}\n\n`); }
        });
      });
      streamData.on('end', () => res.end());
    } else {
      const nimData = response.data || {};
      const choices = (nimData.choices || []).map((choice) => {
        const msg = choice.message || {};
        let text = msg.content || '';
        if (SHOW_REASONING && msg.reasoning_content) {
          text = '<think>\n' + msg.reasoning_content + '\n</think>\n\n' + text;
        }
        return { index: choice.index, message: { role: msg.role, content: text }, finish_reason: choice.finish_reason };
      });
      res.json({ ...nimData, choices });
    }
  } catch (err) {
    res.status(err.response?.status || 500).json({ error: err.message });
  }
}

// =============================================================================
// ELECTRONHUB - CHAT COMPLETIONS
// =============================================================================
async function handleEhubCompletion(req, res) {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  // 1. Check for the tag and clean it out
  const { shouldThink: tagDetected, cleanedMessages } = processThinkingTag(messages);

  const ehubModel = EHUB_MODEL_MAPPING[model] || model;
  
  // 2. Determine thinking mode strictly based on Tag or Global Toggle
  const shouldEnableThinking = ENABLE_THINKING_MODE || tagDetected;

  const ehubRequest = {
    model: ehubModel,
    messages: cleanedMessages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.7,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
    // Apply thinking if tag is detected
    ...(shouldEnableThinking ? { chat_template_kwargs: { thinking: true } } : {}),
    stream: !!stream
  };

  try {
    const response = await axios.post(`${EHUB_API_BASE}/chat/completions`, ehubRequest, {
      headers: { 'Authorization': `Bearer ${EHUB_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      validateStatus: status => status < 500
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      response.data.pipe(res);
    } else {
      res.json(response.data || {});
    }
  } catch (err) {
    res.status(err.response?.status || 500).json({ error: err.message });
  }
}

// =============================================================================
// UNIFIED CHAT COMPLETIONS ENDPOINT
// =============================================================================
app.post(['/v1/chat/completions', '/nvidia/v1/chat/completions', '/ehub/v1/chat/completions'], async (req, res) => {
  const provider = detectProvider(req.path);
  if (provider === 'ehub') await handleEhubCompletion(req, res);
  else await handleNvidiaCompletion(req, res);
});

app.get('/health', (req, res) => res.json({ status: 'ok', thinking_via_tag_only: true }));

app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}. Thinking mode now requires <ENABLETHINKING> tag.`);
});
