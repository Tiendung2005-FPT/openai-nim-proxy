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
const ENABLE_THINKING_MODE = true; // Changed to true: All models think by default

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
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
  'glm-4.7': 'z-ai/glm4.7',
  'glm-5': 'z-ai/glm5'
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
 * Scans entire request body for <NOTHINK> tag.
 * Thinking is ENABLED by default. If <NOTHINK> is found, it disables thinking.
 * Returns { shouldThink: boolean, cleanedMessages: Array }
 */
function processThinkingTag(requestBody) {
  let shouldThink = true; // Thinking is on by default

  const bodyString = JSON.stringify(requestBody);
  if (bodyString.includes('<NOTHINK>')) {
    shouldThink = false;
  }

  const messages = requestBody.messages;
  if (!messages || !Array.isArray(messages)) {
    return { shouldThink, cleanedMessages: messages };
  }

  const cleanedMessages = messages.map(msg => {
    if (msg.content && typeof msg.content === 'string' && msg.content.includes('<NOTHINK>')) {
      return { ...msg, content: msg.content.replace(/<NOTHINK>/g, '').trim() };
    }
    return msg;
  });

  return { shouldThink, cleanedMessages };
}

/**
 * Builds the chat_template_kwargs for thinking mode.
 * Unified kwargs config proven to work for models like Deepseek and GLM5
 */
function buildThinkingKwargs() {
  return { 
    thinking: true, 
    clear_thinking: true, 
    do_sample: true, 
    enable_thinking: true 
  };
}

// =============================================================================
// ENDPOINTS
// =============================================================================
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Multi-Provider API Proxy',
    providers: ['nvidia', 'ehub'],
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get(['/v1/models', '/nvidia/v1/models', '/ehub/v1/models'], (req, res) => {
  const provider = detectProvider(req.path);
  const modelMapping = provider === 'ehub' ? EHUB_MODEL_MAPPING : NVIDIA_MODEL_MAPPING;

  const models = Object.keys(modelMapping).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: `${provider}-proxy`
  }));

  res.json({
    object: 'list',
    data: models,
    provider: provider
  });
});

// =============================================================================
// NVIDIA NIM - CHAT COMPLETIONS
// =============================================================================
async function handleNvidiaCompletion(req, res) {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  const { shouldThink, cleanedMessages } = processThinkingTag(req.body);

  // Smart model mapping / fallback selection
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

      if (probeResp.status >= 200 && probeResp.status < 300) {
        nimModel = model;
      }
    } catch (err) {
      // ignore probe error
    }
  }

  if (!nimModel) {
    const modelLower = (model || '').toString().toLowerCase();
    if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
      nimModel = 'meta/llama-3.1-405b-instruct';
    } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
      nimModel = 'meta/llama-3.1-70b-instruct';
    } else {
      nimModel = 'meta/llama-3.1-8b-instruct';
    }
  }

  const nimRequest = {
    model: nimModel,
    messages: cleanedMessages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.6,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
    // Inject the unified thinking arguments if shouldThink is true
    ...(shouldThink ? { chat_template_kwargs: buildThinkingKwargs() } : {}),
    stream: !!stream
  };

  try {
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      validateStatus: status => status < 500
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      if (res.flushHeaders) res.flushHeaders();

      const streamData = response.data;
      let buffer = '';
      let reasoningStarted = false;

      streamData.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line) return;
          if (line.startsWith('data: ')) {
            const payload = line.slice(6);
            if (payload === '[DONE]') {
              res.write(`data: [DONE]\n\n`);
              return;
            }

            try {
              const parsed = JSON.parse(payload);
              if (parsed.choices?.[0]?.delta) {
                const delta = parsed.choices[0].delta;
                const reasoning = delta.reasoning_content;
                const content = delta.content;

                if (SHOW_REASONING) {
                  let combined = '';
                  if (reasoning && !reasoningStarted) {
                    combined = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combined = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combined += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combined += content;
                  }

                  if (combined) {
                    parsed.choices[0].delta.content = combined;
                    delete parsed.choices[0].delta.reasoning_content;
                  }
                } else {
                  parsed.choices[0].delta.content = content || '';
                  delete parsed.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(parsed)}\n\n`);
            } catch (err) {
              res.write(`data: ${line}\n\n`);
            }
          } else {
            res.write(`data: ${line}\n\n`);
          }
        });
      });

      streamData.on('end', () => { try { res.end(); } catch (e) {} });
      streamData.on('error', (err) => { try { res.end(); } catch (e) {} });

    } else {
      const nimData = response.data || {};
      const choices = Array.isArray(nimData.choices) ? nimData.choices : [];
      const openaiChoices = choices.map((choice) => {
        const msg = choice.message || {};
        let text = msg.content || '';
        if (SHOW_REASONING && msg.reasoning_content) {
          text = '<think>\n' + msg.reasoning_content + '\n</think>\n\n' + text;
        }
        return {
          index: choice.index ?? 0,
          message: { role: msg.role || 'assistant', content: text },
          finish_reason: choice.finish_reason || null
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
