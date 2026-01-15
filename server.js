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
const ENABLE_THINKING_MODE = false;

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

  // Smart model mapping / fallback selection
  let nimModel = NVIDIA_MODEL_MAPPING[model];
  if (!nimModel && typeof model === 'string') {
    // Try to probe the model on NIM - use a different variable name to avoid shadowing `res`
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
      // ignore probe error and fall back
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
    messages: messages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.6,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
    // pass thinking parameter if enabled (wrapped)
    ...(ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : {}),
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
      // STREAMING: NIM returns an event-stream-like stream. Forward to client as SSE.
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      // flush headers early if available
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
          // If line begins with 'data: ' treat as SSE event
          if (line.startsWith('data: ')) {
            const payload = line.slice(6);
            if (payload === '[DONE]') {
              res.write(`data: [DONE]\n\n`);
              return;
            }

            try {
              const parsed = JSON.parse(payload);

              // transform reasoning in deltas if present
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
                  // remove reasoning field and keep content (or empty)
                  parsed.choices[0].delta.content = content || '';
                  delete parsed.choices[0].delta.reasoning_content;
                }
              }

              res.write(`data: ${JSON.stringify(parsed)}\n\n`);
            } catch (err) {
              // if parsing fails, forward raw line
              res.write(`data: ${line}\n\n`);
            }
          } else {
            // not an SSE 'data: ' line — forward raw
            res.write(`data: ${line}\n\n`);
          }
        });
      });

      streamData.on('end', () => {
        try { res.end(); } catch (e) {}
      });
      streamData.on('error', (err) => {
        console.error('NIM stream error:', err);
        try { res.end(); } catch (e) {}
      });

    } else {
      // NON-STREAM: transform NIM response to OpenAI-compatible format
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
          message: {
            role: msg.role || 'assistant',
            content: text
          },
          finish_reason: choice.finish_reason || null
        };
      });

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model || nimModel,
        choices: openaiChoices,
        usage: nimData.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }

  } catch (err) {
    console.error('NIM request error:', err && err.message);
    const status = err.response?.status || 500;
    res.status(status).json({
      error: {
        message: err.response?.data?.error || err.message || 'NIM provider error',
        type: 'provider_error',
        code: status
      }
    });
  }
}

// =============================================================================
// ELECTRONHUB - CHAT COMPLETIONS
// =============================================================================
async function handleEhubCompletion(req, res) {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  const ehubModel = EHUB_MODEL_MAPPING[model] || model;

  const ehubRequest = {
    model: ehubModel,
    messages: messages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.7,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
    stream: !!stream
  };

  try {
    const response = await axios.post(`${EHUB_API_BASE}/chat/completions`, ehubRequest, {
      headers: {
        'Authorization': `Bearer ${EHUB_API_KEY}`,
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

      // response.data is a stream — pipe to client. Also listen for errors.
      response.data.pipe(res);
      response.data.on('error', (err) => {
        console.error('ElectronHub stream error:', err);
        try { res.end(); } catch (e) {}
      });
    } else {
      res.json(response.data || {});
    }
  } catch (err) {
    console.error('ElectronHub request error:', err && err.message);
    const status = err.response?.status || 500;
    res.status(status).json({
      error: {
        message: err.response?.data?.error || err.message || 'ElectronHub provider error',
        type: 'provider_error',
        code: status
      }
    });
  }
}

// =============================================================================
// UNIFIED CHAT COMPLETIONS ENDPOINT
// =============================================================================
app.post(['/v1/chat/completions', '/nvidia/v1/chat/completions', '/ehub/v1/chat/completions'], async (req, res) => {
  try {
    const provider = detectProvider(req.path);
    if (provider === 'ehub') {
      await handleEhubCompletion(req, res);
    } else {
      await handleNvidiaCompletion(req, res);
    }
  } catch (error) {
    console.error('Proxy error:', error && error.message);
    const status = error.response?.status || 500;
    res.status(status).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 Multi-Provider API Proxy running on port ${PORT}`);
  console.log(`   Health:       http://localhost:${PORT}/health`);
  console.log(`   NVIDIA NIM:   http://localhost:${PORT}/nvidia/v1/chat/completions`);
  console.log(`   ElectronHub:  http://localhost:${PORT}/ehub/v1/chat/completions`);
  console.log(`   Default:      http://localhost:${PORT}/v1/chat/completions (uses NVIDIA)`);
  console.log(`   Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
