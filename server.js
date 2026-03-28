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
const EHUB_API_KEY = process.env.EHUB_API_KEY;

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

function processThinkingTag(requestBody) {
  let shouldThink = false;

  const bodyString = JSON.stringify(requestBody);
  if (bodyString.includes('<ENABLETHINKING>')) {
    shouldThink = true;
  }

  const messages = requestBody.messages;
  if (!messages || !Array.isArray(messages)) {
    return { shouldThink, cleanedMessages: messages };
  }

  const cleanedMessages = messages.map(msg => {
    if (msg.content && typeof msg.content === 'string' && msg.content.includes('<ENABLETHINKING>')) {
      return {
        ...msg,
        content: msg.content.replace(/<ENABLETHINKING>/g, '').trim()
      };
    }
    return msg;
  });

  return { shouldThink, cleanedMessages };
}

// Merge thinking config (core logic)
function buildThinkingConfig(req, shouldEnableThinking) {
  const clientConfig = req.body.chat_template_kwargs || null;

  if (!shouldEnableThinking && !clientConfig) return {};

  return {
    chat_template_kwargs: {
      enable_thinking: true,
      clear_thinking: false,
      ...(clientConfig || {})
    }
  };
}

// =============================================================================
// HEALTH
// =============================================================================
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// =============================================================================
// NVIDIA HANDLER
// =============================================================================
async function handleNvidiaCompletion(req, res) {
  const { model, temperature, max_tokens, stream } = req.body || {};

  const { shouldThink: tagDetected, cleanedMessages } = processThinkingTag(req.body);

  let nimModel = NVIDIA_MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct';

  const shouldEnableThinking = ENABLE_THINKING_MODE || tagDetected;

  const nimRequest = {
    model: nimModel,
    messages: cleanedMessages || [],
    temperature: temperature ?? 0.6,
    max_tokens: max_tokens ?? 1024,
    ...buildThinkingConfig(req, shouldEnableThinking),
    stream: !!stream
  };

  try {
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      response.data.pipe(res);
    } else {
      const data = response.data;

      const choices = data.choices.map(choice => {
        let content = choice.message.content || '';

        if (SHOW_REASONING && choice.message.reasoning_content) {
          content =
            `<think>\n${choice.message.reasoning_content}\n</think>\n\n` +
            content;
        }

        return {
          index: choice.index,
          message: {
            role: 'assistant',
            content
          }
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        choices
      });
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}

// =============================================================================
// EHUB HANDLER
// =============================================================================
async function handleEhubCompletion(req, res) {
  const { model, temperature, max_tokens, stream } = req.body || {};

  const { shouldThink: tagDetected, cleanedMessages } = processThinkingTag(req.body);

  const ehubModel = EHUB_MODEL_MAPPING[model] || model;

  const ehubRequest = {
    model: ehubModel,
    messages: cleanedMessages || [],
    temperature: temperature ?? 0.7,
    max_tokens: max_tokens ?? 1024,
    ...buildThinkingConfig(req, tagDetected),
    stream: !!stream
  };

  try {
    const response = await axios.post(
      `${EHUB_API_BASE}/chat/completions`,
      ehubRequest,
      {
        headers: {
          Authorization: `Bearer ${EHUB_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    if (stream) {
      response.data.pipe(res);
    } else {
      res.json(response.data);
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}

// =============================================================================
// ROUTES
// =============================================================================
app.post('/v1/chat/completions', async (req, res) => {
  const provider = detectProvider(req.path);
  if (provider === 'ehub') {
    await handleEhubCompletion(req, res);
  } else {
    await handleNvidiaCompletion(req, res);
  }
});

// =============================================================================
// START
// =============================================================================
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
