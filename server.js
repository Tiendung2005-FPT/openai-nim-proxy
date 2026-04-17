// server.js - Multi-Provider API Proxy (NVIDIA NIM + ElectronHub)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const BODY_LIMIT = process.env.BODY_LIMIT || '50mb';   // ← Changeable via Render env var

app.use(cors());
app.use(express.json({ limit: BODY_LIMIT }));           // ← FIXED
app.use(express.urlencoded({ limit: BODY_LIMIT, extended: true }));

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
const ENABLE_THINKING_MODE = true; 

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
  'glm-5': 'z-ai/glm5',
  'kimi-k2.5': 'moonshotai/kimi-k2.5'
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
// HELPERS & DECONSTRUCTION
// =============================================================================

function deconstructPreset(messages) {
    if (!messages || !Array.isArray(messages)) return {};

    const systemMsg = messages.find(m => m.role === 'system')?.content || '';

    // 1. Extract Dynamic CharName and CharPersona
    const personaRegex = /<([^']+)'s Persona>(.*?)<\/\1's Persona>/s;
    const personaMatch = systemMsg.match(personaRegex);
    const CharName = personaMatch ? personaMatch[1].trim() : 'Character';
    const CharPersona = personaMatch ? personaMatch[2].trim() : '';

    // 2. Extract standard tags helper
    const extractTag = (tag, text) => {
        const regex = new RegExp(`<${tag}>(.*?)</${tag}>`, 's');
        const match = text.match(regex);
        return match ? match[1].trim() : '';
    };

    const Scenario = extractTag("Scenario", systemMsg);
    const UserPersona = extractTag("UserPersona", systemMsg);
    const ExampleDialogs = extractTag("example_dialogs", systemMsg);
    const Summary = extractTag("summary", systemMsg); // <--- New Extraction

    // 3. Extract "Extra" (Everything else in systemMsg)
    let extraText = systemMsg;
    extraText = extraText.replace(personaRegex, '');
    extraText = extraText.replace(/<Scenario>.*?<\/Scenario>/gs, '');
    extraText = extraText.replace(/<UserPersona>.*?<\/UserPersona>/gs, '');
    extraText = extraText.replace(/<example_dialogs>.*?<\/example_dialogs>/gs, '');
    extraText = extraText.replace(/<summary>.*?<\/summary>/gs, ''); // <--- Strip Summary from Extra
    
    const Extra = extraText.trim();

    // 4. Extract User Name from last message
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    const nameMatch = lastUserMsg.match(/^([^:]+):/);
    const UserName = nameMatch ? nameMatch[1].trim() : 'User';

    // 5. Extract and Split History
    const markerIndex = messages.findIndex(m => m.content === '.' && m.role === 'user');
    const History = markerIndex !== -1 ? messages.slice(markerIndex + 1) : [];
    const HistoryBeforeLast = History.length > 1 ? History.slice(0, -1) : [];
    const HistoryLast = History.length > 0 ? History.slice(-1) : [];

    return { 
        CharName, 
        CharPersona, 
        Scenario, 
        UserPersona, 
        ExampleDialogs, 
        Summary, // <--- Added to return object
        Extra, 
        History, 
        HistoryBeforeLast, 
        HistoryLast, 
        UserName 
    };
}

function detectProvider(path) {
  if (path.startsWith('/nvidia/')) return 'nvidia';
  if (path.startsWith('/ehub/')) return 'ehub';
  return 'nvidia';
}

function processThinkingTag(requestBody) {
  let shouldThink = true; 
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

function buildThinkingKwargs() {
  return { clear_thinking: true, enable_thinking: true };
}

// =============================================================================
// TEMPLATE PARSER
// =============================================================================

const fillTemplate = (templateStr, extracted) => {
    if (!templateStr || typeof templateStr !== 'string') return null;

    // 1. Helper for safe string escaping inside JSON
    const safeStr = (str) => str ? JSON.stringify(str).slice(1, -1) : "";

    // 2. Replace simple text variables first
     let processedStr = templateStr
        .replace(/%CHARPERSONA%/g, safeStr(extracted.CharPersona))
        .replace(/%SCENARIO%/g, safeStr(extracted.Scenario))
        .replace(/%USERPERSONA%/g, safeStr(extracted.UserPersona))
        .replace(/%EXAMPLEDIALOGS%/g, safeStr(extracted.ExampleDialogs))
        .replace(/%SUMMARY%/g, safeStr(extracted.Summary)) // <--- New Replacement
        .replace(/%EXTRA%/g, safeStr(extracted.Extra))
        .replace(/{{user}}/g, safeStr(extracted.UserName))
        .replace(/{{char}}/g, safeStr(extracted.CharName));

    // 2. Pre-parsing: Ensure all History tags are wrapped in quotes so JSON.parse doesn't crash
    // This regex matches %HISTORY%, %HISTORY_BEFORE_LAST%, or %HISTORY_LAST%
    processedStr = processedStr.replace(/(?<!")(%HISTORY%|%HISTORY_BEFORE_LAST%|%HISTORY_LAST%)(?!")/g, '"$1"');

    try {
        const tempArray = JSON.parse(processedStr);
        const finalMessages = [];

        // 3. Post-parsing: Flatten the arrays back into the message list
        tempArray.forEach(item => {
            if (item === "%HISTORY%") {
                finalMessages.push(...extracted.History);
            } 
            else if (item === "%HISTORY_BEFORE_LAST%") {
                finalMessages.push(...extracted.HistoryBeforeLast);
            } 
            else if (item === "%HISTORY_LAST%") {
                finalMessages.push(...extracted.HistoryLast);
            } 
            else {
                finalMessages.push(item);
            }
        });

        return finalMessages;
    } catch (error) {
        console.error("--- JSON PARSE ERROR ---");
        console.error(error.message);

        // Try to find the position from the error message
        const posMatch = error.message.match(/at position (\d+)/);
        if (posMatch) {
            const pos = parseInt(posMatch[1]);
            const start = Math.max(0, pos - 50);
            const end = Math.min(processedStr.length, pos + 50);
            
            console.error("Context around error:");
            console.error("...");
            console.error(processedStr.substring(start, end));
            console.error(" ".repeat(pos - start) + "^--- ERROR IS HERE");
            console.error("...");
            
            // Log the exact character code
            console.error(`Character at position ${pos}: "${processedStr[pos]}" (Code: ${processedStr.charCodeAt(pos)})`);
        }
        return null;
    } // End of Catch
}; 

// =============================================================================
// HANDLERS (NVIDIA & EHUB)
// =============================================================================

async function handleNvidiaCompletion(req, res) {
  const { model, temperature, max_tokens, stream } = req.body || {};
  const { shouldThink, cleanedMessages } = processThinkingTag(req.body);

  let nimModel = NVIDIA_MODEL_MAPPING[model] || model;
  
  const nimRequest = {
    model: nimModel,
    messages: cleanedMessages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.6,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 9999,
    ...(shouldThink ? { chat_template_kwargs: buildThinkingKwargs() } : {}),
    stream: !!stream
  };

  try {
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        response.data.pipe(res);
    } else {
        res.json(response.data);
    }
  } catch (err) {
    console.error('Backend error details:', {
    status: err.response?.status,
    data: err.response?.data,
    message: err.message
  });
  const status = err.response?.status || 500;
  res.status(status).json({ 
    error: err.message,
    details: err.response?.data || null 
  });
  }
}

async function handleEhubCompletion(req, res) {
  const { model, temperature, max_tokens, stream } = req.body || {};
  const { shouldThink, cleanedMessages } = processThinkingTag(req.body);
  const ehubModel = EHUB_MODEL_MAPPING[model] || model;

  const ehubRequest = {
    model: ehubModel,
    messages: cleanedMessages || [],
    temperature: typeof temperature === 'number' ? temperature : 0.7,
    max_tokens: typeof max_tokens === 'number' ? max_tokens : 9999,
    ...(shouldThink ? { chat_template_kwargs: buildThinkingKwargs() } : {}),
    stream: !!stream
  };

  try {
    const response = await axios.post(`${EHUB_API_BASE}/chat/completions`, ehubRequest, {
      headers: { 'Authorization': `Bearer ${EHUB_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });
    if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        response.data.pipe(res);
    } else {
        res.json(response.data);
    }
  } catch (err) {
    console.error('Backend error details:', {
    status: err.response?.status,
    data: err.response?.data,
    message: err.message
  });
  const status = err.response?.status || 500;
  res.status(status).json({ 
    error: err.message,
    details: err.response?.data || null 
  });
  }
}

// =============================================================================
// UNIFIED ENDPOINT (WITH PRESET SUPPORT)
// =============================================================================

const chatEndpoints = [
    '/v1/chat/completions',
    '/nvidia/v1/chat/completions',
    '/ehub/v1/chat/completions',
    // Preset Endpoints
    '/v1/chat/completions/preset/:presetName',
    '/nvidia/v1/chat/completions/preset/:presetName',
    '/ehub/v1/chat/completions/preset/:presetName'
];

app.post(chatEndpoints, async (req, res) => {
  try {
    const { presetName } = req.params;
    
    // If it's a preset endpoint, process the preset logic
    if (presetName) {
        const extracted = deconstructPreset(req.body.messages);
        console.log(`--- Loading Preset: [${presetName}] ---`);

        // Dynamically grab the template from ENV using the preset name
        // Ex: presetName 'megumin' -> looks for process.env.MEGUMIN_PRESET
        const envVarName = `${presetName.toUpperCase()}_PRESET`;
        const template = process.env[envVarName];

        if (!template) {
            return res.status(404).json({ error: `Template for preset '${presetName}' not found in environment.` });
        }

        // Fill the template and parse it into an array
        const parsedPresetMessages = fillTemplate(template, extracted);

        if (!parsedPresetMessages) {
            return res.status(500).json({ error: `Failed to parse filled template for '${presetName}'. Check server logs.` });
        }

        // Overwrite the incoming messages with the new template
        req.body.messages = parsedPresetMessages;
        console.log(`Successfully mapped variables to template.`);
    }


    const provider = detectProvider(req.path);
    if (provider === 'ehub') {
      await handleEhubCompletion(req, res);
    } else {
      await handleNvidiaCompletion(req, res);
    }
  } catch (err) {
    console.error('Backend error details:', {
    status: err.response?.status,
    data: err.response?.data,
    message: err.message
  });
  const status = err.response?.status || 500;
  res.status(status).json({ 
    error: err.message,
    details: err.response?.data || null 
  });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
  console.log(`📍 Normal Example: http://localhost:${PORT}/v1/chat/completions`);
  console.log(`📍 Preset Example: http://localhost:${PORT}/v1/chat/completions/preset/megumin`);
});
