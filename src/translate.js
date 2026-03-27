/**
 * Anthropic <-> OpenAI translation logic.
 *
 * Pure functions — no I/O, no side effects.
 * Direct port of the tested Python implementation.
 */

// ---------------------------------------------------------------------------
// Text-based tool call parsing
// ---------------------------------------------------------------------------
// Some models (Mistral, Qwen, etc.) emit tool calls as text instead of using
// the OpenAI function calling API. These parsers detect and extract them.
// Approach inspired by Ollama: tag detection + balanced-brace JSON extraction.

/** Extract a balanced JSON object from text starting at startIdx. */
function extractJsonObject(text, startIdx) {
  if (startIdx >= text.length || text[startIdx] !== '{') return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = startIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) { escape = false; continue; }
    if (ch === '\\' && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return { json: text.slice(startIdx, i + 1), end: i + 1 };
    }
  }
  return null;
}

/** Generate a 9-character alphanumeric ID compatible with vLLM/Mistral backends. */
export function makeToolCallId() {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const bytes = new Uint8Array(9);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, b => chars[b % chars.length]).join('');
}

/**
 * Recursively search a parsed JSON object for tool call name + arguments,
 * following Ollama's findObject strategy.
 */
function findToolCallInObject(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
  const name = obj.name || obj.function?.name;
  if (name) {
    let args = obj.arguments ?? obj.parameters ?? obj.function?.arguments ?? obj.function?.parameters;
    if (typeof args === 'string') {
      try { args = JSON.parse(args); } catch { /* keep as string */ }
    }
    if (args !== undefined) return { name, arguments: args };
  }
  for (const val of Object.values(obj)) {
    if (typeof val === 'object' && val !== null) {
      const found = findToolCallInObject(val);
      if (found) return found;
    }
  }
  return null;
}

/** Parse Mistral [TOOL_CALLS] format. */
function parseMistralToolCalls(text) {
  if (!text.includes('[TOOL_CALLS]')) return { text, toolCalls: [] };

  const toolCalls = [];
  const parts = text.split('[TOOL_CALLS]');
  const prefix = parts[0];
  const leftover = [];

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i].trimStart();

    // Try JSON array: [{"name": ..., "arguments": ...}, ...]
    if (part.startsWith('[')) {
      const bracketEnd = part.indexOf(']');
      if (bracketEnd >= 0) {
        try {
          const arr = JSON.parse(part.slice(0, bracketEnd + 1));
          if (Array.isArray(arr)) {
            for (const item of arr) {
              const found = findToolCallInObject(item);
              if (found) {
                toolCalls.push({
                  id: item.id || makeToolCallId(),
                  type: 'function',
                  function: {
                    name: found.name,
                    arguments: typeof found.arguments === 'string' ? found.arguments : JSON.stringify(found.arguments ?? {}),
                  },
                });
              }
            }
            const after = part.slice(bracketEnd + 1).trim();
            if (after) leftover.push(after);
            continue;
          }
        } catch { /* fall through to Name{json} parsing */ }
      }
    }

    // Try Name{json} format
    const nameMatch = part.match(/^(\w+)\s*/);
    if (nameMatch) {
      const name = nameMatch[1];
      const afterName = part.slice(nameMatch[0].length);
      const jsonResult = extractJsonObject(afterName, 0);
      if (jsonResult) {
        try {
          JSON.parse(jsonResult.json);
          toolCalls.push({
            id: makeToolCallId(),
            type: 'function',
            function: { name, arguments: jsonResult.json },
          });
          const after = afterName.slice(jsonResult.end).trim();
          if (after) leftover.push(after);
          continue;
        } catch { /* fall through */ }
      }
    }
    leftover.push(parts[i]);
  }

  const remaining = [prefix, ...leftover].join('').trim();
  return { text: remaining, toolCalls };
}

/** Parse <tool_call>...</tool_call> tags (Hermes JSON or Qwen3 XML). */
function parseToolCallTags(text) {
  if (!text.includes('<tool_call>') && !text.includes('<function=')) {
    return { text, toolCalls: [] };
  }

  const toolCalls = [];
  let remaining = text;

  // <tool_call>...</tool_call> blocks
  const tagRe = /<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g;
  let tagMatch;
  const matched = [];
  while ((tagMatch = tagRe.exec(remaining)) !== null) {
    const body = tagMatch[1].trim();
    matched.push(tagMatch[0]);

    // Try JSON (Hermes format)
    const jsonStart = body.indexOf('{');
    if (jsonStart >= 0) {
      const jsonResult = extractJsonObject(body, jsonStart);
      if (jsonResult) {
        try {
          const parsed = JSON.parse(jsonResult.json);
          const found = findToolCallInObject(parsed);
          if (found) {
            toolCalls.push({
              id: makeToolCallId(),
              type: 'function',
              function: {
                name: found.name,
                arguments: typeof found.arguments === 'string' ? found.arguments : JSON.stringify(found.arguments ?? {}),
              },
            });
            continue;
          }
        } catch { /* fall through to XML parsing */ }
      }
    }

    // Try Qwen3 XML inside <tool_call>
    const fnCalls = parseQwenFunctionCalls(body);
    toolCalls.push(...fnCalls);
  }
  for (const m of matched) remaining = remaining.replace(m, '');

  // Bare <function=Name>...</function> (without <tool_call> wrapper)
  if (remaining.includes('<function=') && toolCalls.length === 0) {
    const fnCalls = parseQwenFunctionCalls(remaining);
    if (fnCalls.length > 0) {
      toolCalls.push(...fnCalls);
      remaining = remaining.replace(/<function=\w+>[\s\S]*?(?:<\/function>|$)/g, '');
    }
  }

  return { text: remaining.trim(), toolCalls };
}

/** Parse Qwen3 <function=Name><parameter=key>value</parameter></function>. */
function parseQwenFunctionCalls(text) {
  const toolCalls = [];
  const fnRe = /<function=(\w+)>([\s\S]*?)(?:<\/function>|$)/g;
  let fnMatch;
  while ((fnMatch = fnRe.exec(text)) !== null) {
    const name = fnMatch[1];
    const body = fnMatch[2];

    // Try JSON body first
    const jsonStart = body.indexOf('{');
    if (jsonStart >= 0) {
      const jsonResult = extractJsonObject(body, jsonStart);
      if (jsonResult) {
        try {
          JSON.parse(jsonResult.json);
          toolCalls.push({
            id: makeToolCallId(),
            type: 'function',
            function: { name, arguments: jsonResult.json },
          });
          continue;
        } catch { /* fall through */ }
      }
    }

    // Parse <parameter=key>value</parameter> (or value terminated by next param/end)
    const params = {};
    const paramRe = /<parameter=(\w+)>([\s\S]*?)(?:<\/parameter>|(?=<parameter=)|(?=<\/function>)|$)/g;
    let pm;
    while ((pm = paramRe.exec(body)) !== null) {
      params[pm[1]] = pm[2].trim();
    }
    if (Object.keys(params).length > 0) {
      toolCalls.push({
        id: makeToolCallId(),
        type: 'function',
        function: { name, arguments: JSON.stringify(params) },
      });
    }
  }
  return toolCalls;
}

/**
 * Parse tool calls embedded in text by models that don't use the OpenAI
 * function calling API. Returns { text, toolCalls } in OpenAI format.
 *
 * Handles Mistral [TOOL_CALLS], Qwen3 <function=>, Hermes <tool_call>.
 */
export function parseTextToolCalls(text) {
  if (!text || typeof text !== 'string') return { text: text || '', toolCalls: [] };

  // Try tag-based formats first (more specific)
  let result = parseToolCallTags(text);
  if (result.toolCalls.length > 0) return result;

  // Try Mistral format
  result = parseMistralToolCalls(text);
  if (result.toolCalls.length > 0) return result;

  return { text, toolCalls: [] };
}

/**
 * Fix an OpenAI response that has tool calls embedded in text content.
 * Only applies when message.content has text but message.tool_calls is empty.
 */
export function fixTextToolCalls(openaiData) {
  const choice = (openaiData.choices || [])[0];
  if (!choice?.message) return openaiData;

  const message = choice.message;
  if (!message.content || (message.tool_calls && message.tool_calls.length > 0)) {
    return openaiData;
  }

  const { text, toolCalls } = parseTextToolCalls(message.content);
  if (toolCalls.length === 0) return openaiData;

  message.content = text || null;
  message.tool_calls = toolCalls;
  choice.finish_reason = 'tool_calls';

  return openaiData;
}

/** Prefixes used to detect text-based tool calls during streaming. */
export const TOOL_CALL_TEXT_PREFIXES = ['[TOOL_CALLS]', '<tool_call>', '<function='];

// ---------------------------------------------------------------------------
// Schema utilities
// ---------------------------------------------------------------------------

/** Remove `format: "uri"` from JSON schemas (some backends reject it). */
export function stripUriFormat(schema) {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) return schema;
  if (schema.type === 'string' && schema.format === 'uri') {
    const { format, ...rest } = schema;
    return rest;
  }
  const result = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === 'properties' && typeof value === 'object' && value !== null) {
      result[key] = {};
      for (const [k, v] of Object.entries(value)) {
        result[key][k] = stripUriFormat(v);
      }
    } else if ((key === 'items' || key === 'additionalProperties') && typeof value === 'object') {
      result[key] = stripUriFormat(value);
    } else if (['anyOf', 'allOf', 'oneOf'].includes(key) && Array.isArray(value)) {
      result[key] = value.map(item => stripUriFormat(item));
    } else {
      result[key] = value;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Request translation: Anthropic -> OpenAI
// ---------------------------------------------------------------------------

function translateMessages(payload) {
  const messages = [];

  // System messages
  const system = payload.system;
  if (typeof system === 'string' && system) {
    messages.push({ role: 'system', content: system });
  } else if (Array.isArray(system)) {
    for (const block of system) {
      const text = block.text || block.content || '';
      if (text) messages.push({ role: 'system', content: text });
    }
  }

  // Conversation messages
  for (const msg of (payload.messages || [])) {
    const role = msg.role || 'user';
    const content = msg.content;

    const contentParts = [];
    const toolCalls = [];
    const toolResults = [];

    if (typeof content === 'string') {
      contentParts.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        const btype = block.type;
        if (btype === 'text') {
          contentParts.push({ type: 'text', text: block.text || '' });
        } else if (btype === 'image') {
          const source = block.source || {};
          if (source.type === 'base64') {
            const media = source.media_type || 'image/png';
            contentParts.push({
              type: 'image_url',
              image_url: { url: `data:${media};base64,${source.data || ''}` },
            });
          }
        } else if (btype === 'tool_use') {
          toolCalls.push({
            id: block.id || makeToolCallId(),
            type: 'function',
            function: {
              name: block.name || '',
              arguments: JSON.stringify(block.input || {}),
            },
          });
        } else if (btype === 'server_tool_use') {
          // Server tool use in history — treat like a regular tool call
          toolCalls.push({
            id: block.id || makeToolCallId(),
            type: 'function',
            function: {
              name: block.name || '',
              arguments: JSON.stringify(block.input || {}),
            },
          });
        } else if (btype === 'web_search_tool_result') {
          // Server tool result in history — convert to tool message
          let rc = '';
          const c = block.content;
          if (Array.isArray(c)) {
            rc = c.map(r => `${r.title || ''}: ${r.url || ''}\n${r.encrypted_content || r.description || ''}`).join('\n\n');
          } else if (c && c.type === 'web_search_tool_result_error') {
            rc = `Error: ${c.message || 'search failed'}`;
          } else {
            rc = JSON.stringify(c);
          }
          toolResults.push({
            role: 'tool',
            content: rc,
            tool_call_id: block.tool_use_id || '',
          });
        } else if (btype === 'web_fetch_tool_result') {
          let rc = '';
          const c = block.content;
          if (c && c.type === 'web_fetch_result') {
            rc = c.content?.source?.data || JSON.stringify(c);
          } else if (c && c.type === 'web_fetch_tool_result_error') {
            rc = `Error: ${c.message || 'fetch failed'}`;
          } else {
            rc = JSON.stringify(c);
          }
          toolResults.push({
            role: 'tool',
            content: rc,
            tool_call_id: block.tool_use_id || '',
          });
        } else if (btype === 'tool_result') {
          let rc = block.content || '';
          if (Array.isArray(rc)) {
            rc = rc
              .filter(b => typeof b === 'object' && b.type === 'text')
              .map(b => b.text || '')
              .join(' ');
          }
          if (block.is_error) rc = `Error: ${rc}`;
          toolResults.push({
            role: 'tool',
            content: String(rc),
            tool_call_id: block.tool_use_id || '',
          });
        }
        // thinking blocks, code_execution results silently skipped
      }
    }

    // Build main message — skip empty user messages that only contain tool_result
    // blocks, since the results are emitted as separate tool messages below.
    const openaiMsg = { role };
    if (contentParts.length === 1 && contentParts[0].type === 'text') {
      openaiMsg.content = contentParts[0].text;
    } else if (contentParts.length > 0) {
      openaiMsg.content = contentParts;
    } else {
      openaiMsg.content = '';
    }
    if (toolCalls.length > 0) openaiMsg.tool_calls = toolCalls;
    if (contentParts.length > 0 || toolCalls.length > 0 || toolResults.length === 0) {
      messages.push(openaiMsg);
    }

    // Tool results as separate messages
    messages.push(...toolResults);
  }

  return messages;
}

function translateTools(tools) {
  return (tools || [])
    .filter(t => t.name !== 'BatchTool')
    .map(t => ({
      type: 'function',
      function: {
        name: t.name || '',
        description: t.description || '',
        parameters: stripUriFormat(t.input_schema || {}),
      },
    }));
}

function translateToolChoice(tc) {
  if (!tc || typeof tc !== 'object') return undefined;
  if (tc.type === 'auto') return 'auto';
  if (tc.type === 'any') return 'required';
  if (tc.type === 'none') return 'none';
  if (tc.type === 'tool') return { type: 'function', function: { name: tc.name || '' } };
  return undefined;
}

/** Convert an Anthropic Messages request to OpenAI Chat Completions. */
export function translateRequest(payload, model) {
  const openaiPayload = {
    model,
    messages: translateMessages(payload),
    stream: payload.stream === true,
  };

  if (payload.max_tokens !== undefined) openaiPayload.max_tokens = payload.max_tokens;
  if (payload.temperature !== undefined) openaiPayload.temperature = payload.temperature;
  if (payload.top_p !== undefined) openaiPayload.top_p = payload.top_p;
  if (payload.stop_sequences) openaiPayload.stop = payload.stop_sequences;

  const tools = translateTools(payload.tools);
  if (tools.length > 0) openaiPayload.tools = tools;

  const tc = translateToolChoice(payload.tool_choice);
  if (tc !== undefined) openaiPayload.tool_choice = tc;

  if (openaiPayload.stream) {
    openaiPayload.stream_options = { include_usage: true };
  }

  return openaiPayload;
}

// ---------------------------------------------------------------------------
// Response translation: OpenAI -> Anthropic
// ---------------------------------------------------------------------------

const STOP_REASON_MAP = {
  stop: 'end_turn',
  length: 'max_tokens',
  tool_calls: 'tool_use',
};

export function mapStopReason(finishReason) {
  return STOP_REASON_MAP[finishReason] || 'end_turn';
}

export function makeMsgId() {
  return `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
}

/** Convert an OpenAI Chat Completion response to Anthropic Messages. */
export function translateResponse(openaiData, model) {
  const choice = (openaiData.choices || [{}])[0];
  const message = choice.message || {};
  const usage = openaiData.usage || {};

  const content = [];

  if (message.content) {
    content.push({ type: 'text', text: message.content });
  }

  for (const tc of (message.tool_calls || [])) {
    const func = tc.function || {};
    let inputData;
    try {
      inputData = JSON.parse(func.arguments || '{}');
    } catch {
      inputData = {};
    }
    content.push({
      type: 'tool_use',
      id: tc.id || makeToolCallId(),
      name: func.name || '',
      input: inputData,
    });
  }

  const msgId = (openaiData.id || '').replace('chatcmpl', 'msg') || makeMsgId();

  return {
    id: msgId,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: mapStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
    },
  };
}
