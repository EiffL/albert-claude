/**
 * Anthropic <-> OpenAI translation logic.
 *
 * Pure functions — no I/O, no side effects.
 * Direct port of the tested Python implementation.
 */

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
            id: block.id || `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`,
            type: 'function',
            function: {
              name: block.name || '',
              arguments: JSON.stringify(block.input || {}),
            },
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
        // thinking blocks silently skipped
      }
    }

    // Build main message
    const openaiMsg = { role };
    if (contentParts.length === 1 && contentParts[0].type === 'text') {
      openaiMsg.content = contentParts[0].text;
    } else if (contentParts.length > 0) {
      openaiMsg.content = contentParts;
    } else {
      openaiMsg.content = '';
    }
    if (toolCalls.length > 0) openaiMsg.tool_calls = toolCalls;
    messages.push(openaiMsg);

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
      id: tc.id || `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`,
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
