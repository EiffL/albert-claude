/**
 * HTTP proxy server: accepts Anthropic Messages API, forwards as OpenAI Chat Completions.
 *
 * Uses only Node.js builtins (http, fetch).
 */

import http from 'node:http';
import { translateRequest, translateResponse, fixTextToolCalls } from './translate.js';
import { StreamTranslator, formatSSE } from './stream.js';
import {
  partitionTools,
  serverToolToFunction,
  serverToolLoop,
  injectServerToolBlocks,
} from './server-tools.js';

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

const ERROR_TYPE_MAP = {
  400: 'invalid_request_error',
  401: 'authentication_error',
  403: 'permission_error',
  404: 'not_found_error',
  429: 'rate_limit_error',
  529: 'overloaded_error',
};

function errorResponse(res, status, message) {
  const errorType = ERROR_TYPE_MAP[status] || 'api_error';
  const body = JSON.stringify({
    type: 'error',
    error: { type: errorType, message },
  });
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(body);
}

// ---------------------------------------------------------------------------
// Debug logging
// ---------------------------------------------------------------------------

function truncate(text, limit = 200) {
  if (!text || text.length <= limit) return text;
  return text.slice(0, limit) + `... (${text.length} chars)`;
}

function summarizeMessages(messages) {
  return messages.map(msg => {
    const s = { role: msg.role };
    if (typeof msg.content === 'string') {
      s.content = truncate(msg.content);
    } else if (Array.isArray(msg.content)) {
      s.content = msg.content.map(b => {
        if (b.type === 'text') return { type: 'text', text: truncate(b.text) };
        if (b.type === 'image' || b.type === 'image_url') return { type: b.type, data: '[omitted]' };
        if (b.type === 'tool_use') return { type: 'tool_use', name: b.name };
        if (b.type === 'tool_result') return { type: 'tool_result', tool_use_id: b.tool_use_id };
        return { type: b.type };
      });
    }
    if (msg.tool_calls) s.tool_calls = msg.tool_calls.map(tc => ({ name: tc.function?.name }));
    if (msg.tool_call_id) s.tool_call_id = msg.tool_call_id;
    return s;
  });
}

function debugRequest(log, label, payload) {
  const summary = { ...payload };
  if (summary.messages) summary.messages = summarizeMessages(summary.messages);
  if (summary.tools) summary.tools = summary.tools.map(t => t.name || t.function?.name);
  if (typeof summary.system === 'string') summary.system = truncate(summary.system);
  log(`${label}:\n${JSON.stringify(summary, null, 2)}`);
}

function debugResponse(log, label, payload) {
  const summary = { ...payload };
  if (Array.isArray(summary.content)) {
    summary.content = summary.content.map(b => {
      if (b.type === 'text') return { type: 'text', text: truncate(b.text) };
      if (b.type === 'tool_use') return { type: 'tool_use', name: b.name, id: b.id };
      return b;
    });
  }
  log(`${label}:\n${JSON.stringify(summary, null, 2)}`);
}

// ---------------------------------------------------------------------------
// Request body reader
// ---------------------------------------------------------------------------

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks).toString()));
    req.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// SSE emission from non-streaming response
// ---------------------------------------------------------------------------

/**
 * Convert an Anthropic Messages response to an array of SSE event strings.
 * Used when server tools forced a non-streaming call but the client expects SSE.
 */
function emitResponseAsSSE(anthropicResponse) {
  const events = [];

  // message_start (with empty content — blocks come separately)
  events.push(formatSSE('message_start', {
    type: 'message_start',
    message: {
      ...anthropicResponse,
      content: [],
      stop_reason: null,
    },
  }));
  events.push(formatSSE('ping', { type: 'ping' }));

  // Content blocks
  for (let i = 0; i < (anthropicResponse.content || []).length; i++) {
    const block = anthropicResponse.content[i];

    if (block.type === 'text') {
      events.push(formatSSE('content_block_start', {
        type: 'content_block_start',
        index: i,
        content_block: { type: 'text', text: '' },
      }));
      events.push(formatSSE('content_block_delta', {
        type: 'content_block_delta',
        index: i,
        delta: { type: 'text_delta', text: block.text },
      }));
    } else if (block.type === 'tool_use') {
      events.push(formatSSE('content_block_start', {
        type: 'content_block_start',
        index: i,
        content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} },
      }));
      events.push(formatSSE('content_block_delta', {
        type: 'content_block_delta',
        index: i,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) },
      }));
    } else {
      // server_tool_use, *_tool_result blocks — emit as-is
      events.push(formatSSE('content_block_start', {
        type: 'content_block_start',
        index: i,
        content_block: block,
      }));
    }

    events.push(formatSSE('content_block_stop', {
      type: 'content_block_stop',
      index: i,
    }));
  }

  // message_delta + message_stop
  events.push(formatSSE('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicResponse.stop_reason, stop_sequence: null },
    usage: { output_tokens: anthropicResponse.usage?.output_tokens || 0 },
  }));
  events.push(formatSSE('message_stop', { type: 'message_stop' }));

  return events;
}

// ---------------------------------------------------------------------------
// Proxy server
// ---------------------------------------------------------------------------

/**
 * Create and start the proxy server.
 *
 * @param {object} opts
 * @param {number} opts.port - Port to listen on (0 for random)
 * @param {string} opts.baseUrl - OpenAI-compatible API base URL
 * @param {string} opts.apiKey - Bearer token for upstream
 * @param {string} opts.model - Model name to send upstream
 * @param {boolean} opts.debug - Enable debug logging
 * @returns {Promise<{server: http.Server, port: number}>}
 */
export function startProxy({ port = 0, baseUrl, apiKey, model, debug = false, braveApiKey } = {}) {
  const info = debug ? (...args) => console.error('[proxy]', ...args) : () => {};
  const dbg = debug ? (...args) => console.error('[proxy:debug]', ...args) : () => {};

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const path = url.pathname;

    // Health check
    if ((req.method === 'GET' || req.method === 'HEAD') && (path === '/' || path === '/health')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
      return;
    }

    // Count tokens stub
    if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
      try {
        const body = JSON.parse(await readBody(req));
        const text = JSON.stringify(body.messages || []);
        let system = body.system || '';
        if (Array.isArray(system)) system = system.map(b => b.text || '').join(' ');
        const estimated = Math.max(1, Math.floor((text.length + String(system).length) / 4));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ input_tokens: estimated }));
      } catch {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ input_tokens: 0 }));
      }
      return;
    }

    // Main proxy endpoint
    if (req.method === 'POST' && path.startsWith('/v1/messages')) {
      let payload;
      try {
        payload = JSON.parse(await readBody(req));
      } catch {
        return errorResponse(res, 400, 'Invalid JSON body');
      }

      if (!payload.messages) {
        return errorResponse(res, 400, 'Missing required field: messages');
      }

      if (debug) debugRequest(dbg, 'Anthropic request', payload);

      // --- Server tool handling ---
      // Partition tools into server (web_search, web_fetch, code_execution)
      // and client (Read, Write, Bash, etc.) tools.
      const { server: serverTools, client: clientTools } = partitionTools(payload.tools);
      payload.tools = clientTools;

      const activeServerTools = [];
      if (serverTools.length > 0) {
        for (const st of serverTools) {
          const fn = serverToolToFunction(st);
          if (fn) activeServerTools.push(fn);
        }
        if (activeServerTools.length > 0) {
          info(`Server tools active: ${activeServerTools.map(t => t.function.name).join(', ')}`);
        }
      }

      const openaiPayload = translateRequest(payload, model);
      const isStream = openaiPayload.stream;

      // Inject synthetic function tools for active server tools
      if (activeServerTools.length > 0) {
        openaiPayload.tools = [...(openaiPayload.tools || []), ...activeServerTools];
      }

      info(`stream=${isStream} messages=${openaiPayload.messages.length} tools=${(openaiPayload.tools || []).length}`);

      if (debug) debugRequest(dbg, 'OpenAI request', openaiPayload);

      const upstreamHeaders = { 'Content-Type': 'application/json' };
      if (apiKey) upstreamHeaders['Authorization'] = `Bearer ${apiKey}`;

      /** Call the upstream backend (used by both direct calls and the server tool loop). */
      async function callUpstream(body) {
        const resp = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: upstreamHeaders,
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(300_000),
        });
        if (!resp.ok) {
          const errText = await resp.text().catch(() => '');
          throw Object.assign(new Error(`Backend returned ${resp.status}`), {
            status: resp.status,
            detail: errText,
          });
        }
        return resp;
      }

      // When server tools are active, force non-streaming for the initial call
      // so we can peek at the response and run the multi-turn loop if needed.
      const hasActiveServerTools = activeServerTools.length > 0;
      if (hasActiveServerTools && isStream) {
        openaiPayload.stream = false;
        delete openaiPayload.stream_options;
      }

      let upstream;
      try {
        upstream = await callUpstream(openaiPayload);
      } catch (err) {
        if (err.status) {
          info(`Backend returned ${err.status}: ${(err.detail || '').slice(0, 200)}`);
          return errorResponse(res, err.status, `Backend returned ${err.status}`);
        }
        info(`Backend connection failed: ${err.message}`);
        return errorResponse(res, 502, `Backend connection failed: ${err.message}`);
      }

      // --- Non-streaming (or forced non-streaming for server tools) ---
      if (!isStream || hasActiveServerTools) {
        const data = await upstream.json();
        if (data.error) {
          return errorResponse(res, 500, data.error.message || 'Unknown error');
        }
        fixTextToolCalls(data);
        if (debug) debugResponse(dbg, 'OpenAI response', data);

        // Run server tool loop if the model called any server tools
        let finalData = data;
        let serverToolBlocks = [];

        if (hasActiveServerTools) {
          const serverToolNames = new Set(activeServerTools.map(t => t.function.name));
          const toolCalls = data.choices?.[0]?.message?.tool_calls || [];
          const hasServerCalls = toolCalls.some(tc => serverToolNames.has(tc.function?.name));

          if (hasServerCalls) {
            try {
              const loopResult = await serverToolLoop({
                openaiPayload,
                firstResponse: data,
                callUpstream: async (body) => {
                  const r = await callUpstream(body);
                  const json = await r.json();
                  fixTextToolCalls(json);
                  return json;
                },
                braveApiKey,
                log: info,
              });
              finalData = loopResult.response;
              serverToolBlocks = loopResult.serverToolBlocks;
            } catch (err) {
              info(`Server tool loop failed: ${err.message}`);
              // Fall through with original response
            }
          }
        }

        let result = translateResponse(finalData, model);
        result = injectServerToolBlocks(result, serverToolBlocks);
        if (debug) debugResponse(dbg, 'Anthropic response', result);

        // If the client originally requested streaming, emit as SSE events
        if (isStream) {
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
          });
          for (const evt of emitResponseAsSSE(result)) {
            res.write(evt);
          }
          res.end();
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        }
        return;
      }

      // --- Streaming (no server tools) ---
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });

      const translator = new StreamTranslator(model);
      let chunkCount = 0;

      try {
        const reader = upstream.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buf += decoder.decode(value, { stream: true });

          while (buf.includes('\n')) {
            const nlIdx = buf.indexOf('\n');
            const line = buf.slice(0, nlIdx).trim();
            buf = buf.slice(nlIdx + 1);

            if (!line || !line.startsWith('data:')) continue;
            const dataStr = line.slice(5).trim();

            if (dataStr === '[DONE]') {
              dbg(`Stream [DONE] after ${chunkCount} chunks`);
              for (const evt of translator.finish()) {
                res.write(evt);
              }
              res.end();
              return;
            }

            let parsed;
            try {
              parsed = JSON.parse(dataStr);
            } catch {
              continue;
            }

            if (parsed.error) {
              info(`Backend stream error: ${JSON.stringify(parsed.error)}`);
              continue;
            }

            chunkCount++;
            if (chunkCount === 1) {
              const delta = (parsed.choices || [{}])[0].delta || {};
              dbg(`Stream first chunk: ${JSON.stringify(delta).slice(0, 300)}`);
            }

            for (const evt of translator.processChunk(parsed)) {
              res.write(evt);
            }
          }
        }

        // Stream ended without [DONE] — still close cleanly
        for (const evt of translator.finish()) {
          res.write(evt);
        }
      } catch (err) {
        if (err.code === 'ERR_STREAM_PREMATURE_CLOSE' || err.message?.includes('abort')) {
          dbg(`Client disconnected during stream`);
        } else {
          info(`Stream error: ${err.message}`);
        }
      }

      try { res.end(); } catch { /* client gone */ }
      return;
    }

    // Fallback
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  });

  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => {
      const actualPort = server.address().port;
      resolve({ server, port: actualPort });
    });
  });
}
