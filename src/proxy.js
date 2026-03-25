/**
 * HTTP proxy server: accepts Anthropic Messages API, forwards as OpenAI Chat Completions.
 *
 * Uses only Node.js builtins (http, fetch).
 */

import http from 'node:http';
import { translateRequest, translateResponse } from './translate.js';
import { StreamTranslator, formatSSE } from './stream.js';

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
export function startProxy({ port = 0, baseUrl, apiKey, model, debug = false }) {
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

      const openaiPayload = translateRequest(payload, model);
      const isStream = openaiPayload.stream;

      info(`stream=${isStream} messages=${openaiPayload.messages.length} tools=${(openaiPayload.tools || []).length}`);

      if (debug) debugRequest(dbg, 'OpenAI request', openaiPayload);

      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

      let upstream;
      try {
        upstream = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify(openaiPayload),
          signal: AbortSignal.timeout(300_000),
        });
      } catch (err) {
        info(`Backend connection failed: ${err.message}`);
        return errorResponse(res, 502, `Backend connection failed: ${err.message}`);
      }

      if (!upstream.ok) {
        const errText = await upstream.text().catch(() => '');
        info(`Backend returned ${upstream.status}: ${errText.slice(0, 200)}`);
        return errorResponse(res, upstream.status, `Backend returned ${upstream.status}`);
      }

      // --- Non-streaming ---
      if (!isStream) {
        const data = await upstream.json();
        if (data.error) {
          return errorResponse(res, 500, data.error.message || 'Unknown error');
        }
        if (debug) debugResponse(dbg, 'OpenAI response', data);
        const result = translateResponse(data, model);
        if (debug) debugResponse(dbg, 'Anthropic response', result);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
        return;
      }

      // --- Streaming ---
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
