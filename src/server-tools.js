/**
 * Server-side tool handling.
 *
 * Detects Anthropic server tools (web_search, web_fetch, code_execution),
 * converts them to function tools for OpenAI-compatible backends, executes
 * search/fetch locally, and runs a multi-turn loop to feed results back
 * to the model.
 *
 * Uses only Node.js builtins — zero dependencies.
 */

import crypto from 'node:crypto';

// ---------------------------------------------------------------------------
// Server tool detection
// ---------------------------------------------------------------------------

/** Check if a tool definition is an Anthropic server tool (by type prefix). */
export function isServerTool(tool) {
  if (!tool || typeof tool.type !== 'string') return false;
  return tool.type.startsWith('web_search_')
    || tool.type.startsWith('web_fetch_')
    || tool.type.startsWith('code_execution_');
}

/** Partition a tools array into { server, client }. */
export function partitionTools(tools) {
  const server = [];
  const client = [];
  for (const t of (tools || [])) {
    if (isServerTool(t)) server.push(t);
    else client.push(t);
  }
  return { server, client };
}

/**
 * Convert a server tool definition to an OpenAI function tool.
 * Returns null for tools we strip entirely (code_execution).
 */
export function serverToolToFunction(tool) {
  if (tool.type.startsWith('web_search_')) {
    return {
      type: 'function',
      function: {
        name: 'web_search',
        description:
          'Search the web for current information. Returns relevant web page titles, URLs, and descriptions.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'The search query' },
          },
          required: ['query'],
        },
      },
    };
  }
  if (tool.type.startsWith('web_fetch_')) {
    return {
      type: 'function',
      function: {
        name: 'web_fetch',
        description:
          'Fetch the text contents of a web page at a given URL.',
        parameters: {
          type: 'object',
          properties: {
            url: { type: 'string', description: 'The URL to fetch' },
          },
          required: ['url'],
        },
      },
    };
  }
  // code_execution — strip (Claude Code already has Bash)
  return null;
}

// ---------------------------------------------------------------------------
// Search backends
// ---------------------------------------------------------------------------

/**
 * Search via Marginalia (Swedish, EU-funded, zero config).
 * Uses the public API key — no account needed.
 */
export async function marginaliaSearch(query, count = 5) {
  const url = new URL('https://api.marginalia.nu/public/search/' + encodeURIComponent(query));
  const res = await fetch(url, {
    headers: { 'Accept': 'application/json' },
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Marginalia returned ${res.status}`);
  const data = await res.json();
  return (data.results || []).slice(0, count).map(r => ({
    url: r.url || '',
    title: r.title || '',
    description: r.description || '',
  }));
}

/**
 * Search via Brave Search API (requires API key).
 */
export async function braveSearch(query, apiKey, count = 5) {
  const url = new URL('https://api.search.brave.com/res/v1/web/search');
  url.searchParams.set('q', query);
  url.searchParams.set('count', String(count));
  const res = await fetch(url, {
    headers: {
      'Accept': 'application/json',
      'Accept-Encoding': 'gzip',
      'X-Subscription-Token': apiKey,
    },
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Brave Search returned ${res.status}`);
  const data = await res.json();
  return (data.web?.results || []).slice(0, count).map(r => ({
    url: r.url || '',
    title: r.title || '',
    description: r.description || '',
  }));
}

/**
 * Run a web search using the best available backend.
 * Tries Brave if key provided, otherwise Marginalia.
 */
export async function webSearch(query, { braveApiKey } = {}) {
  if (braveApiKey) return braveSearch(query, braveApiKey);
  return marginaliaSearch(query);
}

// ---------------------------------------------------------------------------
// Web fetch
// ---------------------------------------------------------------------------

/** Fetch a URL and extract text content (strip HTML). */
export async function webFetch(targetUrl) {
  const res = await fetch(targetUrl, {
    signal: AbortSignal.timeout(15_000),
    headers: { 'User-Agent': 'le-claude/0.1 (proxy)' },
    redirect: 'follow',
  });
  if (!res.ok) throw new Error(`Fetch returned ${res.status}`);
  const html = await res.text();
  // Basic HTML to text: strip scripts, styles, tags, collapse whitespace
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 50_000); // cap at ~50k chars
}

// ---------------------------------------------------------------------------
// Server tool names we intercept
// ---------------------------------------------------------------------------

const HANDLED_SERVER_TOOLS = new Set(['web_search', 'web_fetch']);

function isHandledServerToolCall(tc) {
  return HANDLED_SERVER_TOOLS.has(tc.function?.name);
}

// ---------------------------------------------------------------------------
// Multi-turn server tool loop
// ---------------------------------------------------------------------------

function makeSrvToolId() {
  return `srvtoolu_${crypto.randomUUID().replace(/-/g, '').slice(0, 20)}`;
}

/**
 * Execute a single server tool call. Returns { name, input, result, error }.
 */
async function executeOne(tc, opts) {
  const name = tc.function?.name;
  let input;
  try {
    input = JSON.parse(tc.function?.arguments || '{}');
  } catch {
    input = {};
  }
  try {
    if (name === 'web_search') {
      const results = await webSearch(input.query, opts);
      return { name, input, results };
    }
    if (name === 'web_fetch') {
      const content = await webFetch(input.url);
      return { name, input, results: content };
    }
  } catch (err) {
    return { name, input, error: err.message };
  }
  return { name, input, error: `Unknown server tool: ${name}` };
}

/**
 * Build Anthropic server_tool_use + *_tool_result content blocks
 * from an executed server tool call.
 */
function buildServerToolBlocks(exec, srvId) {
  const blocks = [];

  blocks.push({
    type: 'server_tool_use',
    id: srvId,
    name: exec.name,
    input: exec.input,
  });

  if (exec.name === 'web_search') {
    if (exec.error) {
      blocks.push({
        type: 'web_search_tool_result',
        tool_use_id: srvId,
        content: {
          type: 'web_search_tool_result_error',
          error_code: 'api_error',
          message: exec.error,
        },
      });
    } else {
      blocks.push({
        type: 'web_search_tool_result',
        tool_use_id: srvId,
        content: (exec.results || []).map(r => ({
          type: 'web_search_result',
          url: r.url,
          title: r.title,
          encrypted_content: r.description,
          page_age: '',
        })),
      });
    }
  } else if (exec.name === 'web_fetch') {
    if (exec.error) {
      blocks.push({
        type: 'web_fetch_tool_result',
        tool_use_id: srvId,
        content: {
          type: 'web_fetch_tool_result_error',
          error_code: 'api_error',
          message: exec.error,
        },
      });
    } else {
      blocks.push({
        type: 'web_fetch_tool_result',
        tool_use_id: srvId,
        content: {
          type: 'web_fetch_result',
          url: exec.input.url,
          content: {
            type: 'document',
            source: {
              type: 'text',
              media_type: 'text/plain',
              data: exec.results,
            },
          },
        },
      });
    }
  }

  return blocks;
}

/**
 * Run the multi-turn server tool loop.
 *
 * Takes the first model response (which contains server tool calls),
 * executes them locally, feeds results back to the model, and repeats.
 * Returns the final OpenAI response + accumulated server tool blocks.
 *
 * @param {object} opts
 * @param {object} opts.openaiPayload - The original translated OpenAI payload
 * @param {object} opts.firstResponse - The first OpenAI response (with server tool calls)
 * @param {function} opts.callUpstream - async (payload) => openaiResponseJSON
 * @param {string} [opts.braveApiKey] - Optional Brave Search API key
 * @param {number} [opts.maxIterations=3] - Max search loop iterations
 * @param {function} [opts.log] - Debug logger
 * @returns {{ response: object, serverToolBlocks: object[] }}
 */
export async function serverToolLoop({
  openaiPayload,
  firstResponse,
  callUpstream,
  braveApiKey,
  maxIterations = 3,
  log = () => {},
}) {
  const serverToolBlocks = [];
  const payload = { ...openaiPayload, stream: false };
  delete payload.stream_options;

  let currentResponse = firstResponse;

  for (let i = 0; i < maxIterations; i++) {
    const message = currentResponse.choices?.[0]?.message || {};
    const allToolCalls = message.tool_calls || [];

    // Partition into server and client tool calls
    const serverCalls = allToolCalls.filter(isHandledServerToolCall);
    const clientCalls = allToolCalls.filter(tc => !isHandledServerToolCall(tc));

    if (serverCalls.length === 0) {
      // No server tool calls — we're done
      return { response: currentResponse, serverToolBlocks };
    }

    log(`Server tool loop iteration ${i + 1}: executing ${serverCalls.length} server tool(s)`);

    // Execute all server tool calls
    const toolMessages = [];
    for (const tc of serverCalls) {
      const srvId = makeSrvToolId();
      const exec = await executeOne(tc, { braveApiKey });
      serverToolBlocks.push(...buildServerToolBlocks(exec, srvId));

      // Build tool result message for the model
      const resultContent = exec.error
        ? `Error: ${exec.error}`
        : typeof exec.results === 'string'
          ? exec.results
          : JSON.stringify(exec.results);

      toolMessages.push({
        role: 'tool',
        tool_call_id: tc.id,
        content: resultContent,
      });
    }

    // If there are also client tool calls, stop looping —
    // return this response so Claude Code can handle the client tools.
    if (clientCalls.length > 0) {
      return { response: currentResponse, serverToolBlocks };
    }

    // Build continuation: add assistant message + tool results, re-query model
    payload.messages = [
      ...payload.messages,
      {
        role: 'assistant',
        content: message.content || '',
        tool_calls: allToolCalls,
      },
      ...toolMessages,
    ];

    currentResponse = await callUpstream(payload);
  }

  // Max iterations reached — return whatever we have
  log('Server tool loop: max iterations reached');
  return { response: currentResponse, serverToolBlocks };
}

/**
 * Inject server tool blocks into an Anthropic response's content array.
 * Server tool blocks go before the model's text/tool_use content.
 */
export function injectServerToolBlocks(anthropicResponse, serverToolBlocks) {
  if (!serverToolBlocks || serverToolBlocks.length === 0) return anthropicResponse;
  return {
    ...anthropicResponse,
    content: [...serverToolBlocks, ...anthropicResponse.content],
  };
}
