import { describe, it, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { startProxy } from '../src/proxy.js';

// ---------------------------------------------------------------------------
// Mock OpenAI backend
// ---------------------------------------------------------------------------

function createMockOpenAI() {
  return http.createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString());

    const model = body.model || 'mock-model';
    const isStream = body.stream === true;
    const hasTools = (body.tools || []).length > 0;

    // Find last user message
    let lastMsg = '';
    for (const m of [...(body.messages || [])].reverse()) {
      if (m.role === 'user' && typeof m.content === 'string') { lastMsg = m.content; break; }
    }

    if (!isStream) {
      const resp = {
        id: 'chatcmpl-test123',
        model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: `Echo: ${lastMsg}` },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      };

      if (lastMsg === 'use_tool' && hasTools) {
        resp.choices[0].message = {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call_test_001',
            type: 'function',
            function: { name: body.tools[0].function.name, arguments: '{"query": "test"}' },
          }],
        };
        resp.choices[0].finish_reason = 'tool_calls';
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(resp));
      return;
    }

    // Streaming
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });

    const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

    send({ choices: [{ delta: { role: 'assistant' }, finish_reason: null }] });

    if (lastMsg === 'use_tool' && hasTools) {
      send({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_s1', type: 'function', function: { name: body.tools[0].function.name, arguments: '' } }] }, finish_reason: null }] });
      send({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"query":' } }] }, finish_reason: null }] });
      send({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ' "test"}' } }] }, finish_reason: null }] });
      send({ choices: [{ delta: {}, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 10, completion_tokens: 8 } });
    } else {
      for (const word of `Echo: ${lastMsg}`.split(' ')) {
        send({ choices: [{ delta: { content: word + ' ' }, finish_reason: null }] });
      }
      send({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 5 } });
    }

    res.write('data: [DONE]\n\n');
    res.end();
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function postJSON(port, path, body) {
  const res = await fetch(`http://127.0.0.1:${port}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res;
}

function parseSSEEvents(text) {
  const events = [];
  let currentEvent = null;
  for (const line of text.split('\n')) {
    if (line.startsWith('event: ')) currentEvent = line.slice(7);
    else if (line.startsWith('data: ') && currentEvent) {
      events.push([currentEvent, JSON.parse(line.slice(6))]);
      currentEvent = null;
    }
  }
  return events;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('integration', () => {
  let mockServer, mockPort, proxyServer, proxyPort;

  // Start servers before tests
  const setup = (async () => {
    mockServer = createMockOpenAI();
    await new Promise(r => mockServer.listen(0, '127.0.0.1', r));
    mockPort = mockServer.address().port;

    const proxy = await startProxy({
      port: 0,
      baseUrl: `http://127.0.0.1:${mockPort}`,
      apiKey: 'test-key',
      model: 'mock-model',
      debug: false,
    });
    proxyServer = proxy.server;
    proxyPort = proxy.port;
  })();

  after(async () => {
    await setup; // ensure setup completed
    proxyServer?.close();
    mockServer?.close();
  });

  it('non-streaming text', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages', {
      model: 'claude-3',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'Hello world' }],
    });
    assert.equal(res.status, 200);
    const data = await res.json();
    assert.equal(data.type, 'message');
    assert.equal(data.role, 'assistant');
    assert.equal(data.model, 'mock-model');
    assert.equal(data.stop_reason, 'end_turn');
    assert.ok(data.content[0].text.includes('Hello world'));
    assert.equal(data.usage.input_tokens, 10);
  });

  it('non-streaming tool call', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages', {
      model: 'claude-3',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'use_tool' }],
      tools: [{ name: 'search', description: 'Search', input_schema: { type: 'object', properties: { query: { type: 'string' } } } }],
    });
    const data = await res.json();
    assert.equal(data.stop_reason, 'tool_use');
    const tc = data.content.find(b => b.type === 'tool_use');
    assert.equal(tc.name, 'search');
    assert.deepEqual(tc.input, { query: 'test' });
  });

  it('streaming text', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages', {
      model: 'claude-3',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'Hello' }],
    });
    assert.equal(res.status, 200);
    const text = await res.text();
    const events = parseSSEEvents(text);
    const names = events.map(e => e[0]);

    assert.equal(names[0], 'message_start');
    assert.equal(names[1], 'ping');
    assert.ok(names.includes('content_block_start'));
    assert.ok(names.includes('content_block_delta'));
    assert.ok(names.includes('content_block_stop'));
    assert.ok(names.includes('message_delta'));
    assert.ok(names.includes('message_stop'));

    const textDeltas = events
      .filter(e => e[0] === 'content_block_delta' && e[1].delta?.type === 'text_delta')
      .map(e => e[1].delta.text);
    assert.ok(textDeltas.join('').includes('Hello'));
  });

  it('streaming tool call', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages', {
      model: 'claude-3',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'use_tool' }],
      tools: [{ name: 'search', description: 'Search', input_schema: { type: 'object' } }],
    });
    const text = await res.text();
    const events = parseSSEEvents(text);

    const blockStart = events.find(e => e[0] === 'content_block_start');
    assert.equal(blockStart[1].content_block.type, 'tool_use');
    assert.equal(blockStart[1].content_block.name, 'search');

    const jsonDeltas = events
      .filter(e => e[0] === 'content_block_delta' && e[1].delta?.type === 'input_json_delta')
      .map(e => e[1].delta.partial_json);
    assert.deepEqual(JSON.parse(jsonDeltas.join('')), { query: 'test' });

    const msgDelta = events.find(e => e[0] === 'message_delta');
    assert.equal(msgDelta[1].delta.stop_reason, 'tool_use');
  });

  it('count tokens stub', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages/count_tokens', {
      messages: [{ role: 'user', content: 'Hello world, this is a test.' }],
    });
    const data = await res.json();
    assert.ok(data.input_tokens > 0);
  });

  it('health endpoint', async () => {
    await setup;
    const res = await fetch(`http://127.0.0.1:${proxyPort}/health`);
    assert.equal(res.status, 200);
    const data = await res.json();
    assert.equal(data.status, 'ok');
  });

  it('invalid JSON returns 400', async () => {
    await setup;
    const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: 'not json',
    });
    assert.equal(res.status, 400);
    const data = await res.json();
    assert.equal(data.type, 'error');
  });

  it('missing messages returns 400', async () => {
    await setup;
    const res = await postJSON(proxyPort, '/v1/messages', { model: 'x' });
    assert.equal(res.status, 400);
    const data = await res.json();
    assert.ok(data.error.message.includes('messages'));
  });
});
