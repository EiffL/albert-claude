import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { stripUriFormat, translateRequest, translateResponse } from '../src/translate.js';

// ---------------------------------------------------------------------------
// stripUriFormat
// ---------------------------------------------------------------------------

describe('stripUriFormat', () => {
  it('removes uri format from string field', () => {
    const result = stripUriFormat({ type: 'string', format: 'uri', description: 'A URL' });
    assert.equal(result.format, undefined);
    assert.equal(result.type, 'string');
    assert.equal(result.description, 'A URL');
  });

  it('preserves non-uri format', () => {
    const result = stripUriFormat({ type: 'string', format: 'date-time' });
    assert.equal(result.format, 'date-time');
  });

  it('recurses into properties', () => {
    const schema = {
      type: 'object',
      properties: {
        url: { type: 'string', format: 'uri' },
        name: { type: 'string' },
      },
    };
    const result = stripUriFormat(schema);
    assert.equal(result.properties.url.format, undefined);
    assert.deepEqual(result.properties.name, { type: 'string' });
  });

  it('recurses into items', () => {
    const result = stripUriFormat({ type: 'array', items: { type: 'string', format: 'uri' } });
    assert.equal(result.items.format, undefined);
  });

  it('recurses into anyOf', () => {
    const result = stripUriFormat({ anyOf: [{ type: 'string', format: 'uri' }, { type: 'null' }] });
    assert.equal(result.anyOf[0].format, undefined);
  });

  it('passes through non-objects', () => {
    assert.equal(stripUriFormat('hello'), 'hello');
    assert.equal(stripUriFormat(42), 42);
    assert.equal(stripUriFormat(null), null);
  });
});

// ---------------------------------------------------------------------------
// translateRequest
// ---------------------------------------------------------------------------

describe('translateRequest', () => {
  it('translates simple text message', () => {
    const result = translateRequest({
      model: 'claude-3',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    }, 'gpt-4');
    assert.equal(result.model, 'gpt-4');
    assert.equal(result.max_tokens, 1024);
    assert.equal(result.messages[0].content, 'Hello');
    assert.equal(result.stream, false);
  });

  it('translates system string', () => {
    const result = translateRequest({
      system: 'You are helpful.',
      messages: [{ role: 'user', content: 'Hi' }],
    }, 'gpt-4');
    assert.equal(result.messages[0].role, 'system');
    assert.equal(result.messages[0].content, 'You are helpful.');
    assert.equal(result.messages[1].content, 'Hi');
  });

  it('translates system block array', () => {
    const result = translateRequest({
      system: [{ type: 'text', text: 'Rule 1.' }, { type: 'text', text: 'Rule 2.' }],
      messages: [{ role: 'user', content: 'Hi' }],
    }, 'gpt-4');
    assert.equal(result.messages[0].content, 'Rule 1.');
    assert.equal(result.messages[1].content, 'Rule 2.');
  });

  it('translates image blocks', () => {
    const result = translateRequest({
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: "What's this?" },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc123' } },
        ],
      }],
    }, 'gpt-4');
    const msg = result.messages[0];
    assert.equal(msg.content.length, 2);
    assert.equal(msg.content[1].type, 'image_url');
    assert.equal(msg.content[1].image_url.url, 'data:image/png;base64,abc123');
  });

  it('translates tool_use in assistant message', () => {
    const result = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'text', text: 'Let me search.' },
          { type: 'tool_use', id: 'call_abc', name: 'search', input: { query: 'test' } },
        ],
      }],
    }, 'gpt-4');
    const msg = result.messages[0];
    assert.equal(msg.content, 'Let me search.');
    assert.equal(msg.tool_calls.length, 1);
    assert.equal(msg.tool_calls[0].id, 'call_abc');
    assert.equal(msg.tool_calls[0].function.name, 'search');
    assert.deepEqual(JSON.parse(msg.tool_calls[0].function.arguments), { query: 'test' });
  });

  it('translates tool_result to separate message', () => {
    const result = translateRequest({
      messages: [{
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'call_abc', content: 'found it' }],
      }],
    }, 'gpt-4');
    assert.equal(result.messages.length, 2);
    assert.equal(result.messages[1].role, 'tool');
    assert.equal(result.messages[1].content, 'found it');
    assert.equal(result.messages[1].tool_call_id, 'call_abc');
  });

  it('prefixes error tool results', () => {
    const result = translateRequest({
      messages: [{
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'x', content: 'not found', is_error: true }],
      }],
    }, 'gpt-4');
    assert.equal(result.messages[1].content, 'Error: not found');
  });

  it('strips thinking blocks', () => {
    const result = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Let me think...' },
          { type: 'text', text: 'The answer is 42.' },
        ],
      }],
    }, 'gpt-4');
    assert.equal(result.messages[0].content, 'The answer is 42.');
  });

  it('translates tool definitions and strips uri format', () => {
    const result = translateRequest({
      messages: [{ role: 'user', content: 'Hi' }],
      tools: [{
        name: 'get_weather',
        description: 'Get weather',
        input_schema: {
          type: 'object',
          properties: { url: { type: 'string', format: 'uri' } },
        },
      }],
    }, 'gpt-4');
    assert.equal(result.tools[0].function.name, 'get_weather');
    assert.equal(result.tools[0].function.parameters.properties.url.format, undefined);
  });

  it('filters BatchTool', () => {
    const result = translateRequest({
      messages: [{ role: 'user', content: 'Hi' }],
      tools: [
        { name: 'BatchTool', description: 'x', input_schema: {} },
        { name: 'real_tool', description: 'y', input_schema: {} },
      ],
    }, 'gpt-4');
    assert.equal(result.tools.length, 1);
    assert.equal(result.tools[0].function.name, 'real_tool');
  });

  it('maps parameters', () => {
    const result = translateRequest({
      max_tokens: 2048,
      temperature: 0.7,
      top_p: 0.9,
      stop_sequences: ['\n\nHuman:'],
      messages: [{ role: 'user', content: 'Hi' }],
    }, 'gpt-4');
    assert.equal(result.max_tokens, 2048);
    assert.equal(result.temperature, 0.7);
    assert.equal(result.top_p, 0.9);
    assert.deepEqual(result.stop, ['\n\nHuman:']);
  });

  it('maps tool_choice', () => {
    const auto = translateRequest({ messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'auto' } }, 'gpt-4');
    assert.equal(auto.tool_choice, 'auto');

    const any = translateRequest({ messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'any' } }, 'gpt-4');
    assert.equal(any.tool_choice, 'required');

    const specific = translateRequest({ messages: [{ role: 'user', content: 'Hi' }], tool_choice: { type: 'tool', name: 'search' } }, 'gpt-4');
    assert.deepEqual(specific.tool_choice, { type: 'function', function: { name: 'search' } });
  });

  it('includes stream_options when streaming', () => {
    const result = translateRequest({ stream: true, messages: [{ role: 'user', content: 'Hi' }] }, 'gpt-4');
    assert.equal(result.stream, true);
    assert.deepEqual(result.stream_options, { include_usage: true });
  });
});

// ---------------------------------------------------------------------------
// translateResponse
// ---------------------------------------------------------------------------

describe('translateResponse', () => {
  it('translates simple text', () => {
    const result = translateResponse({
      id: 'chatcmpl-abc',
      choices: [{ message: { role: 'assistant', content: 'Hello!' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    }, 'gpt-4');
    assert.equal(result.id, 'msg-abc');
    assert.equal(result.type, 'message');
    assert.equal(result.role, 'assistant');
    assert.deepEqual(result.content, [{ type: 'text', text: 'Hello!' }]);
    assert.equal(result.stop_reason, 'end_turn');
    assert.equal(result.usage.input_tokens, 10);
    assert.equal(result.usage.output_tokens, 5);
  });

  it('translates tool calls', () => {
    const result = translateResponse({
      id: 'chatcmpl-xyz',
      choices: [{
        message: {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call_123',
            type: 'function',
            function: { name: 'search', arguments: '{"query": "test"}' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 20, completion_tokens: 15 },
    }, 'gpt-4');
    assert.equal(result.stop_reason, 'tool_use');
    assert.equal(result.content[0].type, 'tool_use');
    assert.equal(result.content[0].name, 'search');
    assert.deepEqual(result.content[0].input, { query: 'test' });
  });

  it('handles malformed tool arguments', () => {
    const result = translateResponse({
      id: 'chatcmpl-bad',
      choices: [{
        message: {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'call_bad', function: { name: 'tool', arguments: 'not-json' } }],
        },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    }, 'gpt-4');
    assert.deepEqual(result.content[0].input, {});
  });

  it('maps stop reason length', () => {
    const result = translateResponse({
      id: 'x',
      choices: [{ message: { content: 'Truncated' }, finish_reason: 'length' }],
      usage: { prompt_tokens: 5, completion_tokens: 100 },
    }, 'gpt-4');
    assert.equal(result.stop_reason, 'max_tokens');
  });
});
