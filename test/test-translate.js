import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { stripUriFormat, translateRequest, translateResponse, parseTextToolCalls, fixTextToolCalls } from '../src/translate.js';

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

  it('translates tool_result to tool message without empty user message', () => {
    const result = translateRequest({
      messages: [{
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'call_abc', content: 'found it' }],
      }],
    }, 'gpt-4');
    assert.equal(result.messages.length, 1);
    assert.equal(result.messages[0].role, 'tool');
    assert.equal(result.messages[0].content, 'found it');
    assert.equal(result.messages[0].tool_call_id, 'call_abc');
  });

  it('prefixes error tool results', () => {
    const result = translateRequest({
      messages: [{
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'x', content: 'not found', is_error: true }],
      }],
    }, 'gpt-4');
    assert.equal(result.messages[0].content, 'Error: not found');
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

// ---------------------------------------------------------------------------
// parseTextToolCalls
// ---------------------------------------------------------------------------

describe('parseTextToolCalls', () => {
  it('returns empty for plain text', () => {
    const result = parseTextToolCalls('Hello, how are you?');
    assert.equal(result.toolCalls.length, 0);
    assert.equal(result.text, 'Hello, how are you?');
  });

  it('returns empty for null/empty', () => {
    assert.equal(parseTextToolCalls(null).toolCalls.length, 0);
    assert.equal(parseTextToolCalls('').toolCalls.length, 0);
  });

  // --- Mistral format ---

  it('parses Mistral single tool call', () => {
    const result = parseTextToolCalls('[TOOL_CALLS]Bash{"command": "ls"}');
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.deepEqual(JSON.parse(result.toolCalls[0].function.arguments), { command: 'ls' });
    assert.equal(result.text, '');
  });

  it('parses Mistral multiple tool calls', () => {
    const text = '[TOOL_CALLS]Bash{"command": "ls"}[TOOL_CALLS]Read{"file_path": "/tmp/foo"}';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 2);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.equal(result.toolCalls[1].function.name, 'Read');
    assert.deepEqual(JSON.parse(result.toolCalls[1].function.arguments), { file_path: '/tmp/foo' });
  });

  it('parses Mistral JSON array format', () => {
    const text = '[TOOL_CALLS] [{"name": "Bash", "arguments": {"command": "ls"}}]';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.deepEqual(JSON.parse(result.toolCalls[0].function.arguments), { command: 'ls' });
  });

  it('parses Mistral with nested JSON', () => {
    const text = '[TOOL_CALLS]Bash{"command": "echo \\\"hello {world}\\\""}';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
  });

  it('preserves text before Mistral tool calls', () => {
    const text = 'I will run this command.\n[TOOL_CALLS]Bash{"command": "ls"}';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.ok(result.text.includes('I will run this command'));
  });

  // --- Qwen3 / Hermes format ---

  it('parses Hermes <tool_call> JSON', () => {
    const text = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.deepEqual(JSON.parse(result.toolCalls[0].function.arguments), { command: 'ls' });
  });

  it('parses multiple <tool_call> blocks', () => {
    const text = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>\n<tool_call>{"name": "Read", "arguments": {"file_path": "/tmp"}}</tool_call>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 2);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.equal(result.toolCalls[1].function.name, 'Read');
  });

  it('parses Qwen3 XML parameters', () => {
    const text = '<tool_call><function=Bash><parameter=command>ls -la</parameter><parameter=description>List files</parameter></function></tool_call>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    const args = JSON.parse(result.toolCalls[0].function.arguments);
    assert.equal(args.command, 'ls -la');
    assert.equal(args.description, 'List files');
  });

  it('parses bare <function=Name> without <tool_call> wrapper', () => {
    const text = '<function=Agent><parameter=subagent_type>Explore</parameter><parameter=prompt>Explore the repo</parameter></function>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Agent');
    const args = JSON.parse(result.toolCalls[0].function.arguments);
    assert.equal(args.subagent_type, 'Explore');
    assert.equal(args.prompt, 'Explore the repo');
  });

  it('parses Qwen3 parameters without closing tags', () => {
    const text = '<function=Agent>\n<parameter=subagent_type>\nExplore\n\n<parameter=prompt>\nExplore the repository structure</function>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Agent');
    const args = JSON.parse(result.toolCalls[0].function.arguments);
    assert.equal(args.subagent_type, 'Explore');
    assert.ok(args.prompt.includes('Explore the repository'));
  });

  it('parses Qwen3 with JSON body inside <function=>', () => {
    const text = '<function=Bash>{"command": "ls", "description": "list files"}</function>';
    const result = parseTextToolCalls(text);
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].function.name, 'Bash');
    assert.deepEqual(JSON.parse(result.toolCalls[0].function.arguments), { command: 'ls', description: 'list files' });
  });

  it('does not false-positive on HTML-like text', () => {
    const result = parseTextToolCalls('<h1>Hello world</h1>');
    assert.equal(result.toolCalls.length, 0);
  });
});

// ---------------------------------------------------------------------------
// fixTextToolCalls
// ---------------------------------------------------------------------------

describe('fixTextToolCalls', () => {
  it('fixes response with text-based tool calls', () => {
    const data = {
      choices: [{
        message: { role: 'assistant', content: '[TOOL_CALLS]Bash{"command": "ls"}' },
        finish_reason: 'stop',
      }],
    };
    fixTextToolCalls(data);
    assert.equal(data.choices[0].message.tool_calls.length, 1);
    assert.equal(data.choices[0].message.tool_calls[0].function.name, 'Bash');
    assert.equal(data.choices[0].message.content, null);
    assert.equal(data.choices[0].finish_reason, 'tool_calls');
  });

  it('does not modify response with proper tool_calls', () => {
    const data = {
      choices: [{
        message: {
          role: 'assistant',
          content: 'Some text with [TOOL_CALLS] in it',
          tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'Bash', arguments: '{}' } }],
        },
        finish_reason: 'tool_calls',
      }],
    };
    fixTextToolCalls(data);
    assert.equal(data.choices[0].message.content, 'Some text with [TOOL_CALLS] in it');
    assert.equal(data.choices[0].message.tool_calls.length, 1);
  });

  it('does not modify plain text response', () => {
    const data = {
      choices: [{
        message: { role: 'assistant', content: 'Just a normal response.' },
        finish_reason: 'stop',
      }],
    };
    fixTextToolCalls(data);
    assert.equal(data.choices[0].message.content, 'Just a normal response.');
    assert.equal(data.choices[0].message.tool_calls, undefined);
    assert.equal(data.choices[0].finish_reason, 'stop');
  });
});
