import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  isServerTool,
  partitionTools,
  serverToolToFunction,
  injectServerToolBlocks,
} from '../src/server-tools.js';
import { translateRequest } from '../src/translate.js';

// ---------------------------------------------------------------------------
// isServerTool
// ---------------------------------------------------------------------------

describe('isServerTool', () => {
  it('detects web_search tool', () => {
    assert.ok(isServerTool({ type: 'web_search_20250305', name: 'web_search' }));
    assert.ok(isServerTool({ type: 'web_search_20260209', name: 'web_search' }));
  });

  it('detects web_fetch tool', () => {
    assert.ok(isServerTool({ type: 'web_fetch_20250910', name: 'web_fetch' }));
  });

  it('detects code_execution tool', () => {
    assert.ok(isServerTool({ type: 'code_execution_20250825', name: 'code_execution' }));
  });

  it('rejects client tools', () => {
    assert.ok(!isServerTool({ name: 'Read', input_schema: {} }));
    assert.ok(!isServerTool({ name: 'web_search', input_schema: {} }));
  });

  it('handles null/undefined', () => {
    assert.ok(!isServerTool(null));
    assert.ok(!isServerTool(undefined));
    assert.ok(!isServerTool({}));
  });
});

// ---------------------------------------------------------------------------
// partitionTools
// ---------------------------------------------------------------------------

describe('partitionTools', () => {
  it('separates server and client tools', () => {
    const tools = [
      { type: 'web_search_20250305', name: 'web_search' },
      { name: 'Read', description: 'Read files', input_schema: { type: 'object' } },
      { type: 'code_execution_20250825', name: 'code_execution' },
      { name: 'Write', description: 'Write files', input_schema: { type: 'object' } },
    ];
    const { server, client } = partitionTools(tools);
    assert.equal(server.length, 2);
    assert.equal(client.length, 2);
    assert.equal(server[0].name, 'web_search');
    assert.equal(server[1].name, 'code_execution');
    assert.equal(client[0].name, 'Read');
    assert.equal(client[1].name, 'Write');
  });

  it('handles empty/null tools', () => {
    assert.deepEqual(partitionTools(null), { server: [], client: [] });
    assert.deepEqual(partitionTools([]), { server: [], client: [] });
  });
});

// ---------------------------------------------------------------------------
// serverToolToFunction
// ---------------------------------------------------------------------------

describe('serverToolToFunction', () => {
  it('converts web_search to function tool', () => {
    const fn = serverToolToFunction({ type: 'web_search_20250305', name: 'web_search' });
    assert.equal(fn.type, 'function');
    assert.equal(fn.function.name, 'web_search');
    assert.ok(fn.function.parameters.properties.query);
  });

  it('converts web_fetch to function tool', () => {
    const fn = serverToolToFunction({ type: 'web_fetch_20250910', name: 'web_fetch' });
    assert.equal(fn.function.name, 'web_fetch');
    assert.ok(fn.function.parameters.properties.url);
  });

  it('returns null for code_execution', () => {
    const fn = serverToolToFunction({ type: 'code_execution_20250825', name: 'code_execution' });
    assert.equal(fn, null);
  });
});

// ---------------------------------------------------------------------------
// injectServerToolBlocks
// ---------------------------------------------------------------------------

describe('injectServerToolBlocks', () => {
  it('prepends server tool blocks to content', () => {
    const response = {
      content: [{ type: 'text', text: 'Here are the results.' }],
      stop_reason: 'end_turn',
    };
    const blocks = [
      { type: 'server_tool_use', id: 'srvtoolu_123', name: 'web_search', input: { query: 'test' } },
      { type: 'web_search_tool_result', tool_use_id: 'srvtoolu_123', content: [] },
    ];
    const result = injectServerToolBlocks(response, blocks);
    assert.equal(result.content.length, 3);
    assert.equal(result.content[0].type, 'server_tool_use');
    assert.equal(result.content[1].type, 'web_search_tool_result');
    assert.equal(result.content[2].type, 'text');
  });

  it('returns response unchanged when no blocks', () => {
    const response = { content: [{ type: 'text', text: 'Hello' }] };
    assert.equal(injectServerToolBlocks(response, []), response);
    assert.equal(injectServerToolBlocks(response, null), response);
  });
});

// ---------------------------------------------------------------------------
// translateRequest with server tool blocks in history
// ---------------------------------------------------------------------------

describe('server tool blocks in conversation history', () => {
  it('translates server_tool_use as tool_call in assistant message', () => {
    const result = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'server_tool_use', id: 'srvtoolu_abc', name: 'web_search', input: { query: 'test' } },
          { type: 'web_search_tool_result', tool_use_id: 'srvtoolu_abc', content: [
            { type: 'web_search_result', url: 'https://example.com', title: 'Example', encrypted_content: 'A page' },
          ]},
          { type: 'text', text: 'Based on my search...' },
        ],
      }],
    }, 'gpt-4');

    // Assistant message should have text content + tool_calls
    const msg = result.messages[0];
    assert.equal(msg.role, 'assistant');
    assert.equal(msg.content, 'Based on my search...');
    assert.equal(msg.tool_calls.length, 1);
    assert.equal(msg.tool_calls[0].id, 'srvtoolu_abc');
    assert.equal(msg.tool_calls[0].function.name, 'web_search');

    // Tool result should be a separate message
    assert.equal(result.messages[1].role, 'tool');
    assert.equal(result.messages[1].tool_call_id, 'srvtoolu_abc');
    assert.ok(result.messages[1].content.includes('Example'));
    assert.ok(result.messages[1].content.includes('https://example.com'));
  });

  it('translates web_search_tool_result error', () => {
    const result = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'server_tool_use', id: 'srvtoolu_err', name: 'web_search', input: { query: 'fail' } },
          { type: 'web_search_tool_result', tool_use_id: 'srvtoolu_err', content: {
            type: 'web_search_tool_result_error',
            error_code: 'api_error',
            message: 'Search unavailable',
          }},
          { type: 'text', text: 'Search failed.' },
        ],
      }],
    }, 'gpt-4');

    const toolMsg = result.messages[1];
    assert.equal(toolMsg.role, 'tool');
    assert.ok(toolMsg.content.includes('Error: Search unavailable'));
  });

  it('translates web_fetch_tool_result in history', () => {
    const result = translateRequest({
      messages: [{
        role: 'assistant',
        content: [
          { type: 'server_tool_use', id: 'srvtoolu_fetch', name: 'web_fetch', input: { url: 'https://example.com' } },
          { type: 'web_fetch_tool_result', tool_use_id: 'srvtoolu_fetch', content: {
            type: 'web_fetch_result',
            url: 'https://example.com',
            content: { source: { data: 'Page content here' } },
          }},
          { type: 'text', text: 'The page says...' },
        ],
      }],
    }, 'gpt-4');

    const toolMsg = result.messages[1];
    assert.equal(toolMsg.role, 'tool');
    assert.equal(toolMsg.content, 'Page content here');
  });
});
