import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { StreamTranslator } from '../src/stream.js';

/** Parse SSE event strings into [event, data] tuples. */
function parseEvents(eventStrings) {
  const events = [];
  for (const str of eventStrings) {
    let eventName = null;
    let data = null;
    for (const line of str.trim().split('\n')) {
      if (line.startsWith('event: ')) eventName = line.slice(7);
      else if (line.startsWith('data: ')) data = JSON.parse(line.slice(6));
    }
    if (eventName && data) events.push([eventName, data]);
  }
  return events;
}

function makeChunk({ content, tool_calls, reasoning, finish_reason, usage } = {}) {
  const delta = {};
  if (content !== undefined) delta.content = content;
  if (tool_calls !== undefined) delta.tool_calls = tool_calls;
  if (reasoning !== undefined) delta.reasoning = reasoning;
  const chunk = { choices: [{ delta, finish_reason: finish_reason || null }] };
  if (usage) chunk.usage = usage;
  return chunk;
}

describe('StreamTranslator', () => {
  it('simple text stream', () => {
    const t = new StreamTranslator('test-model');

    const events = parseEvents(t.processChunk(makeChunk({ content: 'Hello' })));
    const names = events.map(e => e[0]);
    assert.ok(names.includes('message_start'));
    assert.ok(names.includes('ping'));
    assert.ok(names.includes('content_block_start'));
    assert.ok(names.includes('content_block_delta'));

    const msgStart = events.find(e => e[0] === 'message_start')[1];
    assert.equal(msgStart.message.role, 'assistant');
    assert.equal(msgStart.message.model, 'test-model');

    const blockStart = events.find(e => e[0] === 'content_block_start')[1];
    assert.equal(blockStart.content_block.type, 'text');
    assert.equal(blockStart.index, 0);

    const delta = events.find(e => e[0] === 'content_block_delta')[1];
    assert.equal(delta.delta.type, 'text_delta');
    assert.equal(delta.delta.text, 'Hello');

    // Second chunk: just a delta
    const events2 = parseEvents(t.processChunk(makeChunk({ content: ' world' })));
    assert.equal(events2.length, 1);
    assert.equal(events2[0][1].delta.text, ' world');

    // Finish
    const fin = parseEvents(t.finish());
    const finNames = fin.map(e => e[0]);
    assert.ok(finNames.includes('content_block_stop'));
    assert.ok(finNames.includes('message_delta'));
    assert.ok(finNames.includes('message_stop'));

    const msgDelta = fin.find(e => e[0] === 'message_delta')[1];
    assert.equal(msgDelta.delta.stop_reason, 'end_turn');
  });

  it('tool call stream', () => {
    const t = new StreamTranslator('test-model');

    const tcStart = [{ index: 0, id: 'call_abc', type: 'function', function: { name: 'search', arguments: '' } }];
    const events = parseEvents(t.processChunk(makeChunk({ tool_calls: tcStart })));
    assert.ok(events.map(e => e[0]).includes('content_block_start'));

    const blockStart = events.find(e => e[0] === 'content_block_start')[1];
    assert.equal(blockStart.content_block.type, 'tool_use');
    assert.equal(blockStart.content_block.name, 'search');

    // Arguments
    const events2 = parseEvents(t.processChunk(makeChunk({ tool_calls: [{ index: 0, function: { arguments: '{"query":' } }] })));
    assert.equal(events2[0][1].delta.type, 'input_json_delta');
    assert.equal(events2[0][1].delta.partial_json, '{"query":');

    const events3 = parseEvents(t.processChunk(makeChunk({ tool_calls: [{ index: 0, function: { arguments: ' "test"}' } }] })));
    assert.equal(events3[0][1].delta.partial_json, ' "test"}');

    // Finish
    const fin = parseEvents(t.finish());
    const msgDelta = fin.find(e => e[0] === 'message_delta')[1];
    assert.equal(msgDelta.delta.stop_reason, 'tool_use');
  });

  it('text then tool call closes text block', () => {
    const t = new StreamTranslator('test-model');
    t.processChunk(makeChunk({ content: 'Let me search.' }));

    const tcStart = [{ index: 0, id: 'call_xyz', type: 'function', function: { name: 'search', arguments: '{"q":"x"}' } }];
    const events = parseEvents(t.processChunk(makeChunk({ tool_calls: tcStart })));
    const names = events.map(e => e[0]);
    assert.ok(names.includes('content_block_stop'));
    assert.ok(names.includes('content_block_start'));

    const blockStart = events.find(e => e[0] === 'content_block_start')[1];
    assert.equal(blockStart.index, 1); // text was 0, tool is 1
  });

  it('multiple tool calls', () => {
    const t = new StreamTranslator('test-model');

    t.processChunk(makeChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'tool_a', arguments: '{"a":1}' } }] }));
    const events = parseEvents(t.processChunk(makeChunk({ tool_calls: [{ index: 1, id: 'call_2', type: 'function', function: { name: 'tool_b', arguments: '{"b":2}' } }] })));

    const blockStart = events.find(e => e[0] === 'content_block_start')[1];
    assert.equal(blockStart.index, 1);
    assert.equal(blockStart.content_block.name, 'tool_b');

    const fin = parseEvents(t.finish());
    const stops = fin.filter(e => e[0] === 'content_block_stop');
    assert.equal(stops.length, 2);
  });

  it('usage forwarded', () => {
    const t = new StreamTranslator('test-model');
    t.processChunk(makeChunk({ content: 'Hi' }));
    t.processChunk(makeChunk({ usage: { prompt_tokens: 10, completion_tokens: 5 } }));

    const fin = parseEvents(t.finish());
    const msgDelta = fin.find(e => e[0] === 'message_delta')[1];
    assert.equal(msgDelta.usage.output_tokens, 5);
  });

  it('empty delta produces no content events', () => {
    const t = new StreamTranslator('test-model');
    const events = parseEvents(t.processChunk({ choices: [{ delta: { role: 'assistant' }, finish_reason: null }] }));
    const names = events.map(e => e[0]);
    assert.ok(!names.includes('content_block_start'));
    assert.ok(!names.includes('content_block_delta'));
  });

  it('reasoning as thinking', () => {
    const t = new StreamTranslator('test-model');
    const events = parseEvents(t.processChunk(makeChunk({ reasoning: 'Hmm...' })));
    const blockStart = events.find(e => e[0] === 'content_block_start')[1];
    assert.equal(blockStart.content_block.type, 'thinking');

    const delta = events.find(e => e[0] === 'content_block_delta')[1];
    assert.equal(delta.delta.type, 'thinking_delta');
    assert.equal(delta.delta.thinking, 'Hmm...');
  });

  it('finish with no content does not crash', () => {
    const t = new StreamTranslator('test-model');
    const fin = parseEvents(t.finish());
    const names = fin.map(e => e[0]);
    assert.ok(names.includes('message_delta'));
    assert.ok(names.includes('message_stop'));
    assert.ok(!names.includes('content_block_stop'));
  });

  // --- Text-based tool call detection ---

  it('detects Mistral [TOOL_CALLS] in stream', () => {
    const t = new StreamTranslator('test-model');

    // Simulate streaming [TOOL_CALLS]Bash{"command":"ls"}
    const e1 = parseEvents(t.processChunk(makeChunk({ content: '[TOOL_' })));
    // Should buffer (starts with [, could be [TOOL_CALLS])
    const textEvents1 = e1.filter(e => e[0] === 'content_block_delta');
    assert.equal(textEvents1.length, 0); // no text emitted yet

    const e2 = parseEvents(t.processChunk(makeChunk({ content: 'CALLS]Bash{"command":"ls"}' })));
    // Still buffering (tool call mode detected)
    const textEvents2 = e2.filter(e => e[0] === 'content_block_delta');
    assert.equal(textEvents2.length, 0);

    // Finish: should parse and emit tool_use
    const fin = parseEvents(t.finish());
    const blockStart = fin.find(e => e[0] === 'content_block_start' && e[1].content_block?.type === 'tool_use');
    assert.ok(blockStart, 'should emit tool_use block');
    assert.equal(blockStart[1].content_block.name, 'Bash');

    const jsonDelta = fin.find(e => e[0] === 'content_block_delta' && e[1].delta?.type === 'input_json_delta');
    assert.ok(jsonDelta);
    assert.deepEqual(JSON.parse(jsonDelta[1].delta.partial_json), { command: 'ls' });

    const msgDelta = fin.find(e => e[0] === 'message_delta');
    assert.equal(msgDelta[1].delta.stop_reason, 'tool_use');
  });

  it('detects Qwen3 <function=> in stream', () => {
    const t = new StreamTranslator('test-model');

    t.processChunk(makeChunk({ content: '<function=Bash>' }));
    t.processChunk(makeChunk({ content: '<parameter=command>ls</parameter>' }));
    t.processChunk(makeChunk({ content: '</function>' }));

    const fin = parseEvents(t.finish());
    const blockStart = fin.find(e => e[0] === 'content_block_start' && e[1].content_block?.type === 'tool_use');
    assert.ok(blockStart, 'should emit tool_use block');
    assert.equal(blockStart[1].content_block.name, 'Bash');
  });

  it('flushes normal text immediately when first char is not [ or <', () => {
    const t = new StreamTranslator('test-model');

    const events = parseEvents(t.processChunk(makeChunk({ content: 'Hello world' })));
    const blockStart = events.find(e => e[0] === 'content_block_start');
    assert.ok(blockStart, 'should emit text block immediately');
    assert.equal(blockStart[1].content_block.type, 'text');

    const delta = events.find(e => e[0] === 'content_block_delta');
    assert.equal(delta[1].delta.text, 'Hello world');
  });

  it('flushes buffer after 15 chars if no prefix match', () => {
    const t = new StreamTranslator('test-model');

    // Send short text starting with < (still undecided, buffered)
    const e1 = parseEvents(t.processChunk(makeChunk({ content: '<h1>Hi' })));
    const textEvents1 = e1.filter(e => e[0] === 'content_block_delta');
    assert.equal(textEvents1.length, 0, 'should buffer initially');

    // Send more text to exceed 15 chars — triggers flush
    const e2 = parseEvents(t.processChunk(makeChunk({ content: ' there world!</h1>' })));
    const delta = e2.find(e => e[0] === 'content_block_delta');
    assert.ok(delta, 'should have flushed buffered text');
    assert.ok(delta[1].delta.text.includes('<h1>Hi there world!</h1>'));
  });

  it('emits buffered text as plain text if no tool calls found at finish', () => {
    const t = new StreamTranslator('test-model');

    // Send short text starting with < that doesn't match
    t.processChunk(makeChunk({ content: '<hi>' }));
    // Stream ends before 15 chars

    const fin = parseEvents(t.finish());
    const blockStart = fin.find(e => e[0] === 'content_block_start');
    assert.ok(blockStart);
    assert.equal(blockStart[1].content_block.type, 'text');

    const delta = fin.find(e => e[0] === 'content_block_delta');
    assert.equal(delta[1].delta.text, '<hi>');

    const msgDelta = fin.find(e => e[0] === 'message_delta');
    assert.equal(msgDelta[1].delta.stop_reason, 'end_turn');
  });
});
