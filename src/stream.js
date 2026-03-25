/**
 * StreamTranslator — converts OpenAI streaming chunks to Anthropic SSE events.
 *
 * Direct port of the tested Python StreamTranslator class.
 */

import { makeMsgId, mapStopReason } from './translate.js';
import crypto from 'node:crypto';

/** Format a single SSE event. */
export function formatSSE(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export class StreamTranslator {
  constructor(model) {
    this.model = model;
    this._msgId = makeMsgId();
    this._started = false;
    this._nextIndex = 0;
    this._textIndex = null;       // index of open text/thinking block
    this._toolIndices = new Map(); // OAI tc index -> content block index
    this._toolArgs = new Map();    // OAI tc index -> accumulated args
    this._hasTools = false;
    this._usage = null;
  }

  _allocIndex() {
    return this._nextIndex++;
  }

  _closeTextBlock() {
    if (this._textIndex === null) return [];
    const events = [formatSSE('content_block_stop', {
      type: 'content_block_stop',
      index: this._textIndex,
    })];
    this._textIndex = null;
    return events;
  }

  /** Process one parsed OpenAI chunk. Returns array of SSE event strings. */
  processChunk(chunk) {
    const events = [];

    // Emit message_start on first chunk
    if (!this._started) {
      this._started = true;
      events.push(formatSSE('message_start', {
        type: 'message_start',
        message: {
          id: this._msgId,
          type: 'message',
          role: 'assistant',
          model: this.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }));
      events.push(formatSSE('ping', { type: 'ping' }));
    }

    // Capture usage
    if (chunk.usage) this._usage = chunk.usage;

    const choices = chunk.choices || [];
    if (!choices.length) return events;
    const delta = choices[0].delta || {};

    // --- Tool calls ---
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        const oaiIdx = tc.index ?? 0;
        this._hasTools = true;

        if (!this._toolIndices.has(oaiIdx)) {
          // Close open text block first
          events.push(...this._closeTextBlock());

          const ci = this._allocIndex();
          this._toolIndices.set(oaiIdx, ci);
          this._toolArgs.set(oaiIdx, '');
          events.push(formatSSE('content_block_start', {
            type: 'content_block_start',
            index: ci,
            content_block: {
              type: 'tool_use',
              id: tc.id || `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`,
              name: (tc.function || {}).name || '',
              input: {},
            },
          }));
        }

        // Accumulate and emit argument deltas
        const newArgs = (tc.function || {}).arguments || '';
        if (newArgs) {
          this._toolArgs.set(oaiIdx, (this._toolArgs.get(oaiIdx) || '') + newArgs);
          events.push(formatSSE('content_block_delta', {
            type: 'content_block_delta',
            index: this._toolIndices.get(oaiIdx),
            delta: {
              type: 'input_json_delta',
              partial_json: newArgs,
            },
          }));
        }
      }
    }
    // --- Text content ---
    else if (delta.content) {
      if (this._textIndex === null) {
        const ci = this._allocIndex();
        this._textIndex = ci;
        events.push(formatSSE('content_block_start', {
          type: 'content_block_start',
          index: ci,
          content_block: { type: 'text', text: '' },
        }));
      }
      events.push(formatSSE('content_block_delta', {
        type: 'content_block_delta',
        index: this._textIndex,
        delta: { type: 'text_delta', text: delta.content },
      }));
    }
    // --- Reasoning / thinking ---
    else if (delta.reasoning) {
      if (this._textIndex === null) {
        const ci = this._allocIndex();
        this._textIndex = ci;
        events.push(formatSSE('content_block_start', {
          type: 'content_block_start',
          index: ci,
          content_block: { type: 'thinking', thinking: '' },
        }));
      }
      events.push(formatSSE('content_block_delta', {
        type: 'content_block_delta',
        index: this._textIndex,
        delta: { type: 'thinking_delta', thinking: delta.reasoning },
      }));
    }

    return events;
  }

  /** Emit closing events when the stream ends. */
  finish() {
    const events = [];

    // Close open text/thinking block
    events.push(...this._closeTextBlock());

    // Close all tool blocks
    for (const ci of this._toolIndices.values()) {
      events.push(formatSSE('content_block_stop', {
        type: 'content_block_stop',
        index: ci,
      }));
    }

    // message_delta
    const stopReason = this._hasTools ? 'tool_use' : 'end_turn';
    const outputTokens = this._usage ? (this._usage.completion_tokens || 0) : 0;
    events.push(formatSSE('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }));

    // message_stop
    events.push(formatSSE('message_stop', { type: 'message_stop' }));

    return events;
  }
}
