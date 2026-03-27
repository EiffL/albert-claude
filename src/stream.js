/**
 * StreamTranslator — converts OpenAI streaming chunks to Anthropic SSE events.
 *
 * Direct port of the tested Python StreamTranslator class.
 */

import { makeMsgId, mapStopReason, parseTextToolCalls, TOOL_CALL_TEXT_PREFIXES, makeToolCallId } from './translate.js';

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
    // Text buffering for detecting text-based tool calls (Mistral, Qwen, etc.)
    this._textBuffer = '';
    this._bufferingText = true;    // true while deciding if text is tool calls
    this._textToolCallMode = false; // true once a tool call prefix is detected
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

  /** Flush buffered text as a normal text block. */
  _flushTextBuffer() {
    const events = [];
    if (!this._textBuffer) return events;
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
      delta: { type: 'text_delta', text: this._textBuffer },
    }));
    this._textBuffer = '';
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

    // --- Tool calls (proper API format) ---
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        const oaiIdx = tc.index ?? 0;
        this._hasTools = true;

        if (!this._toolIndices.has(oaiIdx)) {
          // Flush any buffered text before opening tool block
          if (this._bufferingText && this._textBuffer) {
            this._bufferingText = false;
            events.push(...this._flushTextBuffer());
          }
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
              id: tc.id || makeToolCallId(),
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
      // If already detected text-based tool calls, just buffer
      if (this._textToolCallMode) {
        this._textBuffer += delta.content;
        return events;
      }

      // Still deciding: buffer and check for tool call prefixes
      if (this._bufferingText) {
        this._textBuffer += delta.content;
        const trimmed = this._textBuffer.trimStart();

        // Check for known tool call prefixes
        for (const prefix of TOOL_CALL_TEXT_PREFIXES) {
          if (trimmed.startsWith(prefix)) {
            this._textToolCallMode = true;
            return events;
          }
        }

        // First char rules out tool calls — flush immediately
        const fc = trimmed[0];
        if (fc && fc !== '[' && fc !== '<') {
          this._bufferingText = false;
          events.push(...this._flushTextBuffer());
          return events;
        }

        // Enough chars to decide it's not a known prefix
        if (trimmed.length >= 15) {
          this._bufferingText = false;
          events.push(...this._flushTextBuffer());
          return events;
        }

        // Still undecided — keep buffering
        return events;
      }

      // Normal text streaming (buffer already flushed)
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
      // Flush buffered text if switching to reasoning
      if (this._bufferingText && this._textBuffer) {
        this._bufferingText = false;
        events.push(...this._flushTextBuffer());
        events.push(...this._closeTextBlock());
      }
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

    // Handle buffered text (tool-call mode or still undecided at end of stream)
    if (this._textToolCallMode || (this._bufferingText && this._textBuffer)) {
      const { text, toolCalls } = parseTextToolCalls(this._textBuffer);

      if (toolCalls.length > 0) {
        // Emit remaining text if any
        if (text) {
          const ci = this._allocIndex();
          events.push(formatSSE('content_block_start', {
            type: 'content_block_start',
            index: ci,
            content_block: { type: 'text', text: '' },
          }));
          events.push(formatSSE('content_block_delta', {
            type: 'content_block_delta',
            index: ci,
            delta: { type: 'text_delta', text },
          }));
          events.push(formatSSE('content_block_stop', {
            type: 'content_block_stop',
            index: ci,
          }));
        }

        // Emit parsed tool calls as tool_use blocks
        for (const tc of toolCalls) {
          this._hasTools = true;
          const ci = this._allocIndex();
          events.push(formatSSE('content_block_start', {
            type: 'content_block_start',
            index: ci,
            content_block: {
              type: 'tool_use',
              id: tc.id,
              name: tc.function.name,
              input: {},
            },
          }));
          events.push(formatSSE('content_block_delta', {
            type: 'content_block_delta',
            index: ci,
            delta: { type: 'input_json_delta', partial_json: tc.function.arguments },
          }));
          events.push(formatSSE('content_block_stop', {
            type: 'content_block_stop',
            index: ci,
          }));
        }
      } else if (this._textBuffer) {
        // No tool calls found — emit buffered text as plain text
        const ci = this._allocIndex();
        events.push(formatSSE('content_block_start', {
          type: 'content_block_start',
          index: ci,
          content_block: { type: 'text', text: '' },
        }));
        events.push(formatSSE('content_block_delta', {
          type: 'content_block_delta',
          index: ci,
          delta: { type: 'text_delta', text: this._textBuffer },
        }));
        events.push(formatSSE('content_block_stop', {
          type: 'content_block_stop',
          index: ci,
        }));
      }
    } else {
      // Normal path — close any open text/thinking block
      events.push(...this._closeTextBlock());
    }

    // Close all proper API tool blocks
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
