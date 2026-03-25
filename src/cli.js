/**
 * CLI entry point for albert-claude.
 *
 * Orchestrates: config → proxy → claude code → cleanup.
 */

import { spawn, execFileSync } from 'node:child_process';
import { loadConfig, interactiveSetup } from './config.js';
import { startProxy } from './proxy.js';

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { debug: false, model: null, setup: false, help: false, claudeArgs: [] };
  let i = 0;
  while (i < args.length) {
    if (args[i] === '--debug') { opts.debug = true; i++; }
    else if (args[i] === '--setup') { opts.setup = true; i++; }
    else if (args[i] === '--help' || args[i] === '-h') { opts.help = true; i++; }
    else if (args[i] === '--model' && i + 1 < args.length) { opts.model = args[i + 1]; i += 2; }
    else if (args[i] === '--') { opts.claudeArgs = args.slice(i + 1); break; }
    else { opts.claudeArgs = args.slice(i); break; }
  }
  return opts;
}

function printHelp() {
  console.error(`
  albert-claude - Use Claude Code with France's Albert API

  Usage:
    npx albert-claude [options] [-- claude-args...]

  Options:
    --setup         Re-run interactive setup
    --model MODEL   Override the model for this session
    --debug         Enable proxy debug logging
    -h, --help      Show this help message

  Examples:
    npx albert-claude                  Start Claude Code with Albert
    npx albert-claude --debug          Start with debug logging
    npx albert-claude --model gpt-4o   Use a specific model
    npx albert-claude -- --help        Pass --help to Claude Code

  Configuration is stored in ~/.config/albert-claude/config.json
  `);
}

// ---------------------------------------------------------------------------
// Claude Code launcher
// ---------------------------------------------------------------------------

function findClaude() {
  try {
    execFileSync('which', ['claude'], { stdio: 'pipe' });
    return 'claude';
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const opts = parseArgs();

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  // Load or create config
  let config = loadConfig();

  if (!config || opts.setup) {
    config = await interactiveSetup(config);
  }

  // CLI overrides
  const model = opts.model || config.model;

  // Check claude is installed
  const claudeBin = findClaude();
  if (!claudeBin) {
    console.error('  Error: "claude" command not found.');
    console.error('  Install Claude Code first: https://docs.anthropic.com/en/docs/claude-code');
    console.error('');
    process.exit(1);
  }

  // Start proxy
  if (opts.debug) process.stderr.write('  Starting proxy... ');
  const { server, port } = await startProxy({
    port: 0,
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    model,
    debug: opts.debug,
  });
  if (opts.debug) console.error(`ok (port ${port})`);

  const claudeEnv = {
    ...process.env,
    ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`,
    ANTHROPIC_AUTH_TOKEN: 'albert-claude',
    ANTHROPIC_API_KEY: '',
    ANTHROPIC_CUSTOM_MODEL_OPTION: model,
    ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: `Albert ${model}`,
  };

  const claudeArgs = ['--model', model, ...opts.claudeArgs];
  const claude = spawn(claudeBin, claudeArgs, {
    stdio: 'inherit',
    env: claudeEnv,
  });

  // Clean shutdown
  function cleanup() {
    server.close();
  }

  claude.on('exit', (code) => {
    cleanup();
    process.exit(code || 0);
  });

  claude.on('error', (err) => {
    console.error(`  Failed to start Claude Code: ${err.message}`);
    cleanup();
    process.exit(1);
  });

  process.on('SIGINT', () => {
    claude.kill('SIGINT');
  });

  process.on('SIGTERM', () => {
    claude.kill('SIGTERM');
    cleanup();
  });
}

main().catch(err => {
  console.error(`  Error: ${err.message}`);
  process.exit(1);
});
