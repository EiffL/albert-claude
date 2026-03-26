/**
 * Configuration management for le-claude.
 *
 * Stores config in ~/.config/le-claude/config.json
 */

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const DEFAULT_BASE_URL = 'https://albert.api.etalab.gouv.fr/v1';

function configDir() {
  const xdg = process.env.XDG_CONFIG_HOME || path.join(process.env.HOME, '.config');
  return path.join(xdg, 'le-claude');
}

function configPath() {
  return path.join(configDir(), 'config.json');
}

/** Load saved config, or return null if none exists. */
export function loadConfig() {
  try {
    const data = fs.readFileSync(configPath(), 'utf-8');
    return JSON.parse(data);
  } catch {
    return null;
  }
}

/** Save config to disk. */
export function saveConfig(config) {
  const dir = configDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(configPath(), JSON.stringify(config, null, 2) + '\n', {
    mode: 0o600, // readable only by owner
  });
}

// ---------------------------------------------------------------------------
// Interactive setup
// ---------------------------------------------------------------------------

function ask(rl, question) {
  return new Promise(resolve => rl.question(question, resolve));
}

/** Fetch available models from the Albert API. */
export async function fetchModels(baseUrl, apiKey) {
  const headers = {};
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

  const res = await fetch(`${baseUrl}/models`, { headers, signal: AbortSignal.timeout(15_000) });
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  const data = await res.json();

  // Filter to text-generation models and sort by name
  return (data.data || [])
    .filter(m => m.type === 'text-generation' || m.id?.includes('gpt') || m.id?.includes('llama'))
    .sort((a, b) => (a.id || '').localeCompare(b.id || ''));
}

/** Interactive model picker. Shows available models and returns chosen ID. */
async function pickModel(rl, models) {
  if (models.length === 0) {
    console.error('  No text-generation models found. Enter model ID manually.');
    return await ask(rl, '  Model ID: ');
  }
  console.error('');
  console.error('  Available models:');
  models.forEach((m, i) => {
    const type = m.type ? ` (${m.type})` : '';
    console.error(`    ${i + 1}. ${m.id}${type}`);
  });
  console.error('');
  const choice = await ask(rl, '  Select model [1]: ');
  const idx = (parseInt(choice, 10) || 1) - 1;
  return models[Math.max(0, Math.min(idx, models.length - 1))].id;
}

/** Fetch models and interactively select one. Returns model ID string. */
export async function selectModel(baseUrl, apiKey) {
  const models = await fetchModels(baseUrl, apiKey);
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr,
  });
  try {
    return await pickModel(rl, models);
  } finally {
    rl.close();
  }
}

/** Run interactive setup. Returns config object. */
export async function interactiveSetup(existingConfig = null) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr, // use stderr so stdout stays clean for claude
  });

  try {
    console.error('');
    console.error('  le-claude - Use Claude Code with Albert API');
    console.error('');

    if (!existingConfig) {
      console.error('  No configuration found. Let\'s set things up!');
      console.error('');
    }

    // API key
    const apiKey = await ask(rl, '  Albert API Key: ');
    if (!apiKey.trim()) {
      console.error('  API key is required.');
      process.exit(1);
    }

    const baseUrl = existingConfig?.baseUrl || DEFAULT_BASE_URL;

    // Test connection
    process.stderr.write('  Testing connection... ');
    let models;
    try {
      models = await fetchModels(baseUrl, apiKey.trim());
      console.error('ok');
    } catch (err) {
      console.error(`failed: ${err.message}`);
      console.error('  Please check your API key and try again.');
      process.exit(1);
    }

    // Model selection
    const model = await pickModel(rl, models);

    const config = {
      apiKey: apiKey.trim(),
      model,
      baseUrl,
    };

    saveConfig(config);
    console.error('');
    console.error(`  Configuration saved to ${configPath()}`);
    console.error('');

    return config;
  } finally {
    rl.close();
  }
}
