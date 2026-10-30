import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';
import { apiService } from '../services/api.js';

let server, Integrity;
before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, watch: null } });
  ({ default: Integrity } = await server.ssrLoadModule('/src/pages/Integrity.jsx'));
});
after(async () => { await server?.close(); });

test('ingest SHA-256 is Original; Current remains unchecked until verification', () => {
  const hash = 'a1'.repeat(32);
  const html = renderToStaticMarkup(React.createElement(Integrity, {
    activeEvidence: { id: 8, original_filename: 'sample.jpg', sha256_hash: hash,
      mime_type: 'image/jpeg', file_size: 100 },
  }));
  const [original, current] = html.split('Current SHA-256');
  assert.ok(original.includes(hash));
  assert.ok(!current.includes(hash), 'stored ingest hash must not masquerade as current bytes');
  assert.ok(current.includes('Not checked'));
  assert.ok(!html.includes('Calculated on active buffer'));
});

test('without selected evidence verification is disabled', () => {
  const html = renderToStaticMarkup(React.createElement(Integrity, { activeEvidence: null }));
  assert.match(html, /<button[^>]*disabled/);
  assert.ok(html.includes('WAITING FOR EVIDENCE'));
});

test('API verification uses POST and returns distinct original/current hash values', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.ok(url.endsWith('/api/evidence/8/verify'));
    assert.equal(options.method, 'POST');
    return { ok: true, json: async () => ({ status: 'MISMATCH', original_sha256: 'a', current_sha256: 'b' }) };
  };
  try {
    const result = await apiService.verifyIntegrity(8);
    assert.equal(result.status, 'MISMATCH');
    assert.notEqual(result.original_sha256, result.current_sha256);
  } finally { globalThis.fetch = originalFetch; }
});

test('API errors are surfaced instead of reporting verified', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, json: async () => ({ detail: 'Read failed' }) });
  try { await assert.rejects(apiService.verifyIntegrity(8), /Read failed/); }
  finally { globalThis.fetch = originalFetch; }
});
