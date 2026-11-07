import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

// The page imports ../records, ../demo and gate_config.json at build time, so every number is read, never typed.
// The two CSVs are served byte for byte (no transform) so the in-browser SHA-256 matches records/test_frozen.sha256.
const DATA = ['credit.csv', 'test_frozen.csv'];
const dataFile = (f) => path.resolve(__dirname, '..', 'data', f);
const serveData = {
  name: 'serve-repo-data',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const f = DATA.find((d) => req.url.endsWith(`/data/${d}`));
      if (!f) return next();
      res.setHeader('content-type', 'text/csv');
      res.end(fs.readFileSync(dataFile(f)));
    });
  },
  generateBundle() {
    for (const f of DATA) this.emitFile({ type: 'asset', fileName: `data/${f}`, source: fs.readFileSync(dataFile(f)) });
  },
};

export default defineConfig({
  base: '/pattern19-wk1-model-gate/',
  plugins: [react(), serveData],
  server: { fs: { allow: ['..'] } },
});
