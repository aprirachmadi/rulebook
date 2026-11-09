import { defineConfig } from "vite";

// Dev: `npm run dev` proxies /api to the FastAPI server. Prod: `npm run build` -> dist/,
// served by FastAPI at /.
export default defineConfig({
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
