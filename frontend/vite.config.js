import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev:  npm run dev  -> http://localhost:5173  (Electron dev loads this URL)
// /api/* proxies to FastAPI so the browser uses relative URLs.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    allowedHosts: true, // allow sandboxed preview hosts
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
