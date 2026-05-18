import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/files": "http://127.0.0.1:8004",
      "/parse_file": "http://127.0.0.1:8004",
      "/database_build": "http://127.0.0.1:8004",
    },
  },
});
