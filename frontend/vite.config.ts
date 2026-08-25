import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", ["VITE_", "NEXTGRAPH_"]);
  const frontendPort = Number(env.NEXTGRAPH_FRONTEND_PORT || "5173");
  const backendHost = env.NEXTGRAPH_BACKEND_HOST || "127.0.0.1";
  const backendPort = env.NEXTGRAPH_BACKEND_PORT || "5190";
  const queryApiPort = env.NEXTGRAPH_QUERY_API_PORT || "8710";
  const entityGraphPort = env.NEXTGRAPH_ENTITY_GRAPH_PORT || "8711";
  const proxyHost = backendHost === "0.0.0.0" ? "127.0.0.1" : backendHost;
  const backendOrigin = `http://${proxyHost}:${backendPort}`;
  const queryApiOrigin = `http://${proxyHost}:${queryApiPort}`;
  const entityGraphOrigin = `http://${proxyHost}:${entityGraphPort}`;

  return {
    plugins: [react()],
    envDir: "..",
    envPrefix: ["VITE_", "NEXTGRAPH_"],
    server: {
      port: frontendPort,
      proxy: {
        "/api/search/external/query/stream": queryApiOrigin,
        "/api/search/external/query": queryApiOrigin,
        "/api/entity-graph": entityGraphOrigin,
        "/files": backendOrigin,
        "/parse_file": backendOrigin,
        "/database_build": backendOrigin,
        "/api": backendOrigin,
      },
    },
  };
});
