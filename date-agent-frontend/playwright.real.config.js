import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-real",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 7_000 },
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:4174",
    browserName: "chromium",
    locale: "zh-CN",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: ".\\.venv\\Scripts\\python.exe -m evals.legacy.real_backend_smoke_server",
      cwd: "../data-agent",
      url: "http://127.0.0.1:8001/health",
      reuseExistingServer: false,
      timeout: 120_000,
      env: { REAL_BACKEND_SMOKE_PORT: "8001" },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4174 --strictPort",
      url: "http://127.0.0.1:4174",
      reuseExistingServer: false,
      timeout: 120_000,
      env: { VITE_API_PROXY_TARGET: "http://127.0.0.1:8001" },
    },
  ],
});
