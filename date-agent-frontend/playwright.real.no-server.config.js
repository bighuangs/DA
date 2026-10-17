import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-real",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 7_000 },
  reporter: [["list"]],
  use: {
    // The runner assigns isolated ports and points Vite at the matching
    // FastAPI/SQLite smoke backend, so both request and browser fixtures cross
    // the same production proxy/HTTP boundary without touching developer data.
    baseURL: process.env.REAL_FRONTEND_SMOKE_URL || "http://127.0.0.1:4174",
    browserName: "chromium",
    locale: "zh-CN",
    trace: "retain-on-failure",
  },
});
