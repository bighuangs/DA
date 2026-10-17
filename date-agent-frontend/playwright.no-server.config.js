import { defineConfig } from "@playwright/test";

// Used by local/CI orchestrators that own the Vite process lifecycle.  Keeping
// webServer out of this config avoids Windows npm child-process teardown leaks.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  timeout: 30_000,
  expect: { timeout: 7_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:4175",
    browserName: "chromium",
    locale: "zh-CN",
    colorScheme: "light",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
});
