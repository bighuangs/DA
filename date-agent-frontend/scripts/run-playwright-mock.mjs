import { once } from "node:events";
import { spawn } from "node:child_process";
import process from "node:process";


const vite = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "--host", "127.0.0.1",
    "--port", "4175",
    "--strictPort",
  ],
  {
    cwd: process.cwd(),
    stdio: "ignore",
    windowsHide: true,
  },
);


async function waitForServer() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (vite.exitCode !== null) {
      throw new Error(`Vite 提前退出，exit=${vite.exitCode}`);
    }
    try {
      const response = await fetch("http://127.0.0.1:4175");
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("等待 Vite 4175 端口超时");
}


async function stopServer() {
  if (vite.exitCode !== null) return;
  vite.kill();
  await Promise.race([
    once(vite, "exit"),
    new Promise((resolve) => setTimeout(resolve, 2000)),
  ]);
  if (vite.exitCode === null) vite.kill("SIGKILL");
}


let exitCode = 1;
try {
  await waitForServer();
  const playwright = spawn(
    process.execPath,
    [
      "node_modules/@playwright/test/cli.js",
      "test",
      "--config=playwright.no-server.config.js",
      ...process.argv.slice(2),
    ],
    {
      cwd: process.cwd(),
      stdio: "inherit",
      windowsHide: true,
      env: process.env,
    },
  );
  const [code] = await once(playwright, "exit");
  exitCode = Number.isInteger(code) ? code : 1;
} catch (error) {
  console.error(error);
} finally {
  await stopServer();
}
process.exit(exitCode);
