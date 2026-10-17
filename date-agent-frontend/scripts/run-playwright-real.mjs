import { once } from "node:events";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { resolve } from "node:path";
import process from "node:process";


async function reserveFreePort() {
  const server = createServer();
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  await new Promise((resolvePromise, reject) => {
    server.close((error) => error ? reject(error) : resolvePromise());
  });
  if (!port) throw new Error("无法分配真实冒烟后端端口");
  return port;
}


const backendRoot = resolve(process.cwd(), "../data-agent");
const backendPort = await reserveFreePort();
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendPort = await reserveFreePort();
const frontendUrl = `http://127.0.0.1:${frontendPort}`;
const backend = spawn(
  resolve(backendRoot, ".venv/Scripts/python.exe"),
  ["-m", "evals.legacy.real_backend_smoke_server"],
  {
    cwd: backendRoot,
    stdio: "ignore",
    windowsHide: true,
    env: { ...process.env, REAL_BACKEND_SMOKE_PORT: String(backendPort) },
  },
);
const frontend = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "--host", "127.0.0.1",
    "--port", String(frontendPort),
    "--strictPort",
  ],
  {
    cwd: process.cwd(),
    stdio: "ignore",
    windowsHide: true,
    env: { ...process.env, VITE_API_PROXY_TARGET: backendUrl },
  },
);
async function waitFor(url, processHandle, label) {
  const timeoutMs = Number(process.env.REAL_BACKEND_SMOKE_TIMEOUT_MS || 90_000);
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (processHandle.exitCode !== null) {
      throw new Error(`${label} 提前退出，exit=${processHandle.exitCode}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Service is still starting.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 200));
  }
  throw new Error(`等待 ${label} 超时（${timeoutMs} ms）`);
}


async function stop(child) {
  if (child.exitCode !== null) return;
  child.kill();
  await Promise.race([
    once(child, "exit"),
    new Promise((resolvePromise) => setTimeout(resolvePromise, 2000)),
  ]);
  if (child.exitCode === null) child.kill("SIGKILL");
}


let exitCode = 1;
try {
  await waitFor(`${backendUrl}/health`, backend, "真实冒烟后端");
  await waitFor(frontendUrl, frontend, "真实冒烟前端");
  const playwright = spawn(
    process.execPath,
    [
      "node_modules/@playwright/test/cli.js",
      "test",
      "--config=playwright.real.no-server.config.js",
      ...process.argv.slice(2),
    ],
    {
      cwd: process.cwd(),
      stdio: "inherit",
      windowsHide: true,
      env: {
        ...process.env,
        REAL_BACKEND_SMOKE_URL: backendUrl,
        REAL_FRONTEND_SMOKE_URL: frontendUrl,
      },
    },
  );
  const [code] = await once(playwright, "exit");
  exitCode = Number.isInteger(code) ? code : 1;
} catch (error) {
  console.error(error);
} finally {
  await stop(frontend);
  await stop(backend);
}
process.exit(exitCode);
