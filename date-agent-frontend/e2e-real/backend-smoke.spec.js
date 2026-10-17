import { expect, test } from "@playwright/test";

test("真实 HTTP、SQLite、认证、RBAC 与 SSE 契约可联通", async ({ request }) => {
  const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
  const ownerName = `owner-${suffix}`;
  const viewerName = `viewer-${suffix}`;
  const password = "SmokePass123!";

  const ownerRegister = await request.post("/api/auth/register", { data: {
    username: ownerName, display_name: "Smoke Owner", password, workspace_name: "Smoke Workspace",
  }});
  expect(ownerRegister.status()).toBe(201);
  const owner = await ownerRegister.json();
  const ownerHeaders = { Authorization: `Bearer ${owner.access_token}` };

  const me = await request.get("/api/auth/me", { headers: ownerHeaders });
  expect(me.status()).toBe(200);
  expect((await me.json()).workspace.role).toBe("owner");

  const viewerRegister = await request.post("/api/auth/register", { data: {
    username: viewerName, display_name: "Smoke Viewer", password, workspace_name: "Viewer Own Workspace",
  }});
  expect(viewerRegister.status()).toBe(201);
  const viewer = await viewerRegister.json();

  const addMember = await request.post("/api/workspaces/current/members", {
    headers: ownerHeaders, data: { username: viewerName, role: "viewer" },
  });
  expect(addMember.status()).toBe(200);

  const switchWorkspace = await request.post("/api/auth/switch-workspace", {
    headers: { Authorization: `Bearer ${viewer.access_token}` },
    data: { workspace_id: owner.user.workspace.id },
  });
  expect(switchWorkspace.status()).toBe(200);
  const viewerInOwnerWorkspace = await switchWorkspace.json();
  const viewerHeaders = { Authorization: `Bearer ${viewerInOwnerWorkspace.access_token}` };

  const forbidden = await request.get("/api/admin/evaluation/scenarios/matrix", { headers: viewerHeaders });
  expect(forbidden.status()).toBe(403);

  const query = await request.post("/api/query", {
    headers: ownerHeaders, data: { query: "真实后端冒烟" },
  });
  expect(query.status()).toBe(200);
  expect(query.headers()["content-type"]).toContain("text/event-stream");
  const stream = await query.text();
  expect(stream).toContain('"sql": "SELECT 1 AS smoke_value"');
  expect(stream).toContain('"result": [{"smoke_value": 1}]');
  expect(stream).toContain('"trace": {"trace_id": "real-smoke-trace"');
  expect(stream).toContain('"done": true');

  const directory = await request.get("/api/chats", { headers: ownerHeaders });
  expect(directory.status()).toBe(200);
  const directoryBody = await directory.json();
  expect(directoryBody.items).toHaveLength(1);
  expect(directoryBody.items[0].chat_id).toBe("real-smoke-chat");

  const transcript = await request.get("/api/chats/real-smoke-chat/turns", { headers: ownerHeaders });
  expect(transcript.status()).toBe(200);
  const transcriptBody = await transcript.json();
  expect(transcriptBody.items[0].sql).toBe("SELECT 1 AS smoke_value");
  expect(transcriptBody.items[0].result.rows).toEqual([{ smoke_value: 1 }]);

  const isolatedDirectory = await request.get("/api/chats", { headers: viewerHeaders });
  expect(isolatedDirectory.status()).toBe(200);
  expect((await isolatedDirectory.json()).items).toEqual([]);
  const hiddenTranscript = await request.get("/api/chats/real-smoke-chat/turns", { headers: viewerHeaders });
  expect(hiddenTranscript.status()).toBe(404);

  const invalidToken = await request.get("/api/auth/me", { headers: { Authorization: "Bearer invalid" } });
  expect(invalidToken.status()).toBe(401);
});

test("浏览器 Fetch 可接收 SSE trace 与显式终态", async ({ page, request }) => {
  const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
  const registration = await request.post("/api/auth/register", { data: {
    username: `stream-${suffix}`,
    display_name: "Stream Smoke",
    password: "SmokePass123!",
    workspace_name: "Stream Smoke Workspace",
  }});
  expect(registration.status()).toBe(201);
  const session = await registration.json();
  await page.goto("/");
  const stream = await page.evaluate(async (token) => {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: "浏览器原始流" }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let output = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      output += decoder.decode(value, { stream: true });
    }
    return output;
  }, session.access_token);
  expect(stream).toContain('"trace": {"trace_id": "real-smoke-trace"');
  expect(stream).toContain('"done": true');
});

test("真实浏览器完成问数后出现目录并可在刷新后恢复", async ({ page }) => {
  const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
  await page.goto("/");
  await page.getByRole("button", { name: "首次使用？创建账号与工作空间" }).click();
  await page.getByLabel("用户名").fill(`browser-${suffix}`);
  await page.getByLabel("显示名称").fill("Browser Smoke");
  await page.getByLabel("密码").fill("SmokePass123!");
  await page.getByLabel("工作空间名称").fill("Browser Smoke Workspace");
  await page.getByRole("button", { name: "注册并进入" }).click();

  await page.getByLabel("输入问题").fill("验证真实会话恢复");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("SELECT 1 AS smoke_value")).toBeVisible();
  await expect(page.locator(".trace-summary")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "发送问题" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "打开会话：验证真实会话恢复" })).toBeVisible();

  await page.reload();
  await expect(page.locator(".user-bubble").filter({ hasText: "验证真实会话恢复" })).toBeVisible();
  await expect(page.getByText("历史结果快照")).toBeVisible();
  await expect(page.getByRole("cell", { name: "1" })).toBeVisible();
});
