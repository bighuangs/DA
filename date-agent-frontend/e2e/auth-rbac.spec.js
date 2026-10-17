import { expect, test } from "@playwright/test";
import { installMockApi, openAs } from "./support/mock-api.js";

test("失效 token 会清理会话并返回登录页", async ({ page }) => {
  await installMockApi(page, { expiredToken: true });
  await page.addInitScript(() => localStorage.setItem("da_access_token", "expired-token"));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "登录数据助手" })).toBeVisible();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("da_access_token"))).toBeNull();
});

test("用户可以登录并退出", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/");
  await page.getByLabel("用户名").fill("alice");
  await page.getByLabel("密码").fill("alice12345");
  await page.getByRole("button", { name: "登录并进入" }).click();
  await expect(page.getByText("从一个业务问题开始")).toBeVisible();
  await page.getByRole("button", { name: "退出登录" }).click();
  await expect(page.getByRole("heading", { name: "登录数据助手" })).toBeVisible();
});

test("新用户可以注册工作空间", async ({ page }) => {
  await installMockApi(page, { username: "bob" });
  await page.goto("/");
  await page.getByRole("button", { name: "首次使用？创建账号与工作空间" }).click();
  await page.getByLabel("用户名").fill("bob");
  await page.getByLabel("显示名称").fill("Bob");
  await page.getByLabel("密码").fill("bob12345");
  await page.getByLabel("工作空间名称").fill("Bob 工作空间");
  await page.getByRole("button", { name: "注册并进入" }).click();
  await expect(page.getByRole("navigation", { name: "主导航" })).toBeVisible();
});

test("viewer 看不到管理入口与写操作", async ({ page }) => {
  await openAs(page, "viewer");
  const nav = page.getByRole("navigation", { name: "主导航" });
  await expect(nav.getByRole("button", { name: "元知识" })).toHaveCount(0);
  await expect(nav.getByRole("button", { name: "脱敏规则" })).toHaveCount(0);
  await expect(nav.getByRole("button", { name: "数据权限" })).toHaveCount(0);
  const directWriteStatus = await page.evaluate(() => fetch("/api/admin/meta/metrics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "forbidden" }),
  }).then((response) => response.status));
  expect(directWriteStatus).toBe(403);
  await nav.getByRole("button", { name: "仪表盘" }).click();
  await expect(page.getByTitle("新建仪表盘")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "保存" })).toHaveCount(0);
});

test("analyst 可问数和编辑仪表盘但不可管理元知识", async ({ page }) => {
  await openAs(page, "analyst");
  const nav = page.getByRole("navigation", { name: "主导航" });
  await expect(nav.getByRole("button", { name: "元知识" })).toHaveCount(0);
  await nav.getByRole("button", { name: "仪表盘" }).click();
  await expect(page.getByTitle("新建仪表盘")).toBeVisible();
});
