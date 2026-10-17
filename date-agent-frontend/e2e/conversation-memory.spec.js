import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";


test("问数展示分层记忆证据并可查看 Redis、权威层和双投影状态", async ({ page }) => {
  const state = await openAs(page);
  await page.getByLabel("输入问题").fill("各地区的销售总额");
  await page.getByRole("button", { name: "发送问题" }).click();

  await page.getByLabel(/处理完成，查看执行详情/).click();
  await expect(page.getByText(/分层会话记忆.*redis/i)).toBeVisible();
  await expect(page.getByText(/召回摘要 1/)).toBeVisible();
  await expect(page.getByText(/已回 MySQL 复核/)).toBeVisible();
  await page.getByTestId("memory-toggle").click();

  const panel = page.getByTestId("memory-panel");
  await expect(panel).toBeVisible();
  await expect(panel.getByText("命中", { exact: true })).toBeVisible();
  await expect(panel.getByText("运行中", { exact: true })).toBeVisible();
  await expect(panel.getByText("可追溯摘要记忆", { exact: true })).toBeVisible();
  await expect(panel.getByText(/压缩 70%/)).toBeVisible();
  await expect(panel.getByText(/用户持续查看各地区销售额/)).toBeVisible();
  await expect(panel.getByText("身份事实", { exact: true })).toBeVisible();
  await expect(panel.getByText("以后默认回答用中文")).toBeVisible();
  await expect(panel.getByText(/来源会话 chat-older/)).toBeVisible();
  await expect(panel.getByText("就绪", { exact: true })).toHaveCount(4);

  expect(state.calls.some(
    (call) => call.method === "GET" && call.path === "/api/chat/chat-1/memory",
  )).toBe(true);
});


test("用户可显式重置短期上下文并保留跨会话长期记忆", async ({ page }) => {
  const state = await openAs(page);
  await page.getByLabel("输入问题").fill("各地区的销售总额");
  await page.getByRole("button", { name: "发送问题" }).click();
  await page.getByTestId("memory-toggle").click();
  await page.getByRole("button", { name: "重置短期上下文" }).click();
  const confirmDialog = page.getByRole("dialog", { name: "重置短期上下文？" });
  await confirmDialog.getByRole("button", { name: "确认重置" }).click();
  const resultDialog = page.getByRole("dialog", { name: "会话记忆已更新" });
  await expect(resultDialog).toContainText("跨会话长期记忆仍保留");
  await resultDialog.getByRole("button", { name: "知道了" }).click();

  await expect(page.locator(".user-bubble").filter({ hasText: "各地区的销售总额" })).toBeVisible();
  await expect(page.getByText(/当前上下文已重置；可见历史仍完整保留/)).toBeVisible();
  await expect(page.getByText(/上下文曾重置，历史仍完整保留/)).toBeVisible();
  const resetCall = state.calls.find(
    (call) => call.method === "DELETE" && call.path === "/api/chat/chat-1/memory",
  );
  expect(resetCall).toBeTruthy();
  expect(state.memoryStatus.long_term.active_count).toBe(4);
});
