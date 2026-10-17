import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("owner can inspect resilience gates, limits and probe evidence", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "稳定性评测" }).click();
  await expect(page.getByRole("heading", { name: "依赖故障、并发与资源边界" })).toBeVisible();
  await expect(page.getByLabel("稳定性评测摘要").getByText("23", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("确定性门禁通过", { exact: true })).toBeVisible();
  await expect(page.getByText("未运行", { exact: true })).toBeVisible();
  await page.getByLabel("类别").selectOption("concurrency");
  const row = page.locator("details.case").first();
  await expect(row).toBeVisible();
  await row.locator("summary").click();
  await expect(row.getByText(/error_rate/)).toBeVisible();
});

test("viewer cannot see the resilience evaluation entry", async ({ page }) => {
  await openAs(page, "viewer");
  await expect(page.getByRole("button", { name: "稳定性评测" })).toHaveCount(0);
});
