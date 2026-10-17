import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("owner can inspect semantic repair registry, ablation and rollback boundary", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "语义修复" }).click();

  await expect(page.getByRole("heading", { name: "语义守卫与分层纠错" })).toBeVisible();
  await expect(page.getByTestId("semantic-repair-evidence-boundary")).toContainText("不扩大线上自动修复范围");
  await expect(page.getByTestId("semantic-active-profile")).toContainText("P2");
  await expect(page.getByTestId("semantic-active-profile")).toContainText("版本化同规则注册表");
  await expect(page.getByRole("table", { name: "语义修复 P0 到 P6 消融表" }).locator("tbody tr")).toHaveCount(7);
  await expect(page.getByText("唯一元数据连接边", { exact: true })).toBeVisible();
  await expect(page.getByText("唯一分组维度（影子）", { exact: true })).toBeVisible();
  await expect(page.getByText("缺少真实模型定向纠错对", { exact: true })).toBeVisible();
  await expect(page.getByText("100.0%", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("P3–P6 BLOCKED", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "立即回退 P0" }).click();
  await expect(page.getByText(/回切 P0 inline_v1/)).toBeVisible();
  await expect(page.getByRole("button", { name: "当前已在 P0" })).toBeDisabled();
});

test("viewer cannot see the semantic repair evaluation entry", async ({ page }) => {
  await openAs(page, "viewer");
  await expect(page.getByRole("button", { name: "语义修复" })).toHaveCount(0);
});
