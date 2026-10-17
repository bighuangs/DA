import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("owner can inspect planning candidates, feature ablation and release boundary", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "规划路由" }).click();

  await expect(page.getByRole("heading", { name: "SQL 规划复杂度路由" })).toBeVisible();
  await expect(page.getByTestId("planning-evidence-boundary")).toContainText("不是 Full Run 发布结论");
  await expect(page.getByTestId("active-profile")).toContainText("C1");
  await expect(page.getByTestId("active-profile")).toContainText("关键词 + 长度基线");
  await expect(page.getByTestId("recommended-profile")).toContainText("C4");
  await expect(page.getByTestId("recommended-profile")).toContainText("加权评分卡");
  await expect(page.getByRole("table", { name: "SQL 规划候选对照表" }).locator("tbody tr")).toHaveCount(6);
  await expect(page.getByText("A0", { exact: true })).toBeVisible();
  await expect(page.getByText("A4", { exact: true })).toBeVisible();
  await expect(page.getByText("缺少真实 Direct / Plan 配对", { exact: true })).toBeVisible();
  await expect(page.getByText("Time To Last Token", { exact: true })).toBeVisible();
  await expect(page.getByText("Cache TTL", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "已在安全基线" })).toBeDisabled();
});

test("viewer cannot see the planning evaluation entry", async ({ page }) => {
  await openAs(page, "viewer");
  await expect(page.getByRole("button", { name: "规划路由" })).toHaveCount(0);
});
