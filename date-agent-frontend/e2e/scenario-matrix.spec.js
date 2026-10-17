import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("owner can inspect 56 executable business scenarios and filter evidence", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "场景矩阵" }).click();
  await expect(page.getByRole("heading", { name: "核心业务场景矩阵" })).toBeVisible();
  await expect(page.getByLabel("场景矩阵摘要").getByRole("article").filter({ hasText: "场景总数" }).getByText("56", { exact: true })).toBeVisible();
  await expect(page.getByText("门禁通过", { exact: true })).toBeVisible();
  await page.getByLabel("能力").selectOption("access_policy");
  await page.getByLabel("风险").selectOption("critical");
  const row = page.locator("details.case-row").first();
  await expect(row).toBeVisible();
  await row.locator("summary").click();
  await expect(row.getByText(/automated evidence access_policy/)).toBeVisible();
});

test("viewer cannot see the scenario matrix entry", async ({ page }) => {
  await openAs(page, "viewer");
  await expect(page.getByRole("button", { name: "场景矩阵" })).toHaveCount(0);
});
