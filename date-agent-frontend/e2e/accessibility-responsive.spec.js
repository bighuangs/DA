import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { installMockApi, openAs } from "./support/mock-api.js";

async function expectNoSeriousA11y(page) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  const blocking = result.violations.filter((item) => ["serious", "critical"].includes(item.impact));
  expect(blocking, blocking.map((item) => `${item.id}: ${item.help}`).join("\n")).toEqual([]);
}

async function expectNoPageOverflow(page) {
  const metrics = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth }));
  expect(metrics.page).toBeLessThanOrEqual(metrics.viewport);
  await expect(page.getByRole("button", { name: "发送问题" })).toBeVisible();
}

test("登录页 axe serious/critical 为 0", async ({ page }) => {
  await installMockApi(page);
  await page.goto("/");
  await expectNoSeriousA11y(page);
});

test("问数首页 axe serious/critical 为 0 且 Tab/Enter 可操作", async ({ page }) => {
  await openAs(page);
  await page.getByLabel("输入问题").focus();
  await page.getByLabel("输入问题").fill("键盘发送测试");
  await page.getByLabel("输入问题").press("Enter");
  await expect(page.locator(".user-bubble").filter({ hasText: "键盘发送测试" })).toBeVisible();
  await expectNoSeriousA11y(page);
});

test("390px 手机视口无页面级横向溢出", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openAs(page);
  await expectNoPageOverflow(page);
  await page.getByRole("button", { name: "打开主菜单" }).click();
  await expect(page.getByRole("button", { name: "质量运营" })).toBeVisible();
  await page.getByRole("button", { name: "质量运营" }).click();
  await expect(page.getByRole("heading", { name: "质量运营" })).toBeVisible();
  const columns = await page.getByLabel("核心质量指标").evaluate((element) => getComputedStyle(element).gridTemplateColumns);
  expect(columns.trim().split(/\s+/)).toHaveLength(1);
  await expectNoSeriousA11y(page);
});

test("768px 平板视口无页面级横向溢出", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await openAs(page);
  await expectNoPageOverflow(page);
});

test("1366px 桌面视口无页面级横向溢出", async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  await openAs(page);
  await expectNoPageOverflow(page);
});
