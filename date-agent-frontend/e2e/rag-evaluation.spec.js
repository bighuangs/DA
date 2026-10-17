import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test("owner can inspect dynamic retrieval, context budget and case evidence", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "RAG 评测" }).click();
  await expect(page.getByRole("heading", { name: "RAG 召回与上下文质量" })).toBeVisible();
  await expect(page.getByLabel("RAG 评测摘要").getByText("100.0%", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("离线确定性索引", { exact: true })).toBeVisible();
  await expect(page.getByText("27.3% → 51.0%", { exact: true })).toBeVisible();
  await expect(page.getByText("2875.0 → 1760.0", { exact: true })).toBeVisible();
  await expect(page.getByText("字段向量 5.2", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "BGE-M3 Dense + Sparse 实测" })).toBeVisible();
  await expect(page.getByText(/首轮 16 题 Recall/)).toBeVisible();
  await expect(page.getByText("R-H3", { exact: true })).toBeVisible();
  await expect(page.getByText("长尾 Recall +18.0 pp", { exact: true })).toBeVisible();
  await expect(page.getByText("v1 回退：关闭", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Point 投影策略消融" })).toBeVisible();
  await expect(page.getByText("选择已生效", { exact: true })).toBeVisible();
  await expect(page.getByText("35 / 35", { exact: true })).toBeVisible();
  await expect(page.getByText("Point 变化").locator("..").getByText("-96", { exact: true })).toBeVisible();
  const capacity = page.getByLabel("Embedding 调度容量对比");
  await expect(capacity.getByRole("heading", { name: "BGE-M3 Embedding 推理调度" })).toBeVisible();
  await expect(capacity.getByText("容量门禁通过", { exact: true })).toBeVisible();
  await expect(capacity.getByText("吞吐 7.3×", { exact: true })).toBeVisible();
  await expect(capacity.getByText("有界动态微批", { exact: true })).toBeVisible();
  await expect(capacity.getByText("512.0 MB", { exact: true })).toBeVisible();
  await expect(capacity.getByText("推理期 RSS 增量 Δ +6.0 MB", { exact: true })).toBeVisible();
  const signals = page.getByLabel("查询信号消融对比");
  await expect(signals.getByRole("heading", { name: "查询信号提取与路由消融" })).toBeVisible();
  await expect(signals.getByText("信号门禁通过", { exact: true })).toBeVisible();
  await expect(signals.getByText("K1 · jieba TF-IDF", { exact: true })).toBeVisible();
  await expect(signals.getByText("K2 · 领域信号", { exact: true })).toBeVisible();
  await expect(signals.getByText("保留 jieba_tfidf_v1，未自动切换", { exact: true })).toBeVisible();
  await expect(signals.getByText("K2 Column 调用 Δ").locator("..").getByText("-2.55", { exact: true })).toBeVisible();
  await expect(signals.getByText("租户/隐私").locator("..").getByText("通过", { exact: true })).toBeVisible();
  await expect(signals.getByText("已跑真实 RAG", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K1 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("通过", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K2 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("未通过", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K1 查询信号方案").getByText("真实字段 Recall").locator("..").getByText("100.0%", { exact: true })).toBeVisible();
  await expect(page.getByLabel("真实召回缺口").getByText("RAG-018", { exact: true })).toBeVisible();
  await page.getByLabel("真实召回缺口").getByText("RAG-018", { exact: true }).click();
  await expect(page.getByText("fact_order.order_quantity", { exact: false })).toBeVisible();
  const row = page.locator("details.case").first();
  await row.locator("summary").click();
  await expect(row.getByText("dim_date.month", { exact: false }).first()).toBeVisible();
});

test("query-signal live evidence keeps K1 when lower-call challengers regress", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "RAG 评测" }).click();

  const signals = page.getByLabel("查询信号消融对比");
  await expect(signals.getByText("已跑真实 RAG", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K0 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("未通过", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K1 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("通过", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K2 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("未通过", { exact: true })).toBeVisible();
  await expect(page.getByLabel("K3 查询信号方案").getByText("真实 RAG 门禁").locator("..").getByText("未运行", { exact: true })).toBeVisible();
  await expect(signals.getByText("保留 jieba_tfidf_v1，未自动切换", { exact: true })).toBeVisible();
  await expect(signals.getByText("明文 Trace：关闭", { exact: true })).toBeVisible();
});

test("viewer cannot see the RAG evaluation entry", async ({ page }) => {
  await openAs(page, "viewer");
  await expect(page.getByRole("button", { name: "RAG 评测" })).toHaveCount(0);
});
