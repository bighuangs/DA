import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

async function ask(page) {
  await page.getByLabel("输入问题").fill("各地区的销售总额");
  await page.getByRole("button", { name: "发送问题" }).click();
}

test("SSE 问数完整展示阶段、SQL、表格、图表、解读、推荐和 trace", async ({ page }) => {
  await openAs(page);
  await page.getByLabel("AI 数据解读").check();
  await ask(page);
  await expect(page.getByText("处理完成")).toBeVisible();
  await expect(page.getByText("SELECT region, SUM(amount) AS amount FROM fact_sales GROUP BY region")).toBeVisible();
  await expect(page.getByRole("cell", { name: "120,000" })).toBeVisible();
  await expect(page.getByText("地区销售额", { exact: true })).toBeVisible();
  await expect(page.getByText("华东地区销售额最高，领先华南 22,000 元。")).toBeVisible();
  await expect(page.getByRole("button", { name: "查看月度趋势" })).toBeVisible();
  const result = page.getByLabel("查询结果");
  await expect(result).toHaveCount(1);
  await expect(result.getByRole("tab", { name: "查看表格" })).toHaveAttribute("aria-selected", "true");
  await result.getByRole("tab", { name: "查看图表" }).click();
  await expect(result.getByRole("tab", { name: "查看图表" })).toHaveAttribute("aria-selected", "true");
  await page.getByLabel(/处理完成，查看执行详情/).click();
  await expect(page.getByText(/规划路由 · 直接生成/)).toBeVisible();
  await expect(page.getByText(/关键词 \+ 长度 · 决策/)).toBeVisible();
  await expect(page.getByText(/加权评分卡 → 直接生成/)).toBeVisible();
  await expect(page.getByText(/语义守卫 · 校验通过/)).toBeVisible();
  await expect(page.getByText(/P2 同规则注册表/)).toBeVisible();
  await expect(page.getByText(/未触发确定性改写/)).toBeVisible();
  await expect(page.getByText(/候选判优 · A/)).toBeVisible();
  await expect(page.getByText(/高置信单候选快路径/)).toBeVisible();
  await expect(page.getByText("321 tokens")).toBeVisible();
  await page.getByText("召回职责与版本").click();
  await expect(page.getByText("Qdrant · column")).toBeVisible();
  await expect(page.getByText("ES · 真实字段值")).toBeVisible();
  await expect(page.getByText(/Sparse 推断 literal 否/)).toBeVisible();
  await expect(page.getByText(/原值日志 否/)).toBeVisible();
});

test("结构化推荐展示简短文案并提交带当前结果范围的完整问题", async ({ page }) => {
  const state = await openAs(page);
  await ask(page);
  await expect(page.getByText("处理完成")).toBeVisible();

  await expect(page.getByRole("button", { name: "查看月度趋势" })).toBeVisible();
  await expect(page.getByRole("button", { name: "分析品类贡献" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看客户排行" })).toBeVisible();
  await page.getByRole("button", { name: "分析品类贡献" }).click();

  await expect.poll(
    () => state.calls.filter((call) => call.path === "/api/query").length,
  ).toBe(2);
  const queryCalls = state.calls.filter((call) => call.path === "/api/query");
  expect(queryCalls[1].body).toMatchObject({
    query: "查询华东地区各品类销售额，并按销售额从高到低排序",
    chat_id: "chat-1",
  });
  expect(queryCalls[1].body.query).not.toBe("分析品类贡献");
  await expect(
    page.getByRole("main").getByText(
      "查询华东地区各品类销售额，并按销售额从高到低排序",
      { exact: true },
    ),
  ).toBeVisible();
});

test("等待首个 SSE 事件时显示准备中而不是错误的完成态", async ({ page }) => {
  await openAs(page, "owner", { queryDelayMs: 350 });
  await ask(page);
  await expect(page.getByText("正在准备…")).toBeVisible();
  await expect(page.getByText("处理完成")).toHaveCount(0);
  await expect(page.getByText("处理完成")).toBeVisible();
});

test("AI 数据解读默认关闭、可手动开启并记住选择", async ({ page }) => {
  const state = await openAs(page);
  const toggle = page.getByLabel("AI 数据解读");
  await expect(toggle).not.toBeChecked();

  await ask(page);
  await expect(page.getByText("处理完成")).toBeVisible();
  await expect(page.getByText("华东地区销售额最高，领先华南 22,000 元。")).toHaveCount(0);
  let queryCalls = state.calls.filter((call) => call.path === "/api/query");
  expect(queryCalls.at(-1).body.enable_analysis).toBe(false);

  await toggle.check();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("da_analysis_enabled"))).toBe("true");
  await ask(page);
  await expect(page.getByText("处理完成").last()).toBeVisible();
  await expect(page.getByText("华东地区销售额最高，领先华南 22,000 元。")).toBeVisible();
  queryCalls = state.calls.filter((call) => call.path === "/api/query");
  expect(queryCalls.at(-1).body.enable_analysis).toBe(true);

  await page.reload();
  await expect(page.getByLabel("AI 数据解读")).toBeChecked();
});

test("多轮省略追问展示确定性闭合与 standalone question", async ({ page }) => {
  const state = await openAs(page);
  await page.getByLabel("输入问题").fill("各品类的销量总和");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();

  await page.getByLabel("输入问题").fill("那销售额呢");
  await page.getByRole("button", { name: "发送问题" }).click();
  await page.getByLabel(/处理完成，查看执行详情/).last().click();
  await expect(page.getByText("多轮闭合 · replace")).toBeVisible();
  await expect(page.getByText("各品类的销售额总和")).toBeVisible();
  await expect(page.getByText("SELECT category, SUM(amount) AS amount FROM fact_sales GROUP BY category")).toBeVisible();
  const queryCalls = state.calls.filter((call) => call.path === "/api/query");
  expect(queryCalls.at(-1).body.chat_id).toBe("chat-1");
});

test("多轮只修改图表类型时把最新结果从柱状图切换为饼图", async ({ page }) => {
  const state = await openAs(page);

  await page.getByLabel("输入问题").fill("销量前十的商品");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();
  await expect(page.getByLabel("查询结果").last().locator(".view-badge")).toHaveText("柱状图");

  await page.getByLabel("输入问题").fill("改为饼图");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成").last()).toBeVisible();

  const queryCalls = state.calls.filter((call) => call.path === "/api/query");
  expect(queryCalls).toHaveLength(2);
  expect(queryCalls[1].body).toMatchObject({ query: "改为饼图", chat_id: "chat-1" });

  const latestResult = page.getByLabel("查询结果").last();
  await expect(latestResult.locator(".view-badge")).toHaveText("饼图");
  await latestResult.getByRole("tab", { name: "查看图表" }).click();
  await expect(latestResult.locator(".chart-canvas")).toBeVisible();
  await expect.poll(() => latestResult.locator(".chart-canvas").evaluate(async (element) => {
    const echarts = await import("/@id/echarts/core");
    return echarts.getInstanceByDom(element)?.getOption()?.series?.[0]?.type || null;
  })).toBe("pie");
});

test("编辑 SQL 后可以安全重跑并渲染新结果", async ({ page }) => {
  await openAs(page);
  await ask(page);
  await page.getByRole("button", { name: "编辑" }).click();
  await page.getByLabel("编辑 SQL").fill("SELECT region, SUM(amount) AS amount FROM fact_sales GROUP BY region LIMIT 10");
  await page.getByRole("button", { name: "运行 SQL" }).click();
  await expect(page.getByRole("cell", { name: "华北" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "86,000" })).toBeVisible();
});

test("问数接口 5xx 会给出后端错误详情", async ({ page }) => {
  await openAs(page, "owner", { queryError: true });
  await ask(page);
  await expect(page.getByText("受控查询失败")).toBeVisible();
});

test("FastAPI 对象校验错误会转换为可读文本", async ({ page }) => {
  await openAs(page, "owner", { queryObjectError: true });
  await ask(page);
  await expect(page.getByText("问题不能为空")).toBeVisible();
  await expect(page.getByText("[object Object]")).toHaveCount(0);
});

test("必要模型不可用时展示结构化补充建议且不展示简化 SQL", async ({ page }) => {
  await openAs(page, "owner", { queryClarification: true });
  await ask(page);
  await expect(page.getByText("建议补充后重试")).toBeVisible();
  await expect(page.getByText("明确要统计的指标及聚合方式")).toBeVisible();
  await expect(page.getByText("operator:set")).toBeVisible();
  await expect(page.getByText(/SELECT region/)).toHaveCount(0);
});

test("反馈失败保留输入并展示可读服务错误", async ({ page }) => {
  await openAs(page, "owner", { feedbackObjectError: true });
  await page.getByLabel("AI 数据解读").check();
  await ask(page);
  await page.getByRole("button", { name: "无帮助" }).click();
  await page.getByLabel("反馈说明").fill("口径解释不清楚");
  await page.getByRole("button", { name: "提交反馈" }).click();
  await expect(page.getByText(/反馈服务暂时不可用/)).toBeVisible();
  await expect(page.getByLabel("反馈说明")).toHaveValue("口径解释不清楚");
});

test("后端拒绝编辑 SQL 时保留安全错误", async ({ page }) => {
  await openAs(page, "owner", { execError: true });
  await ask(page);
  await page.getByRole("button", { name: "编辑" }).click();
  await page.getByLabel("编辑 SQL").fill("DELETE FROM fact_sales");
  await page.getByRole("button", { name: "运行 SQL" }).click();
  await expect(page.getByText("SQL 只允许只读查询")).toBeVisible();
});
