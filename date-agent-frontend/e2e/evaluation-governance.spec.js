import { expect, test } from "@playwright/test";

import { openAs } from "./support/mock-api.js";


test("three difficulty baselines stay visible while an ineligible easy run is blocked", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "质量归因" }).click();
  const tabs = page.getByRole("tablist", { name: "评测难度" });
  await expect(tabs.getByRole("tab")).toHaveCount(3);
  await expect(tabs.getByRole("tab", { name: /简单/ })).toHaveAttribute("aria-selected", "true");
  const baselines = page.getByRole("region", { name: "三种难度观测基线" });
  await expect(baselines.locator("article")).toHaveCount(3);
  await expect(baselines.getByText("简单观测基线", { exact: true })).toBeVisible();
  await expect(baselines.getByText("中等观测基线", { exact: true })).toBeVisible();
  await expect(baselines.getByText("困难观测基线", { exact: true })).toBeVisible();
  await expect(baselines.getByText("54.0%", { exact: true })).toBeVisible();
  await expect(baselines.getByText("30.0%", { exact: true })).toBeVisible();
  await expect(baselines.getByText("16.0%", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "样本、版本指纹与简单观测基线" })).toBeVisible();
  await expect(page.getByText("50/50 完整")).toBeVisible();
  await expect(page.getByText("未登记", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "登记为简单观测基线" })).toBeDisabled();
  await expect(page.getByRole("region", { name: "建立三轮可比实验" })).toBeVisible();
  await expect(page.getByText(/尚未形成三次同指纹、全量且证据完整的观测/)).toBeVisible();
  await page.getByText("各地区的销售总额").click();
  await page.getByText("语义 IR", { exact: true }).click();
  await expect(page.getByText("AST 编译", { exact: true })).toBeVisible();
  await expect(page.getByText(/semantic-ir-v1/)).toBeVisible();
});


test("TODO97 model identity, strict SQL stages, correction attribution and unknown cost are explicit", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "质量归因" }).click();

  const run = page.locator(".run-strip");
  await expect(run.getByText("Q1", { exact: true })).toBeVisible();
  await expect(run.getByText("qwen3.7-plus", { exact: true })).toBeVisible();
  await expect(run.getByText("dashscope", { exact: true })).toBeVisible();

  const metrics = page.locator(".metric-grid");
  const first = metrics.locator("article").filter({ hasText: "First SQL 严格结果" });
  const final = metrics.locator("article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(first.getByText("48.0%", { exact: true })).toBeVisible();
  await expect(final.getByText("54.0%", { exact: true })).toBeVisible();
  await expect(final).toHaveClass(/primary/);
  await expect(final).toHaveClass(/bad/);
  await expect(final.getByText(/主要指标 · 门槛 ≥ 48\/50（96.0%，严格 >95%） · 套件回退策略/)).toBeVisible();

  const ablation = page.getByRole("region", { name: "SQL 模型消融指标" });
  await expect(ablation.getByText("6 / 50 题触发", { exact: true })).toBeVisible();
  await expect(ablation.getByText("修复结果 3 · 修复执行 1", { exact: true })).toBeVisible();
  await expect(ablation.getByText("有害修改 1 · 不可归因 1", { exact: true })).toBeVisible();

  const performance = page.locator(".performance-panel");
  await expect(performance.getByText("未知（未配置价格）", { exact: true })).toBeVisible();
  await expect(performance.getByText("0 USD", { exact: true })).toHaveCount(0);

  await page.getByText("各地区的销售总额").click();
  const stages = page.getByRole("region", { name: "eval_001 SQL 阶段指标" });
  await expect(stages.locator("article").filter({ hasText: "First SQL 严格结果" }).getByText("结果不一致", { exact: true })).toBeVisible();
  const correction = stages.locator("article").filter({ hasText: "Correction 归因" });
  await expect(correction.getByText("修复严格结果", { exact: true })).toBeVisible();
  await expect(correction.getByText(/直接归因/)).toBeVisible();
  await expect(stages.locator("article").filter({ hasText: "Final SQL 严格结果" }).getByText("结果一致", { exact: true })).toBeVisible();
});


test("evidence-complete evaluation run can be registered as workspace observational baseline", async ({ page }) => {
  const state = await openAs(page, "owner", { evaluationEligible: true });
  await page.getByRole("button", { name: "质量归因" }).click();
  await expect(page.getByRole("region", { name: "三次可比实验" })).toBeVisible();
  const final = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(final.getByText(/主要指标 · 门槛 ≥ 48\/50（96.0%，严格 >95%） · 后端治理策略/)).toBeVisible();
  await expect(page.getByText("3/3", { exact: true })).toBeVisible();
  await expect(page.getByText("观测完整，可登记", { exact: true })).toBeVisible();
  await expect(page.getByText("表路径仅观察，Final 结果为主", { exact: true })).toBeVisible();
  const tableObservation = page.getByRole("region", { name: "三次可比实验" }).locator(".experiment-checklist .observe").filter({ hasText: "表路径 Exact（观察）" });
  await expect(tableObservation).toBeVisible();
  await expect(page.getByRole("region", { name: "三次可比实验" }).getByText("表范围安全 100%", { exact: true })).toBeVisible();
  await expect(page.getByText(/工程参考目标仅用于诊断/)).toBeVisible();
  await expect(page.getByText("核心默认", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("LLM Attempt 对账", { exact: true })).toBeVisible();
  await expect(page.getByText("成功 90 · 失败 0 · 超时 0 · 取消 0", { exact: true })).toBeVisible();
  await expect(page.getByText("Attempt 状态守恒", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("语义槽位覆盖", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("共享截止时间", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("排队 / Provider P95", { exact: true })).toBeVisible();
  await expect(page.getByText("选择 B 1 · 有效 B 2 · 修复 1", { exact: true })).toBeVisible();
  await expect(page.getByLabel("模型路由性能").getByText("test-strong", { exact: false })).toBeVisible();
  await expect(page.getByRole("region", { name: "三次可比实验" }).getByText("Prompt 压缩证据", { exact: false })).toBeVisible();
  await expect(page.getByRole("region", { name: "三次可比实验" }).getByText("调用结果分类完整", { exact: false })).toBeVisible();
  await page.getByText("各地区的销售总额").click();
  await expect(page.getByText("SemanticCoverageV1", { exact: true })).toBeVisible();
  await expect(page.getByText("RequestDeadlineV1", { exact: true })).toBeVisible();
  await page.getByText(/LLM Attempt 证据 · 1 次/).click();
  const attempt = page.locator(".attempt-row").filter({ hasText: "generate_sql · sql" });
  await expect(attempt.getByText("generate_sql · sql", { exact: true })).toBeVisible();
  await expect(page.getByText("物理预算 / 恢复预留", { exact: true })).toBeVisible();
  await expect(attempt.getByText(/服务商请求 ID 不可用/)).toBeVisible();
  await expect(attempt.getByText("服务商 SDK 未提供请求 ID", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "登记为简单观测基线" })).toBeEnabled();
  await page.getByLabel("审核说明").fill("数据负责人和分析负责人已复核");
  await page.getByRole("button", { name: "登记为简单观测基线" }).click();
  await expect(page.getByText("已登记为当前工作空间简单观测基线")).toBeVisible();
  await expect(page.getByText("已登记", { exact: true }).first()).toBeVisible();
  expect(state.calls.some((item) => item.method === "POST" && item.path.includes("/baselines/"))).toBeTruthy();
});


test("difficulty tabs scope runs, report and governance requests", async ({ page }) => {
  const state = await openAs(page, "owner");
  await page.getByRole("button", { name: "质量归因" }).click();
  await page.getByRole("tab", { name: /中等/ }).click();
  await expect(page.getByRole("tab", { name: /中等/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".run-strip").getByText("中等", { exact: true })).toBeVisible();
  await expect(page.locator(".run-strip").getByText("50 条", { exact: true })).toBeVisible();
  await expect(page.locator(".metric-grid").getByText("30.0%", { exact: true })).toBeVisible();
  const mediumFinal = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(mediumFinal).toHaveClass(/bad/);
  await expect(mediumFinal.getByText(/门槛 ≥ 43\/50（86.0%，严格 >85%）/)).toBeVisible();
  await expect(page.getByText("中等 SQL 基线集", { exact: true }).first()).toBeVisible();
  await page.getByRole("tab", { name: /困难/ }).click();
  await expect(page.getByRole("tab", { name: /困难/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".run-strip").getByText("困难", { exact: true })).toBeVisible();
  await expect(page.locator(".run-strip").getByText("50 条", { exact: true })).toBeVisible();
  await expect(page.locator(".metric-grid").getByText("16.0%", { exact: true })).toBeVisible();
  const hardFinal = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(hardFinal).toHaveClass(/observe/);
  await expect(hardFinal.getByText(/主要指标 · 仅观测，不设强制正确率门槛/)).toBeVisible();
  for (const path of ["/api/admin/evaluation/nl2sql/runs", "/api/admin/evaluation/nl2sql/latest", "/api/admin/evaluation/nl2sql/governance"]) {
    expect(state.calls.some((item) => item.path === path && item.query.difficulty === "medium")).toBeTruthy();
    expect(state.calls.some((item) => item.path === path && item.query.difficulty === "hard")).toBeTruthy();
  }
});


test("fresh three-tier governance keeps qualification, incomplete evidence and observations distinct", async ({ page }) => {
  await openAs(page, "owner", { evaluationFreshGovernance: true });
  await page.getByRole("button", { name: "质量归因" }).click();

  let experiment = page.getByRole("region", { name: "三次可比实验" });
  await expect(experiment.getByText("3/3", { exact: true })).toBeVisible();
  await expect(experiment.getByText("三轮已完成，阻断证据未闭合", { exact: true })).toBeVisible();
  let scopeSummary = experiment.locator(".experiment-metrics div").filter({ hasText: "最差范围安全 / 证据完整" });
  await expect(scopeSummary.getByText("48/50 / 48/50", { exact: true })).toBeVisible();
  let scopeEvidenceCheck = experiment.locator(".experiment-checklist > div").filter({ hasText: "表范围证据完整" });
  await expect(scopeEvidenceCheck).toHaveClass(/fail/);
  const securityCheck = experiment.locator(".experiment-checklist > div").filter({ hasText: "安全守卫无绕过" });
  await expect(securityCheck).toHaveClass(/pass/);
  let tableObservation = experiment.locator(".experiment-checklist > div").filter({ hasText: "表路径 Exact（观察）" });
  await expect(tableObservation).toHaveClass(/observe/);
  const easyR3 = experiment.locator(".experiment-run").filter({ hasText: "20260809_123828_451550" });
  await expect(easyR3.locator("span").nth(3)).toHaveText("45/50");
  await expect(easyR3.locator("span").nth(4)).toHaveText("48/50");
  await expect(easyR3.locator("span").nth(5)).toHaveText("48/50");
  await expect(easyR3.locator("span").nth(6)).toHaveText("47/50");
  const easyFinal = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(easyFinal).toHaveClass(/primary/);
  await expect(easyFinal).toHaveClass(/bad/);
  await expect(easyFinal.getByText("94.0%", { exact: true })).toBeVisible();

  await page.getByRole("tab", { name: /中等/ }).click();
  experiment = page.getByRole("region", { name: "三次可比实验" });
  await expect(experiment.getByText("满足工程参考目标", { exact: true })).toBeVisible();
  scopeSummary = experiment.locator(".experiment-metrics div").filter({ hasText: "最差范围安全 / 证据完整" });
  await expect(scopeSummary.getByText("50/50 / 50/50", { exact: true })).toBeVisible();
  scopeEvidenceCheck = experiment.locator(".experiment-checklist > div").filter({ hasText: "表范围证据完整" });
  await expect(scopeEvidenceCheck).toHaveClass(/pass/);
  tableObservation = experiment.locator(".experiment-checklist > div").filter({ hasText: "表路径 Exact（观察）" });
  await expect(tableObservation).toHaveClass(/observe/);
  const mediumR3 = experiment.locator(".experiment-run").filter({ hasText: "20260809_130914_348751" });
  await expect(mediumR3.locator("span").nth(3)).toHaveText("49/50");
  await expect(mediumR3.locator("span").nth(4)).toHaveText("50/50");
  await expect(mediumR3.locator("span").nth(5)).toHaveText("50/50");
  await expect(mediumR3.locator("span").nth(6)).toHaveText("46/50");
  const mediumFinal = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(mediumFinal).toHaveClass(/primary/);
  await expect(mediumFinal).toHaveClass(/good/);
  await expect(mediumFinal.getByText("92.0%", { exact: true })).toBeVisible();

  await page.getByRole("tab", { name: /困难/ }).click();
  experiment = page.getByRole("region", { name: "三次可比实验" });
  await expect(experiment.getByText("三轮已完成，阻断证据未闭合", { exact: true })).toBeVisible();
  scopeSummary = experiment.locator(".experiment-metrics div").filter({ hasText: "最差范围安全 / 证据完整" });
  await expect(scopeSummary.getByText("48/50 / 48/50", { exact: true })).toBeVisible();
  tableObservation = experiment.locator(".experiment-checklist > div").filter({ hasText: "表路径 Exact（观察）" });
  await expect(tableObservation).toHaveClass(/observe/);
  const hardR3 = experiment.locator(".experiment-run").filter({ hasText: "20260809_135357_495896" });
  await expect(hardR3.locator("span").nth(3)).toHaveText("37/50");
  await expect(hardR3.locator("span").nth(4)).toHaveText("48/50");
  await expect(hardR3.locator("span").nth(5)).toHaveText("48/50");
  await expect(hardR3.locator("span").nth(6)).toHaveText("22/50");
  const hardFinal = page.locator(".metric-grid article").filter({ hasText: "Final SQL 最终结果正确率" });
  await expect(hardFinal).toHaveClass(/primary/);
  await expect(hardFinal).toHaveClass(/observe/);
  await expect(hardFinal.getByText("44.0%", { exact: true })).toBeVisible();
  await expect(hardFinal.getByText(/仅观测，不设强制正确率门槛/)).toBeVisible();
});


test("approved baseline with a missing artifact is shown as a governance error", async ({ page }) => {
  await openAs(page, "owner", { evaluationUnresolvableBaseline: "medium" });
  await page.getByRole("button", { name: "质量归因" }).click();
  const baselines = page.getByRole("region", { name: "三种难度观测基线" });
  const medium = baselines.locator("article").filter({ hasText: "中等观测基线" });

  await expect(medium).toHaveClass(/unresolvable/);
  await expect(medium.getByText("已审批但制品不可解析", { exact: true })).toBeVisible();
  await expect(medium.getByText("已审批，但基线制品不存在或无法访问。", { exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: /中等/ })).toContainText("基线制品异常");
});
