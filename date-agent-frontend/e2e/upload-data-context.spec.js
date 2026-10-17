import { expect, test } from "@playwright/test";
import { openAs, sampleDataset } from "./support/mock-api.js";

test("上传隔离页面展示独立分母、证据等级与零污染门禁", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByRole("button", { name: "上传隔离" }).click();

  await expect(page.getByRole("heading", { name: "上传数据资产隔离门禁" })).toBeVisible();
  await expect(page.getByText("隔离合同通过")).toBeVisible();
  await expect(page.getByText("离线合同证据 · 不声明生产等价", { exact: true })).toBeVisible();
  await expect(page.getByText("21 / 21")).toBeVisible();
  await expect(page.getByText("越权数据库执行").locator("..").getByText("0", { exact: true })).toBeVisible();
  await expect(page.getByText("清理成功率").locator("..").getByText("100.0%")).toBeVisible();
  await expect(page.getByText("真实用户夹具").locator("..").getByText("0", { exact: true })).toBeVisible();
});

test("数据上下文选择上传表并独立对比，查询载荷和 Trace 使用同一上下文", async ({ page }) => {
  const state = await openAs(page, "owner");
  const hiddenChartWarnings = [];
  page.on("console", (message) => {
    if (message.type() === "warning" && message.text().includes("Can't get DOM width or height")) {
      hiddenChartWarnings.push(message.text());
    }
  });

  await page.getByTestId("data-context-trigger").click();
  const drawer = page.getByRole("dialog", { name: "本轮使用哪些数据？" });
  await expect(drawer).toBeVisible();
  await drawer.getByRole("button", { name: /sales\.csv/ }).click();
  await drawer.getByRole("button", { name: /与 DW 独立对比/ }).click();
  await drawer.getByRole("button", { name: "应用数据上下文" }).click();

  await expect(page.getByTestId("data-context-trigger")).toContainText("sales.csv");
  await expect(page.getByTestId("data-context-trigger")).toContainText("独立对比");

  await page.getByLabel("输入问题").fill("对比上传销售额与正式数据");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();
  await page.getByLabel(/处理完成.*查看执行详情/).click();
  await expect(page.getByText("数据上下文 · 独立对比")).toBeVisible();
  await expect(page.getByText("确定性来源对比 · compared")).toBeVisible();
  await expect(page.getByText(/按 region 对齐 3 组/)).toBeVisible();
  await page.getByText("查看逐组差异（3）").click();
  const eastComparison = page.getByRole("row", { name: /region：华东/ });
  await expect(eastComparison).toBeVisible();
  await expect(eastComparison).toContainText("25,000");

  const queryCall = state.calls.filter((item) => item.path === "/api/query").at(-1);
  expect(queryCall.body.dataset_ids).toEqual(["dataset-1"]);
  expect(queryCall.body.source_mode).toBe("upload_compare_dw");
  expect(queryCall.body.comparison_intent).toBe("independent_aggregate_compare");
  expect(queryCall.body.mapping_confirmation_id).toBeNull();
  expect(hiddenChartWarnings).toEqual([]);

  const contextSave = state.calls.filter((item) => /\/api\/chats\/[^/]+\/data-context$/.test(item.path) && item.method === "PUT").at(-1);
  expect(contextSave.body.source_mode).toBe("upload_compare_dw");
  expect(contextSave.body.dataset_ids).toEqual(["dataset-1"]);
});

test("已有会话切换上传数据使用应用内确认弹层并保留历史", async ({ page }) => {
  await openAs(page, "owner");
  await page.getByLabel("输入问题").fill("先查询正式数据");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();

  await page.getByRole("button", { name: "临时数据", exact: true }).click();
  const useButton = page.getByRole("button", { name: "用于问数", exact: true });
  await useButton.click();

  const confirmDialog = page.getByRole("dialog", { name: "开始新的数据上下文？" });
  await expect(confirmDialog).toBeVisible();
  await expect(confirmDialog).toContainText("sales.csv");
  await expect(confirmDialog).toContainText("现有会话仍保留在历史列表中");

  await confirmDialog.getByRole("button", { name: "保留当前会话" }).click();
  await expect(confirmDialog).toHaveCount(0);
  await expect(useButton).toBeFocused();

  await useButton.click();
  await confirmDialog.getByRole("button", { name: "开始新对话并选择" }).click();
  const drawer = page.getByRole("dialog", { name: "本轮使用哪些数据？" });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole("button", { name: /sales\.csv/ })).toHaveAttribute("aria-pressed", "true");
});

test("最多两个上传版本可组合，重登后恢复会话默认数据上下文", async ({ page }) => {
  const second = {
    ...structuredClone(sampleDataset),
    dataset_id: "dataset-2",
    version_id: "dataset-2",
    asset_id: "asset-2",
    version_no: 2,
    filename: "targets.csv",
  };
  const state = await openAs(page, "owner", {
    datasets: [structuredClone(sampleDataset), second],
  });

  await page.getByTestId("data-context-trigger").click();
  const drawer = page.getByRole("dialog", { name: "本轮使用哪些数据？" });
  await drawer.getByRole("button", { name: /sales\.csv/ }).click();
  await drawer.getByRole("button", { name: /targets\.csv/ }).click();
  await drawer.getByRole("button", { name: /两个上传表关联/ }).click();
  await drawer.getByRole("button", { name: "应用数据上下文" }).click();

  await page.getByLabel("输入问题").fill("比较两个上传版本");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();
  const queryCall = state.calls.filter((item) => item.path === "/api/query").at(-1);
  expect(queryCall.body.dataset_ids).toEqual(["dataset-1", "dataset-2"]);
  expect(queryCall.body.source_mode).toBe("multi_upload");
  expect(queryCall.body.comparison_intent).toBe("multi_upload");

  await expect.poll(() => state.calls.filter((item) =>
    /\/api\/chats\/[^/]+\/data-context$/.test(item.path)
    && item.method === "PUT").length).toBe(1);
  await page.reload();
  await expect(page.getByTestId("data-context-trigger")).toContainText("sales.csv + targets.csv");
  await expect(page.getByTestId("data-context-trigger")).toContainText("双上传表");
});
