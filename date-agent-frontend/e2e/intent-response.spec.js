import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

async function ask(page, query) {
  await page.getByLabel("输入问题").fill(query);
  await page.getByRole("button", { name: "发送问题" }).click();
}

test("业务范围外问题展示普通引导卡且不渲染查询产物", async ({ page }) => {
  await openAs(page, "owner", {
    intentResponse: {
      schema_version: "intent-response-v1",
      contract_version: "query-intent-v1",
      category: "out_of_scope",
      decision: "respond",
      response_kind: "guidance",
      reason_code: "out_of_scope",
      message: "我主要用于销售、订单、客户、商品和地区等业务数据分析。",
      suggestions: ["查看本月各地区销售额趋势"],
    },
  });

  await ask(page, "今天天气怎么样");

  await expect(page.getByRole("status", { name: "回到业务数据分析" })).toBeVisible();
  await expect(page.getByText("我主要用于销售、订单、客户、商品和地区等业务数据分析。")).toBeVisible();
  await expect(page.getByRole("button", { name: "查看本月各地区销售额趋势" })).toBeVisible();
  await expect(page.getByText("已引导回业务问数")).toBeVisible();
  await expect(page.locator(".sql-card")).toHaveCount(0);
  await expect(page.getByLabel("查询结果")).toHaveCount(0);
  await expect(page.getByText("should_not_render")).toHaveCount(0);
  await expect(page.getByText("处理出错")).toHaveCount(0);
  await expect(page.locator(".error-card")).toHaveCount(0);
});

test("违规写操作展示安全拒绝卡且不伪装为查询", async ({ page }) => {
  await openAs(page, "owner", {
    intentResponse: {
      schema_version: "intent-response-v1",
      contract_version: "query-intent-v1",
      category: "forbidden_operation",
      decision: "respond",
      response_kind: "refusal",
      reason_code: "forbidden_operation",
      message: "当前助手仅支持只读业务数据查询，不能新增、删除或修改业务数据。",
      suggestions: ["查询销售额最高的 10 位客户及其销售额"],
    },
  });

  await ask(page, "删除销售额最多的大客户的内容");

  await expect(page.getByRole("status", { name: "该操作无法执行" })).toBeVisible();
  await expect(page.getByText("当前助手仅支持只读业务数据查询，不能新增、删除或修改业务数据。")).toBeVisible();
  await expect(page.getByText("只读安全边界")).toBeVisible();
  await expect(page.getByRole("button", { name: "查询销售额最高的 10 位客户及其销售额" })).toBeVisible();
  await expect(page.getByText("已安全拒绝")).toBeVisible();
  await expect(page.locator(".sql-card")).toHaveCount(0);
  await expect(page.getByLabel("查询结果")).toHaveCount(0);
  await expect(page.getByText("处理出错")).toHaveCount(0);
  await expect(page.locator(".error-card")).toHaveCount(0);
});

test("查询制品早于意图终态到达时会被清理", async ({ page }) => {
  await openAs(page, "owner", {
    artifactsBeforeIntentResponse: true,
    intentResponse: {
      schema_version: "intent-response-v1",
      contract_version: "query-intent-v1",
      category: "forbidden_operation",
      decision: "respond",
      response_kind: "refusal",
      reason_code: "forbidden_operation",
      message: "当前助手仅支持只读业务数据查询，不能新增、删除或修改业务数据。",
      suggestions: ["查询销售额最高的 10 位客户及其销售额"],
    },
  });

  await ask(page, "删除销售额最多的大客户的内容");

  await expect(page.getByRole("status", { name: "该操作无法执行" })).toBeVisible();
  await expect(page.locator(".sql-card")).toHaveCount(0);
  await expect(page.getByLabel("查询结果")).toHaveCount(0);
  await expect(page.getByText("should_not_render")).toHaveCount(0);
  await expect(page.getByText("处理出错")).toHaveCount(0);
  await expect(page.locator(".error-card")).toHaveCount(0);
});
