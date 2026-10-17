import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

const now = "2026-07-28T08:30:00Z";
const earlier = "2026-07-27T06:00:00Z";

function conversation(chatId, title, status = "active", lastMessageAt = now) {
  return {
    chat_id: chatId,
    title,
    preview: `${title}的最近一次提问`,
    status,
    last_turn_status: "succeeded",
    turn_count: 1,
    created_at: earlier,
    last_message_at: lastMessageAt,
    updated_at: lastMessageAt,
    archived_at: status === "archived" ? lastMessageAt : null,
    context_reset_at: null,
    version: 2,
  };
}

function turn(chatId, question, sql, rows, chart = null) {
  return {
    turn_id: `turn-${chatId}`,
    chat_id: chatId,
    trace_id: `trace-${chatId}`,
    status: "succeeded",
    created_at: now,
    completed_at: now,
    user_text: question,
    assistant_summary: `查询成功，返回 ${rows.length} 行。`,
    sql,
    result: {
      columns: Object.keys(rows[0] || {}),
      rows,
      row_count: rows.length,
      truncated: false,
    },
    chart,
    replay_fidelity: "result_snapshot",
    unavailable_parts: chart ? ["analysis", "recommendation"] : ["chart", "analysis", "recommendation"],
  };
}

function historyFixture() {
  return {
    conversations: [
      conversation("chat-latest", "月度销售趋势", "active", now),
      conversation("chat-older", "大客户名单", "active", earlier),
      conversation("chat-archived", "去年区域复盘", "archived", earlier),
    ],
    conversationTurns: {
      "chat-latest": [
        turn(
          "chat-latest",
          "最近两个月销售趋势",
          "SELECT month, SUM(amount) AS amount FROM fact_sales GROUP BY month",
          [{ month: "2026-06", amount: 120000 }, { month: "2026-07", amount: 135000 }],
          {
            type: "line",
            title: "最近两个月销售趋势",
            axis: {
              x: { name: "month", value: "month" },
              y: [{ name: "amount", value: "amount" }],
            },
          },
        ),
      ],
      "chat-older": [
        turn(
          "chat-older",
          "大客户有哪些",
          "SELECT customer_name, SUM(amount) AS amount FROM fact_sales GROUP BY customer_name",
          [{ customer_name: "客户甲", amount: 88000 }],
        ),
      ],
      "chat-archived": [
        turn(
          "chat-archived",
          "去年各区域销售额",
          "SELECT region, SUM(amount) AS amount FROM fact_sales GROUP BY region",
          [{ region: "华东", amount: 980000 }],
        ),
      ],
    },
  };
}

test("重登自动恢复最近会话，历史快照不写入 localStorage", async ({ page }) => {
  await openAs(page, "owner", historyFixture());

  await expect(page.getByText("月度销售趋势", { exact: true }).last()).toBeVisible();
  await expect(page.getByText("最近两个月销售趋势")).toBeVisible();
  await expect(page.getByText("SELECT month, SUM(amount) AS amount FROM fact_sales GROUP BY month")).toBeVisible();
  await expect(page.getByRole("cell", { name: "135,000" })).toBeVisible();
  await expect(page.getByText(/未保存：AI 解读、追问建议/)).toBeVisible();
  const result = page.getByLabel("查询结果");
  await expect(result.locator(".view-badge")).toHaveText("折线图");
  await result.getByRole("tab", { name: "查看图表" }).click();
  await expect.poll(() => result.locator(".chart-canvas").evaluate(async (element) => {
    const echarts = await import("/@id/echarts/core");
    return echarts.getInstanceByDom(element)?.getOption()?.series?.[0]?.type || null;
  })).toBe("line");

  const storage = await page.evaluate(() => Object.fromEntries(Object.entries(localStorage)));
  expect(Object.values(storage)).not.toContain("最近两个月销售趋势");
  expect(Object.values(storage)).not.toContain("135000");
  expect(storage["da_last_chat:user-alice:workspace-1"]).toBe("chat-latest");

  await page.reload();
  await expect(page.getByText("最近两个月销售趋势")).toBeVisible();
  await expect(page.getByText("月度销售趋势", { exact: true }).last()).toBeVisible();
});

test("切换历史会话后追问复用原 chat_id，归档只读且可恢复", async ({ page }) => {
  const state = await openAs(page, "owner", historyFixture());
  const olderButton = page.getByRole("button", { name: "打开会话：大客户名单" });
  await olderButton.click();
  await expect(page.getByText("大客户有哪些")).toBeVisible();

  await page.getByLabel("输入问题").fill("那按地区拆分呢");
  await page.getByRole("button", { name: "发送问题" }).click();
  await expect(page.getByText("处理完成")).toBeVisible();
  const queryCall = state.calls.filter((call) => call.path === "/api/query").at(-1);
  expect(queryCall.body.chat_id).toBe("chat-older");

  const selectedItem = page.getByRole("button", { name: "打开会话：大客户名单" }).locator("..");
  await selectedItem.getByRole("button", { name: "归档会话" }).click();
  await expect(page.getByText("已归档 · 只读浏览")).toBeVisible();
  await expect(page.getByLabel("输入问题")).toBeDisabled();
  await page.getByRole("button", { name: "恢复并继续" }).click();
  await expect(page.getByLabel("输入问题")).toBeEnabled();
  await expect(page.getByText(/已恢复 \d+ 轮对话/)).toBeVisible();
});

test("归档列表支持恢复与软删除确认", async ({ page }) => {
  await openAs(page, "owner", historyFixture());
  await page.getByRole("tab", { name: "已归档" }).click();
  const archivedButton = page.getByRole("button", { name: "打开会话：去年区域复盘" });
  await expect(archivedButton).toBeVisible();

  await archivedButton.locator("..").getByRole("button", { name: "移除历史会话" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除历史会话？" });
  await expect(deleteDialog).toContainText("去年区域复盘");
  await deleteDialog.getByRole("button", { name: "删除会话" }).click();
  await expect(archivedButton).toHaveCount(0);
  await expect(page.getByText("会话已删除")).toBeVisible();
});
