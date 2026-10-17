import { expect, test } from "@playwright/test";
import { openAs, sampleDataset } from "./support/mock-api.js";

test("元知识可新增指标并完成 Schema 预览同步", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "元知识" }).click();
  const openEditor = page.getByRole("button", { name: "新增指标" });
  await openEditor.click();
  const editor = page.getByRole("dialog", { name: "新增指标口径" });
  await expect(editor).toHaveAttribute("aria-modal", "true");
  await expect(editor.getByLabel("名称")).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(editor.getByRole("button", { name: "关闭元知识编辑弹层" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(editor.getByRole("button", { name: "保存", exact: true })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(editor).toHaveCount(0);
  await expect(openEditor).toBeFocused();

  await openEditor.click();
  await page.getByLabel("名称").fill("GMV");
  await page.getByLabel("业务定义").fill("已支付订单成交总额");
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.locator(".name-cell strong").filter({ hasText: "GMV" })).toBeVisible();
  await page.getByRole("button", { name: "同步 Schema" }).click();
  const schemaDialog = page.getByRole("dialog", { name: "发布 Schema 变更？" });
  await expect(schemaDialog).toContainText("新增表 1");
  await schemaDialog.getByRole("button", { name: "发布并重建" }).click();
  await expect(page.getByText("Schema 同步完成")).toBeVisible();
  expect(state.calls.some((call) => call.path === "/api/admin/meta/schema/sync")).toBeTruthy();

  const metricRow = page.getByRole("row").filter({ hasText: "GMV" });
  const editButton = metricRow.getByRole("button", { name: "编辑" });
  await editButton.click();
  const editDialog = page.getByRole("dialog", { name: "编辑指标口径" });
  await expect(editDialog.getByLabel("业务定义")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(editDialog).toHaveCount(0);
  await expect(editButton).toBeFocused();

  await metricRow.getByRole("button", { name: "删除" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除指标口径？" });
  await expect(deleteDialog).toContainText("不再参与元知识召回");
  await deleteDialog.getByRole("button", { name: "确认删除" }).click();
  await expect(metricRow).toHaveCount(0);
});

test("元知识页明确展示并直接覆盖重建 BGE-M3 双向量索引", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "元知识" }).click();
  const status = page.getByLabel("BGE-M3 混合索引状态");
  await expect(status.getByText("BGE-M3 Dense + Sparse", { exact: true })).toBeVisible();
  await expect(status.getByText("v1 Dense 回退")).toBeVisible();
  await expect(status.getByText("单一结构化主 Point", { exact: true })).toBeVisible();
  await expect(status.getByText("point-identity-v2", { exact: true })).toBeVisible();
  const runtime = page.getByLabel("BGE-M3 推理调度状态");
  await expect(runtime.getByText("有界动态微批", { exact: true })).toBeVisible();
  await runtime.locator("summary").click();
  await expect(runtime.getByText("4 次合批", { exact: true })).toBeVisible();
  await expect(runtime.getByText("28 完成 / 0 拒绝", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "覆盖重建 BGE-M3 混合索引" }).click();
  const rebuildDialog = page.getByRole("dialog", { name: "覆盖重建 BGE-M3 混合索引？" });
  await expect(rebuildDialog).toContainText("不保留 v1 Dense 回退");
  await rebuildDialog.getByRole("button", { name: "覆盖重建" }).click();
  await expect(page.getByText("BGE-M3 混合索引已覆盖")).toBeVisible();
  const call = state.calls.find((item) => item.path === "/api/admin/meta/rebuild");
  expect(call.body).toEqual({ target: "all" });
});

test("元知识页不会把过期内容指纹误报为索引就绪", async ({ page }) => {
  await openAs(page, "owner", {
    hybridStatus: {
      active_mode: "bge_m3_dense_sparse",
      replacement_strategy: "overwrite_existing_collections",
      v1_fallback: false,
      ready: false,
      runtime: { model: "BAAI/bge-m3", device: "cpu", dense_size: 1024 },
      collections: {
        column: { exists: true, compatible: true, content_compatible: false, matching_points_count: 0, points_count: 72 },
        metric: { exists: true, compatible: true, points_count: 18 },
        training: { exists: true, compatible: true, points_count: 5 },
      },
    },
  });
  await page.getByRole("button", { name: "元知识" }).click();
  const status = page.getByLabel("BGE-M3 混合索引状态");
  await expect(status.getByText("需要覆盖重建", { exact: true })).toBeVisible();
  await expect(status.getByText(/当前合同 Point \/ 预期 · 版本不匹配，需覆盖重建/)).toBeVisible();
});

test("字段值页展示独立 ES 版本并按受控模式检索", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "元知识" }).click();
  const status = page.getByLabel("ES 字段值索引状态");
  await expect(status.getByText("ES 字段值索引", { exact: true })).toBeVisible();
  await expect(status.getByText("active 版本就绪", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /字段值检索/ }).click();
  await page.getByLabel("检索词").fill("华东");
  await page.getByLabel("模式").selectOption("exact");
  await page.getByLabel("表").selectOption("table-1");
  await page.getByRole("button", { name: "检索", exact: true }).click();
  await expect(page.getByRole("cell", { name: "华东" })).toBeVisible();
  await expect(page.getByText("ES 查询阶段权限过滤")).toBeVisible();
  await expect(page.getByText("es_field_value", { exact: true })).toBeVisible();
  expect(state.calls.some((call) => call.path === "/api/admin/meta/value-index/search")).toBeTruthy();
});

test("临时 CSV 可上传、用于问数、开启数据库对比并删除", async ({ page }) => {
  const state = await openAs(page, "owner", { datasets: [] });
  await page.getByRole("button", { name: "临时数据", exact: true }).click();
  await page.getByLabel("上传临时数据文件").setInputFiles({ name: "sales.csv", mimeType: "text/csv", buffer: Buffer.from("地区,销售额\n华东,120000") });
  await expect(page.getByText("sales.csv", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "用于问数" }).click();
  const drawer = page.getByRole("dialog", { name: "本轮使用哪些数据？" });
  await expect(drawer.getByRole("button", { name: /sales\.csv/ })).toHaveAttribute("aria-pressed", "true");
  await drawer.getByRole("button", { name: /与 DW 独立对比/ }).click();
  await drawer.getByRole("button", { name: "应用数据上下文" }).click();
  await expect(page.getByTestId("data-context-trigger")).toContainText("sales.csv");
  await expect(page.getByTestId("data-context-trigger")).toContainText("独立对比");
  await page.getByLabel("输入问题").fill("和数据库销售额对比");
  await page.getByRole("button", { name: "发送问题" }).click();
  const queryCall = state.calls.find((call) => call.path === "/api/query");
  expect(queryCall?.body.source_mode).toBe("upload_compare_dw");
  expect(queryCall?.body.comparison_intent).toBe("independent_aggregate_compare");
  await page.getByRole("button", { name: "临时数据", exact: true }).click();
  await page.getByRole("button", { name: "归档", exact: true }).click();
  await expect(page.getByText("archived", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "对话", exact: true }).click();
  await expect(page.getByTestId("data-context-trigger")).toContainText("正式数据仓库");
  await page.getByTestId("data-context-trigger").click();
  await expect(page.getByRole("alert")).toContainText("已安全回退到正式数据仓库");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "临时数据", exact: true }).click();
  await page.getByRole("button", { name: "删除" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除上传版本？" });
  await expect(deleteDialog).toBeVisible();
  await expect(deleteDialog).toContainText("sales.csv");
  await deleteDialog.getByRole("button", { name: "确认删除" }).click();
  await expect(page.getByText("从一个业务文件开始")).toBeVisible();
});

test("脱敏规则可以新增且弹层支持 Escape 与焦点恢复", async ({ page }) => {
  await openAs(page);
  await page.getByRole("button", { name: "脱敏规则" }).click();
  await expect(page.getByRole("heading", { name: "系统内置规则" })).toBeVisible();
  await expect(page.getByText("customer_name")).toBeVisible();
  const open = page.getByRole("button", { name: "新增规则" });
  await open.click();
  await expect(page.getByRole("dialog", { name: "新增脱敏规则" })).toBeVisible();
  await page.getByLabel("字段匹配").fill("*.phone");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(open).toBeFocused();
  await open.click();
  await page.getByLabel("字段匹配").fill("*.phone");
  await page.getByLabel("策略").selectOption("keep");
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("*.phone")).toBeVisible();
  await expect(page.getByText("保留原值", { exact: true })).toBeVisible();
  const ruleRow = page.getByRole("row").filter({ hasText: "*.phone" });
  await ruleRow.getByRole("button", { name: "删除" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除脱敏规则？" });
  await expect(deleteDialog).toContainText("*.phone");
  await deleteDialog.getByRole("button", { name: "删除规则" }).click();
  await expect(ruleRow).toHaveCount(0);
});

test("数据权限策略可以新增并持久化允许列和行过滤", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "数据权限" }).click();
  await page.getByRole("button", { name: "新增策略" }).click();
  await page.getByLabel("角色").selectOption("viewer");
  await page.getByLabel("表匹配规则").fill("fact_sales");
  await page.getByLabel("允许列").fill("region,amount");
  await page.getByLabel("行过滤条件").fill("region = '华东'");
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("fact_sales")).toBeVisible();
  expect(state.calls.some((call) => call.path === "/api/admin/meta/access-policies" && call.method === "POST")).toBeTruthy();
  const policyRow = page.locator(".policy-card").filter({ hasText: "fact_sales" });
  await policyRow.getByRole("button", { name: "删除" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除数据权限策略？" });
  await expect(deleteDialog).toContainText("后续查询");
  await deleteDialog.getByRole("button", { name: "删除策略" }).click();
  await expect(policyRow).toHaveCount(0);
});

test("仪表盘可打开、刷新和保存，owner 拥有编辑动作", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "仪表盘" }).click();
  await page.getByText("销售看板", { exact: true }).click();
  await expect(page.getByText("地区销售额")).toBeVisible();
  await page.getByRole("button", { name: "刷新数据" }).click();
  await page.getByRole("button", { name: "保存" }).click();
  expect(state.calls.some((call) => call.path === "/api/dashboard/dashboard-1" && call.method === "PUT")).toBeTruthy();
});

test("仪表盘以统一 12 列合同规范化恢复、拖拽、缩放和持久化布局", async ({ page }) => {
  const chart = (title) => ({
    type: "column",
    title,
    axis: { x: { name: "地区", value: "region" }, y: [{ name: "销售额", value: "amount" }] },
  });
  const rows = [{ region: "华东", amount: 120000 }];
  const state = await openAs(page, "owner", {
    dashboardBlocks: [
      { x: 11, y: -4, w: 6, h: 0, chart: chart("右边界图表"), rows },
      { x: 8, y: 10, w: 3, h: 3, chart: chart("缩放边界图表"), rows },
      { x: "invalid", y: 20.4, w: 99, h: "invalid", chart: chart("异常布局图表"), rows },
    ],
  });

  await page.getByRole("button", { name: "仪表盘" }).click();
  await page.getByText("销售看板", { exact: true }).click();
  const canvas = page.locator(".canvas-grid");
  const cards = canvas.locator(".block-card");
  await expect(cards).toHaveCount(3);

  await expect(cards.nth(0)).toHaveAttribute("style", /grid-area: 1 \/ 7 \/ span 9 \/ span 6/);
  await expect(cards.nth(1)).toHaveAttribute("style", /grid-area: 11 \/ 9 \/ span 9 \/ span 3/);
  await expect(cards.nth(2)).toHaveAttribute("style", /grid-area: 21 \/ 1 \/ span 9 \/ span 12/);
  expect(await cards.nth(0).evaluate((card) => {
    const chartCanvas = card.querySelector(".chart-canvas")?.getBoundingClientRect();
    const body = card.querySelector(".block-body")?.getBoundingClientRect();
    return Boolean(chartCanvas && body && chartCanvas.top >= body.top - 1 && chartCanvas.bottom <= body.bottom + 1);
  })).toBeTruthy();

  const firstHead = cards.nth(0).locator(".block-head");
  const firstHeadBox = await firstHead.boundingBox();
  const canvasBox = await canvas.boundingBox();
  expect(firstHeadBox).toBeTruthy();
  expect(canvasBox).toBeTruthy();
  await page.mouse.move(firstHeadBox.x + firstHeadBox.width / 2, firstHeadBox.y + firstHeadBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(canvasBox.x + canvasBox.width + 500, 1, { steps: 4 });
  await page.mouse.up();
  await expect(cards.nth(0)).toHaveAttribute("style", /grid-area: 1 \/ 7 \/ span 9 \/ span 6/);

  const resizeHandle = cards.nth(1).locator(".resize-handle");
  await resizeHandle.scrollIntoViewIfNeeded();
  const resizeBox = await resizeHandle.boundingBox();
  expect(resizeBox).toBeTruthy();
  await page.mouse.move(resizeBox.x + resizeBox.width / 2, resizeBox.y + resizeBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(canvasBox.x + canvasBox.width + 500, 1, { steps: 4 });
  await page.mouse.up();
  await expect(cards.nth(1)).toHaveAttribute("style", /grid-area: 11 \/ 9 \/ span 9 \/ span 4/);

  const withinCanvas = await cards.evaluateAll((elements) => {
    const bounds = elements[0].parentElement.getBoundingClientRect();
    return elements.every((element) => {
      const card = element.getBoundingClientRect();
      return card.left >= bounds.left - 1 && card.right <= bounds.right + 1;
    });
  });
  expect(withinCanvas).toBeTruthy();

  await page.getByRole("button", { name: "保存布局" }).click();
  await expect(page.getByText("仪表盘布局已保存")).toBeVisible();
  const saveCall = state.calls.findLast((call) => call.path === "/api/dashboard/dashboard-1" && call.method === "PUT");
  expect(saveCall.body.component_data.map(({ x, y, w, h }) => ({ x, y, w, h }))).toEqual([
    { x: 6, y: 0, w: 6, h: 9 },
    { x: 8, y: 10, w: 4, h: 9 },
    { x: 0, y: 20, w: 12, h: 9 },
  ]);

  await page.getByRole("button", { name: "刷新数据" }).click();
  await expect(cards.nth(1)).toHaveAttribute("style", /grid-area: 11 \/ 9 \/ span 9 \/ span 4/);
});

test("仪表盘按 pid 构树并在当前文件夹创建，级联删除会清除子看板画布", async ({ page }) => {
  const state = await openAs(page, "owner", {
    dashboards: [
      { id: "folder-a", name: "业务目录", node_type: "folder", pid: null },
      { id: "folder-b", name: "月度分析", node_type: "folder", pid: "folder-a" },
      { id: "dashboard-1", name: "销售看板", node_type: "dashboard", pid: "folder-b" },
      { id: "dashboard-independent", name: "独立看板", node_type: "dashboard", pid: null },
    ],
  });

  await page.getByRole("button", { name: "仪表盘" }).click();
  const folderA = page.locator(".tree-row").filter({ hasText: /^业务目录$/ });
  const folderB = page.locator(".tree-row").filter({ hasText: /^月度分析$/ });
  const salesDashboard = page.locator(".tree-row").filter({ hasText: /^销售看板$/ });
  await expect(folderA).toHaveAttribute("data-depth", "0");
  await expect(folderB).toHaveAttribute("data-depth", "1");
  await expect(salesDashboard).toHaveAttribute("data-depth", "2");

  await folderA.getByRole("button", { name: "收起 业务目录" }).click();
  await expect(folderB).toHaveCount(0);
  await folderA.getByRole("button", { name: "展开 业务目录" }).click();
  await expect(folderB).toBeVisible();

  await folderB.focus();
  await page.keyboard.press("Enter");
  await expect(folderB).toHaveClass(/active/);
  await page.getByRole("button", { name: "新建仪表盘" }).first().click();
  const dashboardDialog = page.getByRole("dialog", { name: "新建仪表盘" });
  await dashboardDialog.getByLabel("仪表盘名称").fill("目录内看板");
  await dashboardDialog.getByRole("button", { name: "创建" }).click();
  await expect(page.getByText("目录内看板", { exact: true })).toBeVisible();
  const dashboardCreate = state.calls.findLast((call) => call.path === "/api/dashboard" && call.method === "POST");
  expect(dashboardCreate.body.pid).toBe("folder-b");

  await page.getByRole("button", { name: "新建文件夹" }).click();
  const folderDialog = page.getByRole("dialog", { name: "新建文件夹" });
  await folderDialog.getByLabel("文件夹名称").fill("归档目录");
  await folderDialog.getByRole("button", { name: "创建" }).click();
  await expect(page.getByText("归档目录", { exact: true })).toBeVisible();
  const folderCreate = state.calls.findLast((call) => call.path === "/api/dashboard" && call.method === "POST");
  expect(folderCreate.body).toMatchObject({ node_type: "folder", pid: "folder-b" });

  await salesDashboard.click();
  await expect(page.getByText("地区销售额", { exact: true })).toBeVisible();
  await folderA.hover();
  await folderA.getByRole("button", { name: "删除 业务目录" }).click();
  const deleteDialog = page.getByRole("dialog", { name: "删除文件夹？" });
  await deleteDialog.getByRole("button", { name: "确认删除" }).click();
  await expect(folderA).toHaveCount(0);
  await expect(page.getByText("月度分析", { exact: true })).toHaveCount(0);
  await expect(page.getByText("销售看板", { exact: true })).toHaveCount(0);
  await expect(page.getByText("目录内看板", { exact: true })).toHaveCount(0);
  await expect(page.getByText("归档目录", { exact: true })).toHaveCount(0);
  await expect(page.locator(".canvas-toolbar")).toHaveCount(0);
  await expect(page.locator(".canvas-grid")).toHaveCount(0);
  await expect(page.getByText("选择一个仪表盘开始查看")).toBeVisible();
  await expect(page.getByText("独立看板", { exact: true })).toBeVisible();
});

test("仪表盘快速切换只提交最后一次选择并让文件夹选择失效在途请求", async ({ page }) => {
  const tableBlock = (title) => [{
    x: 0, y: 0, w: 6, h: 9,
    chart: { type: "table", title, columns: [{ name: "值", value: "value" }] },
    rows: [{ value: title }],
  }];
  await openAs(page, "owner", {
    dashboards: [
      { id: "folder-1", name: "分析目录", node_type: "folder", pid: null },
      { id: "dashboard-slow", name: "慢看板", node_type: "dashboard", pid: null },
      { id: "dashboard-fast", name: "快看板", node_type: "dashboard", pid: null },
    ],
    dashboardGetDelays: { "dashboard-slow": 350, "dashboard-fast": 10 },
    dashboardBlocksById: {
      "dashboard-slow": tableBlock("慢结果"),
      "dashboard-fast": tableBlock("快结果"),
    },
  });

  await page.getByRole("button", { name: "仪表盘" }).click();
  await page.getByText("慢看板", { exact: true }).click();
  await page.getByText("快看板", { exact: true }).click();
  await expect(page.locator(".canvas-title")).toHaveText("快看板");
  await page.waitForTimeout(450);
  await expect(page.locator(".canvas-title")).toHaveText("快看板");
  await expect(page.locator(".block-title")).toHaveText("快结果");

  await page.getByText("慢看板", { exact: true }).click();
  await page.getByText("分析目录", { exact: true }).click();
  await page.waitForTimeout(450);
  await expect(page.locator(".canvas-toolbar")).toHaveCount(0);
  await expect(page.getByText("选择一个仪表盘开始查看")).toBeVisible();
});

test("工作空间所有者可以创建空间并添加已注册成员", async ({ page }) => {
  const state = await openAs(page);
  await page.getByRole("button", { name: "工作空间" }).click();
  await page.getByText("创建新的工作空间", { exact: true }).click();
  await page.getByLabel("新工作空间名称").fill("分析组");
  await page.getByRole("button", { name: "创建" }).click();
  await page.getByText("邀请已注册成员", { exact: true }).click();
  await page.getByLabel("已注册用户名").fill("bob");
  await page.getByLabel("成员角色").selectOption("analyst");
  await page.getByRole("button", { name: "添加成员" }).click();
  expect(state.calls.some((call) => call.path === "/api/workspaces/current/members" && call.method === "POST")).toBeTruthy();
  expect(sampleDataset.row_count).toBe(2);
});

test("仪表盘空状态提供创建与问数引导", async ({ page }) => {
  await openAs(page, "owner", { emptyDashboards: true });
  await page.getByRole("button", { name: "仪表盘" }).click();
  await expect(page.getByText("还没有仪表盘")).toBeVisible();
  await expect(page.getByRole("button", { name: "新建仪表盘" }).first()).toBeVisible();
  await expect(page.getByText("创建你的第一个业务看板")).toBeVisible();
});

test("数据权限空状态解释已有保护并可创建首条策略", async ({ page }) => {
  await openAs(page);
  await page.getByRole("button", { name: "数据权限" }).click();
  await expect(page.getByText("当前工作空间还没有细粒度策略")).toBeVisible();
  await expect(page.getByText(/工作空间表白名单、只读 SQL 和脱敏规则/)).toBeVisible();
  await page.getByRole("button", { name: "创建第一条策略" }).click();
  await expect(page.getByRole("dialog", { name: "新增数据权限策略" })).toBeVisible();
});
