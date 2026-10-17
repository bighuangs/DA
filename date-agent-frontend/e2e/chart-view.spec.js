import { expect, test } from "@playwright/test";
import { openAs } from "./support/mock-api.js";

test.beforeEach(async ({ page }) => {
  await openAs(page);
});

test("bar 使用横向数值轴，column 保持纵向数值轴", async ({ page }) => {
  const axes = await page.evaluate(async () => {
    const { buildChartOption } = await import("/src/components/ChartView.vue");
    const rows = [
      { region: "华东", amount: 120000 },
      { region: "华南", amount: 98000 },
    ];
    const base = {
      axis: {
        x: { name: "地区", value: "region" },
        y: [{ name: "销售额", value: "amount" }],
      },
    };
    const column = buildChartOption({ ...base, type: "column" }, rows);
    const bar = buildChartOption({ ...base, type: "bar" }, rows);
    return {
      column: { x: column.xAxis.type, y: column.yAxis.type, categories: column.xAxis.data },
      bar: { x: bar.xAxis.type, y: bar.yAxis.type, categories: bar.yAxis.data },
    };
  });

  expect(axes.column).toEqual({ x: "category", y: "value", categories: ["华东", "华南"] });
  expect(axes.bar).toEqual({ x: "value", y: "category", categories: ["华东", "华南"] });
});

test("pie 点击用切片维度生成业务点，不依赖空 seriesName", async ({ page }) => {
  const point = await page.evaluate(async () => {
    const { buildChartOption, resolveChartPoint } = await import("/src/components/ChartView.vue");
    const chart = {
      type: "pie",
      axis: {
        series: { name: "地区", value: "region" },
        y: { name: "销售额", value: "amount" },
      },
    };
    const rows = [
      { region: "华东", amount: 120000 },
      { region: "华南", amount: 98000 },
    ];
    const option = buildChartOption(chart, rows);
    return resolveChartPoint(chart, rows, {
      componentType: "series",
      seriesName: "",
      name: "华南",
      dataIndex: 1,
      data: option.series[0].data[1],
      value: 98000,
    });
  });

  expect(point).toMatchObject({
    field: "region",
    label: "地区",
    value: "华南",
    row: { region: "华南", amount: 98000 },
    dataIndex: 1,
  });
});

test("分组预测系列点击保留原始业务维度，不把预测后缀写入筛选值", async ({ page }) => {
  const point = await page.evaluate(async () => {
    const { buildChartOption, resolveChartPoint } = await import("/src/components/ChartView.vue");
    const chart = {
      type: "line",
      axis: {
        x: { name: "月份", value: "month" },
        series: { name: "地区", value: "region" },
        y: [{ name: "销售额", value: "amount" }],
      },
    };
    const rows = [
      { month: "2026-01", region: "华东", amount: 100 },
      { month: "2026-01", region: "华南", amount: 80 },
      { month: "2026-02", region: "华东", amount: 110 },
      { month: "2026-02", region: "华南", amount: 90 },
    ];
    const option = buildChartOption(chart, rows, 2);
    const predictedSeries = option.series.find((item) => item.name === "华南（预测）");
    const dataIndex = predictedSeries.data.findIndex((item) => item?.__daRowIndex === 3);
    return resolveChartPoint(chart, rows, {
      componentType: "series",
      seriesName: predictedSeries.name,
      name: "2026-02",
      dataIndex,
      data: predictedSeries.data[dataIndex],
      value: 90,
    });
  });

  expect(point).toMatchObject({
    field: "region",
    label: "地区",
    value: "华南",
    row: { month: "2026-02", region: "华南", amount: 90 },
  });
});

test("真实 DOM 在 chart 与 table 间重复切换时销毁旧实例并绑定新 canvas", async ({ page }) => {
  await page.evaluate(async () => {
    const { createApp, h, nextTick, ref } = await import("/@id/vue");
    const { default: ChartView } = await import("/src/components/ChartView.vue");
    const host = document.createElement("section");
    host.id = "chart-lifecycle-harness";
    document.body.appendChild(host);

    const rows = [
      { region: "华东", amount: 120000 },
      { region: "华南", amount: 98000 },
    ];
    const columnChart = () => ({
      type: "column",
      title: "地区销售额",
      axis: {
        x: { name: "地区", value: "region" },
        y: [{ name: "销售额", value: "amount" }],
      },
    });
    const tableChart = () => ({
      type: "table",
      columns: [
        { name: "地区", value: "region" },
        { name: "销售额", value: "amount" },
      ],
    });
    const chart = ref(columnChart());
    const retiredRoots = [];
    const app = createApp({
      setup: () => () => h(ChartView, { chart: chart.value, rows }),
    });
    app.mount(host);

    async function settle() {
      await nextTick();
      await nextTick();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }

    async function switchTo(type) {
      const previousRoot = host.querySelector(".chart-canvas");
      if (previousRoot) retiredRoots.push(previousRoot);
      chart.value = type === "table" ? tableChart() : columnChart();
      await settle();
      const activeRoot = host.querySelector(".chart-canvas");
      return {
        activeCanvasCount: host.querySelectorAll("canvas").length,
        activeInstanceId: activeRoot?.getAttribute("_echarts_instance_") || null,
        previousDisposed: !previousRoot || !previousRoot.getAttribute("_echarts_instance_"),
        leakedRetiredInstances: retiredRoots.filter(
          (root) => root.getAttribute("_echarts_instance_"),
        ).length,
      };
    }

    await settle();
    window.__chartLifecycleHarness = { host, app, switchTo, retiredRoots };
  });

  const host = page.locator("#chart-lifecycle-harness");
  await expect(host.locator("canvas")).toHaveCount(1);
  const initialId = await host.locator(".chart-canvas").getAttribute("_echarts_instance_");
  expect(initialId).toBeTruthy();

  const activeIds = [];
  for (let iteration = 0; iteration < 3; iteration += 1) {
    const tableState = await page.evaluate(() => window.__chartLifecycleHarness.switchTo("table"));
    expect(tableState).toMatchObject({
      activeCanvasCount: 0,
      activeInstanceId: null,
      previousDisposed: true,
      leakedRetiredInstances: 0,
    });
    await expect(host.getByRole("table")).toBeVisible();

    const chartState = await page.evaluate(() => window.__chartLifecycleHarness.switchTo("column"));
    expect(chartState.activeCanvasCount).toBe(1);
    expect(chartState.activeInstanceId).toBeTruthy();
    expect(chartState.leakedRetiredInstances).toBe(0);
    activeIds.push(chartState.activeInstanceId);
  }

  expect(new Set([initialId, ...activeIds]).size).toBe(4);
  await expect(host.locator("canvas")).toHaveCount(1);

  const unmounted = await page.evaluate(async () => {
    const harness = window.__chartLifecycleHarness;
    const activeRoot = harness.host.querySelector(".chart-canvas");
    harness.retiredRoots.push(activeRoot);
    harness.app.unmount();
    await Promise.resolve();
    return {
      canvasCount: harness.host.querySelectorAll("canvas").length,
      leakedInstances: harness.retiredRoots.filter(
        (root) => root?.getAttribute("_echarts_instance_"),
      ).length,
    };
  });
  expect(unmounted).toEqual({ canvasCount: 0, leakedInstances: 0 });
});
