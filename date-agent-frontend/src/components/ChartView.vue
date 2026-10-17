<template>
  <div class="chart-view">
    <!-- table 类型：直接渲染表格（标题由外层卡片头部展示，此处不重复） -->
    <div v-if="chart.type === 'table'" class="table-wrap">
      <table class="result-table">
        <thead>
          <tr>
            <th v-for="col in chart.columns" :key="col.value">{{ col.name }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rIdx) in rows" :key="rIdx">
            <td v-for="col in chart.columns" :key="col.value">{{ row[col.value] }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <!-- 其它类型：ECharts 渲染 -->
    <div v-else ref="chartEl" class="chart-canvas"></div>
  </div>
</template>

<script>
function dataPoint(value, rowIndex, xValue, seriesValue) {
  const point = { value };
  if (Number.isInteger(rowIndex) && rowIndex >= 0) point.__daRowIndex = rowIndex;
  if (xValue !== undefined) point.__daXValue = xValue;
  if (seriesValue !== undefined) point.__daSeriesValue = seriesValue;
  return point;
}

// 纯函数同时供组件渲染和浏览器回归测试使用，避免测试依赖 Canvas 像素坐标。
export function buildChartOption(chart = {}, sourceRows = [], predictStart = 0) {
  const { type, axis } = chart;
  const rows = Array.isArray(sourceRows) ? sourceRows : [];
  const xField = axis?.x?.value;
  // axis.y 在 pie 里是单个对象 {name,value}，在 column/bar/line 里是数组。
  const yRaw = axis?.y;
  const yList = Array.isArray(yRaw) ? yRaw : (yRaw ? [yRaw] : []);
  const yFields = yList.map((y) => y.value);
  const yNames = yList.map((y) => y.name);
  const seriesField = axis?.series?.value;

  if (type === "pie") {
    const valField = yFields[0];
    return {
      tooltip: { trigger: "item" },
      legend: { orient: "vertical", left: "left", top: 8 },
      series: [{
        type: "pie",
        radius: "60%",
        data: rows.map((row, rowIndex) => ({
          name: String(row?.[seriesField] ?? ""),
          ...dataPoint(Number(row?.[valField]), rowIndex, undefined, row?.[seriesField]),
        })),
      }],
    };
  }

  // column / bar / line（含 series 分组或多个 y 指标）
  const baseType = type === "line" ? "line" : "bar";
  const xData = [...new Set(rows.map((row) => row?.[xField]))];
  const inPredict = predictStart > 0 && predictStart < rows.length;
  const splitIdx = inPredict ? predictStart : rows.length;

  let series = [];
  if (seriesField) {
    const groups = [...new Set(rows.map((row) => row?.[seriesField]))];
    const yField = yFields[0];
    const buildSegments = (group) => {
      const actualIndexes = xData.map((xValue) => rows.findIndex((row, rowIndex) => (
        rowIndex < splitIdx
        && row?.[xField] === xValue
        && row?.[seriesField] === group
      )));
      const actualValues = actualIndexes.map((rowIndex) => (
        rowIndex >= 0 ? Number(rows[rowIndex]?.[yField]) : null
      ));
      const actualData = actualValues.map((value, index) => dataPoint(
        inPredict ? value : (value ?? 0),
        actualIndexes[index],
        xData[index],
        group,
      ));
      if (!inPredict) return [{ name: String(group), type: baseType, data: actualData }];

      // 预测段从最后一个实际点开始，保证折线连续过渡到虚线段。
      const predicted = new Array(xData.length).fill(null);
      const lastActualIdx = actualValues.findLastIndex((value) => value !== null);
      if (lastActualIdx >= 0) {
        predicted[lastActualIdx] = dataPoint(
          actualValues[lastActualIdx],
          actualIndexes[lastActualIdx],
          xData[lastActualIdx],
          group,
        );
      }
      xData.forEach((xValue, index) => {
        const rowIndex = rows.findIndex((row, sourceIndex) => (
          sourceIndex >= splitIdx
          && row?.[xField] === xValue
          && row?.[seriesField] === group
        ));
        if (rowIndex >= 0) {
          predicted[index] = dataPoint(Number(rows[rowIndex]?.[yField]), rowIndex, xValue, group);
        }
      });
      return [
        { name: String(group), type: baseType, data: actualData },
        {
          name: String(group) + "（预测）",
          type: baseType,
          data: predicted,
          lineStyle: { type: "dashed" },
          itemStyle: { opacity: 0.7 },
        },
      ];
    };
    for (const group of groups) series = series.concat(buildSegments(group));
  } else {
    yFields.forEach((yField, fieldIndex) => {
      const name = yNames[fieldIndex] || yField;
      const actualIndexes = xData.map((xValue) => rows.findIndex((row, rowIndex) => (
        rowIndex < splitIdx && row?.[xField] === xValue
      )));
      const actualValues = actualIndexes.map((rowIndex) => (
        rowIndex >= 0 ? Number(rows[rowIndex]?.[yField]) : null
      ));
      const actualData = actualValues.map((value, index) => dataPoint(
        inPredict ? value : (value ?? 0),
        actualIndexes[index],
        xData[index],
        undefined,
      ));
      if (!inPredict) {
        series.push({ name, type: baseType, data: actualData });
        return;
      }

      const predicted = new Array(xData.length).fill(null);
      const lastActualIdx = actualValues.findLastIndex((value) => value !== null);
      if (lastActualIdx >= 0) {
        predicted[lastActualIdx] = dataPoint(
          actualValues[lastActualIdx],
          actualIndexes[lastActualIdx],
          xData[lastActualIdx],
          undefined,
        );
      }
      xData.forEach((xValue, index) => {
        const rowIndex = rows.findIndex((row, sourceIndex) => (
          sourceIndex >= splitIdx && row?.[xField] === xValue
        ));
        if (rowIndex >= 0) {
          predicted[index] = dataPoint(Number(rows[rowIndex]?.[yField]), rowIndex, xValue, undefined);
        }
      });
      series.push({ name, type: baseType, data: actualData });
      series.push({
        name: name + "（预测）",
        type: baseType,
        data: predicted,
        lineStyle: { type: "dashed" },
        itemStyle: { opacity: 0.7 },
      });
    });
  }

  const categoryAxis = {
    type: "category",
    data: xData.map(String),
    axisLabel: { interval: 0, rotate: xData.length > 8 ? 30 : 0 },
  };
  const valueAxis = { type: "value" };
  const horizontal = type === "bar";

  return {
    tooltip: { trigger: "axis" },
    legend: { top: 8 },
    grid: { top: 45, left: 50, right: 30, bottom: 40, containLabel: true },
    xAxis: horizontal ? valueAxis : categoryAxis,
    yAxis: horizontal
      ? { ...categoryAxis, axisLabel: { interval: 0, rotate: 0 } }
      : valueAxis,
    series,
  };
}

// 把 ECharts 点击参数归一化为稳定的业务维度值。
export function resolveChartPoint(chart = {}, sourceRows = [], params = {}) {
  if (params?.componentType && params.componentType !== "series") return null;

  const axis = chart.axis || {};
  const rows = Array.isArray(sourceRows) ? sourceRows : [];
  const xField = axis.x?.value;
  const seriesField = axis.series?.value;
  const encoded = params?.data && typeof params.data === "object" ? params.data : {};
  const encodedRowIndex = Number.isInteger(encoded.__daRowIndex) ? encoded.__daRowIndex : -1;
  const xValue = encoded.__daXValue ?? (chart.type === "pie" ? undefined : params?.name);
  const seriesValue = encoded.__daSeriesValue
    ?? (chart.type === "pie" ? params?.name : params?.seriesName);

  let row = encodedRowIndex >= 0 ? (rows[encodedRowIndex] || null) : null;
  if (!row && seriesField) {
    row = rows.find((item) => (
      (!xField || xValue === undefined || String(item?.[xField]) === String(xValue))
      && String(item?.[seriesField]) === String(seriesValue)
    )) || null;
  } else if (!row && xField && xValue !== undefined) {
    row = rows.find((item) => String(item?.[xField]) === String(xValue)) || null;
  } else if (!row && Number.isInteger(params?.dataIndex)) {
    row = rows[params.dataIndex] || null;
  }

  const field = seriesField || xField || "value";
  const value = row?.[field]
    ?? (seriesField ? seriesValue : xValue)
    ?? params?.value;
  if (value === undefined || value === null || value === "") return null;

  return {
    field,
    label: seriesField ? (axis.series?.name || field) : (axis.x?.name || field),
    value,
    series: params?.seriesName || "",
    row,
    dataIndex: params?.dataIndex,
  };
}
</script>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from "vue";
import * as echarts from "echarts/core";
import { BarChart, LineChart, PieChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart, LineChart, PieChart,
  GridComponent, LegendComponent, TooltipComponent,
  CanvasRenderer,
]);

const props = defineProps({
  chart: { type: Object, required: true },
  rows: { type: Array, default: () => [] },
  // 预测段起点索引：≥1 时把行划分为"实际(0..start-1)"与"预测(start..)"，
  // 实际段实线、预测段虚线，共享 x 轴（仅 line/column/bar 生效）
  predictStart: { type: Number, default: 0 },
});
const emit = defineEmits(["point-click"]);

const chartEl = ref(null);
let instance = null;
let instanceElement = null;

function disposeChart() {
  if (instance) {
    instance.off("click", handleChartClick);
    instance.dispose();
  }
  instance = null;
  instanceElement = null;
}

function render() {
  // v-if 会在 table / chart 之间替换真实 DOM；旧实例不能复用到新 canvas。
  if (props.chart.type === "table" || !chartEl.value) {
    disposeChart();
    return;
  }
  if (instance && instanceElement !== chartEl.value) disposeChart();
  if (!instance) {
    instance = echarts.init(chartEl.value);
    instanceElement = chartEl.value;
    instance.on("click", handleChartClick);
  }
  try {
    instance.setOption(buildChartOption(props.chart, props.rows, props.predictStart), true);
  } catch (e) {
    // 配置异常时打印到控制台，便于定位（如 axis.y 形态不符等）
    console.error("[ChartView] 构建图表配置失败：", e, props.chart);
  }
}

// 把 ECharts 的点击事件归一化为业务维度值，交给 App.vue 生成下钻/联动追问。
function handleChartClick(params) {
  const point = resolveChartPoint(props.chart, props.rows, params);
  if (point) emit("point-click", point);
}

onMounted(async () => {
  await nextTick();
  render();
  // 窗口尺寸变化时同步重绘
  window.addEventListener("resize", onResize);
});

function onResize() {
  if (instance) instance.resize();
}

watch(() => [props.chart, props.rows, props.predictStart], async () => {
  await nextTick();
  render();
}, { deep: true });

onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
  disposeChart();
});
</script>

<style scoped>
.chart-view {
  width: 100%;
}
.chart-canvas {
  width: 100%;
  height: 340px;
}
.table-wrap {
  max-width: 100%;
  overflow-x: auto;
}
.result-table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.result-table th,
.result-table td {
  border-bottom: 1px solid #e8e9eb;
  padding: 9px 14px;
  white-space: nowrap;
  text-align: left;
}
.result-table th {
  background: #f5f6f7;
  font-weight: 600;
  color: #646a73;
}
.result-table tbody tr:hover {
  background: #f5f6f7;
}
.chart-canvas { min-height: 300px; height: 340px; }
.table-wrap { border: 1px solid var(--da-border-light); border-radius: 10px; }
.result-table th,.result-table td { border-color: var(--da-border-light); padding: 10px 13px; }
.result-table th { position: sticky; top: 0; z-index: 1; background: var(--da-bg-soft); color: var(--da-text-secondary); }
.result-table tbody tr:nth-child(even) { background: rgba(244,247,251,.58); }
@media(max-width:640px){.chart-canvas{height:290px;min-height:260px}}
</style>
