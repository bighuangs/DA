<template>
  <section class="result-workbench" aria-label="查询结果">
    <header class="result-head">
      <div class="result-heading">
        <span class="result-icon"><ChartNoAxesCombined :size="18" aria-hidden="true" /></span>
        <span>
          <strong>{{ msg.chart?.title || "查询结果" }}</strong>
          <small>{{ msg.rows?.length || 0 }} 行 · {{ msg.columns?.length || 0 }} 个字段</small>
        </span>
      </div>
      <div class="result-actions" aria-label="结果操作">
        <button v-if="canEditDashboard" type="button" @click="emit('add-dashboard', msg)">
          <Pin :size="15" aria-hidden="true" /><span>加入仪表盘</span>
        </button>
        <button type="button" :disabled="msg._exporting" @click="emit('export', msg)">
          <Download :size="15" aria-hidden="true" /><span>{{ msg._exporting ? "导出中…" : "导出" }}</span>
        </button>
        <button type="button" :disabled="msg._predicting" @click="emit('predict', msg)">
          <TrendingUp :size="15" aria-hidden="true" /><span>{{ msg._predicting ? "预测中…" : "预测" }}</span>
        </button>
        <button type="button" :disabled="msg._anomaly" @click="emit('anomaly', msg)">
          <ScanSearch :size="15" aria-hidden="true" /><span>{{ msg._anomaly ? "检测中…" : "异常检测" }}</span>
        </button>
      </div>
    </header>

    <div v-if="hasChart" class="view-tabs" role="tablist" aria-label="结果视图">
      <button
        type="button"
        role="tab"
        :aria-selected="activeView === 'table'"
        :class="{ active: activeView === 'table' }"
        @click="activeView = 'table'"
      >
        <Table2 :size="15" aria-hidden="true" />查看表格
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeView === 'chart'"
        :class="{ active: activeView === 'chart' }"
        @click="activeView = 'chart'"
      >
        <ChartColumn :size="15" aria-hidden="true" />查看图表
      </button>
      <span class="view-badge">{{ chartTypeLabel(msg.chart?.type) }}</span>
    </div>

    <div v-if="msg._actionError" class="action-status error" role="alert">
      <CircleAlert :size="15" aria-hidden="true" />{{ msg._actionError }}
    </div>
    <div v-else-if="msg._actionNotice" class="action-status success" role="status">
      <CircleCheck :size="15" aria-hidden="true" />{{ msg._actionNotice }}
    </div>

    <div v-show="activeView === 'table' || !hasChart" class="table-wrap" role="tabpanel">
      <table class="result-table">
        <caption class="sr-only">查询结果，共 {{ msg.rows?.length || 0 }} 行</caption>
        <thead>
          <tr>
            <th v-for="col in msg.columns || []" :key="col" :title="`原始字段：${col}`">
              <span>{{ columnLabel(col) }}</span>
              <small v-if="columnLabel(col) !== col">{{ col }}</small>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in msg.rows || []" :key="rowIndex">
            <td v-for="col in msg.columns || []" :key="col">{{ formatCell(row[col]) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="!msg.rows?.length" class="empty-result">
        <Inbox :size="24" aria-hidden="true" />
        <span>查询执行成功，但没有返回数据</span>
      </div>
    </div>

    <div v-if="hasChart && activeView === 'chart'" class="chart-panel" role="tabpanel">
      <ChartView :chart="msg.chart" :rows="msg.rows || []" @point-click="emit('point-click', msg, $event)" />
    </div>

    <footer v-if="msg._selectedPoint" class="chart-interaction">
      <span>
        已选 {{ msg._selectedPoint.label }}：
        <strong>{{ formatCell(msg._selectedPoint.value) }}</strong>
      </span>
      <div>
        <button type="button" :disabled="loading" @click="emit('drill-down', msg)">下钻</button>
        <button type="button" :disabled="loading" @click="emit('linked-filter', msg)">联动筛选</button>
        <button type="button" :disabled="loading" @click="emit('period-compare', msg, '环比')">环比</button>
        <button type="button" :disabled="loading" @click="emit('period-compare', msg, '同比')">同比</button>
        <button class="clear" type="button" @click="emit('clear-point', msg)">清除选择</button>
      </div>
    </footer>
  </section>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import {
  ChartColumn,
  ChartNoAxesCombined,
  CircleAlert,
  CircleCheck,
  Download,
  Inbox,
  Pin,
  ScanSearch,
  Table2,
  TrendingUp,
} from "@lucide/vue";
import ChartView from "./ChartView.vue";

const props = defineProps({
  msg: { type: Object, required: true },
  canEditDashboard: { type: Boolean, default: false },
  loading: { type: Boolean, default: false },
});
const emit = defineEmits([
  "add-dashboard",
  "export",
  "predict",
  "anomaly",
  "point-click",
  "drill-down",
  "linked-filter",
  "period-compare",
  "clear-point",
]);

const activeView = ref("table");
const hasChart = computed(() => Boolean(props.msg.chart && props.msg.chart.type !== "table"));
watch(hasChart, (available) => {
  if (!available) activeView.value = "table";
});

const columnLabels = {
  region: "地区",
  region_id: "地区编号",
  region_name: "地区",
  province: "省份",
  city: "城市",
  category: "品类",
  category_name: "品类",
  product: "商品",
  product_name: "商品",
  customer: "客户",
  customer_name: "客户",
  customer_count: "客户数",
  order_count: "订单数",
  sales: "销售额",
  amount: "销售额",
  total_amount: "销售总额",
  sales_amount: "销售额",
  total_sales: "销售总额",
  quantity: "数量",
  sales_quantity: "销量",
  total_quantity: "总销量",
  ratio: "占比",
  rate: "比率",
  repurchase_rate: "复购率",
  period: "周期",
  date: "日期",
  month: "月份",
  year: "年份",
};

function columnLabel(column) {
  const key = String(column || "").toLowerCase();
  if (columnLabels[key]) return columnLabels[key];
  return String(column || "")
    .replace(/_id$/i, "编号")
    .replace(/_/g, " ")
    .replace(/\btotal\b/gi, "总计")
    .replace(/\bcount\b/gi, "数量")
    .replace(/\bname\b/gi, "名称")
    .trim() || column;
}

function formatCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value);
  }
  const text = String(value);
  const match = text.match(/^(-?)(\d{10,})(\.\d+)?$/);
  if (!match) return text;
  const integer = match[2].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${match[1]}${integer}${match[3] || ""}`;
}

function chartTypeLabel(type) {
  return ({ column: "柱状图", bar: "条形图", line: "折线图", pie: "饼图", table: "表格" })[type] || "图表";
}
</script>

<style scoped>
.result-workbench {
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--da-border-light);
  border-radius: var(--da-radius-md);
  background: #fff;
  box-shadow: var(--da-shadow-sm);
}
.result-head {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 11px 14px;
  border-bottom: 1px solid var(--da-border-light);
  background: linear-gradient(180deg, #fff, #fbfcfe);
}
.result-heading { min-width: 0; display: flex; align-items: center; gap: 10px; }
.result-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.result-heading > span:last-child { min-width: 0; display: grid; gap: 3px; }
.result-heading strong {
  overflow: hidden;
  color: var(--da-text);
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.result-heading small { color: var(--da-text-muted); font-size: 10px; }
.result-actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 6px; }
.result-actions button,
.view-tabs button,
.chart-interaction button {
  min-height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 5px 9px;
  border: 1px solid var(--da-border);
  border-radius: 8px;
  background: #fff;
  color: var(--da-text-secondary);
  font-size: 11px;
}
.result-actions button:hover:not(:disabled),
.chart-interaction button:hover:not(:disabled) {
  border-color: var(--da-accent);
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.result-actions button:disabled, .chart-interaction button:disabled { opacity: 0.55; }
.view-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--da-border-light);
  background: var(--da-bg-soft);
}
.view-tabs button { min-height: 30px; border-color: transparent; background: transparent; }
.view-tabs button.active {
  border-color: var(--da-border-light);
  background: #fff;
  color: var(--da-accent-strong);
  box-shadow: var(--da-shadow-sm);
  font-weight: 650;
}
.view-badge {
  margin-left: auto;
  padding: 3px 8px;
  border-radius: 999px;
  background: #eaf2ff;
  color: var(--da-accent-strong);
  font-size: 9px;
}
.action-status {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 9px 13px;
  border-bottom: 1px solid;
  font-size: 11px;
}
.action-status.error { border-color: #f0c5bf; background: #fff3f1; color: #b83d31; }
.action-status.success { border-color: #b9e2c8; background: #f0fbf4; color: #287b4a; }
.table-wrap { max-height: 430px; overflow: auto; }
.result-table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px; }
.result-table th,
.result-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--da-border-light);
  color: var(--da-text);
  text-align: left;
  white-space: nowrap;
}
.result-table th {
  position: sticky;
  z-index: 1;
  top: 0;
  background: #f6f8fb;
  color: var(--da-text-secondary);
  font-weight: 650;
}
.result-table th span, .result-table th small { display: block; }
.result-table th small { margin-top: 2px; color: #526176; font: 9px/1.2 Consolas, monospace; font-weight: 400; }
.result-table tbody tr:nth-child(even) { background: rgba(244, 247, 251, 0.62); }
.result-table tbody tr:hover { background: #eef4ff; }
.chart-panel { min-height: 340px; padding: 12px; }
.chart-interaction {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 13px;
  border-top: 1px solid var(--da-border-light);
  background: var(--da-bg-soft);
  color: var(--da-text-secondary);
  font-size: 11px;
}
.chart-interaction > div { display: flex; flex-wrap: wrap; gap: 6px; }
.chart-interaction strong { color: var(--da-accent-strong); }
.chart-interaction button.clear { border-color: transparent; background: transparent; color: var(--da-danger); }
.empty-result {
  min-height: 150px;
  display: grid;
  place-items: center;
  align-content: center;
  gap: 8px;
  color: var(--da-text-muted);
  font-size: 11px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
}

@media (max-width: 640px) {
  .result-head { align-items: flex-start; flex-direction: column; }
  .result-actions { width: 100%; justify-content: flex-start; }
  .result-actions button { flex: 1 1 calc(50% - 4px); }
  .chart-panel { min-height: 300px; padding: 6px; }
  .chart-interaction { align-items: flex-start; flex-direction: column; }
}
</style>
