<template>
  <section class="matrix-page">
    <header class="matrix-head">
      <div><div class="eyebrow">ADMIN / BUSINESS SCENARIOS</div><h1>核心业务场景矩阵</h1><p>按身份、入口、资源状态和预期结果追踪组合路径，每一行都绑定可执行证据。</p></div>
      <div class="actions"><button class="btn" @click="copyCommand">复制执行命令</button><button class="btn primary" :disabled="loading" @click="load">{{ loading ? "刷新中…" : "刷新报告" }}</button></div>
    </header>
    <div v-if="error" class="alert" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-if="notice" class="notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <div v-if="loading && !report" class="loading">正在读取场景矩阵报告…</div>
    <template v-else-if="report">
      <section class="summary-grid" aria-label="场景矩阵摘要">
        <article><span>场景总数</span><strong>{{ report.summary.total }}</strong><small>门槛 ≥ 50</small></article>
        <article class="good"><span>已通过</span><strong>{{ report.summary.passed }}</strong><small>{{ percent(report.summary.pass_rate) }}</small></article>
        <article :class="report.summary.failed ? 'bad' : 'good'"><span>失败</span><strong>{{ report.summary.failed }}</strong><small>自动化执行结果</small></article>
        <article :class="report.summary.not_run ? 'warn' : 'good'"><span>未执行</span><strong>{{ report.summary.not_run }}</strong><small>不是默认通过</small></article>
        <article :class="report.contract.valid ? 'good' : 'bad'"><span>矩阵契约</span><strong>{{ report.contract.valid ? "完整" : "缺失" }}</strong><small>{{ shortHash(report.contract.matrix_version) }}</small></article>
      </section>

      <section class="panel">
        <div class="panel-head"><div><h2>能力覆盖</h2><p>每项能力至少 5 个正反例；失败或未执行会在能力条中显式保留。</p></div><span :class="report.summary.gate_passed ? 'pass' : 'fail'">{{ report.summary.gate_passed ? "门禁通过" : "门禁未通过" }}</span></div>
        <div class="capability-grid">
          <article v-for="item in report.capabilities" :key="item.capability"><div><strong>{{ capabilityLabel(item.capability) }}</strong><span>{{ item.passed }}/{{ item.total }}</span></div><div class="bar"><i :style="{ width: `${item.total ? item.passed / item.total * 100 : 0}%` }"></i></div><small v-if="item.failed">{{ item.failed }} 失败</small><small v-else-if="item.not_run">{{ item.not_run }} 未执行</small><small v-else>全部通过</small></article>
        </div>
        <details v-if="report.contract.errors?.length" class="contract-errors"><summary>查看契约错误</summary><ul><li v-for="item in report.contract.errors" :key="item">{{ item }}</li></ul></details>
      </section>

      <section class="panel">
        <div class="panel-head"><div><h2>场景明细</h2><p>筛选不会改变统计口径，只缩小当前核查列表。</p></div><span>{{ filteredCases.length }} / {{ report.cases.length }}</span></div>
        <form class="filters" aria-label="场景矩阵筛选" @submit.prevent>
          <label>能力<select v-model="filters.capability"><option value="">全部能力</option><option v-for="item in report.capabilities" :key="item.capability" :value="item.capability">{{ capabilityLabel(item.capability) }}</option></select></label>
          <label>身份<select v-model="filters.principal"><option value="">全部身份</option><option v-for="item in principals" :key="item" :value="item">{{ roleLabel(item) }}</option></select></label>
          <label>风险<select v-model="filters.risk"><option value="">全部风险</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label>
          <label>结果<select v-model="filters.status"><option value="">全部结果</option><option value="passed">通过</option><option value="failed">失败</option><option value="not_run">未执行</option></select></label>
          <label>失败层<select v-model="filters.layer"><option value="">全部层级</option><option v-for="item in failureLayers" :key="item" :value="item">{{ layerLabel(item) }}</option></select></label>
        </form>
        <div class="case-list">
          <details v-for="item in filteredCases" :key="item.id" class="case-row">
            <summary><span class="status-dot" :class="item.status"></span><strong class="mono">{{ item.id }}</strong><span class="title">{{ item.title }}</span><span class="chip">{{ capabilityLabel(item.capability) }}</span><span class="risk" :class="item.risk">{{ item.risk }}</span><span class="role">{{ roleLabel(item.principal) }}</span></summary>
            <div class="case-body">
              <dl><div><dt>入口</dt><dd class="mono">{{ item.entrypoint }}</dd></div><div><dt>资源状态</dt><dd>{{ item.resource_state }}</dd></div><div><dt>预期</dt><dd>{{ decisionLabel(item.expected.decision) }} · HTTP {{ item.expected.http_status }} · {{ item.expected.side_effect }}</dd></div><div><dt>归因层</dt><dd>{{ layerLabel(item.expected.failure_layer) }}</dd></div><div><dt>Trace</dt><dd>{{ item.expected.trace }}</dd></div><div><dt>自动证据</dt><dd><code>{{ item.evidence.type }} · {{ item.evidence.target }}</code></dd></div></dl>
              <div class="tags"><span v-for="tag in item.tags" :key="tag">{{ tag }}</span></div>
              <p v-if="item.error" class="case-error">{{ item.error }}</p>
            </div>
          </details>
          <p v-if="!filteredCases.length" class="empty">当前筛选没有场景。</p>
        </div>
      </section>
      <section class="command"><div><strong>重新执行矩阵</strong><small>在后端目录运行；会执行绑定的单测和 Playwright，并更新本页报告。</small></div><code>{{ command }}</code></section>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { CircleAlert, CircleCheck } from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const command = "python -m app.eval.run_scenarios";
const report = ref(null), loading = ref(false), error = ref(""), notice = ref("");
const filters = reactive({ capability: "", principal: "", risk: "", status: "", layer: "" });
const principals = computed(() => [...new Set((report.value?.cases || []).map(item => item.principal))]);
const failureLayers = computed(() => [...new Set((report.value?.cases || []).map(item => item.expected.failure_layer))]);
const filteredCases = computed(() => (report.value?.cases || []).filter(item =>
  (!filters.capability || item.capability === filters.capability) && (!filters.principal || item.principal === filters.principal) &&
  (!filters.risk || item.risk === filters.risk) && (!filters.status || item.status === filters.status) &&
  (!filters.layer || item.expected.failure_layer === filters.layer)
));
async function load() { loading.value = true; error.value = ""; try { report.value = await requestJson("/api/admin/evaluation/scenarios/matrix", {}, "场景矩阵加载失败"); } catch (e) { error.value = e.message || "场景矩阵加载失败"; } finally { loading.value = false; } }
async function copyCommand() { try { await navigator.clipboard.writeText(command); notice.value = "执行命令已复制"; window.setTimeout(() => { notice.value = ""; }, 2200); } catch { error.value = "剪贴板权限不可用，请手动复制页面底部命令。"; } }
function percent(value) { const n = Number(value); return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—"; }
function shortHash(value) { return value ? `${value.slice(0, 14)}…${value.slice(-6)}` : "未生成"; }
function capabilityLabel(value) { return ({ conversation: "多轮对话", chart: "图表交互", knowledge: "元知识", adhoc: "文件查询", masking: "脱敏出口", rbac: "RBAC", access_policy: "行列权限", dashboard: "仪表盘" })[value] || value; }
function roleLabel(value) { return ({ owner: "所有者", admin: "管理员", analyst: "分析员", viewer: "查看者", removed: "已移除成员" })[value] || value; }
function layerLabel(value) { return ({ none: "正常", identity: "身份", resource: "资源", policy: "策略", sql: "SQL", dependency: "依赖", presentation: "展示" })[value] || value; }
function decisionLabel(value) { return ({ allow: "允许", deny: "拒绝", pass: "契约通过" })[value] || value; }
onMounted(load);
</script>

<style scoped>
.matrix-page{flex:1 1 auto;min-height:0;overflow:auto;padding:36px 34px 70px;color:var(--da-text)}.matrix-head,.panel,.summary-grid,.alert,.notice,.command{max-width:1320px;margin-left:auto;margin-right:auto}.matrix-head{display:flex;justify-content:space-between;gap:24px;margin-bottom:22px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:750;letter-spacing:.17em}.matrix-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.matrix-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.actions{display:flex;gap:8px;align-items:start}.btn{min-height:38px;padding:8px 13px;border:1px solid var(--da-border);border-radius:9px;background:#fff;color:var(--da-text-secondary);font:12px inherit}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.alert,.notice{display:flex;align-items:center;gap:7px;margin-bottom:13px;padding:11px 14px;border-radius:10px;font-size:12px}.alert{border:1px solid #f0c5bf;background:#fff3f1;color:#bd3b2c}.notice{border:1px solid #b9e2c8;background:#f0fbf4;color:#287b4a}.loading{min-height:360px;display:grid;place-items:center;color:var(--da-text-muted)}.summary-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:17px}.summary-grid article{position:relative;overflow:hidden;padding:17px;border:1px solid var(--da-border-light);border-radius:14px;background:#fff;box-shadow:var(--da-shadow-sm)}.summary-grid article:before{position:absolute;inset:0 auto 0 0;width:3px;background:#9aa6b6;content:""}.summary-grid .good:before{background:#35a56a}.summary-grid .bad:before{background:#dc6656}.summary-grid .warn:before{background:#e0a33a}.summary-grid span,.summary-grid small{display:block;color:var(--da-text-muted);font-size:10px}.summary-grid strong{display:block;margin:7px 0 4px;font-size:22px}.panel{margin-bottom:17px;padding:21px;border:1px solid var(--da-border-light);border-radius:16px;background:rgba(255,255,255,.92);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;justify-content:space-between;gap:15px;margin-bottom:15px}.panel-head h2{margin:0;font-size:16px}.panel-head>span{align-self:start;padding:5px 9px;border-radius:99px;font-size:10px}.panel-head .pass{background:#e9f8ef;color:#257949}.panel-head .fail{background:#fff0ee;color:#b63e31}.capability-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.capability-grid article{padding:13px;border:1px solid var(--da-border-light);border-radius:11px;background:#fbfcfe}.capability-grid article>div:first-child{display:flex;justify-content:space-between;font-size:11px}.capability-grid small{color:var(--da-text-muted);font-size:9px}.bar{height:6px;margin:9px 0 6px;overflow:hidden;border-radius:99px;background:#e9edf3}.bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2d75e8,#60a0ff)}.contract-errors{margin-top:13px;color:#b33f32;font-size:11px}.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:13px}.filters label{display:grid;gap:4px;color:var(--da-text-muted);font-size:9px}.filters select{min-height:35px;padding:7px 9px;border:1px solid var(--da-border);border-radius:8px;background:#fff;font-size:11px}.case-list{display:grid;gap:7px}.case-row{border:1px solid var(--da-border-light);border-radius:10px;background:#fff}.case-row summary{display:flex;align-items:center;gap:9px;padding:12px;cursor:pointer;list-style:none}.case-row summary::-webkit-details-marker{display:none}.status-dot{width:8px;height:8px;border-radius:50%;background:#98a4b4}.status-dot.passed{background:#35a56a}.status-dot.failed{background:#dc6656}.case-row .title{min-width:0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.mono,code{font-family:Consolas,monospace}.case-row summary strong{font-size:10px}.chip,.risk,.role{padding:4px 7px;border-radius:99px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:9px}.risk.critical{background:#fff0ee;color:#b33f32}.risk.high{background:#fff5e5;color:#956314}.case-body{padding:13px;border-top:1px solid var(--da-border-light)}.case-body dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0}.case-body dl div{padding:9px;border-radius:8px;background:var(--da-bg-soft)}.case-body dt{color:var(--da-text-muted);font-size:9px}.case-body dd{overflow-wrap:anywhere;margin:4px 0 0;font-size:10px}.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.tags span{padding:3px 6px;border-radius:5px;background:#eef5ff;color:#2b65bd;font-size:9px}.case-error{color:#b33f32;font-size:10px}.empty{padding:25px;text-align:center;color:var(--da-text-muted);font-size:11px}.command{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:18px;padding:15px 17px;border:1px solid #cfe0f4;border-radius:13px;background:#f4f8fd}.command strong,.command small{display:block}.command small{margin-top:4px;color:var(--da-text-muted);font-size:10px}.command code{overflow:auto;padding:9px 11px;border-radius:8px;background:#202634;color:#e8edf5;font-size:11px;white-space:nowrap}
@media(max-width:1000px){.matrix-head{display:block}.actions{margin-top:13px}.summary-grid{grid-template-columns:repeat(3,1fr)}.capability-grid{grid-template-columns:1fr 1fr}.case-body dl{grid-template-columns:1fr 1fr}}
@media(max-width:700px){.matrix-page{padding:25px 14px 50px}.summary-grid{grid-template-columns:1fr 1fr}.capability-grid,.case-body dl{grid-template-columns:1fr}.chip,.role{display:none}.command{grid-template-columns:1fr}.matrix-head h1{font-size:25px}}
</style>
