<template>
  <section class="isolation-page">
    <header class="page-head">
      <div>
        <span class="eyebrow">EVALUATION / UPLOAD ISOLATION</span>
        <h1>上传数据资产隔离门禁</h1>
        <p>独立验证租户授权、正式 Schema/索引零污染、失败补偿、最终清理与合成夹具隐私。</p>
      </div>
      <div class="actions">
        <button class="btn" type="button" :disabled="loading" @click="load">
          <RefreshCw :size="15" :class="{ spinning: loading }" aria-hidden="true" />
          刷新
        </button>
        <button class="btn primary" type="button" @click="copyCommand">
          <Copy :size="15" aria-hidden="true" />
          复制命令
        </button>
      </div>
    </header>

    <p v-if="error" class="alert error" role="alert">{{ error }}</p>
    <p v-if="notice" class="alert notice" role="status">{{ notice }}</p>
    <div v-if="loading && !report" class="loading">正在读取上传隔离证据…</div>

    <template v-else-if="report">
      <section class="run-banner" :class="gatePassed ? 'passed' : 'failed'">
        <div class="gate-icon">
          <ShieldCheck v-if="gatePassed" :size="25" aria-hidden="true" />
          <ShieldAlert v-else :size="25" aria-hidden="true" />
        </div>
        <div>
          <strong>{{ gatePassed ? "隔离合同通过" : "隔离合同尚未通过" }}</strong>
          <span>
            {{ report.evidence_level === "offline_contract" ? "离线合同证据" : evidenceLabel(report.evidence_level) }}
            · {{ report.production_equivalent ? "生产等价" : "不声明生产等价" }}
          </span>
        </div>
        <dl>
          <div><dt>Run</dt><dd>{{ report.run_id || "尚未运行" }}</dd></div>
          <div><dt>生成时间</dt><dd>{{ dateTime(report.generated_at) }}</dd></div>
        </dl>
      </section>

      <section class="metric-grid" aria-label="上传隔离核心门禁">
        <article :class="zeroClass(metrics.unauthorized_db_execution_count)">
          <span>越权数据库执行</span><strong>{{ metrics.unauthorized_db_execution_count ?? "—" }}</strong>
          <small>必须为 0</small>
        </article>
        <article :class="zeroClass(metrics.schema_index_pollution_count)">
          <span>Schema / 索引污染</span><strong>{{ metrics.schema_index_pollution_count ?? "—" }}</strong>
          <small>必须为 0</small>
        </article>
        <article :class="zeroClass(metrics.sandbox_residual_resources)">
          <span>清理后资源残留</span><strong>{{ metrics.sandbox_residual_resources ?? "—" }}</strong>
          <small>必须为 0</small>
        </article>
        <article :class="zeroClass(metrics.real_upload_content_occurrences)">
          <span>真实上传内容命中</span><strong>{{ metrics.real_upload_content_occurrences ?? "—" }}</strong>
          <small>必须为 0</small>
        </article>
        <article :class="oneClass(metrics.cleanup_success_rate)">
          <span>清理成功率</span><strong>{{ percent(metrics.cleanup_success_rate) }}</strong>
          <small>二次扫描确认</small>
        </article>
        <article :class="oneClass(metrics.compensation_recovery_rate)">
          <span>补偿恢复率</span><strong>{{ percent(metrics.compensation_recovery_rate) }}</strong>
          <small>失败不得伪装删除</small>
        </article>
      </section>

      <section class="panel denominator-panel">
        <div class="panel-head">
          <div><h2>独立分母与证据等级</h2><p>该分母不会计入 NL2SQL 60 题 exact/result accuracy。</p></div>
          <span :class="metrics.denominator_consistency ? 'pass' : 'fail'">
            {{ metrics.denominator_consistency ? "分母守恒" : "分母不一致" }}
          </span>
        </div>
        <div class="denominator">
          <div v-for="key in denominatorKeys" :key="key">
            <span>{{ denominatorLabel(key) }}</span>
            <strong>{{ denominator[key] ?? 0 }}</strong>
          </div>
        </div>
        <div class="level-note">
          <Info :size="16" aria-hidden="true" />
          <p>
            当前页面如标记为“离线合同证据”，仅证明合同、反例和报告守恒。
            发布前可在本机 Docker 上执行 <code>isolated_dependencies</code>，真实创建并精确清理独立 MySQL、Qdrant、ES 与 Redis 资源。
          </p>
        </div>
      </section>

      <section class="panel fixture-panel">
        <div class="panel-head">
          <div><h2>匿名合成夹具</h2><p>只记录来源、固定 seed 与不可逆 hash；不保留文件名、预览行和物理表名。</p></div>
          <b>{{ fixture.fixture_count || 0 }} 套</b>
        </div>
        <div class="fixture-summary">
          <div><span>来源完整率</span><strong>{{ percent(fixture.provenance_completeness) }}</strong></div>
          <div><span>真实用户夹具</span><strong>{{ fixture.real_user_fixture_count ?? 0 }}</strong></div>
          <div><span>合同来源</span><strong>{{ originText(fixture.origin_counts) }}</strong></div>
          <div><span>原始行持久化</span><strong>{{ report.privacy?.raw_rows_persisted ? "异常" : "关闭" }}</strong></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><h2>场景明细</h2><p>六类 P0 反例每类至少三条，并保留正常 CSV/XLSX 生命周期合同。</p></div>
          <b>{{ filteredCases.length }} / {{ report.cases?.length || 0 }}</b>
        </div>
        <form class="filters" aria-label="上传隔离场景筛选" @submit.prevent>
          <label>
            类别
            <select v-model="category">
              <option value="">全部类别</option>
              <option v-for="item in categories" :key="item" :value="item">{{ categoryLabel(item) }}</option>
            </select>
          </label>
          <label>
            状态
            <select v-model="status">
              <option value="">全部状态</option>
              <option value="passed">通过</option>
              <option value="failed">失败</option>
            </select>
          </label>
        </form>
        <div class="case-list">
          <details v-for="item in filteredCases" :key="item.scenario_id" class="case">
            <summary>
              <span class="case-state" :class="item.status"></span>
              <code>{{ item.scenario_id }}</code>
              <strong>{{ probeLabel(item.probe) }}</strong>
              <em>{{ categoryLabel(item.category) }}</em>
              <small>{{ number(item.duration_ms) }} ms</small>
            </summary>
            <div class="case-body">
              <div><span>Fixture</span><code>{{ item.fixture_id }}</code></div>
              <div><span>Fixture hash</span><code>{{ shortHash(item.fixture_hash) }}</code></div>
              <div class="evidence"><span>最小化证据</span><pre>{{ pretty(item.evidence) }}</pre></div>
            </div>
          </details>
          <p v-if="!filteredCases.length" class="empty">当前筛选没有场景。</p>
        </div>
      </section>

      <section class="command-card">
        <div>
          <strong>重新生成同口径证据</strong>
          <small>命令只使用仓库内从零生成的匿名合成夹具。</small>
        </div>
        <code>{{ command }}</code>
      </section>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { Copy, Info, RefreshCw, ShieldAlert, ShieldCheck } from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const report = ref(null);
const loading = ref(false);
const error = ref("");
const notice = ref("");
const category = ref("");
const status = ref("");
const denominatorKeys = ["declared", "eligible", "selected", "attempted", "passed", "failed", "cleanup_failed"];
const metrics = computed(() => report.value?.metrics || {});
const denominator = computed(() => report.value?.denominator || {});
const fixture = computed(() => report.value?.fixture_manifest || {});
const command = computed(() => report.value?.run_command || "python -m app.eval.adhoc_isolation_suite --evidence-level offline_contract --enforce");
const gatePassed = computed(() => Boolean(report.value?.summary?.gate_passed));
const categories = computed(() => [...new Set((report.value?.cases || []).map((item) => item.category))]);
const filteredCases = computed(() => (report.value?.cases || []).filter((item) =>
  (!category.value || item.category === category.value)
  && (!status.value || item.status === status.value)));

async function load() {
  loading.value = true;
  error.value = "";
  try {
    report.value = await requestJson(
      "/api/admin/evaluation/adhoc-isolation/latest",
      {},
      "上传隔离报告加载失败",
    );
  } catch (exception) {
    error.value = exception.message || "上传隔离报告加载失败";
  } finally {
    loading.value = false;
  }
}
async function copyCommand() {
  try {
    await navigator.clipboard.writeText(command.value);
    notice.value = "评测命令已复制";
    setTimeout(() => { notice.value = ""; }, 1800);
  } catch {
    error.value = "剪贴板不可用，请手动复制页面底部命令。";
  }
}
function percent(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
function number(value) { return Number(value || 0).toLocaleString("zh-CN", { maximumFractionDigits: 2 }); }
function shortHash(value) {
  const text = String(value || "—");
  return text.length > 25 ? `${text.slice(0, 16)}…${text.slice(-7)}` : text;
}
function dateTime(value) { return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—"; }
function pretty(value) { return JSON.stringify(value || {}, null, 2); }
function zeroClass(value) { return Number(value || 0) === 0 ? "good" : "bad"; }
function oneClass(value) { return Number(value || 0) === 1 ? "good" : "bad"; }
function evidenceLabel(value) {
  return ({ isolated_dependencies: "本机隔离依赖证据", controlled_smoke: "受控 API Smoke" })[value] || value || "未运行";
}
function denominatorLabel(value) {
  return ({ declared: "声明", eligible: "可运行", selected: "已选择", attempted: "已尝试", passed: "通过", failed: "失败", cleanup_failed: "清理失败" })[value] || value;
}
function categoryLabel(value) {
  return ({
    lifecycle: "生命周期",
    authorization: "授权旁路",
    schema_index_isolation: "Schema / 索引隔离",
    ttl_cleanup: "TTL 与清理",
    fault_compensation: "故障补偿",
    multi_dataset_concurrency: "多数据集隔离",
    privacy_artifact: "隐私制品",
  })[value] || value;
}
function probeLabel(value) {
  return ({
    csv_xlsx_roundtrip: "CSV / XLSX 同口径解析",
    duplicate_header_normalization: "重复与空表头稳定规范化",
    conservative_type_inference: "边界类型保守推断",
    ordinary_entry_denied: "普通 SQL 入口拒绝内部表",
    cross_owner_denied: "跨用户数据集拒绝",
    cross_dataset_denied: "携带 A 身份引用 B 表拒绝",
    schema_snapshot_excludes_upload: "正式 Schema 快照排除上传表",
    meta_projection_excludes_upload: "Meta 与索引投影排除上传表",
    shared_namespace_unchanged: "共享资源前后快照一致",
    expired_context_denied: "过期版本拒绝新查询",
    stale_version_denied: "旧版本不可继续执行",
    cache_namespace_cleanup: "缓存命名空间回收",
    create_failure_compensated: "创建中断后补偿",
    drop_failure_reconciled: "DROP 首次失败后协调恢复",
    cancellation_finally_cleanup: "取消仍进入最终清理",
    parallel_namespace_isolation: "并行 Run 命名空间无交叉",
    mixed_asset_sql_denied: "多数据集越界 SQL 拒绝",
    immutable_version_isolation: "不可变版本关闭式授权",
    fixture_provenance_complete: "合成夹具来源完整",
    report_contains_no_rows: "报告不含原始行值",
    unknown_origin_rejected: "未知来源拒绝准入",
  })[value] || value;
}
function originText(value) {
  const labels = {
    synthetic_adhoc_contract: "合同夹具",
    synthetic_negative_payload: "安全反例",
  };
  const entries = Object.entries(value || {});
  return entries.length ? entries.map(([key, count]) => `${labels[key] || key} ${count}`).join(" · ") : "—";
}
onMounted(load);
</script>

<style scoped>
.isolation-page{flex:1 1 auto;min-height:0;overflow:auto;padding:36px 34px 70px;color:var(--da-text)}.page-head,.run-banner,.metric-grid,.panel,.alert,.command-card{max-width:1320px;margin-left:auto;margin-right:auto}.page-head{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:22px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:800;letter-spacing:.17em}.page-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.page-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.actions{display:flex;gap:8px}.btn{display:inline-flex;align-items:center;gap:7px;min-height:38px;padding:8px 13px;border:1px solid var(--da-border);border-radius:10px;background:#fff;color:var(--da-text-secondary);font:12px inherit;cursor:pointer}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.alert{margin-bottom:13px;padding:11px 14px;border-radius:10px;font-size:12px}.alert.error{border:1px solid #f0c5bf;background:#fff3f1;color:#bd3b2c}.alert.notice{border:1px solid #b9e2c8;background:#f0fbf4;color:#287b4a}.loading{min-height:380px;display:grid;place-items:center;color:var(--da-text-muted)}.run-banner{display:flex;align-items:center;gap:14px;margin-bottom:14px;padding:16px 18px;border:1px solid #b9dfca;border-radius:15px;background:linear-gradient(135deg,#f0fbf5,#fff)}.run-banner.failed{border-color:#efc4be;background:linear-gradient(135deg,#fff2f0,#fff)}.gate-icon{width:45px;height:45px;display:grid;place-items:center;border-radius:13px;background:#dff5e8;color:#267b4b}.failed .gate-icon{background:#ffe1dd;color:#b63c30}.run-banner>div:nth-child(2){min-width:0;flex:1}.run-banner strong,.run-banner span{display:block}.run-banner strong{font-size:15px}.run-banner span{margin-top:4px;color:var(--da-text-muted);font-size:11px}.run-banner dl{display:flex;gap:25px;margin:0}.run-banner dl div{max-width:260px}.run-banner dt{color:var(--da-text-muted);font-size:9px}.run-banner dd{overflow:hidden;margin:4px 0 0;font:10px Consolas,monospace;text-overflow:ellipsis;white-space:nowrap}.metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px}.metric-grid article{position:relative;overflow:hidden;padding:16px;border:1px solid var(--da-border-light);border-radius:14px;background:#fff;box-shadow:var(--da-shadow-sm)}.metric-grid article:before{position:absolute;inset:0 auto 0 0;width:3px;background:#8d99a9;content:""}.metric-grid article.good:before{background:#35a56a}.metric-grid article.bad:before{background:#dc6656}.metric-grid span,.metric-grid small{display:block;color:var(--da-text-muted);font-size:9px}.metric-grid strong{display:block;margin:7px 0 4px;font-size:21px}.panel{margin-bottom:16px;padding:21px;border:1px solid var(--da-border-light);border-radius:16px;background:rgba(255,255,255,.94);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:15px}.panel-head h2{margin:0;font-size:16px}.panel-head>b,.panel-head>span{padding:5px 9px;border-radius:999px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:10px}.panel-head .pass{background:#e8f8ee;color:#26794a}.panel-head .fail{background:#fff0ee;color:#b63e31}.denominator{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.denominator div,.fixture-summary div{padding:12px;border-radius:10px;background:var(--da-bg-soft)}.denominator span,.fixture-summary span{display:block;color:var(--da-text-muted);font-size:9px}.denominator strong,.fixture-summary strong{display:block;margin-top:5px;font-size:14px}.level-note{display:flex;align-items:flex-start;gap:9px;margin-top:13px;padding:11px 12px;border:1px solid #d5e5f7;border-radius:10px;background:#f6faff;color:#49627e}.level-note svg{flex:0 0 auto;margin-top:1px}.level-note p{margin:0;font-size:10px;line-height:1.6}.level-note code{font-family:Consolas,monospace}.fixture-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.filters{display:flex;gap:9px;margin-bottom:13px}.filters label{display:grid;gap:4px;color:var(--da-text-muted);font-size:9px}.filters select{min-height:35px;padding:7px 9px;border:1px solid var(--da-border);border-radius:8px;background:#fff}.case-list{display:grid;gap:7px}.case{border:1px solid var(--da-border-light);border-radius:10px;background:#fff}.case summary{display:flex;align-items:center;gap:9px;padding:12px;cursor:pointer;list-style:none}.case summary::-webkit-details-marker{display:none}.case-state{width:8px;height:8px;border-radius:50%;background:#dc6656}.case-state.passed{background:#35a56a}.case summary code{font:10px Consolas,monospace}.case summary strong{min-width:0;flex:1;font-size:12px}.case summary em{padding:4px 7px;border-radius:99px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:9px;font-style:normal}.case summary small{color:var(--da-text-muted);font-size:9px}.case-body{display:grid;grid-template-columns:1fr 1fr 2fr;gap:9px;padding:12px;border-top:1px solid var(--da-border-light)}.case-body>div{min-width:0;padding:10px;border-radius:8px;background:var(--da-bg-soft)}.case-body span,.case-body code{display:block}.case-body span{color:var(--da-text-muted);font-size:9px}.case-body code{overflow:hidden;margin-top:5px;font:9px Consolas,monospace;text-overflow:ellipsis}.case-body pre{overflow:auto;max-height:180px;margin:6px 0 0;font:9px/1.5 Consolas,monospace}.empty{text-align:center;color:var(--da-text-muted);font-size:12px}.command-card{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:18px;padding:15px 17px;border:1px solid #cfe0f4;border-radius:13px;background:#f4f8fd}.command-card strong,.command-card small{display:block}.command-card small{margin-top:4px;color:var(--da-text-muted);font-size:10px}.command-card code{overflow:auto;padding:9px 11px;border-radius:8px;background:#202634;color:#e8edf5;font:10px Consolas,monospace;white-space:nowrap}.spinning{animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:1120px){.metric-grid{grid-template-columns:repeat(3,1fr)}.denominator{grid-template-columns:repeat(4,1fr)}}
@media(max-width:760px){.isolation-page{padding:25px 14px 50px}.page-head{display:block}.actions{margin-top:13px}.run-banner{align-items:flex-start;flex-wrap:wrap}.run-banner dl{width:100%}.metric-grid,.fixture-summary{grid-template-columns:1fr 1fr}.denominator{grid-template-columns:repeat(2,1fr)}.case summary em{display:none}.case-body,.command-card{grid-template-columns:1fr}.page-head h1{font-size:25px}}
</style>
