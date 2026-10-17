<template>
  <div class="meta-view">
    <div class="meta-toolbar">
      <div><div class="meta-eyebrow">ADMIN / KNOWLEDGE</div><h1>元知识管理</h1><p>维护指标口径、业务术语和 SQL 样例，让问数召回随业务变化同步更新。</p></div>
      <div class="toolbar-actions"><label class="schema-sync-option"><input v-model="rebuildValueIndex" type="checkbox" />同步字段值索引</label><button class="meta-btn with-icon" :disabled="loading || schemaSyncing" @click="syncSchema"><RefreshCw :size="14" aria-hidden="true" />{{ schemaSyncing ? "Schema 同步中…" : "同步 Schema" }}</button><button class="meta-btn" :disabled="loading" @click="loadAll">刷新</button><button class="meta-btn primary" :disabled="rebuilding || schemaSyncing" @click="rebuildAll">{{ rebuilding ? "BGE-M3 索引重建中…" : "覆盖重建 BGE-M3 混合索引" }}</button></div>
    </div>
    <div v-if="error" class="meta-alert error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div><div v-if="notice" class="meta-alert success" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <section v-if="hybridStatus" class="hybrid-status" aria-label="BGE-M3 混合索引状态">
      <div class="hybrid-summary"><div><span class="hybrid-kicker">ACTIVE RETRIEVAL</span><strong>BGE-M3 Dense + Sparse</strong><small>{{ hybridStatus.runtime?.model }} · {{ hybridStatus.runtime?.device }} · {{ hybridStatus.runtime?.dense_size }} 维</small></div><span :class="hybridStatus.ready ? 'hybrid-ready' : 'hybrid-pending'">{{ hybridStatus.ready ? "索引就绪" : "需要覆盖重建" }}</span></div>
      <div class="hybrid-contract"><span>命名向量 <code>dense</code> + <code>sparse</code></span><span>融合 <code>entity RRF</code></span><span>Point 投影 <code>{{ strategyLabel(hybridStatus.point_projection?.active_strategy) }}</code></span><span>身份 <code>{{ hybridStatus.point_projection?.identity_version || "—" }}</code></span><span>v1 Dense 回退 <code>关闭</code></span></div>
      <details v-if="hybridStatus.runtime?.scheduling" class="embedding-runtime" aria-label="BGE-M3 推理调度状态">
        <summary><span>推理调度 <code>{{ runtimeProfileLabel(hybridStatus.runtime.runtime_profile) }}</code></span><span :class="runtimeHealthy ? 'hybrid-ready' : 'hybrid-pending'">{{ runtimeStateLabel(hybridStatus.runtime.scheduling.state) }}</span><small>队列 {{ runtimeQueue.depth || 0 }} / {{ runtimeQueue.capacity || "—" }} · 最近批次 {{ runtimeBatching.last_batch_texts || 0 }} texts</small></summary>
        <div class="runtime-grid">
          <article><span>调度策略</span><b>{{ runtimePolicy.batch_window_ms || 0 }} ms</b><small>机会式窗口 · 最多 {{ runtimePolicy.max_batch_texts || "—" }} texts / {{ runtimePolicy.max_batch_tokens || "—" }} tokens</small></article>
          <article><span>批处理</span><b>{{ runtimeBatching.merged_batch_count || 0 }} 次合批</b><small>最大批次 {{ runtimeBatching.max_batch_texts_seen || 0 }} · 最大排队 {{ runtimeQueue.max_depth_seen || 0 }}</small></article>
          <article><span>运行时 P95</span><b>{{ runtimeLatency.total?.p95 || 0 }} ms</b><small>排队 {{ runtimeLatency.queue_wait?.p95 || 0 }} · 推理 {{ runtimeLatency.inference?.p95 || 0 }} ms</small></article>
          <article><span>容量结果</span><b>{{ runtimeRequests.completed || 0 }} 完成 / {{ runtimeRequests.rejected || 0 }} 拒绝</b><small>取消 {{ runtimeRequests.cancelled || 0 }} · 超时 {{ runtimeRequests.timed_out || 0 }} · 失败 {{ runtimeRequests.failed || 0 }}</small></article>
        </div>
        <p>单进程模型实例 {{ hybridStatus.runtime.model_instances || 0 }}；运行时切换不改变 dense+sparse 向量指纹，也不会记录原始问题文本。</p>
      </details>
      <div class="hybrid-collections"><article v-for="(item, name) in hybridStatus.collections" :key="name"><b>{{ collectionLabel(name) }}</b><strong>{{ item.matching_points_count ?? item.points_count ?? 0 }} <em>/ {{ item.expected_points_count ?? "—" }}</em></strong><small>当前合同 Point / 预期 · {{ collectionState(item) }}</small><small v-if="routeManifest(name)">{{ routeManifest(name).entity_count }} 实体 · {{ Number(routeManifest(name).points_per_entity || 0).toFixed(2) }} Point/实体 · {{ roleSummary(routeManifest(name).point_roles) }}</small></article></div>
      <div v-if="hybridStatus.point_projection" class="point-contract"><span>canonical 模板 <code>{{ hybridStatus.point_projection.canonical_template_version }}</code></span><span>别名策略 <code>{{ hybridStatus.point_projection.alias_policy_version }}</code></span><span>批准别名 <b>{{ hybridStatus.point_projection.approved_alias_count }}</b></span><span>支持对照 <code>{{ hybridStatus.point_projection.supported_strategies?.join(" / ") }}</code></span></div>
      <p v-if="hybridStatus.runtime?.error" class="hybrid-error">{{ hybridStatus.runtime.error }}</p>
    </section>
    <section v-if="valueIndexStatus" class="value-index-status" aria-label="ES 字段值索引状态">
      <div><span class="hybrid-kicker">GOVERNED VALUE RETRIEVAL</span><strong>ES 字段值索引</strong><small>真实字段值只从 ES 检索；Qdrant Dense/Sparse 仅召回元知识实体</small></div>
      <div class="value-index-facts"><span :class="valueIndexStatus.ready ? 'hybrid-ready' : 'hybrid-pending'">{{ valueIndexStatus.ready ? "active 版本就绪" : "需要同步字段值索引" }}</span><span v-if="valueIndexStatus.drift_reasons?.length" class="hybrid-pending" :title="valueIndexStatus.drift_reasons.join('、')">漂移 {{ valueIndexStatus.drift_reasons.length }}</span><span>文档 <b>{{ valueIndexStatus.document_count || 0 }}</b></span><span>白名单字段 <b>{{ valueIndexStatus.sync_column_count || 0 }}</b></span><span>预检拦截 <b>{{ valueIndexStatus.policy_precheck_blocked?.length || 0 }}</b></span><code :title="valueIndexStatus.index_version || ''">{{ shortVersion(valueIndexStatus.index_version) }}</code></div>
    </section>
    <div class="summary-grid"><div class="summary-card"><span class="summary-icon blue"><Sigma :size="20" aria-hidden="true" /></span><div><strong>{{ overview.metrics || 0 }}</strong><span>可计算指标</span></div></div><div class="summary-card"><span class="summary-icon violet"><BookOpenText :size="20" aria-hidden="true" /></span><div><strong>{{ overview.terminology || 0 }}</strong><span>业务术语</span></div></div><div class="summary-card"><span class="summary-icon green"><FileCode2 :size="20" aria-hidden="true" /></span><div><strong>{{ overview.training || 0 }}</strong><span>SQL 样例</span></div></div><div class="summary-card"><span class="summary-icon orange"><Table2 :size="20" aria-hidden="true" /></span><div><strong>{{ overview.tables || 0 }}</strong><span>已同步数据表</span></div></div></div>
    <div class="meta-tabs"><button :class="{ active: tab === 'metrics' }" @click="tab = 'metrics'">指标口径 <em>{{ metricItems.length }}</em></button><button :class="{ active: tab === 'terminology' }" @click="tab = 'terminology'">业务术语 <em>{{ terminologyItems.length }}</em></button><button :class="{ active: tab === 'training' }" @click="tab = 'training'">SQL 样例 <em>{{ trainingItems.length }}</em></button><button :class="{ active: tab === 'values' }" @click="tab = 'values'">字段值检索 <em>{{ valueIndexStatus?.document_count || 0 }}</em></button><button :class="{ active: tab === 'tables' }" @click="tab = 'tables'">Schema 预览 <em>{{ tables.length }}</em></button></div>
    <section class="meta-panel">
      <div class="panel-head"><div><h2>{{ tabTitle }}</h2><span class="panel-sub">{{ tabDescription }}</span></div><div class="panel-actions" v-if="tab !== 'tables' && tab !== 'values'"><input v-model.trim="search" class="search-input" placeholder="搜索名称或问题…" /><button class="meta-btn primary with-icon" @click="openCreate"><Plus :size="14" aria-hidden="true" />新增{{ tab === 'training' ? '样例' : tab === 'terminology' ? '术语' : '指标' }}</button></div><form v-else-if="tab === 'values'" class="value-search-form" @submit.prevent="searchValues"><label>检索词<input v-model.trim="valueQuery.keyword" required maxlength="255" placeholder="例如：华东" /></label><label>模式<select v-model="valueQuery.queryMode"><option value="exact">精确</option><option value="prefix">前缀</option><option value="full_text">全文</option></select></label><label>表<select v-model="valueQuery.tableId" @change="valueQuery.columnId = ''"><option value="">全部允许表</option><option v-for="item in tables" :key="item.id" :value="item.id">{{ item.name }}</option></select></label><label>字段<select v-model="valueQuery.columnId"><option value="">全部允许字段</option><option v-for="item in valueColumns" :key="item.id" :value="item.id">{{ item.id }}</option></select></label><button class="meta-btn primary" type="submit" :disabled="valueSearching">{{ valueSearching ? "检索中…" : "检索" }}</button></form></div>
      <div v-if="loading" class="meta-empty">正在加载元知识…</div>
      <div v-else-if="tab === 'metrics' || tab === 'terminology'" class="table-scroll"><table class="meta-table"><thead><tr><th>名称</th><th>业务定义</th><th>别名</th><th>关联字段</th><th>状态</th><th class="ops-col">操作</th></tr></thead><tbody><tr v-for="item in visibleMetrics" :key="item.id"><td><div class="name-cell"><strong>{{ item.name }}</strong><small>{{ item.id }}</small></div></td><td class="description-cell">{{ item.description || '—' }}</td><td><div class="tag-list"><span v-for="alias in item.alias" :key="alias" class="tag">{{ alias }}</span><span v-if="!item.alias.length" class="muted">—</span></div></td><td><div class="field-list"><code v-for="field in item.relevant_columns" :key="field">{{ field }}</code><span v-if="!item.relevant_columns.length" class="term-label">纯术语</span></div></td><td><button class="status-toggle" :class="{ on: item.enabled }" @click="toggleMetric(item)"><i></i>{{ item.enabled ? '已启用' : '已停用' }}</button></td><td class="ops"><button @click="openEdit(item)">编辑</button><button class="danger-text" @click="removeMetric(item)">删除</button></td></tr></tbody></table><div v-if="!visibleMetrics.length" class="meta-empty">暂无匹配的{{ tab === 'terminology' ? '业务术语' : '指标' }}，点击右上角新增。</div></div>
      <div v-else-if="tab === 'training'" class="training-list"><article v-for="item in visibleTraining" :key="item.id" class="training-card"><div class="training-main"><div class="training-question">Q · {{ item.question }}</div><pre>{{ item.sql }}</pre></div><div class="training-side"><button class="status-toggle" :class="{ on: item.enabled }" @click="toggleTraining(item)"><i></i>{{ item.enabled ? '已启用' : '已停用' }}</button><small>{{ formatDate(item.created_at) }}</small><div><button @click="openEdit(item)">编辑</button><button class="danger-text" @click="removeTraining(item)">删除</button></div></div></article><div v-if="!visibleTraining.length" class="meta-empty">暂无匹配的 SQL 样例，点击右上角新增。</div></div>
      <div v-else-if="tab === 'values'" class="value-results"><details v-if="valueIndexStatus?.retained_versions?.length" class="value-history"><summary>已保留版本（ES 可独立回退）</summary><div v-for="item in valueIndexStatus.retained_versions" :key="item.physical_index"><code>{{ shortVersion(item.index_version) }}</code><span>{{ item.document_count }} 条</span><b v-if="item.active">当前 active</b><button v-else class="meta-btn" @click="rollbackValueIndex(item)">回退到此版本</button></div></details><div v-if="valueSearchEvidence" class="value-evidence"><span>职责 <code>{{ valueSearchEvidence.responsibility }}</code></span><span>模式 <code>{{ valueSearchEvidence.query_mode }}</code></span><span>版本 <code>{{ shortVersion(valueSearchEvidence.index_version) }}</code></span><span>ES 查询阶段权限过滤 <b>{{ valueSearchEvidence.filter_stage === 'elasticsearch_query' ? '已应用' : '缺失' }}</b></span><span>命中 {{ valueSearchEvidence.hit_count || 0 }}</span><span>{{ valueSearchEvidence.latency_ms || 0 }} ms</span><span>原值写日志 <b>{{ valueSearchEvidence.raw_value_logged ? '是' : '否' }}</b></span></div><div v-if="!valueSearchRan" class="meta-empty">输入检索词，可验证 exact / prefix / full_text 三种受控模式以及工作空间、表、列和版本过滤。</div><div v-else-if="!valueResults.length" class="meta-empty">没有命中；请确认字段已勾选同步、未被敏感/高基数策略拦截，并已重建字段值索引。</div><div v-else class="table-scroll"><table class="meta-table"><thead><tr><th>字段值</th><th>表</th><th>字段</th><th>匹配模式</th><th>版本</th></tr></thead><tbody><tr v-for="item in valueResults" :key="item.document_id || item.id"><td><strong>{{ item.value }}</strong></td><td><code>{{ item.table_id }}</code></td><td><code>{{ item.column_id }}</code></td><td>{{ item.query_mode }}</td><td><code>{{ shortVersion(item.index_version) }}</code></td></tr></tbody></table></div></div>
      <div v-else class="schema-list"><div v-if="schemaSyncSummary" class="schema-sync-summary">最近同步：{{ schemaSyncSummary.source }} · {{ schemaSyncSummary.table_count }} 张表 / {{ schemaSyncSummary.column_count }} 个字段 · 新增表 {{ schemaSyncSummary.diff.counts.added_tables }} · 变更字段 {{ schemaSyncSummary.diff.counts.changed_columns }} · 删除字段 {{ schemaSyncSummary.diff.counts.removed_columns }}</div><details v-for="table in tables" :key="table.id" class="schema-card"><summary><span class="schema-name">{{ table.name }}</span><span class="schema-role">{{ table.role }}</span><span class="schema-description">{{ table.description || '暂无表描述' }}</span><span class="schema-count">{{ table.columns.length }} 个字段</span></summary><div class="schema-columns"><div v-for="column in table.columns" :key="column.id" class="column-row"><code>{{ column.name }} <em v-if="column.is_display" class="display-badge">默认展示</em></code><span class="column-type">{{ column.type }}</span><span class="column-role">{{ column.role }}</span><span>{{ column.description || '—' }}</span></div></div></details><div v-if="!tables.length" class="meta-empty">暂无已构建的 Schema，请先点击“同步 Schema”。</div></div>
    </section>
    <div
      v-if="modal.open"
      class="modal-mask"
      @click.self="closeModal"
      @keydown.esc.stop.prevent="closeModal"
    >
      <div
        ref="modalElement"
        class="meta-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="meta-edit-dialog-title"
        aria-describedby="meta-edit-dialog-description"
        tabindex="-1"
        @keydown.tab="trapModalFocus"
      >
        <div class="modal-head">
          <div>
            <h2 id="meta-edit-dialog-title">
              {{ modal.id ? '编辑' : '新增' }}{{ modal.type === 'training' ? ' SQL 样例' : modal.type === 'terminology' ? '业务术语' : '指标口径' }}
            </h2>
            <p id="meta-edit-dialog-description">
              {{ modal.type === 'terminology' ? '术语不绑定具体字段，只提供业务定义给模型参考。' : modal.type === 'training' ? '问题与 SQL 会作为 few-shot 样例参与召回。' : '关联字段用于把指标口径桥接到真实 Schema。' }}
            </p>
          </div>
          <button type="button" class="close-btn" aria-label="关闭元知识编辑弹层" @click="closeModal">
            <X :size="18" aria-hidden="true" />
          </button>
        </div>
        <form @submit.prevent="saveItem">
          <template v-if="modal.type !== 'training'">
            <label>
              名称
              <input ref="modalFirstField" v-model.trim="form.name" required maxlength="128" :disabled="!!modal.id" placeholder="例如：GMV" />
            </label>
            <label>业务定义 <textarea v-model.trim="form.description" rows="4" required placeholder="描述统计口径、适用范围或业务含义"></textarea></label>
            <label>别名 <input v-model="form.aliasText" placeholder="用逗号分隔，例如：成交额,交易总额" /></label>
            <label v-if="modal.type === 'metric'">关联字段 <input v-model="form.columnsText" placeholder="用逗号分隔，例如：fact_order.order_amount" /></label>
          </template>
          <template v-else>
            <label>
              自然语言问题
              <textarea ref="modalFirstField" v-model.trim="form.question" rows="3" required maxlength="255" placeholder="例如：统计各地区的销售总额"></textarea>
            </label>
            <label>目标 SQL <textarea v-model.trim="form.sql" rows="8" required spellcheck="false" placeholder="SELECT …"></textarea></label>
          </template>
          <label class="form-check"><input v-model="form.enabled" type="checkbox" /> 保存后启用并参与召回</label>
          <div class="modal-foot">
            <span class="modal-tip">保存后请点击“同步索引”使改动进入向量检索。</span>
            <button type="button" class="meta-btn" @click="closeModal">取消</button>
            <button type="submit" class="meta-btn primary" :disabled="saving">{{ saving ? '保存中…' : '保存' }}</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref } from "vue";
import {
  BookOpenText,
  CircleAlert,
  CircleCheck,
  FileCode2,
  Plus,
  RefreshCw,
  Sigma,
  Table2,
  X,
} from "@lucide/vue";
import { requestJson } from "../utils/http.js";
import { confirmDialog } from "../composables/appDialog.js";
const API = "/api/admin/meta";
const tab = ref("metrics"), loading = ref(false), saving = ref(false), rebuilding = ref(false), schemaSyncing = ref(false), rebuildValueIndex = ref(false), valueSearching = ref(false), valueSearchRan = ref(false), error = ref(""), notice = ref(""), search = ref("");
const metrics = ref([]), trainingItems = ref([]), tables = ref([]), overview = reactive({ metrics: 0, terminology: 0, training: 0, tables: 0 });
const schemaSyncSummary = ref(null);
const hybridStatus = ref(null);
const valueIndexStatus = ref(null), valueResults = ref([]), valueSearchEvidence = ref(null);
const valueQuery = reactive({ keyword: "", queryMode: "exact", tableId: "", columnId: "" });
const modal = reactive({ open: false, type: "metric", id: null });
const modalElement = ref(null), modalFirstField = ref(null);
let modalTrigger = null;
const form = reactive({ name: "", description: "", aliasText: "", columnsText: "", question: "", sql: "", enabled: true });
const metricItems = computed(() => metrics.value.filter((item) => item.kind === "metric"));
const terminologyItems = computed(() => metrics.value.filter((item) => item.kind === "terminology"));
const visibleMetrics = computed(() => (tab.value === "terminology" ? terminologyItems.value : metricItems.value).filter((item) => `${item.name} ${item.description} ${(item.alias || []).join(" ")}`.toLowerCase().includes(search.value.toLowerCase())));
const visibleTraining = computed(() => trainingItems.value.filter((item) => `${item.question} ${item.sql}`.toLowerCase().includes(search.value.toLowerCase())));
const valueColumns = computed(() => tables.value
  .filter((table) => !valueQuery.tableId || table.id === valueQuery.tableId)
  .flatMap((table) => table.columns || []));
const runtimeQueue = computed(() => hybridStatus.value?.runtime?.scheduling?.queue || {});
const runtimePolicy = computed(() => hybridStatus.value?.runtime?.scheduling?.policy || {});
const runtimeMetrics = computed(() => hybridStatus.value?.runtime?.scheduling?.metrics || {});
const runtimeBatching = computed(() => runtimeMetrics.value.batching || {});
const runtimeLatency = computed(() => runtimeMetrics.value.latency_ms || {});
const runtimeRequests = computed(() => runtimeMetrics.value.requests || {});
const runtimeHealthy = computed(() => ["idle", "running"].includes(hybridStatus.value?.runtime?.scheduling?.state) && Number(runtimeRequests.value.failed || 0) === 0);
const tabTitle = computed(() => ({ metrics: "指标口径", terminology: "业务术语", training: "SQL 样例", values: "字段值检索", tables: "Schema 预览" }[tab.value]));
const tabDescription = computed(() => ({ metrics: "可计算指标绑定真实字段，帮助模型遵循统一口径。", terminology: "纯术语型指标，不绑定字段，只用于理解业务黑话。", training: "高质量问题-SQL 对会以 few-shot 形式注入 SQL 生成。", values: "ES 是真实字段值的唯一召回面；检索在工作空间、active 版本及行列权限内完成。", tables: "当前已构建的表与字段元数据，仅供查看。" }[tab.value]));
async function request(path, options = {}) { return requestJson(`${API}${path}`, options, "元知识请求失败"); }
function showNotice(text) { notice.value = text; window.clearTimeout(showNotice.timer); showNotice.timer = window.setTimeout(() => { notice.value = ""; }, 3600); }
async function loadAll() { loading.value = true; error.value = ""; try { const [summary, allMetrics, allTraining, allTables, hybrid, valueStatus] = await Promise.all([request("/overview"), request("/metrics"), request("/training"), request("/tables"), request("/hybrid/status"), request("/value-index/status")]); Object.assign(overview, summary); metrics.value = allMetrics; trainingItems.value = allTraining; tables.value = allTables; hybridStatus.value = hybrid; valueIndexStatus.value = valueStatus; } catch (e) { error.value = e.message || "元知识加载失败"; } finally { loading.value = false; } }
function resetForm() { Object.assign(form, { name: "", description: "", aliasText: "", columnsText: "", question: "", sql: "", enabled: true }); }
function modalFocusableElements() {
  if (!modalElement.value) return [];
  return [...modalElement.value.querySelectorAll(
    'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
  )].filter((element) => !element.hidden && element.getClientRects().length > 0);
}
async function focusModal() {
  await nextTick();
  const firstFormField = modalElement.value?.querySelector(
    'form input:not([disabled]), form textarea:not([disabled]), form select:not([disabled])',
  );
  const target = modalFirstField.value && !modalFirstField.value.disabled
    ? modalFirstField.value
    : firstFormField || modalFocusableElements()[0] || modalElement.value;
  target?.focus();
}
function trapModalFocus(event) {
  const elements = modalFocusableElements();
  if (!elements.length) {
    event.preventDefault();
    modalElement.value?.focus();
    return;
  }
  const first = elements[0];
  const last = elements[elements.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}
function openCreate() {
  modalTrigger = document.activeElement;
  resetForm();
  modal.id = null;
  modal.type = tab.value === "training" ? "training" : tab.value === "terminology" ? "terminology" : "metric";
  modal.open = true;
  focusModal();
}
function openEdit(item) {
  modalTrigger = document.activeElement;
  resetForm();
  modal.id = item.id;
  modal.type = item.question !== undefined ? "training" : item.kind;
  if (modal.type === "training") Object.assign(form, { question: item.question, sql: item.sql, enabled: item.enabled });
  else Object.assign(form, { name: item.name, description: item.description, aliasText: (item.alias || []).join(","), columnsText: (item.relevant_columns || []).join(","), enabled: item.enabled });
  modal.open = true;
  focusModal();
}
function closeModal() {
  if (!modal.open) return;
  const trigger = modalTrigger;
  modalTrigger = null;
  modal.open = false;
  nextTick(() => {
    if (!modal.open && trigger?.isConnected) trigger.focus();
  });
}
function listText(value) { return value.split(",").map((item) => item.trim()).filter(Boolean); }
async function saveItem() { saving.value = true; error.value = ""; try { if (modal.type === "training") { const payload = { question: form.question, sql: form.sql, enabled: form.enabled }; await request(modal.id ? `/training/${modal.id}` : "/training", { method: modal.id ? "PATCH" : "POST", body: JSON.stringify(payload) }); } else { const payload = { description: form.description, alias: listText(form.aliasText), relevant_columns: modal.type === "metric" ? listText(form.columnsText) : [], enabled: form.enabled }; if (modal.id) await request(`/metrics/${encodeURIComponent(modal.id)}`, { method: "PATCH", body: JSON.stringify(payload) }); else await request("/metrics", { method: "POST", body: JSON.stringify({ name: form.name, ...payload }) }); } closeModal(); await loadAll(); showNotice("已保存。请同步对应索引后，新内容才会参与召回。"); } catch (e) { error.value = e.message || "保存失败"; } finally { saving.value = false; } }
async function toggleMetric(item) { try { await request(`/metrics/${encodeURIComponent(item.id)}`, { method: "PATCH", body: JSON.stringify({ description: item.description, alias: item.alias, relevant_columns: item.relevant_columns, enabled: !item.enabled }) }); item.enabled = !item.enabled; showNotice(item.kind === "terminology" ? "状态已保存，请同步术语索引。" : "状态已保存，请同步指标索引。"); } catch (e) { error.value = e.message || "状态更新失败"; } }
async function toggleTraining(item) { try { await request(`/training/${item.id}`, { method: "PATCH", body: JSON.stringify({ question: item.question, sql: item.sql, enabled: !item.enabled }) }); item.enabled = !item.enabled; showNotice("状态已保存，请同步样例索引。"); } catch (e) { error.value = e.message || "状态更新失败"; } }
async function removeMetric(item) {
  const itemLabel = item.kind === "terminology" ? "业务术语" : "指标口径";
  const confirmed = await confirmDialog({
    title: `删除${itemLabel}？`,
    message: `“${item.name}”将不再参与元知识召回；同步索引后变更生效。`,
    confirmLabel: "确认删除",
    tone: "danger",
  });
  if (!confirmed) return;
  try {
    await request(`/metrics/${encodeURIComponent(item.id)}`, { method: "DELETE" });
    await loadAll();
    showNotice(`${itemLabel}已删除，请同步${item.kind === "terminology" ? "术语" : "指标"}索引。`);
  } catch (e) {
    error.value = e.message || "删除失败";
  }
}
async function removeTraining(item) {
  const confirmed = await confirmDialog({
    title: "删除 SQL 样例？",
    message: "该问题与 SQL 将不再作为 few-shot 样例参与召回。",
    confirmLabel: "删除样例",
    tone: "danger",
  });
  if (!confirmed) return;
  try {
    await request(`/training/${item.id}`, { method: "DELETE" });
    await loadAll();
    showNotice("样例已删除，请同步样例索引。");
  } catch (e) {
    error.value = e.message || "删除失败";
  }
}
async function rebuild(target) { rebuilding.value = true; error.value = ""; try { const result = await request("/rebuild", { method: "POST", body: JSON.stringify({ target }) }); hybridStatus.value = result.status; const indexed = result.indexed || {}; showNotice(`BGE-M3 混合索引已覆盖：字段 ${indexed.columns || 0}、指标/术语 ${indexed.metrics || 0}、SQL 样例 ${indexed.training || 0} 个实体；Point 投影 ${strategyLabel(result.status?.point_projection?.active_strategy)}。`); } catch (e) { error.value = e.message || "BGE-M3 混合索引重建失败"; } finally { rebuilding.value = false; } }
async function rebuildAll() {
  const confirmed = await confirmDialog({
    title: "覆盖重建 BGE-M3 混合索引？",
    message: `该操作会覆盖现有 3 个 Qdrant collection，并按“${strategyLabel(hybridStatus.value?.point_projection?.active_strategy)}”生成 BGE-M3 dense+sparse Point；不保留 v1 Dense 回退。`,
    confirmLabel: "覆盖重建",
    tone: "danger",
  });
  if (confirmed) await rebuild("all");
}
async function syncSchema() {
  schemaSyncing.value = true;
  error.value = "";
  try {
    const preview = await request("/schema/preview");
    const counts = preview.diff?.counts || {};
    const summary = `发现新增表 ${counts.added_tables || 0}、删除表 ${counts.removed_tables || 0}、变更表 ${counts.changed_tables || 0}；新增字段 ${counts.added_columns || 0}、删除字段 ${counts.removed_columns || 0}、变更字段 ${counts.changed_columns || 0}。`;
    const confirmed = await confirmDialog({
      title: "发布 Schema 变更？",
      message: `${summary}\n确认后将发布到元库并重建字段索引。`,
      confirmLabel: "发布并重建",
      tone: "danger",
    });
    if (!confirmed) return;
    const result = await request("/schema/sync", { method: "POST", body: JSON.stringify({ rebuild_value_index: rebuildValueIndex.value }) });
    schemaSyncSummary.value = { source: result.source, table_count: result.table_count, column_count: result.column_count, diff: result.diff };
    await loadAll();
    const valueSummary = result.value_index ? `；ES 发布 ${result.value_index.document_count || 0} 条，准入 ${result.value_index.admitted_column_count || 0} 列，拦截 ${result.value_index.blocked_column_count || 0} 列` : "";
    showNotice(`Schema 同步完成：${result.table_count} 张表、${result.column_count} 个字段；${result.value_index_rebuilt ? "字段值索引已重建" : "未重建字段值索引"}${valueSummary}。`);
  } catch (e) {
    error.value = e.message || "Schema 同步失败";
  } finally {
    schemaSyncing.value = false;
  }
}
async function searchValues() { valueSearching.value = true; valueSearchRan.value = true; error.value = ""; try { const params = new URLSearchParams({ keyword: valueQuery.keyword, query_mode: valueQuery.queryMode, page: "1", page_size: "50" }); if (valueQuery.tableId) params.set("table_id", valueQuery.tableId); if (valueQuery.columnId) params.set("column_id", valueQuery.columnId); const result = await request(`/value-index/search?${params.toString()}`); valueResults.value = result.items || []; valueSearchEvidence.value = result.evidence || null; } catch (e) { valueResults.value = []; valueSearchEvidence.value = null; error.value = e.message || "字段值检索失败"; } finally { valueSearching.value = false; } }
async function rollbackValueIndex(item) {
  const version = shortVersion(item.index_version);
  const confirmed = await confirmDialog({
    title: "回退 ES 字段值索引？",
    message: `ES 字段值索引将独立回退到 ${version}；Qdrant 元知识索引不会改变。`,
    confirmLabel: "确认回退",
    tone: "danger",
  });
  if (!confirmed) return;
  error.value = "";
  try {
    await request("/value-index/rollback", { method: "POST", body: JSON.stringify({ physical_index: item.physical_index }) });
    await loadAll();
    showNotice(`ES 字段值索引已回退到 ${version}；Qdrant profile 保持不变。`);
  } catch (e) {
    error.value = e.message || "字段值索引回退失败";
  }
}
function formatDate(value) { return value ? String(value).slice(0, 16).replace("T", " ") : "未记录时间"; }
function collectionLabel(value) { return ({ column: "字段索引", metric: "指标/术语索引", training: "SQL 样例索引" })[value] || value; }
function collectionState(item) { if (item.content_compatible === false) return "版本不匹配，需覆盖重建"; if (item.compatible) return "双向量兼容"; return item.exists ? "旧 Schema" : "未创建"; }
function strategyLabel(value) { return ({ single_canonical: "单一结构化主 Point", multi_all: "名称/描述/全别名多 Point", mixed_selective: "主 Point + 批准别名" })[value] || value || "—"; }
function runtimeProfileLabel(value) { return ({ microbatch_inprocess_v1: "有界动态微批", serial_inprocess_v1: "串行回退基线" })[value] || value || "—"; }
function runtimeStateLabel(value) { return ({ idle: "等待首个请求", running: "运行中", draining: "正在排空", closed: "已关闭" })[value] || value || "未知"; }
function routeManifest(name) { return hybridStatus.value?.point_projection?.routes?.[name] || null; }
function roleSummary(roles) { return Object.entries(roles || {}).map(([name, count]) => `${name} ${count}`).join(" · ") || "无角色"; }
function shortVersion(value) { if (!value) return "未发布"; const text = String(value).replace("sha256:", ""); return `${text.slice(0, 12)}…`; }
onMounted(loadAll);
</script>

<style scoped>
.meta-view{flex:1 1 auto;min-height:0;overflow:auto;padding:28px 34px 50px;background:#f7f8fa;color:var(--da-text)}.meta-toolbar,.panel-head,.modal-head,.modal-foot{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.meta-toolbar{max-width:1360px;margin:0 auto 24px}.meta-eyebrow{color:var(--da-accent);font-size:11px;font-weight:700;letter-spacing:.14em}h1,h2,p{margin:0}h1{margin-top:5px;font-size:26px;letter-spacing:-.03em}.meta-toolbar p,.panel-sub,.modal-head p{display:block;margin-top:7px;color:var(--da-text-muted);font-size:13px}.toolbar-actions,.panel-actions{display:flex;align-items:center;gap:9px}.meta-btn{border:1px solid var(--da-border);border-radius:8px;background:#fff;color:var(--da-text-secondary);padding:8px 13px;font-size:13px;transition:.15s}.meta-btn:hover:not(:disabled){border-color:var(--da-accent);color:var(--da-accent);background:var(--da-accent-soft)}.meta-btn.primary{color:#fff;border-color:var(--da-accent);background:var(--da-accent)}.meta-btn.primary:hover:not(:disabled){color:#fff;background:var(--da-accent-hover)}.meta-btn:disabled{cursor:not-allowed;opacity:.55}.meta-alert{max-width:1360px;margin:-8px auto 18px;padding:10px 14px;border-radius:8px;font-size:13px}.meta-alert.error{color:#bd3b2c;border:1px solid #f0c5bf;background:#fff3f1}.meta-alert.success{color:#287b4a;border:1px solid #b9e2c8;background:#f0fbf4}.summary-grid{max-width:1360px;margin:0 auto 25px;display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.summary-card{display:flex;align-items:center;gap:12px;padding:17px;border:1px solid var(--da-border-light);border-radius:11px;background:#fff;box-shadow:0 2px 8px rgba(31,35,41,.025)}.summary-icon{width:35px;height:35px;display:grid;place-items:center;border-radius:10px;font-size:18px;font-weight:700}.summary-icon.blue{color:#327fd1;background:#eaf4ff}.summary-icon.violet{color:#8054cf;background:#f2ebff}.summary-icon.green{color:#2e9a60;background:#eaf9f0}.summary-icon.orange{color:#cf7d27;background:#fff4e6}.summary-card strong{display:block;font-size:20px;line-height:22px}.summary-card span:not(.summary-icon){color:var(--da-text-muted);font-size:12px}.meta-tabs{max-width:1360px;margin:0 auto;display:flex;gap:25px;border-bottom:1px solid var(--da-border)}.meta-tabs button{position:relative;padding:12px 2px;border:0;background:transparent;color:var(--da-text-muted);font-size:14px;cursor:pointer}.meta-tabs button.active{color:var(--da-accent);font-weight:600}.meta-tabs button.active:after{content:'';position:absolute;right:0;bottom:-1px;left:0;height:2px;background:var(--da-accent)}.meta-tabs em{display:inline-block;margin-left:4px;padding:1px 6px;border-radius:10px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:11px;font-style:normal}.meta-panel{max-width:1360px;min-height:430px;margin:20px auto 0;padding:21px;border:1px solid var(--da-border-light);border-radius:12px;background:#fff;box-shadow:0 3px 14px rgba(31,35,41,.03)}.panel-head{align-items:center;margin-bottom:17px}.panel-head h2{font-size:17px}.search-input{width:210px;padding:8px 11px;border:1px solid var(--da-border);border-radius:8px;outline:0;color:var(--da-text);font:inherit;font-size:13px}.search-input:focus{border-color:var(--da-accent)}.table-scroll{overflow-x:auto}.meta-table{width:100%;border-collapse:collapse;min-width:850px;font-size:13px}.meta-table th,.meta-table td{padding:13px 11px;border-bottom:1px solid var(--da-border-light);text-align:left;vertical-align:top}.meta-table th{color:var(--da-text-muted);font-size:12px;font-weight:500}.meta-table tr:hover td{background:#fbfcfe}.name-cell strong,.name-cell small{display:block}.name-cell small{margin-top:4px;color:var(--da-text-muted);font-size:11px}.description-cell{max-width:340px;line-height:1.65;color:var(--da-text-secondary)}.tag-list,.field-list{display:flex;flex-wrap:wrap;gap:4px}.tag,.field-list code,.term-label{padding:2px 7px;border-radius:5px;background:var(--da-bg-soft);color:var(--da-text-secondary);font-size:11px}.field-list code{color:#5379a6}.term-label{color:#8054cf;background:#f2ebff}.muted{color:var(--da-text-muted)}.status-toggle{display:inline-flex;align-items:center;gap:6px;border:0;background:transparent;color:var(--da-text-muted);font:inherit;font-size:12px;cursor:pointer}.status-toggle i{width:24px;height:14px;position:relative;display:block;border-radius:9px;background:#cfd3d8;transition:.15s}.status-toggle i:after{content:'';width:10px;height:10px;position:absolute;top:2px;left:2px;border-radius:50%;background:#fff;transition:.15s}.status-toggle.on{color:#2e9a60}.status-toggle.on i{background:var(--da-success)}.status-toggle.on i:after{left:12px}.ops{white-space:nowrap}.ops button,.training-side button{padding:0 5px;border:0;background:transparent;color:var(--da-accent);font:inherit;font-size:12px;cursor:pointer}.danger-text{color:var(--da-danger)!important}.meta-empty{padding:70px 20px;text-align:center;color:var(--da-text-muted);font-size:13px}.training-list{display:grid;gap:10px}.training-card{display:flex;justify-content:space-between;gap:20px;padding:15px;border:1px solid var(--da-border-light);border-radius:9px;background:#fff}.training-main{min-width:0;flex:1}.training-question{margin-bottom:9px;font-size:14px;font-weight:600}.training-main pre{max-height:115px;overflow:auto;margin:0;padding:11px;border-radius:7px;background:#20252b;color:#dbe7f1;font:12px/1.55 Consolas,monospace;white-space:pre-wrap}.training-side{flex:0 0 125px;display:flex;flex-direction:column;align-items:flex-end;gap:12px}.training-side small{color:var(--da-text-muted);font-size:11px}.schema-list{display:grid;gap:8px}.schema-card{border:1px solid var(--da-border-light);border-radius:8px;overflow:hidden}.schema-card summary{display:flex;align-items:center;gap:12px;padding:13px 15px;cursor:pointer;list-style:none}.schema-card summary::-webkit-details-marker{display:none}.schema-name{font-weight:600}.schema-role,.schema-count{padding:2px 7px;border-radius:4px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:11px}.schema-description{flex:1;overflow:hidden;color:var(--da-text-secondary);font-size:12px;text-overflow:ellipsis;white-space:nowrap}.schema-columns{border-top:1px solid var(--da-border-light)}.column-row{display:grid;grid-template-columns:1.1fr 110px 100px 2fr;gap:12px;padding:9px 15px;border-bottom:1px solid var(--da-border-light);color:var(--da-text-secondary);font-size:12px}.column-row:last-child{border-bottom:0}.column-row code{color:#5379a6}.column-type,.column-role{color:var(--da-text-muted)}.modal-mask{position:fixed;z-index:20;inset:0;display:grid;place-items:center;padding:20px;background:rgba(31,35,41,.35)}.meta-modal{width:min(620px,100%);max-height:calc(100vh - 40px);overflow:auto;padding:23px;border-radius:13px;background:#fff;box-shadow:0 18px 60px rgba(31,35,41,.22)}.modal-head{margin-bottom:20px}.modal-head h2{font-size:18px}.close-btn{border:0;background:transparent;color:var(--da-text-muted);font-size:25px;cursor:pointer}.meta-modal label{display:block;margin:14px 0;color:var(--da-text-secondary);font-size:12px}.meta-modal input:not([type=checkbox]),.meta-modal textarea{display:block;width:100%;margin-top:6px;padding:9px 11px;border:1px solid var(--da-border);border-radius:7px;outline:0;color:var(--da-text);font:inherit;font-size:13px;resize:vertical}.meta-modal input:focus,.meta-modal textarea:focus{border-color:var(--da-accent)}.form-check{display:flex!important;align-items:center;gap:8px;color:var(--da-text)!important}.form-check input{width:15px;height:15px}.modal-foot{align-items:center;margin-top:22px;padding-top:16px;border-top:1px solid var(--da-border-light)}.modal-tip{flex:1;color:var(--da-text-muted);font-size:11px;line-height:1.5}@media(max-width:850px){.meta-view{padding:22px 15px 40px}.summary-grid{grid-template-columns:repeat(2,1fr)}.meta-toolbar{display:block}.toolbar-actions{margin-top:15px}.panel-head{display:block}.panel-actions{margin-top:13px}.training-card{display:block}.training-side{width:100%;flex-direction:row;align-items:center;margin-top:12px}.column-row{grid-template-columns:1fr 90px}.column-row span:last-child{grid-column:1/-1}.schema-description{display:none}}
.meta-btn.with-icon,.meta-alert{display:inline-flex;align-items:center;gap:7px}.meta-alert{display:flex}.summary-icon{display:grid;place-items:center}
</style>

<style scoped>
.hybrid-status{max-width:1420px;margin:0 auto 18px;padding:15px 17px;border:1px solid #cfe0f4;border-radius:14px;background:linear-gradient(135deg,#f4f8fd,#fbfdff);box-shadow:var(--da-shadow-sm)}
.hybrid-summary{display:flex;align-items:center;justify-content:space-between;gap:15px}.hybrid-summary strong,.hybrid-summary small{display:block}.hybrid-summary strong{margin-top:3px;font-size:16px}.hybrid-summary small{margin-top:3px;color:var(--da-text-muted);font-size:10px}.hybrid-kicker{color:var(--da-accent);font-size:9px;font-weight:750;letter-spacing:.14em}
.hybrid-ready,.hybrid-pending{padding:5px 9px;border-radius:99px;font-size:10px}.hybrid-ready{background:#e9f8ef;color:#257949}.hybrid-pending{background:#fff0db;color:#9a650f}
.hybrid-contract{display:flex;flex-wrap:wrap;gap:8px;margin-top:11px}.hybrid-contract span{padding:5px 8px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:10px}
.embedding-runtime{margin-top:10px;border:1px solid #dce8f5;border-radius:10px;background:rgba(255,255,255,.72)}.embedding-runtime summary{display:flex;align-items:center;gap:9px;padding:9px 10px;cursor:pointer;list-style:none}.embedding-runtime summary::-webkit-details-marker{display:none}.embedding-runtime summary span:first-child{font-size:10px;font-weight:650}.embedding-runtime summary small{margin-left:auto;color:var(--da-text-muted);font-size:9px}.embedding-runtime summary .hybrid-ready,.embedding-runtime summary .hybrid-pending{padding:3px 7px}.runtime-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;padding:0 10px 9px}.runtime-grid article{padding:9px;border-radius:8px;background:#f6f9fd}.runtime-grid span,.runtime-grid b,.runtime-grid small{display:block}.runtime-grid span,.runtime-grid small{color:var(--da-text-muted);font-size:8px}.runtime-grid b{margin:4px 0;font-size:11px}.embedding-runtime>p{margin:0;padding:8px 10px;border-top:1px solid #e6eef7;color:var(--da-text-muted);font-size:8px}
.hybrid-collections{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}.hybrid-collections article{padding:10px;border:1px solid #dce8f5;border-radius:9px;background:#fff}.hybrid-collections b,.hybrid-collections strong,.hybrid-collections small{display:block}.hybrid-collections b,.hybrid-collections small{color:var(--da-text-muted);font-size:9px}.hybrid-collections strong{margin:4px 0;font-size:17px}.hybrid-collections strong em{color:var(--da-text-muted);font-size:11px;font-style:normal;font-weight:500}.hybrid-collections small+small{margin-top:4px;color:#55708c}.hybrid-error{margin-top:9px;color:#bd3b2c;font-size:10px}
.point-contract{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid #dce8f5}.point-contract span{color:var(--da-text-muted);font-size:9px}.point-contract code{color:#45637e}.point-contract b{color:var(--da-text)}
@media(max-width:850px){.hybrid-collections,.runtime-grid{grid-template-columns:1fr}.embedding-runtime summary{align-items:flex-start;flex-wrap:wrap}.embedding-runtime summary small{width:100%;margin-left:0}}
</style>

<style scoped>
.schema-sync-option{display:flex;align-items:center;gap:5px;color:var(--da-text-muted);font-size:12px;white-space:nowrap}.schema-sync-option input{accent-color:var(--da-accent)}.schema-sync-summary{padding:10px 12px;border:1px solid #cfe0f4;border-radius:7px;background:#f4f8fd;color:var(--da-text-secondary);font-size:12px}
.display-badge{display:inline-block;padding:2px 5px;border-radius:5px;background:#e9f8ef;color:#257949;font:9px sans-serif;font-style:normal}
.value-index-status{max-width:1420px;margin:0 auto 18px;padding:14px 17px;border:1px solid #d8e5dc;border-radius:14px;background:linear-gradient(135deg,#f4fbf7,#fcfefd);box-shadow:var(--da-shadow-sm);display:flex;align-items:center;justify-content:space-between;gap:20px}.value-index-status strong,.value-index-status small{display:block}.value-index-status strong{margin-top:3px;font-size:15px}.value-index-status small{margin-top:3px;color:var(--da-text-muted);font-size:10px}.value-index-facts{display:flex;align-items:center;justify-content:flex-end;flex-wrap:wrap;gap:8px}.value-index-facts>span,.value-index-facts>code{padding:5px 8px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:10px}.value-index-facts b{color:var(--da-text)}.value-search-form{display:flex;align-items:flex-end;justify-content:flex-end;flex-wrap:wrap;gap:8px}.value-search-form label{color:var(--da-text-muted);font-size:10px}.value-search-form input,.value-search-form select{display:block;min-width:125px;height:35px;margin-top:4px;padding:0 9px;border:1px solid var(--da-border);border-radius:8px;background:#fff;color:var(--da-text);font:12px inherit}.value-search-form label:first-child input{min-width:180px}.value-evidence{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:15px;padding:10px 12px;border:1px solid #d8e5dc;border-radius:9px;background:#f4fbf7}.value-evidence span{color:var(--da-text-muted);font-size:11px}.value-evidence code,.value-evidence b{color:#287b4a}.value-results .meta-table{min-width:760px}
.value-history{margin-bottom:14px;padding:10px 12px;border:1px solid var(--da-border-light);border-radius:9px;background:#fbfcfe}.value-history summary{cursor:pointer;color:var(--da-accent);font-size:12px}.value-history>div{display:grid;grid-template-columns:minmax(130px,1fr) 80px 110px;align-items:center;gap:10px;padding:7px 0;border-bottom:1px dashed var(--da-border-light);font-size:11px}.value-history>div:last-child{border-bottom:0}.value-history b{color:#287b4a}.value-history .meta-btn{min-height:28px;padding:4px 8px;font-size:10px}
/* visual refresh */
.meta-view{padding:38px 34px 64px;background:transparent}.meta-toolbar,.summary-grid,.meta-tabs,.meta-panel,.meta-alert{max-width:1420px}.meta-toolbar{margin-bottom:25px}h1{margin-top:7px;font-size:29px;letter-spacing:-.04em}.meta-eyebrow{font-size:10px;letter-spacing:.17em}.toolbar-actions{flex-wrap:wrap;justify-content:flex-end}.meta-btn{min-height:36px;border-radius:9px;font-weight:550}.meta-btn.primary{box-shadow:0 6px 14px rgba(52,120,246,.16)}.summary-grid{gap:14px}.summary-card{min-height:82px;padding:18px;border-radius:15px;box-shadow:var(--da-shadow-sm);transition:.15s}.summary-card:hover{box-shadow:var(--da-shadow-md);transform:translateY(-1px)}.summary-icon{width:39px;height:39px;border-radius:12px}.summary-card strong{font-size:22px}.meta-tabs{gap:30px}.meta-tabs button{padding:13px 3px}.meta-panel{margin-top:20px;padding:23px;border-radius:16px;box-shadow:var(--da-shadow-sm)}.search-input{height:37px;border-radius:9px;background:#fbfcfe}.meta-table tbody tr:nth-child(even) td{background:rgba(244,247,251,.42)}.training-card,.schema-card{border-radius:11px}.training-card{padding:17px;box-shadow:0 1px 2px rgba(24,39,75,.025)}.training-card:hover,.schema-card:hover{border-color:#d5deea}.schema-card summary{min-height:50px}.modal-mask{backdrop-filter:blur(4px)}.meta-modal{border:1px solid rgba(255,255,255,.75);border-radius:18px;box-shadow:var(--da-shadow-lg)}.meta-modal input:not([type=checkbox]),.meta-modal textarea{border-radius:9px;background:#fbfcfe}@media(max-width:850px){.meta-view{padding:25px 15px 44px}.toolbar-actions{justify-content:flex-start}.meta-tabs{gap:18px;overflow-x:auto}.meta-tabs button{flex:0 0 auto}.meta-panel{padding:16px}.panel-actions{flex-wrap:wrap}.search-input{width:100%;flex-basis:100%}}@media(max-width:520px){.summary-grid{grid-template-columns:1fr 1fr;gap:9px}.summary-card{min-height:72px;padding:12px}.summary-icon{width:34px;height:34px}.summary-card strong{font-size:19px}.schema-sync-option{flex-basis:100%}}
</style>
