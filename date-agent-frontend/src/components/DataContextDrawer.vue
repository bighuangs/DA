<template>
  <div
    v-if="store.state.drawerOpen"
    class="context-backdrop"
    @click.self="close"
    @keydown.esc.stop="close"
  >
    <aside
      class="context-drawer"
      role="dialog"
      aria-modal="true"
      aria-labelledby="data-context-title"
    >
      <header class="drawer-head">
        <div>
          <span class="eyebrow">DATA CONTEXT</span>
          <h2 id="data-context-title">本轮使用哪些数据？</h2>
          <p>选择与授权在服务端生效；推荐候选不会自动加入查询。</p>
        </div>
        <button ref="closeButton" class="icon-button" type="button" aria-label="关闭数据上下文" @click="close">
          <X :size="18" aria-hidden="true" />
        </button>
      </header>

      <div v-if="store.state.invalidation" class="context-alert" role="alert">
        <TriangleAlert :size="16" aria-hidden="true" />
        <span>
          <strong>上下文已失效</strong>
          {{ store.state.invalidation.message }}
        </span>
      </div>

      <section class="context-section">
        <div class="section-title">
          <span>选择数据</span>
          <button type="button" class="text-button" :disabled="store.state.catalogLoading" @click="store.loadCatalog">
            <RefreshCw :size="13" :class="{ spinning: store.state.catalogLoading }" aria-hidden="true" />
            刷新
          </button>
        </div>
        <button
          type="button"
          class="source-card official"
          :class="{ active: !store.state.selectedIds.length }"
          @click="selectDw"
        >
          <span class="source-icon"><Database :size="18" aria-hidden="true" /></span>
          <span>
            <strong>正式数据仓库</strong>
            <small>受元知识、行列权限和查询成本守卫保护</small>
          </span>
          <CheckCircle2 v-if="!store.state.selectedIds.length" :size="17" aria-hidden="true" />
        </button>

        <div class="catalog-label">
          <span>我的上传数据</span>
          <em>最多 2 个</em>
        </div>
        <div v-if="store.state.catalogLoading && !store.state.catalog.length" class="drawer-empty">
          <LoaderCircle class="spinning" :size="18" aria-hidden="true" />正在加载目录…
        </div>
        <div v-else-if="store.state.catalogError" class="drawer-empty error" role="alert">
          {{ store.state.catalogError }}
        </div>
        <div v-else-if="!readyDatasets.length" class="drawer-empty">
          暂无可查询的上传版本，可从输入框回形针或“临时数据”页上传。
        </div>
        <div v-else class="source-list">
          <button
            v-for="dataset in readyDatasets"
            :key="dataset.dataset_id"
            type="button"
            class="source-card"
            :class="{ active: store.state.selectedIds.includes(dataset.dataset_id) }"
            :aria-pressed="store.state.selectedIds.includes(dataset.dataset_id)"
            @click="toggle(dataset)"
          >
            <span class="source-icon upload"><FileSpreadsheet :size="18" aria-hidden="true" /></span>
            <span>
              <strong>{{ dataset.filename }}</strong>
              <small>v{{ dataset.version_no }} · {{ dataset.row_count }} 行 · {{ dataset.column_count }} 列</small>
            </span>
            <CheckCircle2
              v-if="store.state.selectedIds.includes(dataset.dataset_id)"
              :size="17"
              aria-hidden="true"
            />
          </button>
        </div>
        <div v-if="store.state.quota" class="quota-line">
          <span>版本 {{ store.state.quota.usage.active_versions }}/{{ store.state.quota.limits.active_versions }}</span>
          <span>容量 {{ formatBytes(store.state.quota.usage.bytes) }}/{{ formatBytes(store.state.quota.limits.bytes) }}</span>
        </div>
      </section>

      <section v-if="store.state.selectedIds.length" class="context-section">
        <div class="section-title"><span>分析方式</span><em>显式切换</em></div>
        <div class="mode-grid">
          <button
            type="button"
            :class="{ active: store.state.sourceMode === 'upload_only' }"
            @click="setMode('upload_only')"
          >
            <FileSearch :size="17" aria-hidden="true" />
            <span><strong>仅查询上传表</strong><small>不调用正式 DW 召回</small></span>
          </button>
          <button
            v-if="store.state.selectedIds.length === 1"
            type="button"
            :class="{ active: store.state.sourceMode === 'upload_compare_dw' }"
            @click="setMode('upload_compare_dw')"
          >
            <Columns3 :size="17" aria-hidden="true" />
            <span><strong>与 DW 独立对比</strong><small>分别聚合，不做行级 JOIN</small></span>
          </button>
          <button
            v-if="store.state.selectedIds.length === 1"
            type="button"
            :class="{ active: store.state.sourceMode === 'upload_join_dw' }"
            @click="setMode('upload_join_dw')"
          >
            <GitMerge :size="17" aria-hidden="true" />
            <span><strong>与 DW 受控关联</strong><small>需确认映射并通过质量门禁</small></span>
          </button>
          <button
            v-if="store.state.selectedIds.length === 2"
            type="button"
            :class="{ active: store.state.sourceMode === 'multi_upload' }"
            @click="setMode('multi_upload')"
          >
            <Table2 :size="17" aria-hidden="true" />
            <span><strong>两个上传表关联</strong><small>最多两表，需确认映射</small></span>
          </button>
        </div>
      </section>

      <section v-if="store.state.mappingProposal" class="context-section mapping-section">
        <div class="section-title"><span>关联候选待确认</span><em>不会自动应用</em></div>
        <article
          v-for="item in store.state.mappingProposal.mappings || []"
          :key="`${item.left.table}.${item.left.column}-${item.right.table}.${item.right.column}`"
          class="mapping-card"
        >
          <div>
            <strong>{{ item.left.display_name }}</strong>
            <ArrowRight :size="14" aria-hidden="true" />
            <strong>{{ item.right.display_name }}</strong>
          </div>
          <small>
            {{ item.left.table }}.{{ item.left.column }} ↔
            {{ item.right.table }}.{{ item.right.column }}
          </small>
          <span>候选置信度 {{ Math.round(Number(item.score || 0) * 100) }}%</span>
        </article>
        <button
          type="button"
          class="primary-action"
          :disabled="mappingConfirming"
          @click="confirmMapping"
        >
          <LoaderCircle v-if="mappingConfirming" class="spinning" :size="15" aria-hidden="true" />
          <ShieldCheck v-else :size="15" aria-hidden="true" />
          {{ mappingConfirming ? "正在执行质量探针…" : "确认并运行关联质量门禁" }}
        </button>
        <p v-if="mappingMessage" class="mapping-message" :class="{ error: mappingBlocked }" role="status">
          {{ mappingMessage }}
        </p>
      </section>

      <footer class="drawer-footer">
        <button type="button" class="secondary-action" @click="resetContext">重置为正式 DW</button>
        <button
          type="button"
          class="primary-action"
          :disabled="store.state.saving || !store.state.dirty"
          @click="applyContext"
        >
          <LoaderCircle v-if="store.state.saving" class="spinning" :size="15" aria-hidden="true" />
          <Check v-else :size="15" aria-hidden="true" />
          {{ store.state.dirty ? "应用数据上下文" : "已应用" }}
        </button>
      </footer>
    </aside>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";
import {
  ArrowRight,
  Check,
  CheckCircle2,
  Columns3,
  Database,
  FileSearch,
  FileSpreadsheet,
  GitMerge,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Table2,
  TriangleAlert,
  X,
} from "@lucide/vue";
import { useDataContextStore } from "../stores/dataContext.js";

const props = defineProps({
  chatId: { type: String, default: "" },
});
const emit = defineEmits(["notice"]);
const store = useDataContextStore();
const mappingConfirming = ref(false);
const mappingMessage = ref("");
const mappingBlocked = ref(false);
const closeButton = ref(null);
let returnFocus = null;
const readyDatasets = computed(() => store.state.catalog.filter((item) => item.status === "ready"));

onMounted(() => store.loadCatalog());

watch(
  () => store.state.drawerOpen,
  async (open) => {
    if (!open) return;
    returnFocus = document.activeElement;
    await nextTick();
    closeButton.value?.focus();
  },
  { immediate: true },
);

async function close() {
  store.state.drawerOpen = false;
  await nextTick();
  if (returnFocus?.isConnected) returnFocus.focus();
  returnFocus = null;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function selectDw() {
  store.state.selectedIds = [];
  store.setAnalysisMode("dw_only");
}

function toggle(dataset) {
  try {
    store.toggleDataset(dataset);
  } catch (error) {
    emit("notice", { text: error?.message || "选择失败", tone: "error" });
  }
}

function setMode(mode) {
  try {
    store.setAnalysisMode(mode);
  } catch (error) {
    emit("notice", { text: error?.message || "模式切换失败", tone: "error" });
  }
}

async function applyContext() {
  try {
    await store.apply(props.chatId);
    emit("notice", { text: "数据上下文已冻结，后续每轮会保存独立快照。", tone: "success" });
    close();
  } catch (error) {
    emit("notice", { text: error?.message || "应用失败", tone: "error" });
  }
}

async function resetContext() {
  try {
    await store.reset(props.chatId);
    emit("notice", { text: "已重置为正式数据仓库。", tone: "success" });
    close();
  } catch (error) {
    emit("notice", { text: error?.message || "重置失败", tone: "error" });
  }
}

async function confirmMapping() {
  mappingConfirming.value = true;
  mappingMessage.value = "";
  mappingBlocked.value = false;
  try {
    const result = await store.confirmMapping(store.state.mappingProposal.mapping_id);
    if (result.status === "confirmed") {
      mappingMessage.value = "唯一性、覆盖率、扇出与粒度门禁均已通过，请应用上下文后重新提问。";
    } else {
      mappingBlocked.value = true;
      mappingMessage.value = "质量门禁未通过，系统已阻止 JOIN。请改用独立对比或修正上传键。";
    }
  } catch (error) {
    mappingBlocked.value = true;
    mappingMessage.value = error?.message || "映射确认失败";
  } finally {
    mappingConfirming.value = false;
  }
}
</script>

<style scoped>
.context-backdrop { position: fixed; inset: 0; z-index: 90; display: flex; justify-content: flex-end; background: rgba(15,23,42,.28); backdrop-filter: blur(3px); }
.context-drawer { width: min(460px, 100vw); height: 100%; overflow: auto; padding: 26px; color: var(--da-text); background: rgba(248,250,253,.98); border-left: 1px solid var(--da-border); box-shadow: -24px 0 60px rgba(15,23,42,.16); }
.drawer-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; padding-bottom: 20px; border-bottom: 1px solid var(--da-border-light); }
.drawer-head h2 { margin: 4px 0 6px; font-size: 22px; letter-spacing: -.03em; }
.drawer-head p { margin: 0; max-width: 350px; color: var(--da-text-muted); font-size: 12px; line-height: 1.6; }
.eyebrow { color: var(--da-accent); font-size: 10px; font-weight: 800; letter-spacing: .16em; }
.icon-button, .text-button { border: 0; color: var(--da-text-muted); background: transparent; cursor: pointer; }
.icon-button { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--da-border); border-radius: 10px; background: #fff; }
.context-alert { display: flex; gap: 10px; margin-top: 16px; padding: 12px; color: #9a3412; border: 1px solid #fed7aa; border-radius: 12px; background: #fff7ed; font-size: 12px; }
.context-alert span { display: grid; gap: 2px; }
.context-section { padding: 20px 0; border-bottom: 1px solid var(--da-border-light); }
.section-title, .catalog-label, .quota-line { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.section-title { margin-bottom: 10px; font-size: 12px; font-weight: 800; }
.section-title em, .catalog-label em { color: var(--da-text-muted); font-size: 10px; font-style: normal; font-weight: 500; }
.text-button { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; }
.source-list { display: grid; gap: 8px; max-height: 245px; overflow: auto; }
.source-card { width: 100%; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 10px; padding: 12px; text-align: left; color: var(--da-text); border: 1px solid var(--da-border-light); border-radius: 13px; background: rgba(255,255,255,.88); cursor: pointer; transition: .15s ease; }
.source-card:hover { border-color: rgba(52,120,246,.34); transform: translateY(-1px); }
.source-card.active { color: #174ea6; border-color: rgba(52,120,246,.48); background: linear-gradient(135deg, #eff6ff, #f8fbff); box-shadow: 0 0 0 2px rgba(52,120,246,.08); }
.source-card.official { margin-bottom: 14px; }
.source-card strong, .source-card small { display: block; }
.source-card strong { overflow: hidden; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.source-card small { margin-top: 3px; color: var(--da-text-muted); font-size: 10px; }
.source-icon { display: grid; place-items: center; width: 34px; height: 34px; color: #2563eb; border-radius: 10px; background: #dbeafe; }
.source-icon.upload { color: #0f766e; background: #ccfbf1; }
.catalog-label { margin: 3px 2px 8px; color: var(--da-text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
.drawer-empty { display: flex; align-items: center; justify-content: center; gap: 8px; min-height: 76px; padding: 14px; color: var(--da-text-muted); border: 1px dashed var(--da-border); border-radius: 12px; font-size: 11px; text-align: center; }
.drawer-empty.error { color: #b91c1c; border-color: #fecaca; background: #fff7f7; }
.quota-line { margin-top: 10px; color: var(--da-text-muted); font-size: 10px; }
.mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mode-grid button { display: flex; align-items: flex-start; gap: 8px; min-height: 72px; padding: 11px; text-align: left; color: var(--da-text); border: 1px solid var(--da-border-light); border-radius: 12px; background: #fff; cursor: pointer; }
.mode-grid button.active { color: #174ea6; border-color: rgba(52,120,246,.48); background: #eff6ff; }
.mode-grid strong, .mode-grid small { display: block; }
.mode-grid strong { font-size: 11px; }
.mode-grid small { margin-top: 4px; color: var(--da-text-muted); font-size: 9px; line-height: 1.4; }
.mapping-section { padding: 16px; margin-top: 16px; border: 1px solid #fde68a; border-radius: 14px; background: #fffbeb; }
.mapping-card { padding: 10px; margin-bottom: 8px; border: 1px solid #fef3c7; border-radius: 10px; background: rgba(255,255,255,.8); }
.mapping-card div { display: flex; align-items: center; gap: 7px; font-size: 11px; }
.mapping-card small, .mapping-card > span { display: block; margin-top: 5px; color: var(--da-text-muted); font-size: 9px; }
.drawer-footer { position: sticky; bottom: -26px; display: grid; grid-template-columns: 1fr 1.35fr; gap: 9px; padding: 16px 0 26px; background: linear-gradient(180deg, rgba(248,250,253,0), rgba(248,250,253,.98) 22%); }
.primary-action, .secondary-action { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 38px; padding: 0 13px; border-radius: 11px; cursor: pointer; font-size: 11px; font-weight: 700; }
.primary-action { color: #fff; border: 1px solid #3478f6; background: linear-gradient(135deg, #3478f6, #2563eb); box-shadow: 0 8px 18px rgba(37,99,235,.18); }
.primary-action:disabled { opacity: .5; cursor: default; box-shadow: none; }
.secondary-action { color: var(--da-text); border: 1px solid var(--da-border); background: #fff; }
.mapping-section .primary-action { width: 100%; margin-top: 4px; }
.mapping-message { margin: 9px 0 0; color: #047857; font-size: 10px; line-height: 1.5; }
.mapping-message.error { color: #b91c1c; }
.spinning { animation: context-spin .8s linear infinite; }
@keyframes context-spin { to { transform: rotate(360deg); } }
@media (max-width: 640px) {
  .context-backdrop { align-items: flex-end; }
  .context-drawer { width: 100%; height: min(86vh, 760px); padding: 20px 16px; border-top: 1px solid var(--da-border); border-left: 0; border-radius: 22px 22px 0 0; }
  .mode-grid { grid-template-columns: 1fr; }
  .drawer-footer { bottom: -20px; padding-bottom: 20px; }
}
</style>
