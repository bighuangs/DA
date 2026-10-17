<template>
  <section class="adhoc-page">
    <div class="adhoc-card">
      <div class="adhoc-head">
        <div>
          <div class="eyebrow">AD-HOC DATA</div>
          <h1>上传数据资产</h1>
          <p>CSV / XLSX 会形成受当前用户与工作空间保护的不可变版本；物理表名不会暴露给浏览器。</p>
        </div>
        <label class="upload-btn" :class="{ disabled: uploading }">
          <LoaderCircle v-if="uploading" :size="15" class="spin" aria-hidden="true" />
          <Upload v-else :size="15" aria-hidden="true" />
          {{ uploading ? "导入中…" : "上传文件" }}
          <input type="file" aria-label="上传临时数据文件" accept=".csv,.xlsx" :disabled="uploading" @change="uploadFile" />
        </label>
      </div>
      <div v-if="error" class="error-box" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
      <div v-else-if="notice" class="notice-box" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
      <div v-if="store.state.quota" class="quota-board">
        <span><strong>{{ store.state.quota.usage.active_versions }}</strong> 活跃版本</span>
        <span><strong>{{ store.state.quota.usage.rows.toLocaleString() }}</strong> 总行数</span>
        <span><strong>{{ Math.round(store.state.quota.usage.bytes / 1024) }}</strong> KB 已用</span>
        <em>同名文件可创建新版本；删除失败会进入可重试清理状态</em>
      </div>
      <div v-if="!datasets.length && !error" class="empty">
        <span class="empty-icon"><FileSpreadsheet :size="24" aria-hidden="true" /></span>
        <strong>从一个业务文件开始</strong>
        <p>支持 CSV 与 XLSX。系统会执行类型推断、Schema 剖析、配额检查与补偿式物化，随后可在对话侧选择数据上下文。</p>
        <label class="empty-upload">
          <Upload :size="15" aria-hidden="true" />选择文件
          <input type="file" aria-label="选择第一个临时数据文件" accept=".csv,.xlsx" :disabled="uploading" @change="uploadFile" />
        </label>
      </div>
      <div v-for="dataset in datasets" :key="dataset.dataset_id" class="dataset-item" :class="{ selected: selected?.dataset_id === dataset.dataset_id, archived: dataset.status === 'archived' }">
        <div class="dataset-title">
          <div>
            <div class="dataset-name-line">
              <strong>{{ dataset.filename }}</strong>
              <span class="version-badge">v{{ dataset.version_no }}</span>
              <span class="status-badge" :class="dataset.status">{{ dataset.status }}</span>
            </div>
            <span class="dataset-meta">{{ dataset.row_count }} 行 · {{ dataset.column_count }} 列 · {{ dataset.format.toUpperCase() }} · {{ dataset.retention_mode === "managed" ? "受管版本" : "临时版本" }}</span>
          </div>
          <div class="dataset-actions">
            <button v-if="dataset.status === 'ready'" class="mini-btn primary" @click="useDataset(dataset)"><MessageSquareText :size="13" aria-hidden="true" />用于问数</button>
            <button v-if="dataset.status === 'ready'" class="mini-btn" title="续期 24 小时" @click="extendDataset(dataset)"><Clock3 :size="13" aria-hidden="true" />续期</button>
            <button v-if="dataset.status === 'ready'" class="mini-btn" @click="archiveDataset(dataset)"><Archive :size="13" aria-hidden="true" />归档</button>
            <button class="mini-btn danger" @click="requestDelete(dataset, $event.currentTarget)"><Trash2 :size="13" aria-hidden="true" />删除</button>
          </div>
        </div>
        <div class="schema-list">
          <span v-for="column in dataset.columns" :key="column.name" class="schema-chip">{{ column.display_name }} <small>{{ column.type }}</small></span>
        </div>
        <div class="preview-wrap">
          <table v-if="dataset.preview?.length" class="preview-table">
            <thead><tr><th v-for="column in dataset.columns" :key="column.name">{{ column.display_name }}</th></tr></thead>
            <tbody><tr v-for="(row, rowIndex) in dataset.preview" :key="rowIndex"><td v-for="column in dataset.columns" :key="column.name">{{ row[column.display_name] ?? "—" }}</td></tr></tbody>
          </table>
          <div v-else class="empty-preview">文件只有表头，没有数据行。</div>
        </div>
        <div class="expires">
          <span>有效期至 {{ dataset.expires_at }}</span>
          <span>Schema {{ String(dataset.schema_fingerprint || "—").replace("sha256:", "").slice(0, 12) }}</span>
          <span>版本 {{ String(dataset.version_id || "").slice(0, 8) }}</span>
        </div>
      </div>
    </div>
    <ConfirmDatasetDeleteDialog
      v-if="deleteTarget"
      :dataset="deleteTarget"
      :busy="deleteBusy"
      @cancel="cancelDelete"
      @confirm="confirmDelete"
    />
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import {
  Archive,
  CircleAlert,
  CircleCheck,
  Clock3,
  FileSpreadsheet,
  LoaderCircle,
  MessageSquareText,
  Trash2,
  Upload,
} from "@lucide/vue";
import { requestJson } from "../utils/http.js";
import { useDataContextStore } from "../stores/dataContext.js";
import ConfirmDatasetDeleteDialog from "./ConfirmDatasetDeleteDialog.vue";

const emit = defineEmits(["use-dataset"]);
const store = useDataContextStore();
const datasets = computed(() => store.state.catalog);
const selected = computed(() => (
  datasets.value.find((item) => store.state.selectedIds.includes(item.dataset_id)) || null
));
const uploading = ref(false);
const error = ref("");
const notice = ref("");
const deleteTarget = ref(null);
const deleteBusy = ref(false);
let deleteTrigger = null;
let noticeTimer = null;

function showNotice(message) {
  if (noticeTimer) window.clearTimeout(noticeTimer);
  notice.value = message;
  noticeTimer = window.setTimeout(() => {
    notice.value = "";
    noticeTimer = null;
  }, 3500);
}

async function loadDatasets() {
  error.value = "";
  try {
    await store.loadCatalog();
    if (store.state.catalogError) throw new Error(store.state.catalogError);
  } catch (err) { error.value = err?.message || "临时数据集加载失败"; }
}

async function uploadFile(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  uploading.value = true;
  error.value = "";
  try {
    const data = await store.upload(file, "managed");
    showNotice(`“${data.filename || file.name}”v${data.version_no} 已就绪，可直接用于问数`);
  } catch (err) { error.value = err?.message || "文件导入失败"; }
  finally { uploading.value = false; }
}

function useDataset(dataset) {
  showNotice(`已选择“${dataset.filename}”，正在返回对话`);
  emit("use-dataset", dataset);
}

function requestDelete(dataset, trigger) {
  deleteTarget.value = dataset;
  deleteTrigger = trigger || null;
}

async function cancelDelete() {
  if (deleteBusy.value) return;
  deleteTarget.value = null;
  await nextTick();
  deleteTrigger?.focus();
  deleteTrigger = null;
}

async function confirmDelete() {
  const dataset = deleteTarget.value;
  if (!dataset || deleteBusy.value) return;
  deleteBusy.value = true;
  error.value = "";
  try {
    await requestJson(`/api/uploaded-datasets/${dataset.dataset_id}`, { method: "DELETE" }, "删除失败");
    await store.loadCatalog();
    showNotice(`“${dataset.filename}”已删除`);
  } catch (err) { error.value = err?.message || "删除失败"; }
  finally {
    deleteBusy.value = false;
    deleteTarget.value = null;
    deleteTrigger = null;
  }
}

async function archiveDataset(dataset) {
  try {
    await requestJson(`/api/uploaded-datasets/${dataset.dataset_id}/archive`, { method: "POST" }, "归档失败");
    await store.loadCatalog();
    showNotice(`“${dataset.filename}”已归档`);
  } catch (err) { error.value = err?.message || "归档失败"; }
}

async function extendDataset(dataset) {
  try {
    await requestJson(`/api/uploaded-datasets/${dataset.dataset_id}/extend`, {
      method: "POST",
      body: JSON.stringify({ seconds: 86400 }),
    }, "续期失败");
    await store.loadCatalog();
    showNotice(`“${dataset.filename}”已续期 24 小时`);
  } catch (err) { error.value = err?.message || "续期失败"; }
}

onMounted(loadDatasets);
</script>

<style scoped>
.adhoc-page { flex: 1 1 auto; overflow: auto; padding: 32px 20px 60px; background: var(--da-bg); }
.adhoc-card { max-width: 980px; margin: 0 auto; }
.adhoc-head { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 24px; }
.eyebrow { color: var(--da-accent); font-size: 11px; letter-spacing: 2px; font-weight: 700; }
h1 { margin: 8px 0; color: var(--da-text); font-size: 26px; }
p { margin: 0; color: var(--da-text-secondary); font-size: 13px; }
.upload-btn { padding: 9px 16px; border-radius: 9px; color: #fff; background: var(--da-accent); cursor: pointer; font-size: 13px; white-space: nowrap; }
.upload-btn.disabled { opacity: .6; cursor: wait; }
.upload-btn input { display: none; }
.error-box, .notice-box { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; padding: 12px 14px; border-radius: 9px; font-size: 13px; }
.error-box { color: var(--da-danger); background: rgba(231, 76, 60, .08); border: 1px solid rgba(231, 76, 60, .25); }
.notice-box { color: #287b4a; background: #f0fbf4; border: 1px solid #b9e2c8; }
.quota-board { display: grid; grid-template-columns: repeat(3, auto) 1fr; align-items: center; gap: 18px; margin-bottom: 18px; padding: 14px 16px; color: var(--da-text-muted); border: 1px solid var(--da-border-light); border-radius: 13px; background: rgba(255,255,255,.8); font-size: 10px; }
.quota-board span { display: grid; gap: 2px; }
.quota-board strong { color: var(--da-text); font-size: 16px; }
.quota-board em { justify-self: end; max-width: 360px; font-style: normal; line-height: 1.5; text-align: right; }
.empty, .empty-preview { padding: 26px; text-align: center; color: var(--da-text-muted); font-size: 13px; }
.dataset-item { margin-bottom: 18px; padding: 16px; border: 1px solid var(--da-border); border-radius: 12px; background: var(--da-bg); }
.dataset-item.selected { border-color: var(--da-accent); box-shadow: 0 0 0 2px var(--da-accent-soft); }
.dataset-item.archived { opacity: .72; }
.dataset-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.dataset-title strong { display: block; color: var(--da-text); font-size: 14px; }
.dataset-name-line { display: flex; align-items: center; gap: 7px; }
.version-badge, .status-badge { padding: 2px 6px; border-radius: 999px; color: #1d4ed8; background: #dbeafe; font-size: 9px; font-weight: 700; }
.status-badge { color: #047857; background: #d1fae5; text-transform: uppercase; }
.status-badge.archived { color: #64748b; background: #e2e8f0; }
.status-badge.cleanup_failed { color: #b91c1c; background: #fee2e2; }
.dataset-meta, .expires { color: var(--da-text-muted); font-size: 11px; }
.dataset-meta { display: block; margin-top: 5px; }
.dataset-actions { display: flex; gap: 7px; }
.mini-btn { display: inline-flex; align-items: center; gap: 5px; padding: 5px 10px; border: 1px solid var(--da-border); border-radius: 7px; background: var(--da-bg); color: var(--da-text-secondary); cursor: pointer; font-size: 12px; }
.mini-btn.primary { color: #fff; background: var(--da-accent); border-color: var(--da-accent); }
.mini-btn.danger { color: var(--da-danger); }
.schema-list { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0; }
.schema-chip { padding: 4px 8px; border-radius: 5px; color: var(--da-text-secondary); background: var(--da-bg-soft); font-size: 11px; }
.schema-chip small { margin-left: 4px; color: var(--da-accent); }
.preview-wrap { overflow-x: auto; border: 1px solid var(--da-border-light); border-radius: 8px; }
.preview-table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px; }
.preview-table th, .preview-table td { padding: 8px 10px; border-bottom: 1px solid var(--da-border-light); white-space: nowrap; text-align: left; color: var(--da-text); }
.preview-table th { background: var(--da-bg-soft); color: var(--da-text-secondary); }
.expires { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; margin-top: 10px; }
@media (max-width: 640px) { .adhoc-head, .dataset-title { flex-direction: column; align-items: stretch; } .dataset-actions { justify-content: flex-end; } }
.adhoc-page { padding: 38px 24px 64px; background: transparent; }
.adhoc-card { max-width: 1120px; }
.adhoc-head { margin-bottom: 26px; }
h1 { margin: 8px 0 6px; font-size: 29px; letter-spacing: -.04em; }
.eyebrow { font-size: 10px; letter-spacing: .17em; }
.upload-btn { min-height: 38px; display: inline-flex; align-items: center; gap: 7px; padding: 9px 15px; border-radius: 10px; font-weight: 650; box-shadow: 0 7px 16px rgba(52,120,246,.18); transition: .15s; }
.upload-btn:hover:not(.disabled) { background: var(--da-accent-hover); transform: translateY(-1px); }
.empty { min-height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 9px; border: 1px dashed var(--da-border); border-radius: 16px; background: rgba(255,255,255,.65); }
.empty strong { color: var(--da-text-secondary); font-size: 15px; }
.empty p { max-width: 500px; line-height: 1.7; }
.empty-icon { width: 48px; height: 48px; display: grid; place-items: center; border-radius: 14px; background: var(--da-accent-soft); color: var(--da-accent); }
.empty-upload { min-height: 36px; display: inline-flex; align-items: center; gap: 6px; margin-top: 5px; padding: 8px 13px; border-radius: 9px; background: var(--da-accent); color: #fff; cursor: pointer; font-weight: 650; }
.empty-upload input { display: none; }
.dataset-item { margin-bottom: 14px; padding: 20px; border-color: var(--da-border-light); border-radius: 16px; background: rgba(255,255,255,.96); box-shadow: var(--da-shadow-sm); transition: .15s; }
.dataset-item:hover { box-shadow: var(--da-shadow-md); transform: translateY(-1px); }
.dataset-item.selected { border-color: rgba(52,120,246,.45); box-shadow: 0 0 0 3px var(--da-accent-soft), var(--da-shadow-md); }
.dataset-title strong { font-size: 15px; }
.mini-btn { min-height: 32px; border-radius: 8px; transition: .15s; }
.mini-btn:hover { border-color: var(--da-accent); color: var(--da-accent); background: var(--da-accent-soft); }
.schema-chip { padding: 5px 9px; border-radius: 7px; }
.preview-wrap { border-radius: 10px; }
.preview-table th { position: sticky; top: 0; }
.preview-table tbody tr:nth-child(even) { background: rgba(244,247,251,.6); }
@media (max-width: 640px) { .adhoc-page { padding: 25px 15px 44px; } .dataset-item { padding: 16px; } }
@media (max-width: 760px) { .quota-board { grid-template-columns: repeat(3, 1fr); } .quota-board em { grid-column: 1 / -1; justify-self: start; text-align: left; } }
.spin { animation: adhoc-spin .8s linear infinite; }
@keyframes adhoc-spin { to { transform: rotate(360deg); } }
</style>
