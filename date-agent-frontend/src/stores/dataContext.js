import { computed, reactive } from "vue";
import { responseError } from "../utils/http.js";

const MODES = new Set([
  "dw_only",
  "upload_only",
  "upload_compare_dw",
  "upload_join_dw",
  "multi_upload",
]);

const state = reactive({
  identity: "",
  catalog: [],
  catalogLoading: false,
  catalogError: "",
  quota: null,
  selectedIds: [],
  sourceMode: "dw_only",
  comparisonIntent: null,
  mappingConfirmationId: null,
  mappingProposal: null,
  mappingEvidence: null,
  revision: 0,
  applied: {
    datasetIds: [],
    sourceMode: "dw_only",
    comparisonIntent: null,
    mappingConfirmationId: null,
    revision: 0,
  },
  dirty: false,
  saving: false,
  uploadState: "idle",
  uploadProgressLabel: "",
  drawerOpen: false,
  lastResolved: null,
  invalidation: null,
});

function snapshotDraft() {
  return {
    datasetIds: [...state.selectedIds],
    sourceMode: state.sourceMode,
    comparisonIntent: state.comparisonIntent,
    mappingConfirmationId: state.mappingConfirmationId,
    revision: state.revision,
  };
}

function inferMode() {
  if (!state.selectedIds.length) {
    state.sourceMode = "dw_only";
    state.comparisonIntent = null;
    state.mappingConfirmationId = null;
  } else if (state.selectedIds.length > 1) {
    state.sourceMode = "multi_upload";
    state.comparisonIntent = "multi_upload";
  } else if (state.sourceMode === "dw_only" || state.sourceMode === "multi_upload") {
    state.sourceMode = "upload_only";
    state.comparisonIntent = "upload_only";
  }
}

function markDirty() {
  inferMode();
  state.dirty = JSON.stringify(snapshotDraft()) !== JSON.stringify(state.applied);
}

function fallbackToDwOnly() {
  const revision = state.revision;
  resetLocal();
  state.revision = revision;
  state.applied.revision = revision;
  state.invalidation = {
    reason_code: "dataset_catalog_changed",
    message: "已应用的上传版本不再可用，数据上下文已安全回退到正式数据仓库。",
  };
}

function reconcileCatalogReferences() {
  const active = new Set(
    state.catalog
      .filter((item) => item.status === "ready")
      .map((item) => item.dataset_id),
  );
  if (state.applied.datasetIds.some((id) => !active.has(id))) {
    fallbackToDwOnly();
    return;
  }
  if (state.selectedIds.some((id) => !active.has(id))) {
    state.selectedIds = state.selectedIds.filter((id) => active.has(id));
    state.mappingConfirmationId = null;
    state.invalidation = {
      reason_code: "dataset_catalog_changed",
      message: "已选择的数据集版本发生变化，请重新应用上下文。",
    };
    markDirty();
  }
}

function setIdentity(identity) {
  const value = String(identity || "");
  if (state.identity === value) return;
  state.identity = value;
  resetLocal({ keepCatalog: false });
}

function resetLocal({ keepCatalog = true } = {}) {
  if (!keepCatalog) {
    state.catalog = [];
    state.quota = null;
    state.catalogError = "";
  }
  state.selectedIds = [];
  state.sourceMode = "dw_only";
  state.comparisonIntent = null;
  state.mappingConfirmationId = null;
  state.mappingProposal = null;
  state.mappingEvidence = null;
  state.revision = 0;
  state.applied = {
    datasetIds: [],
    sourceMode: "dw_only",
    comparisonIntent: null,
    mappingConfirmationId: null,
    revision: 0,
  };
  state.dirty = false;
  state.lastResolved = null;
  state.invalidation = null;
}

async function loadCatalog() {
  if (state.catalogLoading) return;
  state.catalogLoading = true;
  state.catalogError = "";
  try {
    const [catalogResponse, quotaResponse] = await Promise.all([
      fetch("/api/uploaded-datasets?limit=50"),
      fetch("/api/uploaded-datasets/quota"),
    ]);
    if (!catalogResponse.ok) {
      throw new Error(await responseError(catalogResponse, "上传数据目录加载失败"));
    }
    const catalog = await catalogResponse.json();
    state.catalog = Array.isArray(catalog.items) ? catalog.items : [];
    if (quotaResponse.ok) state.quota = await quotaResponse.json();
    reconcileCatalogReferences();
  } catch (error) {
    state.catalogError = error?.message || "上传数据目录加载失败";
  } finally {
    state.catalogLoading = false;
  }
}

function selectOnly(dataset) {
  if (!dataset?.dataset_id) return;
  if (!state.catalog.some((item) => item.dataset_id === dataset.dataset_id)) {
    state.catalog = [dataset, ...state.catalog];
  }
  state.selectedIds = [dataset.dataset_id];
  state.sourceMode = "upload_only";
  state.comparisonIntent = "upload_only";
  state.mappingConfirmationId = null;
  state.mappingProposal = null;
  markDirty();
}

function toggleDataset(dataset) {
  const id = dataset?.dataset_id;
  if (!id) return;
  if (!state.catalog.some((item) => item.dataset_id === id)) {
    state.catalog = [dataset, ...state.catalog];
  }
  if (state.selectedIds.includes(id)) {
    state.selectedIds = state.selectedIds.filter((value) => value !== id);
  } else if (state.selectedIds.length < 2) {
    state.selectedIds = [...state.selectedIds, id];
  } else {
    throw new Error("当前受控关联最多选择两个上传数据集");
  }
  state.mappingConfirmationId = null;
  state.mappingProposal = null;
  markDirty();
}

function setAnalysisMode(mode, intent = null) {
  if (!MODES.has(mode)) throw new Error("不支持的数据源模式");
  if (mode !== "dw_only" && !state.selectedIds.length) {
    throw new Error("请先选择上传数据集");
  }
  if (mode === "multi_upload" && state.selectedIds.length !== 2) {
    throw new Error("多上传表模式需要选择两个数据集");
  }
  state.sourceMode = mode;
  state.comparisonIntent = intent || ({
    dw_only: null,
    upload_only: "upload_only",
    upload_compare_dw: "independent_aggregate_compare",
    upload_join_dw: "upload_dw_row_join",
    multi_upload: "multi_upload",
  })[mode];
  if (!["upload_join_dw", "multi_upload"].includes(mode)) {
    state.mappingConfirmationId = null;
    state.mappingProposal = null;
  }
  markDirty();
}

function applyLocal() {
  state.applied = { ...snapshotDraft(), revision: state.revision };
  state.dirty = false;
  return state.applied;
}

async function apply(chatId) {
  if (!chatId) return applyLocal();
  state.saving = true;
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}/data-context`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_mode: state.sourceMode,
        dataset_ids: state.selectedIds,
        comparison_intent: state.comparisonIntent,
        mapping_confirmation_id: state.mappingConfirmationId,
        expected_revision: state.revision,
      }),
    });
    if (!response.ok) throw new Error(await responseError(response, "数据上下文应用失败"));
    const data = await response.json();
    state.revision = Number(data.revision || 1);
    state.applied = { ...snapshotDraft(), revision: state.revision };
    state.dirty = false;
    state.lastResolved = data.resolved_context || null;
    return data;
  } finally {
    state.saving = false;
  }
}

async function hydrate(chatId) {
  if (!chatId) {
    resetLocal();
    return;
  }
  await loadCatalog();
  const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}/data-context`);
  if (!response.ok) throw new Error(await responseError(response, "会话数据上下文恢复失败"));
  const data = await response.json();
  state.selectedIds = Array.isArray(data.dataset_ids) ? data.dataset_ids : [];
  state.sourceMode = data.source_mode || "dw_only";
  state.comparisonIntent = data.comparison_intent || null;
  state.mappingConfirmationId = data.mapping_confirmation_id || null;
  state.revision = Number(data.revision || 0);
  state.applied = { ...snapshotDraft(), revision: state.revision };
  state.dirty = false;
  state.invalidation = data.invalidation_reason
    ? { reason_code: "server_invalidation", message: data.invalidation_reason }
    : null;
  reconcileCatalogReferences();
}

async function reset(chatId) {
  if (chatId) {
    const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}/data-context/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_revision: state.revision }),
    });
    if (!response.ok) throw new Error(await responseError(response, "上下文重置失败"));
    const data = await response.json();
    resetLocal();
    state.revision = Number(data.revision || 1);
    state.applied.revision = state.revision;
    return data;
  }
  resetLocal();
}

async function upload(file, retentionMode = "managed") {
  state.uploadState = "uploading";
  state.uploadProgressLabel = "正在解析并物化数据…";
  const body = new FormData();
  body.append("file", file);
  const idempotencyKey = crypto.randomUUID();
  try {
    const response = await fetch(`/api/uploaded-datasets?retention_mode=${encodeURIComponent(retentionMode)}`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body,
    });
    if (!response.ok) throw new Error(await responseError(response, "上传失败"));
    const dataset = await response.json();
    state.catalog = [dataset, ...state.catalog.filter((item) => item.dataset_id !== dataset.dataset_id)];
    selectOnly(dataset);
    applyLocal();
    state.uploadState = "ready";
    state.uploadProgressLabel = "上传完成";
    await loadCatalog();
    return dataset;
  } catch (error) {
    state.uploadState = "failed";
    state.uploadProgressLabel = error?.message || "上传失败";
    throw error;
  }
}

async function confirmMapping(mappingId) {
  const response = await fetch(`/api/source-mappings/${encodeURIComponent(mappingId)}/confirm`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await responseError(response, "关联映射确认失败"));
  const data = await response.json();
  state.mappingEvidence = data;
  if (data.status === "confirmed") {
    state.mappingConfirmationId = data.mapping_id;
    state.mappingProposal = null;
    markDirty();
  }
  return data;
}

function handleStreamEvent(data) {
  if (data.source_context_resolved) {
    state.lastResolved = data.source_context_resolved;
  }
  if (data.source_mapping?.status === "proposed") {
    state.mappingProposal = data.source_mapping;
    state.drawerOpen = true;
  } else if (data.source_mapping) {
    state.mappingEvidence = data.source_mapping;
  }
  if (data.source_context_invalidated) {
    state.invalidation = data.source_context_invalidated;
    state.selectedIds = [];
    state.sourceMode = "dw_only";
    state.comparisonIntent = null;
    state.mappingConfirmationId = null;
    applyLocal();
  }
}

function queryPayload() {
  const context = state.applied;
  // DW-only 不再作为“显式上下文”发送。这样会话中持久化的上传版本若在
  // 另一页面/设备失效，服务端会走默认上下文校验、原子重置并回退 DW。
  if (context.sourceMode === "dw_only") return {};
  return {
    dataset_ids: [...context.datasetIds],
    source_mode: context.sourceMode,
    comparison_intent: context.comparisonIntent,
    mapping_confirmation_id: context.mappingConfirmationId,
    data_context_revision: context.revision,
  };
}

const selectedDatasets = computed(() => {
  const selected = new Set(state.selectedIds);
  return state.catalog.filter((item) => selected.has(item.dataset_id));
});
const appliedDatasets = computed(() => {
  const selected = new Set(state.applied.datasetIds);
  return state.catalog.filter((item) => selected.has(item.dataset_id));
});
const contextLabel = computed(() => {
  if (state.applied.sourceMode === "dw_only") return "正式数据仓库";
  const names = appliedDatasets.value.map((item) => item.filename).join(" + ");
  return names || "上传数据上下文";
});

export function useDataContextStore() {
  return {
    state,
    selectedDatasets,
    appliedDatasets,
    contextLabel,
    setIdentity,
    resetLocal,
    loadCatalog,
    selectOnly,
    toggleDataset,
    setAnalysisMode,
    applyLocal,
    apply,
    hydrate,
    reset,
    upload,
    confirmMapping,
    handleStreamEvent,
    queryPayload,
  };
}
