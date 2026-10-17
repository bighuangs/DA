<template>
  <div class="dash-view">
    <div v-if="error" class="dash-error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-else-if="notice" class="dash-notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <!-- 左：资源树 -->
    <aside class="dash-tree">
      <div class="tree-head">
        <span class="tree-title"><LayoutDashboard :size="17" aria-hidden="true" />仪表盘</span>
        <div class="tree-actions">
          <button v-if="canEdit" class="mini-btn icon" aria-label="新建文件夹" title="在当前目录新建文件夹" @click="createNode('folder', selectedFolderId || null)"><FolderPlus :size="16" aria-hidden="true" /></button>
          <button v-if="canEdit" class="mini-btn icon" aria-label="新建仪表盘" title="在当前目录新建仪表盘" @click="createNode('dashboard', selectedFolderId || null)"><LayoutDashboard :size="16" aria-hidden="true" /></button>
          <button class="mini-btn icon" aria-label="刷新仪表盘列表" title="刷新仪表盘列表" @click="loadTree"><RefreshCw :size="16" aria-hidden="true" /></button>
        </div>
      </div>
      <div class="tree-body" role="tree" aria-label="仪表盘资源树">
        <div v-if="!tree.length" class="tree-empty">
          <FolderOpen :size="24" aria-hidden="true" />
          <strong>还没有仪表盘</strong>
          <span v-if="canEdit">先创建一个仪表盘，再从问数结果中加入图表。</span>
          <span v-else>当前工作空间尚未发布可查看的仪表盘。</span>
          <button v-if="canEdit" class="mini-btn primary" @click="createNode('dashboard', null)">新建仪表盘</button>
        </div>
        <button v-if="tree.length" type="button" role="treeitem" aria-level="1" class="tree-row tree-root"
                 :class="{ active: activeNodeId === '' }" @click="selectRootFolder">
          <span class="tree-toggle-spacer" aria-hidden="true"></span>
          <FolderOpen class="tree-icon" :size="15" aria-hidden="true" />
          <span class="tree-name">根目录</span>
        </button>
        <div v-for="row in visibleTreeRows" :key="row.node.id" class="tree-row"
             :class="{ active: row.node.id === activeNodeId }"
             :data-depth="row.depth"
             role="treeitem"
             :aria-level="row.depth + 1"
             :aria-expanded="row.node.node_type === 'folder' && row.hasChildren ? isFolderExpanded(row.node.id) : undefined"
             tabindex="0"
             :style="{ paddingLeft: `${10 + row.depth * 18}px` }"
             @click="selectNode(row.node)"
             @keydown.enter.self.prevent="selectNode(row.node)"
             @keydown.space.self.prevent="selectNode(row.node)">
          <button
            v-if="row.node.node_type === 'folder' && row.hasChildren"
            type="button"
            class="tree-toggle"
            :aria-label="`${isFolderExpanded(row.node.id) ? '收起' : '展开'} ${row.node.name}`"
            @click.stop="toggleFolder(row.node.id)"
          >
            <ChevronDown v-if="isFolderExpanded(row.node.id)" :size="14" aria-hidden="true" />
            <ChevronRight v-else :size="14" aria-hidden="true" />
          </button>
          <span v-else class="tree-toggle-spacer" aria-hidden="true"></span>
          <component :is="row.node.node_type === 'folder' && isFolderExpanded(row.node.id) ? FolderOpen : row.node.node_type === 'folder' ? Folder : LayoutDashboard" class="tree-icon" :size="15" aria-hidden="true" />
          <span class="tree-name">{{ row.node.name }}</span>
          <span v-if="canEdit" class="tree-ops">
            <button class="mini-btn x" :aria-label="`重命名 ${row.node.name}`" title="重命名" @click.stop="renameNode(row.node)"><Pencil :size="14" aria-hidden="true" /></button>
            <button class="mini-btn x danger" :aria-label="`删除 ${row.node.name}`" title="删除" @click.stop="deleteNode(row.node)"><Trash2 :size="14" aria-hidden="true" /></button>
          </span>
        </div>
      </div>
    </aside>

    <!-- 右：画布 -->
    <section class="dash-canvas">
      <div class="canvas-toolbar" v-if="selected">
        <span class="canvas-title">{{ selected.name }}</span>
        <div class="canvas-actions">
          <button v-if="canEdit" class="mini-btn with-icon" @click="togglePreview">
            <Pencil v-if="preview" :size="14" aria-hidden="true" /><Eye v-else :size="14" aria-hidden="true" />{{ preview ? '编辑布局' : '预览' }}
          </button>
          <button class="mini-btn with-icon" @click="refreshData"><RefreshCw :size="14" aria-hidden="true" />刷新数据</button>
          <button v-if="canEdit" class="mini-btn primary" :disabled="saving || preview" @click="save">
            <Save :size="14" aria-hidden="true" />{{ saving ? '保存中…' : '保存布局' }}
          </button>
        </div>
      </div>

      <div class="canvas-empty" v-if="!selected">
        <div class="empty-logo"><LayoutDashboard :size="28" aria-hidden="true" /></div>
        <div class="empty-tip">{{ tree.length ? "选择一个仪表盘开始查看" : "创建你的第一个业务看板" }}</div>
        <div class="empty-sub">{{ tree.length ? "左侧目录保存了当前工作空间的全部看板。" : "看板用于集中查看问数结果，支持刷新和布局调整。" }}</div>
        <button v-if="canEdit && !tree.length" class="empty-cta" @click="createNode('dashboard', selectedFolderId || null)"><Plus :size="15" aria-hidden="true" />新建仪表盘</button>
      </div>

      <div class="canvas-empty" v-else-if="!blocks.length">
        <div class="empty-logo"><PanelTopDashed :size="28" aria-hidden="true" /></div>
        <div class="empty-tip">画布为空</div>
        <div class="empty-sub">完成一次问数后，在结果工作台选择“加入仪表盘”。</div>
        <button class="empty-cta" @click="emit('open-chat')"><MessageSquare :size="15" aria-hidden="true" />前往对话问数</button>
      </div>

      <!-- 画布：CSS Grid 12 列 + 原生指针拖拽/缩放 -->
      <div v-else ref="canvasEl" class="canvas-grid">
        <div
          v-for="(b, i) in blocks"
          :key="i"
          class="block-card"
          :class="{ dragging: drag.idx === i, preview }"
          :style="blockStyle(b)"
        >
          <div
            class="block-head"
            :class="{ draggable: canEdit && !preview }"
            @pointerdown="onDragStart($event, i)"
          >
            <span class="block-title">{{ b.chart?.title || '图表' }}</span>
            <span class="block-badge">{{ chartTypeLabel(b.chart?.type) }}</span>
          </div>
          <div class="block-body">
            <div v-if="b.error" class="block-error" role="alert"><CircleAlert :size="15" aria-hidden="true" />{{ b.error }}</div>
            <ChartView v-else :chart="b.chart" :rows="b.rows || []" />
          </div>
          <div
            v-if="canEdit && !preview"
            class="resize-handle"
            @pointerdown.stop="onResizeStart($event, i)"
          ></div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from "vue";
import {
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Eye,
  Folder,
  FolderOpen,
  FolderPlus,
  LayoutDashboard,
  MessageSquare,
  PanelTopDashed,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Trash2,
} from "@lucide/vue";
import ChartView from "./ChartView.vue";
import { responseError } from "../utils/http.js";
import { confirmDialog, promptDialog } from "../composables/appDialog.js";

const props = defineProps({
  // 父组件可指定要加载的仪表盘ID（来自"加入仪表盘"后的跳转）
  selectId: { type: String, default: "" },
  canEdit: { type: Boolean, default: false },
});
const emit = defineEmits(["open-chat"]);

const COLS = 12;
const ROW_H = 40;   // 与 CSS grid-auto-rows 保持一致
const MIN_W = 2;
// 9 行在 40px 行高、10px gap 下为 440px，可完整容纳 340px ChartView、卡头和内边距。
const MIN_H = 9;
const DEFAULT_W = 6;
const DEFAULT_H = 9;

const API = "/api/dashboard";
const tree = ref([]);
const activeNodeId = ref("");
const selectedFolderId = ref("");
const expandedFolderIds = ref(new Set());
let knownFolderIds = new Set();
let dashboardLoadRevision = 0;
const selectedId = ref("");
const selected = ref(null);      // 当前仪表盘节点元信息
const blocks = ref([]);          // 画布块（含 rows/error）
const preview = ref(false);
const saving = ref(false);
const error = ref("");
const notice = ref("");
const canvasEl = ref(null);
let noticeTimer = null;

function showNotice(message) {
  if (noticeTimer) window.clearTimeout(noticeTimer);
  notice.value = message;
  noticeTimer = window.setTimeout(() => {
    notice.value = "";
    noticeTimer = null;
  }, 2500);
}

// 拖拽/缩放进行中的状态
const drag = ref({ idx: -1, startX: 0, startY: 0, origX: 0, origY: 0, mode: "" });

function gridInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.round(parsed) : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// 后端历史数据、拖拽中的临时值和保存载荷统一走同一套 12 列合同。
// 宽度先规范化，再据此收紧 x，保证任意时刻都满足 x + w <= 12。
function normalizeBlock(block) {
  const source = block && typeof block === "object" ? block : {};
  const w = clamp(gridInteger(source.w, DEFAULT_W), MIN_W, COLS);
  return {
    ...source,
    x: clamp(gridInteger(source.x, 0), 0, COLS - w),
    y: Math.max(0, gridInteger(source.y, 0)),
    w,
    h: Math.max(MIN_H, gridInteger(source.h, DEFAULT_H)),
  };
}

function normalizeBlocks(value) {
  return Array.isArray(value) ? value.map(normalizeBlock) : [];
}

function blockStyle(b) {
  const layout = normalizeBlock(b);
  return {
    gridColumn: `${layout.x + 1} / span ${layout.w}`,
    gridRow: `${layout.y + 1} / span ${layout.h}`,
  };
}

function chartTypeLabel(t) {
  return { column: "柱状图", bar: "条形图", line: "折线图", pie: "饼图", table: "表格" }[t] || "图表";
}

// 后端返回 pid 扁平列表；这里生成当前展开状态下的稳定可见行。
// 孤儿节点会成为根节点；只有无法抵达根节点的异常循环才走末尾兜底。
const visibleTreeRows = computed(() => {
  const nodes = tree.value;
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const children = new Map();
  const roots = [];
  for (const node of nodes) {
    const parent = byId.get(node.pid);
    if (parent?.node_type === "folder") {
      const siblings = children.get(parent.id) || [];
      siblings.push(node);
      children.set(parent.id, siblings);
    } else {
      roots.push(node);
    }
  }

  const rows = [];
  const visited = new Set();
  const reachesRoot = (node) => {
    const ancestors = new Set();
    let cursor = node;
    while (cursor && !ancestors.has(cursor.id)) {
      ancestors.add(cursor.id);
      const parent = byId.get(cursor.pid);
      if (parent?.node_type !== "folder") return true;
      cursor = parent;
    }
    return false;
  };
  const append = (node, depth) => {
    if (!node || visited.has(node.id)) return;
    visited.add(node.id);
    const descendants = children.get(node.id) || [];
    rows.push({ node, depth, hasChildren: descendants.length > 0 });
    if (node.node_type === "folder" && expandedFolderIds.value.has(node.id)) {
      for (const child of descendants) append(child, depth + 1);
    }
  };
  for (const root of roots) append(root, 0);
  // 不得把“因祖先折叠而不可见”的正常后代重新平铺到根；仅展示异常循环兜底。
  for (const node of nodes) {
    if (!visited.has(node.id) && !reachesRoot(node)) append(node, 0);
  }
  return rows;
});

function isFolderExpanded(id) {
  return expandedFolderIds.value.has(id);
}

function toggleFolder(id) {
  const next = new Set(expandedFolderIds.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  expandedFolderIds.value = next;
}

function expandFolder(id) {
  if (!id || expandedFolderIds.value.has(id)) return;
  expandedFolderIds.value = new Set([...expandedFolderIds.value, id]);
}

function clearDashboardSelection({ invalidatePending = true } = {}) {
  if (invalidatePending) dashboardLoadRevision += 1;
  selected.value = null;
  blocks.value = [];
  selectedId.value = "";
  preview.value = false;
}

function selectRootFolder() {
  activeNodeId.value = "";
  selectedFolderId.value = "";
  clearDashboardSelection();
}

function subtreeIds(rootId) {
  const ids = new Set([rootId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of tree.value) {
      if (!ids.has(node.id) && ids.has(node.pid)) {
        ids.add(node.id);
        changed = true;
      }
    }
  }
  return ids;
}

async function loadTree() {
  try {
    const res = await fetch(`${API}/tree`);
    if (!res.ok) throw new Error(await responseError(res, "仪表盘列表加载失败"));
    const body = await res.json();
    const nextTree = Array.isArray(body)
      ? body.filter((node) => node && typeof node === "object" && node.id != null)
      : [];
    tree.value = nextTree;

    const nextFolderIds = new Set(nextTree.filter((node) => node.node_type === "folder").map((node) => node.id));
    const nextExpanded = new Set([...expandedFolderIds.value].filter((id) => nextFolderIds.has(id)));
    for (const id of nextFolderIds) {
      if (!knownFolderIds.has(id)) nextExpanded.add(id);
    }
    knownFolderIds = nextFolderIds;
    expandedFolderIds.value = nextExpanded;

    if (selectedFolderId.value && !nextFolderIds.has(selectedFolderId.value)) selectedFolderId.value = "";
    const nodeIds = new Set(nextTree.map((node) => node.id));
    if (selectedId.value && !nodeIds.has(selectedId.value)) clearDashboardSelection();
    const selectedNode = nextTree.find((node) => node.id === selectedId.value);
    if (selectedNode?.node_type === "dashboard") {
      selectedFolderId.value = nextFolderIds.has(selectedNode.pid) ? selectedNode.pid : "";
    }
    if (activeNodeId.value && !nodeIds.has(activeNodeId.value)) activeNodeId.value = "";
    error.value = "";
  } catch (e) { error.value = e.message || "仪表盘列表加载失败"; }
}

async function selectNode(node) {
  activeNodeId.value = node.id;
  if (node.node_type === "folder") {
    selectedFolderId.value = node.id;
    expandFolder(node.id);
    clearDashboardSelection();
    return;
  }
  selectedFolderId.value = knownFolderIds.has(node.pid) ? node.pid : "";
  selectedId.value = node.id;
  await loadDashboard(node.id);
}

async function loadDashboard(id) {
  const revision = ++dashboardLoadRevision;
  try {
    const res = await fetch(`${API}/${id}`);
    if (revision !== dashboardLoadRevision) return;
    if (!res.ok) {
      const message = await responseError(res, "仪表盘加载失败");
      if (revision !== dashboardLoadRevision) return;
      error.value = message;
      clearDashboardSelection({ invalidatePending: false });
      return;
    }
    const data = await res.json();
    if (revision !== dashboardLoadRevision) return;
    selected.value = { id: data.id, name: data.name };
    selectedId.value = data.id;
    activeNodeId.value = data.id;
    blocks.value = normalizeBlocks(data.component_data);
    preview.value = false;
    error.value = "";
  } catch (e) {
    if (revision !== dashboardLoadRevision) return;
    error.value = e?.message || "仪表盘加载失败";
    clearDashboardSelection({ invalidatePending: false });
  }
}

// 相邻列起点的像素步长；扣除内边距并计入 gap，保持指针位移与 CSS Grid 对齐。
function colStep() {
  const canvas = canvasEl.value;
  if (!canvas) return 80;
  const style = window.getComputedStyle(canvas);
  const padding = (Number.parseFloat(style.paddingLeft) || 0) + (Number.parseFloat(style.paddingRight) || 0);
  const gap = Number.parseFloat(style.columnGap) || 0;
  const tracksWidth = Math.max(0, canvas.clientWidth - padding - gap * (COLS - 1));
  return tracksWidth / COLS + gap;
}

function rowStep() {
  const canvas = canvasEl.value;
  const gap = canvas ? Number.parseFloat(window.getComputedStyle(canvas).rowGap) || 0 : 0;
  return ROW_H + gap;
}

function onDragStart(e, i) {
  if (preview.value || !props.canEdit) return;
  e.preventDefault();
  const b = normalizeBlock(blocks.value[i]);
  blocks.value[i] = b;
  drag.value = { idx: i, startX: e.clientX, startY: e.clientY, origX: b.x, origY: b.y, mode: "drag" };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
}

function onResizeStart(e, i) {
  if (preview.value || !props.canEdit) return;
  e.preventDefault();
  const b = normalizeBlock(blocks.value[i]);
  blocks.value[i] = b;
  drag.value = { idx: i, startX: e.clientX, startY: e.clientY, origX: b.w, origY: b.h, mode: "resize" };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
}

function onPointerMove(e) {
  const d = drag.value;
  if (d.idx < 0) return;
  const b = blocks.value[d.idx];
  if (!b) return;
  const dx = e.clientX - d.startX;
  const dy = e.clientY - d.startY;
  if (d.mode === "drag") {
    const dCol = Math.round(dx / colStep());
    const dRow = Math.round(dy / rowStep());
    b.x = clamp(d.origX + dCol, 0, COLS - b.w);
    b.y = Math.max(0, d.origY + dRow);
  } else {
    const dCol = Math.round(dx / colStep());
    const dRow = Math.round(dy / rowStep());
    b.w = clamp(d.origX + dCol, MIN_W, COLS - b.x);
    b.h = Math.max(MIN_H, d.origY + dRow);
  }
}

function onPointerUp() {
  const activeIndex = drag.value.idx;
  if (activeIndex >= 0 && blocks.value[activeIndex]) {
    blocks.value[activeIndex] = normalizeBlock(blocks.value[activeIndex]);
  }
  drag.value = { idx: -1, startX: 0, startY: 0, origX: 0, origY: 0, mode: "" };
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerup", onPointerUp);
  // 容器尺寸变化后通知 ECharts 重绘
  window.dispatchEvent(new Event("resize"));
}

onBeforeUnmount(() => {
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerup", onPointerUp);
  if (noticeTimer) window.clearTimeout(noticeTimer);
});

// 保存：仅存 x/y/w/h/chart/sql/query，剥离 rows/error 等运行时字段
async function save() {
  if (!selected.value || preview.value) return;
  saving.value = true;
  try {
    blocks.value = normalizeBlocks(blocks.value);
    const payload = blocks.value.map((b) => ({
      x: b.x, y: b.y, w: b.w, h: b.h,
      chart: b.chart, sql: b.sql, query: b.query || "",
    }));
    const res = await fetch(`${API}/${selected.value.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ component_data: payload }),
    });
    if (!res.ok) {
      throw new Error(await responseError(res, "保存失败"));
    }
    error.value = "";
    showNotice("仪表盘布局已保存");
  } catch (e) {
    error.value = e.message || "保存失败";
  } finally {
    saving.value = false;
  }
}

async function refreshData() {
  if (selectedId.value) await loadDashboard(selectedId.value);
}

function togglePreview() {
  preview.value = !preview.value;
  if (preview.value) window.dispatchEvent(new Event("resize"));
}

// 树操作：新建文件夹/仪表盘
async function createNode(nodeType, pid) {
  const isFolder = nodeType === "folder";
  const parentId = knownFolderIds.has(pid) ? pid : null;
  const name = (await promptDialog({
    title: isFolder ? "新建文件夹" : "新建仪表盘",
    message: isFolder ? "文件夹用于整理当前工作空间中的仪表盘。" : "创建后即可从问数结果中加入图表。",
    label: isFolder ? "文件夹名称" : "仪表盘名称",
    value: isFolder ? "新文件夹" : "新仪表盘",
    confirmLabel: "创建",
  }))?.trim();
  if (!name) return;
  const res = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, node_type: nodeType, pid: parentId }),
  });
  if (!res.ok) {
    error.value = await responseError(res, "创建失败");
    return;
  }
  if (parentId) expandFolder(parentId);
  await loadTree();
  showNotice(nodeType === "folder" ? "文件夹已创建" : "仪表盘已创建");
}

async function renameNode(node) {
  const name = (await promptDialog({
    title: `重命名${node.node_type === "folder" ? "文件夹" : "仪表盘"}`,
    label: "新名称",
    value: node.name,
    confirmLabel: "保存名称",
  }))?.trim();
  if (!name || name === node.name) return;
  const res = await fetch(`${API}/${node.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    error.value = await responseError(res, "重命名失败");
    return;
  }
  await loadTree();
  if (selectedId.value === node.id) selected.value = { ...selected.value, name };
  showNotice("名称已更新");
}

async function deleteNode(node) {
  const confirmed = await confirmDialog({
    title: `删除${node.node_type === "folder" ? "文件夹" : "仪表盘"}？`,
    message: `“${node.name}”及其全部子项将被永久删除。`,
    confirmLabel: "确认删除",
    tone: "danger",
  });
  if (!confirmed) return;
  const removedIds = subtreeIds(node.id);
  const res = await fetch(`${API}/${node.id}`, { method: "DELETE" });
  if (!res.ok) {
    error.value = await responseError(res, "删除失败");
    return;
  }
  if (removedIds.has(selectedId.value)) clearDashboardSelection();
  if (removedIds.has(selectedFolderId.value) || removedIds.has(activeNodeId.value)) {
    const parentStillExists = node.pid && !removedIds.has(node.pid) && knownFolderIds.has(node.pid);
    selectedFolderId.value = parentStillExists ? node.pid : "";
    activeNodeId.value = selectedFolderId.value;
  }
  await loadTree();
  showNotice("资源已删除");
}

// 父组件指定 selectId 时自动加载（来自"加入仪表盘"跳转）
// immediate: 组件挂载时若 selectId 已存在（跳转进入），立即加载
watch(() => props.selectId, async (id) => {
  if (id) await loadDashboard(id);
}, { immediate: true });

onMounted(loadTree);
</script>

<style scoped>
.dash-view {
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
}
.dash-error,
.dash-notice { position: absolute; z-index: 5; left: 250px; right: 12px; top: 8px; display:flex;align-items:center;gap:7px;padding: 8px 12px; color: var(--da-danger); background: #fff1f0; border: 1px solid #ffccc7; border-radius: 8px; }
.dash-notice{color:#287b4a;background:#f0fbf4;border-color:#b9e2c8}

/* 左侧资源树 */
.dash-tree {
  flex: 0 0 240px;
  border-right: 1px solid var(--da-border-light);
  background: var(--da-bg);
  display: flex;
  flex-direction: column;
}
.tree-head {
  padding: 12px 14px;
  border-bottom: 1px solid var(--da-border-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.tree-title {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 600;
  color: var(--da-text);
}
.tree-actions {
  display: flex;
  gap: 4px;
}
.tree-body {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 6px;
}
.tree-empty {
  padding: 24px 12px;
  display: grid;
  justify-items: center;
  gap: 8px;
  text-align: center;
  font-size: 12px;
  color: var(--da-text-muted);
}
.tree-empty strong { color: var(--da-text); font-size: 13px; }
.tree-empty span { line-height: 1.6; }
.tree-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.12s;
}
.tree-root {
  width: 100%;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
}
.tree-toggle,
.tree-toggle-spacer {
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
}
.tree-toggle {
  display: grid;
  place-items: center;
  padding: 0;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: var(--da-text-muted);
  cursor: pointer;
}
.tree-toggle:hover,
.tree-toggle:focus-visible { background: var(--da-bg-soft); color: var(--da-accent); }
.tree-row:hover {
  background: var(--da-bg-soft);
}
.tree-row.active {
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.tree-icon {
  flex: 0 0 auto;
  color: var(--da-text-muted);
}
.tree-name {
  flex: 1 1 auto;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tree-ops {
  display: flex;
  gap: 2px;
  opacity: 0;
  pointer-events: none;
  transition: opacity .12s;
}
.tree-row:hover .tree-ops,
.tree-row:focus-within .tree-ops {
  opacity: 1;
  pointer-events: auto;
}

/* 右侧画布 */
.dash-canvas {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--da-bg-soft);
}
.canvas-toolbar {
  flex: 0 0 auto;
  height: 52px;
  padding: 0 18px;
  background: var(--da-bg);
  border-bottom: 1px solid var(--da-border-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.canvas-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--da-text);
}
.canvas-actions {
  display: flex;
  gap: 8px;
}
.canvas-empty {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--da-text-muted);
  gap: 6px;
}
.empty-logo {
  font-size: 44px;
}
.empty-tip {
  font-size: 15px;
  color: var(--da-text-secondary);
}
.empty-sub {
  font-size: 12px;
}
.empty-cta {
  min-height: 36px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  padding: 7px 12px;
  border: 1px solid var(--da-accent);
  border-radius: 9px;
  background: var(--da-accent);
  color: #fff;
  font-size: 12px;
  font-weight: 650;
}

/* 画布网格：12 列 CSS Grid，auto-rows 40px */
.canvas-grid {
  flex: 1 1 auto;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 16px;
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  grid-auto-rows: 40px;
  gap: 8px;
  align-content: start;
  width: 100%;
  box-sizing: border-box;
}

/* 单个画布块卡片 */
.block-card {
  background: var(--da-bg);
  border: 1px solid var(--da-border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  min-width: 0;
  min-height: 0;
}
.block-card.dragging {
  opacity: 0.85;
  box-shadow: 0 6px 20px rgba(31, 35, 41, 0.18);
  z-index: 10;
}
.block-head {
  flex: 0 0 auto;
  padding: 8px 12px;
  border-bottom: 1px solid var(--da-border-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
}
.block-head.draggable {
  cursor: move;
}
.block-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--da-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.block-badge {
  flex: 0 0 auto;
  padding: 1px 8px;
  font-size: 11px;
  border-radius: 999px;
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.block-body {
  flex: 1 1 auto;
  overflow: hidden;
  padding: 8px;
  position: relative;
  min-height: 0;
}
.block-error {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px;
  font-size: 12px;
  color: var(--da-danger);
}

/* 右下角缩放手柄 */
.resize-handle {
  position: absolute;
  right: 0;
  bottom: 0;
  width: 16px;
  height: 16px;
  cursor: nwse-resize;
  background: linear-gradient(135deg, transparent 50%, var(--da-border) 50%);
  border-bottom-right-radius: 10px;
  z-index: 5;
}
.resize-handle:hover {
  background: linear-gradient(135deg, transparent 50%, var(--da-accent) 50%);
}
.block-card.preview .resize-handle {
  display: none;
}

/* mini-btn 视觉 */
.mini-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 4px 9px;
  font-size: 12px;
  line-height: 18px;
  border-radius: 8px;
  border: 1px solid var(--da-border);
  background: var(--da-bg);
  color: var(--da-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
}
.mini-btn.icon { width: 32px; padding: 0; }
.mini-btn.x.danger:hover { color: var(--da-danger); background: #fff1f0; }
.mini-btn:hover:not(:disabled) {
  border-color: var(--da-accent);
  color: var(--da-accent);
  background: var(--da-accent-soft);
}
.mini-btn.x {
  padding: 2px 6px;
  border: none;
  background: transparent;
}
.mini-btn.x:hover {
  background: var(--da-bg-soft);
}
.mini-btn.primary {
  background: var(--da-accent);
  color: #fff;
  border-color: var(--da-accent);
}
.mini-btn.primary:hover:not(:disabled) {
  background: var(--da-accent-hover);
  color: #fff;
}
.mini-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.dash-view { position: relative; gap: 0; background: var(--da-bg-soft); }
.dash-tree { flex-basis: 270px; border-right: 0; background: rgba(255,255,255,.92); box-shadow: 4px 0 18px rgba(31,55,96,.04); z-index: 2; }
.tree-head { min-height: 60px; padding: 13px 16px; }
.tree-title { font-size: 15px; letter-spacing: -.01em; }
.tree-body { padding: 10px; }
.tree-row { min-height: 40px; padding: 9px 10px; border: 1px solid transparent; border-radius: 10px; }
.tree-row.active { border-color: rgba(52,120,246,.13); background: var(--da-accent-soft); font-weight: 600; }
.tree-empty { margin: 8px; padding: 38px 14px; border: 1px dashed var(--da-border); border-radius: 12px; line-height: 1.7; }
.dash-canvas { background: radial-gradient(circle at 50% 0, rgba(52,120,246,.05), transparent 32%), var(--da-bg-soft); }
.canvas-toolbar { height: 60px; padding: 0 22px; background: rgba(255,255,255,.88); backdrop-filter: blur(12px); }
.canvas-title { font-size: 16px; letter-spacing: -.02em; }
.canvas-empty { margin: 24px; border: 1px dashed var(--da-border); border-radius: 18px; background: rgba(255,255,255,.58); }
.empty-logo { width: 64px; height: 64px; display: grid; place-items: center; margin-bottom: 8px; border-radius: 20px; background: #fff; box-shadow: var(--da-shadow-md); }
.canvas-grid { padding: 20px; gap: 10px; }
.block-card { border-color: var(--da-border-light); border-radius: 14px; box-shadow: var(--da-shadow-sm); transition: box-shadow .15s, transform .15s; }
.block-card:hover { box-shadow: var(--da-shadow-md); }
.block-head { min-height: 42px; padding: 9px 13px; background: linear-gradient(180deg,#fff,#fbfcfe); }
.block-body { padding: 10px; }
.block-badge { padding: 3px 8px; }
.mini-btn { min-height: 32px; border-radius: 9px; }
.dash-error,.dash-notice { left: 286px; right: 16px; top: 10px; padding: 10px 13px; box-shadow: var(--da-shadow-sm); }
@media(max-width:760px){.dash-view{flex-direction:column}.dash-tree{flex:0 0 auto;max-height:245px;border-bottom:1px solid var(--da-border-light)}.tree-body{overflow:auto}.dash-error,.dash-notice{position:fixed;left:12px;right:12px;top:70px}.canvas-toolbar{height:auto;min-height:58px;padding:10px 14px;gap:10px}.canvas-actions{flex-wrap:wrap;justify-content:flex-end}.canvas-grid{padding:12px;display:block}.block-card{min-height:360px!important;margin-bottom:12px;transform:none!important}.canvas-empty{min-height:330px;margin:12px}.tree-row .tree-ops{opacity:1;pointer-events:auto}}
</style>
