<template>
  <header class="mobile-header">
    <button class="icon-button" type="button" aria-label="打开主菜单" @click="drawerOpen = true">
      <Menu :size="20" aria-hidden="true" />
    </button>
    <div class="mobile-brand">
      <span class="brand-mark"><ChartNoAxesCombined :size="20" aria-hidden="true" /></span>
      <span>DA 数据助手</span>
    </div>
    <button class="icon-button primary" type="button" aria-label="开始新对话" @click="startConversation">
      <Plus :size="19" aria-hidden="true" />
    </button>
  </header>

  <button
    v-if="drawerOpen"
    class="drawer-backdrop"
    type="button"
    aria-label="关闭主菜单"
    @click="drawerOpen = false"
  ></button>

  <aside class="app-sidebar" :class="{ open: drawerOpen }" aria-label="应用导航">
    <div class="sidebar-brand">
      <span class="brand-mark"><ChartNoAxesCombined :size="23" aria-hidden="true" /></span>
      <span class="brand-copy">
        <strong>DA 数据助手</strong>
        <small>DATA COPILOT</small>
      </span>
      <button class="drawer-close" type="button" aria-label="关闭主菜单" @click="drawerOpen = false">
        <X :size="19" aria-hidden="true" />
      </button>
    </div>

    <button class="new-chat" type="button" @click="startConversation">
      <Plus :size="17" aria-hidden="true" />
      <span>新对话</span>
      <kbd>Alt N</kbd>
    </button>

    <ConversationHistoryList
      :items="conversations"
      :active-chat-id="activeChatId"
      :status="historyStatus"
      :loading="historyLoading"
      :error="historyError"
      :has-more="historyHasMore"
      :retention-days="historyRetentionDays"
      @select="emit('open-chat', $event)"
      @refresh="emit('refresh-history')"
      @load-more="emit('load-more-history')"
      @archive="emit('archive-chat', $event)"
      @restore="emit('restore-chat', $event)"
      @delete="emit('delete-chat', $event)"
      @status-change="emit('history-status-change', $event)"
    />

    <nav class="nav-groups" aria-label="主导航">
      <section v-for="group in visibleGroups" :key="group.label" class="nav-group">
        <h2>{{ group.label }}</h2>
        <button
          v-for="item in group.items"
          :key="item.mode"
          type="button"
          class="nav-item"
          :class="{ active: mode === item.mode }"
          :aria-current="mode === item.mode ? 'page' : undefined"
          @click="selectMode(item.mode)"
        >
          <component :is="item.icon" :size="17" :stroke-width="1.9" aria-hidden="true" />
          <span>{{ item.label }}</span>
          <ChevronRight v-if="mode === item.mode" :size="15" aria-hidden="true" />
        </button>
      </section>
    </nav>

    <footer class="sidebar-footer">
      <button class="workspace-card" type="button" @click="selectMode('workspace')">
        <span class="workspace-avatar">{{ workspaceInitial }}</span>
        <span>
          <strong>{{ currentUser?.workspace?.name || "当前工作空间" }}</strong>
          <small>{{ roleLabel(currentUser?.workspace?.role) }}</small>
        </span>
        <ChevronRight :size="15" aria-hidden="true" />
      </button>
      <button class="logout" type="button" @click="emit('logout')">
        <LogOut :size="16" aria-hidden="true" />
        <span>退出登录</span>
      </button>
    </footer>
  </aside>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import ConversationHistoryList from "./ConversationHistoryList.vue";
import {
  Activity,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronRight,
  FileSpreadsheet,
  Gauge,
  Grid3X3,
  KeyRound,
  LayoutDashboard,
  LogOut,
  Menu,
  MessageSquare,
  Plus,
  Search,
  ShieldCheck,
  Target,
  Users,
  Waypoints,
  Wrench,
  X,
} from "@lucide/vue";

const props = defineProps({
  mode: { type: String, required: true },
  currentUser: { type: Object, default: null },
  conversations: { type: Array, default: () => [] },
  activeChatId: { type: String, default: "" },
  historyStatus: { type: String, default: "active" },
  historyLoading: { type: Boolean, default: false },
  historyError: { type: String, default: "" },
  historyHasMore: { type: Boolean, default: false },
  historyRetentionDays: { type: Number, default: 30 },
});
const emit = defineEmits([
  "update:mode",
  "new-chat",
  "logout",
  "open-chat",
  "refresh-history",
  "load-more-history",
  "archive-chat",
  "restore-chat",
  "delete-chat",
  "history-status-change",
]);
const drawerOpen = ref(false);

const role = computed(() => props.currentUser?.workspace?.role || "viewer");
const canManage = computed(() => ["owner", "admin"].includes(role.value));
const workspaceInitial = computed(() => String(props.currentUser?.workspace?.name || "DA").trim().slice(0, 1).toUpperCase());

const visibleGroups = computed(() => [
  {
    label: "工作区",
    items: [
      { mode: "chat", label: "对话", icon: MessageSquare },
      { mode: "dashboard", label: "仪表盘", icon: LayoutDashboard },
      { mode: "adhoc", label: "临时数据", icon: FileSpreadsheet },
    ],
  },
  {
    label: "数据治理",
    items: [
      ...(canManage.value ? [{ mode: "meta", label: "元知识", icon: BrainCircuit }] : []),
      ...(canManage.value ? [{ mode: "masking", label: "脱敏规则", icon: ShieldCheck }] : []),
      ...(canManage.value ? [{ mode: "access", label: "数据权限", icon: KeyRound }] : []),
      { mode: "workspace", label: "工作空间", icon: Users },
    ],
  },
  {
    label: "质量验证",
    items: canManage.value
      ? [
          { mode: "evaluation", label: "质量归因", icon: Target },
          { mode: "quality-operations", label: "质量运营", icon: Activity },
          { mode: "scenario-matrix", label: "场景矩阵", icon: Grid3X3 },
          { mode: "resilience", label: "稳定性评测", icon: Gauge },
          { mode: "adhoc-isolation", label: "上传隔离", icon: ShieldCheck },
          { mode: "rag-evaluation", label: "RAG 评测", icon: Search },
          { mode: "planning-evaluation", label: "规划路由", icon: Waypoints },
          { mode: "semantic-repair", label: "语义修复", icon: Wrench },
        ]
      : [],
  },
].filter((group) => group.items.length));

function roleLabel(value) {
  return ({ owner: "所有者", admin: "管理员", analyst: "分析员", viewer: "查看者" })[value] || value || "";
}

function selectMode(value) {
  emit("update:mode", value);
  drawerOpen.value = false;
}

function startConversation() {
  emit("update:mode", "chat");
  emit("new-chat");
  drawerOpen.value = false;
}

function handleKeydown(event) {
  if (event.key === "Escape") drawerOpen.value = false;
  if (event.altKey && event.key.toLowerCase() === "n") {
    event.preventDefault();
    startConversation();
  }
}

onMounted(() => window.addEventListener("keydown", handleKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", handleKeydown));
</script>

<style scoped>
.app-sidebar {
  position: relative;
  z-index: 50;
  flex: 0 0 244px;
  width: 244px;
  min-height: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--da-border-light);
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 6px 0 24px rgba(31, 55, 96, 0.035);
}
.sidebar-brand {
  height: 76px;
  flex: 0 0 76px;
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 0 18px;
  border-bottom: 1px solid var(--da-border-light);
}
.brand-mark {
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  display: grid;
  place-items: center;
  border-radius: 13px;
  background: linear-gradient(145deg, #3e86fa, #1f58c8);
  color: #fff;
  box-shadow: 0 10px 24px rgba(36, 99, 212, 0.22);
}
.brand-copy {
  min-width: 0;
  display: grid;
  gap: 4px;
}
.brand-copy strong {
  color: var(--da-text);
  font-size: 15px;
  letter-spacing: -0.02em;
}
.brand-copy small {
  color: var(--da-text-muted);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.13em;
}
.new-chat {
  min-height: 42px;
  margin: 14px 14px 9px;
  padding: 0 12px;
  display: flex;
  align-items: center;
  gap: 9px;
  border: 1px solid rgba(36, 99, 212, 0.17);
  border-radius: 11px;
  background: var(--da-accent);
  color: #fff;
  font-size: 13px;
  font-weight: 650;
  box-shadow: 0 8px 18px rgba(36, 99, 212, 0.18);
}
.new-chat:hover {
  background: var(--da-accent-hover);
  transform: translateY(-1px);
}
.new-chat span { flex: 1; text-align: left; }
.new-chat kbd {
  padding: 2px 5px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 5px;
  color: #fff;
  font: 9px/1.4 inherit;
}
.nav-groups {
  min-height: 0;
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 4px 10px 16px;
}
.nav-group { margin-top: 13px; }
.nav-group h2 {
  margin: 0 10px 6px;
  color: var(--da-text-muted);
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.13em;
}
.nav-item {
  width: 100%;
  min-height: 39px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: transparent;
  color: var(--da-text-secondary);
  font-size: 12px;
  text-align: left;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.nav-item span { flex: 1; }
.nav-item:hover {
  border-color: var(--da-border-light);
  background: var(--da-bg-soft);
  color: var(--da-text);
}
.nav-item.active {
  border-color: rgba(36, 99, 212, 0.13);
  background: var(--da-accent-soft);
  color: var(--da-accent-strong);
  font-weight: 650;
}
.sidebar-footer {
  flex: 0 0 auto;
  display: grid;
  gap: 7px;
  padding: 12px;
  border-top: 1px solid var(--da-border-light);
}
.workspace-card,
.logout {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 9px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  text-align: left;
}
.workspace-card { min-height: 50px; padding: 6px; }
.workspace-card:hover, .logout:hover { background: var(--da-bg-soft); }
.workspace-avatar {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: #eaf2ff;
  color: var(--da-accent-strong);
  font-size: 13px;
  font-weight: 750;
}
.workspace-card > span:nth-child(2) {
  min-width: 0;
  flex: 1;
  display: grid;
  gap: 2px;
}
.workspace-card strong {
  overflow: hidden;
  color: var(--da-text);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.workspace-card small { color: var(--da-text-muted); font-size: 9px; }
.logout {
  min-height: 34px;
  padding: 0 10px;
  color: var(--da-text-muted);
  font-size: 11px;
}
.logout:hover { color: var(--da-danger); }
.mobile-header,
.drawer-backdrop,
.drawer-close { display: none; }

@media (max-width: 900px) {
  .mobile-header {
    position: relative;
    z-index: 45;
    height: 58px;
    flex: 0 0 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    border-bottom: 1px solid var(--da-border-light);
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(16px);
  }
  .mobile-brand { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; }
  .mobile-brand .brand-mark { width: 32px; height: 32px; flex-basis: 32px; border-radius: 10px; }
  .icon-button {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border: 1px solid var(--da-border-light);
    border-radius: 11px;
    background: #fff;
    color: var(--da-text-secondary);
  }
  .icon-button.primary { border-color: var(--da-accent); background: var(--da-accent); color: #fff; }
  .app-sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 70;
    width: min(300px, calc(100vw - 48px));
    transform: translateX(-104%);
    box-shadow: 18px 0 50px rgba(18, 34, 61, 0.2);
    transition: transform 0.2s ease;
  }
  .app-sidebar.open { transform: translateX(0); }
  .drawer-backdrop {
    position: fixed;
    inset: 0;
    z-index: 65;
    display: block;
    border: 0;
    background: rgba(18, 28, 45, 0.44);
  }
  .drawer-close {
    width: 34px;
    height: 34px;
    margin-left: auto;
    display: grid;
    place-items: center;
    border: 0;
    border-radius: 9px;
    background: var(--da-bg-soft);
    color: var(--da-text-secondary);
  }
}
</style>
