<template>
  <section class="history-panel" aria-labelledby="conversation-history-title">
    <header class="history-header">
      <div>
        <span class="history-kicker">CONVERSATIONS</span>
        <h2 id="conversation-history-title">会话历史</h2>
      </div>
      <button
        class="history-icon-button"
        type="button"
        aria-label="刷新会话历史"
        :disabled="loading"
        @click="emit('refresh')"
      >
        <RotateCw :size="14" :class="{ spinning: loading }" aria-hidden="true" />
      </button>
    </header>

    <div class="history-tabs" role="tablist" aria-label="会话状态">
      <button
        type="button"
        role="tab"
        :aria-selected="status === 'active'"
        :class="{ active: status === 'active' }"
        @click="emit('status-change', 'active')"
      >
        最近
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="status === 'archived'"
        :class="{ active: status === 'archived' }"
        @click="emit('status-change', 'archived')"
      >
        已归档
      </button>
    </div>

    <div class="history-body" aria-live="polite">
      <div v-if="loading && !items.length" class="history-skeleton" aria-label="正在加载会话">
        <span v-for="index in 3" :key="index"></span>
      </div>

      <div v-else-if="error && !items.length" class="history-state error" role="alert">
        <CircleAlert :size="15" aria-hidden="true" />
        <span>{{ error }}</span>
        <button type="button" @click="emit('refresh')">重试</button>
      </div>

      <div v-else-if="!items.length" class="history-state empty">
        <MessagesSquare :size="18" aria-hidden="true" />
        <span>{{ status === "archived" ? "暂无已归档会话" : "完成一次问数后，会话会出现在这里" }}</span>
      </div>

      <ul v-else class="history-list">
        <li
          v-for="item in items"
          :key="item.chat_id"
          class="history-item"
          :class="{ selected: item.chat_id === activeChatId }"
        >
          <button
            type="button"
            class="history-main"
            :aria-current="item.chat_id === activeChatId ? 'true' : undefined"
            :aria-label="`打开会话：${item.title}`"
            @click="emit('select', item)"
          >
            <span class="history-status-dot" :class="item.last_turn_status"></span>
            <span class="history-copy">
              <strong :title="item.title">{{ item.title }}</strong>
              <small :title="item.preview">{{ item.preview || statusText(item.last_turn_status) }}</small>
            </span>
            <time :datetime="item.last_message_at || undefined">{{ relativeTime(item.last_message_at) }}</time>
          </button>
          <div class="history-actions">
            <button
              v-if="item.status === 'archived'"
              type="button"
              aria-label="恢复会话"
              title="恢复会话"
              @click.stop="emit('restore', item)"
            >
              <ArchiveRestore :size="13" aria-hidden="true" />
            </button>
            <button
              v-else
              type="button"
              aria-label="归档会话"
              title="归档会话"
              @click.stop="emit('archive', item)"
            >
              <Archive :size="13" aria-hidden="true" />
            </button>
            <button
              type="button"
              class="danger"
              aria-label="移除历史会话"
              title="删除会话"
              @click.stop="emit('delete', item)"
            >
              <Trash2 :size="13" aria-hidden="true" />
            </button>
          </div>
        </li>
      </ul>

      <button
        v-if="hasMore"
        class="history-more"
        type="button"
        :disabled="loading"
        @click="emit('load-more')"
      >
        <LoaderCircle v-if="loading" :size="13" class="spinning" aria-hidden="true" />
        {{ loading ? "加载中…" : "加载更多" }}
      </button>
    </div>

    <footer v-if="retentionDays > 0" class="history-retention">
      <Clock3 :size="12" aria-hidden="true" />
      保留期 {{ retentionDays }} 天
    </footer>
  </section>
</template>

<script setup>
import {
  Archive,
  ArchiveRestore,
  CircleAlert,
  Clock3,
  LoaderCircle,
  MessagesSquare,
  RotateCw,
  Trash2,
} from "@lucide/vue";

defineProps({
  items: { type: Array, default: () => [] },
  activeChatId: { type: String, default: "" },
  status: { type: String, default: "active" },
  loading: { type: Boolean, default: false },
  error: { type: String, default: "" },
  hasMore: { type: Boolean, default: false },
  retentionDays: { type: Number, default: 30 },
});

const emit = defineEmits([
  "select",
  "refresh",
  "load-more",
  "archive",
  "restore",
  "delete",
  "status-change",
]);

function relativeTime(value) {
  const date = new Date(value || "");
  if (Number.isNaN(date.getTime())) return "刚刚";
  const seconds = Math.round((date.getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" });
  if (Math.abs(seconds) < 60) return formatter.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return formatter.format(hours, "hour");
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 7) return formatter.format(days, "day");
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function statusText(value) {
  return ({
    pending: "处理中",
    succeeded: "查询已完成",
    failed: "本轮未完成",
    cancelled: "请求已取消",
  })[value] || "会话记录";
}
</script>

<style scoped>
.history-panel {
  flex: 0 0 auto;
  margin: 1px 10px 0;
  padding: 11px 9px 8px;
  border: 1px solid rgba(36, 99, 212, 0.09);
  border-radius: 13px;
  background:
    linear-gradient(145deg, rgba(242, 247, 255, 0.92), rgba(255, 255, 255, 0.96));
}
.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 3px 8px;
}
.history-header > div { min-width: 0; }
.history-kicker {
  display: block;
  color: var(--da-accent);
  font-size: 7px;
  font-weight: 800;
  letter-spacing: 0.14em;
}
.history-header h2 {
  margin: 2px 0 0;
  color: var(--da-text);
  font-size: 11px;
  letter-spacing: -0.01em;
}
.history-icon-button {
  width: 27px;
  height: 27px;
  display: grid;
  place-items: center;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--da-text-muted);
}
.history-icon-button:hover:not(:disabled) {
  border-color: var(--da-border-light);
  background: #fff;
  color: var(--da-accent);
}
.history-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3px;
  padding: 3px;
  border-radius: 9px;
  background: rgba(215, 226, 242, 0.52);
}
.history-tabs button {
  min-height: 25px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--da-text-muted);
  font-size: 9px;
  font-weight: 650;
}
.history-tabs button.active {
  background: #fff;
  color: var(--da-accent-strong);
  box-shadow: 0 2px 8px rgba(30, 66, 121, 0.08);
}
.history-body {
  max-height: 212px;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  padding-top: 6px;
}
.history-list {
  display: grid;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.history-item {
  position: relative;
  min-width: 0;
  border: 1px solid transparent;
  border-radius: 10px;
  transition: background 0.14s, border-color 0.14s;
}
.history-item:hover {
  border-color: rgba(36, 99, 212, 0.09);
  background: rgba(255, 255, 255, 0.82);
}
.history-item.selected {
  border-color: rgba(36, 99, 212, 0.16);
  background: #fff;
  box-shadow: 0 5px 14px rgba(37, 75, 132, 0.07);
}
.history-main {
  width: 100%;
  min-width: 0;
  min-height: 48px;
  display: grid;
  grid-template-columns: 6px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
  padding: 6px 7px;
  border: 0;
  border-radius: 9px;
  background: transparent;
  text-align: left;
}
.history-status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #91a0b3;
  box-shadow: 0 0 0 3px rgba(145, 160, 179, 0.12);
}
.history-status-dot.succeeded {
  background: #2ba977;
  box-shadow: 0 0 0 3px rgba(43, 169, 119, 0.12);
}
.history-status-dot.pending {
  background: #e49b30;
  box-shadow: 0 0 0 3px rgba(228, 155, 48, 0.14);
}
.history-status-dot.failed,
.history-status-dot.cancelled {
  background: #d45c58;
  box-shadow: 0 0 0 3px rgba(212, 92, 88, 0.12);
}
.history-copy {
  min-width: 0;
  display: grid;
  gap: 3px;
}
.history-copy strong,
.history-copy small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.history-copy strong {
  color: var(--da-text-secondary);
  font-size: 10px;
  font-weight: 680;
}
.history-copy small {
  color: var(--da-text-muted);
  font-size: 8px;
}
.history-main time {
  align-self: start;
  padding-top: 2px;
  color: #667085;
  font-size: 8px;
  white-space: nowrap;
}
.history-actions {
  position: absolute;
  right: 5px;
  bottom: 4px;
  display: flex;
  gap: 2px;
  opacity: 0;
  transform: translateY(2px);
  transition: opacity 0.14s, transform 0.14s;
}
.history-item:hover .history-actions,
.history-item:focus-within .history-actions {
  opacity: 1;
  transform: translateY(0);
}
.history-actions button {
  width: 24px;
  height: 22px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 6px;
  background: #f0f4fa;
  color: var(--da-text-muted);
}
.history-actions button:hover {
  background: var(--da-accent-soft);
  color: var(--da-accent-strong);
}
.history-actions button.danger:hover {
  background: rgba(214, 69, 69, 0.09);
  color: var(--da-danger);
}
.history-state {
  min-height: 78px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 12px 8px;
  color: var(--da-text-muted);
  font-size: 9px;
  line-height: 1.5;
  text-align: center;
}
.history-state.error { color: var(--da-danger); }
.history-state button {
  border: 0;
  background: transparent;
  color: var(--da-accent);
  font-size: 9px;
  font-weight: 650;
}
.history-skeleton {
  display: grid;
  gap: 6px;
  padding: 3px 0;
}
.history-skeleton span {
  height: 46px;
  border-radius: 9px;
  background: linear-gradient(90deg, #edf2f8 20%, #f9fbfe 48%, #edf2f8 76%);
  background-size: 220% 100%;
  animation: shimmer 1.3s ease-in-out infinite;
}
.history-more {
  width: 100%;
  min-height: 27px;
  margin-top: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--da-accent);
  font-size: 9px;
  font-weight: 650;
}
.history-more:hover:not(:disabled) { background: rgba(255, 255, 255, 0.75); }
.history-retention {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 3px 0;
  color: #667085;
  font-size: 8px;
}
.spinning { animation: spin 0.75s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes shimmer { to { background-position: -120% 0; } }

@media (prefers-reduced-motion: reduce) {
  .spinning,
  .history-skeleton span { animation: none; }
}
</style>
