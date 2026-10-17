<template>
  <aside class="memory-panel" aria-label="会话记忆详情" data-testid="memory-panel">
    <div class="memory-panel-head">
      <div>
        <strong>分层会话记忆</strong>
        <small>近期原文、摘要段与长期事实均由 MySQL 复核，Redis/ES/Qdrant 只承担加速与召回</small>
      </div>
      <button aria-label="关闭会话记忆详情" @click="$emit('close')"><X :size="16" aria-hidden="true" /></button>
    </div>
    <div v-if="state.loading" class="memory-panel-empty">正在读取…</div>
    <div v-else-if="state.error" class="memory-panel-error">{{ state.error }}</div>
    <template v-else-if="state.data">
      <div class="memory-stats">
        <span>Redis <b>{{ state.data.working_memory?.available ? (state.data.working_memory?.exists ? "命中" : "未命中") : "降级" }}</b></span>
        <span>Checkpoint <b>v{{ state.data.checkpoint?.version || 0 }}</b></span>
        <span>摘要段 <b>{{ state.data.summary_memory?.active_segment_count || 0 }}</b></span>
        <span>长期记忆 <b>{{ state.data.long_term?.active_count || 0 }}（本会话 {{ state.data.long_term?.current_chat_count || 0 }}）</b></span>
      </div>
      <section class="summary-card" aria-label="摘要记忆状态">
        <header>
          <div>
            <strong>可追溯摘要记忆</strong>
            <small>覆盖至消息 #{{ state.data.summary_memory?.covered_through_message_id ?? "—" }} · 水位 v{{ state.data.summary_memory?.watermark_version || 0 }}</small>
          </div>
          <b>{{ compressionLabel(state.data.summary_memory) }}</b>
        </header>
        <div class="summary-facts">
          <span v-for="(count, level) in state.data.summary_memory?.by_level || {}" :key="level">{{ level }} · {{ count }}</span>
          <span>信号 · {{ state.data.summary_memory?.signal_count || 0 }}</span>
          <span>近期 TTL · {{ state.data.working_memory?.ttl_seconds ?? "—" }}s</span>
        </div>
        <div v-if="state.data.summary_memory?.items?.length" class="summary-items">
          <article v-for="item in state.data.summary_memory.items.slice(0, 4)" :key="item.summary_id">
            <div><b>L{{ item.level }}</b><span>消息 {{ item.source_start_message_id }}–{{ item.source_end_message_id }}</span></div>
            <p>{{ item.content }}</p>
            <small>{{ item.source_message_count }} 条来源 · {{ item.summary_tokens }} tokens</small>
          </article>
        </div>
        <p v-else class="summary-empty">尚未越过摘要高水位；近期精确消息仍会按统一预算进入上下文。</p>
      </section>
      <div class="memory-projection" aria-label="长期记忆投影状态">
        <span>Outbox <b>{{ state.data.projection?.outbox?.backlog || 0 }}</b></span>
        <span>Worker <b>{{ state.data.projection?.worker?.running ? "运行中" : "已停止" }}</b></span>
        <span>长期 ES <b>{{ projectionLabel(state.data.projection?.elasticsearch, "alias_ready") }}</b></span>
        <span>长期 Qdrant <b>{{ projectionLabel(state.data.projection?.qdrant, "exists") }}</b></span>
        <span>摘要 ES <b>{{ projectionLabel(state.data.projection?.summary_elasticsearch, "alias_ready") }}</b></span>
        <span>摘要 Qdrant <b>{{ projectionLabel(state.data.projection?.summary_qdrant, "exists") }}</b></span>
      </div>
      <div v-if="state.data.long_term?.items?.length" class="memory-items">
        <article v-for="item in state.data.long_term.items" :key="item.memory_id">
          <span>{{ memoryTypeLabel(item.memory_type) }}</span>
          <p>{{ item.content }}</p>
          <small>{{ item.verification_status }} · 来源会话 {{ item.source_chat_id }} · {{ item.created_at }}</small>
        </article>
      </div>
      <div v-else class="memory-panel-empty">当前工作空间还没有经确认的长期事实；摘要记忆与长期事实相互独立。</div>
      <div class="memory-panel-actions">
        <button :disabled="state.resetting" @click="reset(false)">重置短期上下文</button>
        <button class="danger" :disabled="state.resetting" @click="reset(true)">同时撤销长期记忆</button>
      </div>
    </template>
  </aside>
</template>

<script setup>
import { onMounted, reactive } from "vue";
import { X } from "@lucide/vue";
import { requestJson } from "../utils/http.js";
import { alertDialog, confirmDialog } from "../composables/appDialog.js";

const props = defineProps({ chatId: { type: String, required: true } });
const emit = defineEmits(["close", "reset-complete"]);
const state = reactive({ loading: true, resetting: false, data: null, error: "" });

function projectionLabel(projection, readyKey) {
  if (!projection || projection.status === "unavailable") return "未启用";
  if (projection.status === "degraded") return "降级";
  return projection[readyKey] ? "就绪" : "待建立";
}

function compressionLabel(summary) {
  const source = Number(summary?.source_tokens || 0);
  const target = Number(summary?.summary_tokens || 0);
  if (!source) return "等待生成";
  return `压缩 ${Math.max(0, Math.round((1 - target / source) * 100))}%`;
}

function memoryTypeLabel(type) {
  return ({
    preference: "偏好",
    correction: "修正",
    query_episode: "查询语义",
    profile_fact: "身份事实",
    long_term_goal: "长期目标",
  })[type] || type;
}

async function load() {
  state.loading = true;
  state.error = "";
  try {
    state.data = await requestJson(
      `/api/chat/${encodeURIComponent(props.chatId)}/memory`,
      {},
      "记忆读取失败",
    );
  } catch (error) {
    state.error = error?.message || "记忆读取失败";
  } finally {
    state.loading = false;
  }
}

async function reset(includeLongTerm) {
  if (state.resetting) return;
  const message = includeLongTerm
    ? "确认撤销当前会话的短期状态及全部长期记忆吗？"
    : "确认清空当前会话的短期上下文吗？跨会话长期偏好会保留。";
  const confirmed = await confirmDialog({
    title: includeLongTerm ? "重置并撤销长期记忆？" : "重置短期上下文？",
    message,
    confirmLabel: includeLongTerm ? "重置并撤销" : "确认重置",
    tone: includeLongTerm ? "danger" : "default",
  });
  if (!confirmed) return;
  state.resetting = true;
  try {
    const body = await requestJson(
      `/api/chat/${encodeURIComponent(props.chatId)}/memory?include_long_term=${includeLongTerm ? "true" : "false"}`,
      { method: "DELETE" },
      "记忆重置失败",
    );
    await alertDialog({
      title: "会话记忆已更新",
      message: includeLongTerm
        ? `已重置会话并撤销 ${body.long_term_revoked || 0} 条长期记忆。`
        : "已重置短期上下文；跨会话长期记忆仍保留。",
    });
    emit("reset-complete");
  } catch (error) {
    state.error = error?.message || "记忆重置失败";
  } finally {
    state.resetting = false;
  }
}

onMounted(load);
</script>

<style scoped>
.memory-panel { position: fixed; z-index: 45; right: max(18px, calc((100vw - 900px) / 2)); bottom: 126px; width: min(600px, calc(100vw - 32px)); max-height: min(650px, 72vh); overflow: auto; padding: 18px; border: 1px solid #cddbf0; border-radius: 18px; background: rgba(255, 255, 255, .985); box-shadow: 0 22px 58px rgba(31, 62, 103, .22); color: var(--da-text); }
.memory-panel-head { display: flex; align-items: start; justify-content: space-between; gap: 12px; }
.memory-panel-head > div { display: grid; gap: 4px; }
.memory-panel-head small { color: var(--da-text-muted); font-size: 11px; }
.memory-panel-head button { width: 28px; height: 28px; border: 0; border-radius: 8px; background: var(--da-bg-soft); font-size: 18px; cursor: pointer; }
.memory-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; margin: 14px 0; }
.memory-stats span { display: grid; gap: 4px; padding: 9px; border-radius: 9px; background: var(--da-bg-soft); color: var(--da-text-muted); font-size: 11px; }
.memory-stats b { color: var(--da-text-secondary); font-size: 12px; }
.summary-card { display: grid; gap: 10px; margin: 0 0 13px; padding: 13px; border: 1px solid #cfe0f7; border-radius: 13px; background: linear-gradient(145deg, #f8fbff, #f2f7ff); }
.summary-card header { display: flex; align-items: start; justify-content: space-between; gap: 12px; }
.summary-card header > div { display: grid; gap: 3px; }
.summary-card header strong { color: #245fae; font-size: 12px; }
.summary-card header small, .summary-card header > b { color: var(--da-text-muted); font-size: 10px; }
.summary-card header > b { padding: 4px 7px; border-radius: 999px; background: #fff; color: #2e67ae; white-space: nowrap; }
.summary-facts { display: flex; flex-wrap: wrap; gap: 5px; }
.summary-facts span { padding: 4px 7px; border-radius: 999px; background: #e7f1ff; color: #366baf; font-size: 10px; }
.summary-items { display: grid; gap: 6px; }
.summary-items article { display: grid; gap: 4px; padding: 8px 9px; border: 1px solid rgba(176, 202, 236, .7); border-radius: 9px; background: rgba(255, 255, 255, .86); }
.summary-items article > div { display: flex; justify-content: space-between; gap: 8px; color: var(--da-text-muted); font-size: 10px; }
.summary-items article > div b { color: #2e67ae; }
.summary-items p, .summary-empty { margin: 0; color: var(--da-text-secondary); font-size: 11px; line-height: 1.55; overflow-wrap: anywhere; }
.summary-items small { color: var(--da-text-muted); font-size: 9px; }
.summary-empty { color: var(--da-text-muted); }
.memory-projection { display: flex; flex-wrap: wrap; gap: 6px; margin: -5px 0 12px; }
.memory-projection span { padding: 5px 8px; border: 1px solid var(--da-border-light); border-radius: 999px; color: var(--da-text-muted); font-size: 11px; }
.memory-projection b { color: var(--da-text-secondary); }
.memory-items { display: grid; gap: 7px; }
.memory-items article { display: grid; grid-template-columns: 90px 1fr; gap: 3px 8px; padding: 9px; border: 1px solid var(--da-border-light); border-radius: 9px; }
.memory-items article > span { color: #2e67ae; font-size: 11px; }
.memory-items p { overflow-wrap: anywhere; margin: 0; font-size: 12px; }
.memory-items small { grid-column: 2; color: var(--da-text-muted); font-size: 10px; }
.memory-panel-empty, .memory-panel-error { padding: 20px; text-align: center; color: var(--da-text-muted); font-size: 12px; }
.memory-panel-error { color: #b63e31; }
.memory-panel-actions { display: flex; justify-content: flex-end; gap: 7px; margin-top: 12px; }
.memory-panel-actions button { padding: 7px 10px; border: 1px solid var(--da-border); border-radius: 8px; background: #fff; color: var(--da-text-secondary); font-size: 11px; cursor: pointer; }
.memory-panel-actions .danger { border-color: #efc2bd; color: #b63e31; }
@media (max-width: 640px) {
  .memory-panel { right: 16px; bottom: 145px; }
  .memory-stats { grid-template-columns: 1fr 1fr; }
}
</style>
