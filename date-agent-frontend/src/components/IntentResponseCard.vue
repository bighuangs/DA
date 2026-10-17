<template>
  <section
    class="intent-response-card"
    :class="tone"
    role="status"
    :aria-label="title"
    data-testid="intent-response-card"
  >
    <div class="intent-response-icon" aria-hidden="true">{{ glyph }}</div>
    <div class="intent-response-body">
      <div class="intent-response-heading">
        <strong>{{ title }}</strong>
        <span>{{ categoryLabel }}</span>
      </div>
      <p>{{ response.message }}</p>
      <div v-if="suggestions.length" class="intent-response-suggestions">
        <span>{{ suggestionLabel }}</span>
        <button
          v-for="suggestion in suggestions"
          :key="suggestion"
          type="button"
          @click="$emit('suggest', suggestion)"
        >
          {{ suggestion }}<i aria-hidden="true">→</i>
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  response: {
    type: Object,
    required: true,
  },
});

defineEmits(["suggest"]);

const kind = computed(() => String(props.response?.response_kind || "guidance"));
const tone = computed(() => ({
  refusal: "refusal",
  clarification: "clarification",
})[kind.value] || "guidance");
const title = computed(() => ({
  refusal: "该操作无法执行",
  clarification: "需要补充信息",
})[kind.value] || "回到业务数据分析");
const glyph = computed(() => ({
  refusal: "!",
  clarification: "?",
})[kind.value] || "↗");
const suggestionLabel = computed(() => (
  kind.value === "refusal" ? "可以改为只读查询" : "你可以这样问"
));
const categoryLabel = computed(() => ({
  business_help: "使用帮助",
  out_of_scope: "业务范围外",
  forbidden_operation: "只读安全边界",
  ambiguous: "意图待确认",
})[String(props.response?.category || "")] || "业务问数");
const suggestions = computed(() => [
  ...new Set(
    (Array.isArray(props.response?.suggestions) ? props.response.suggestions : [])
      .map((item) => String(item || "").trim())
      .filter(Boolean),
  ),
].slice(0, 6));
</script>

<style scoped>
.intent-response-card {
  width: 100%;
  display: flex;
  gap: 13px;
  padding: 17px;
  border: 1px solid rgba(52, 120, 246, .2);
  border-radius: 14px;
  background: linear-gradient(145deg, #f8fbff, #f1f7ff);
  box-shadow: var(--da-shadow-sm);
}
.intent-response-card.refusal {
  border-color: rgba(196, 125, 33, .27);
  background: linear-gradient(145deg, #fffdf8, #fff7e9);
}
.intent-response-card.clarification {
  border-color: rgba(112, 88, 176, .22);
  background: linear-gradient(145deg, #fcfbff, #f6f2ff);
}
.intent-response-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  display: grid;
  place-items: center;
  border-radius: 11px;
  background: #dfeeff;
  color: #2869c5;
  font-size: 16px;
  font-weight: 800;
}
.refusal .intent-response-icon { background: #ffebc9; color: #9c641b; }
.clarification .intent-response-icon { background: #ece4ff; color: #6d50ab; }
.intent-response-body { min-width: 0; flex: 1; }
.intent-response-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.intent-response-heading strong { color: var(--da-text); font-size: 14px; }
.intent-response-heading span {
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, .76);
  color: var(--da-text-muted);
  font-size: 9px;
  white-space: nowrap;
}
.intent-response-body p {
  margin: 8px 0 0;
  color: var(--da-text-secondary);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}
.intent-response-suggestions { display: grid; gap: 7px; margin-top: 13px; }
.intent-response-suggestions > span { color: var(--da-text-muted); font-size: 10px; }
.intent-response-suggestions button {
  min-height: 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 11px;
  border: 1px solid rgba(52, 120, 246, .16);
  border-radius: 9px;
  background: rgba(255, 255, 255, .84);
  color: var(--da-text-secondary);
  font: 11px/1.45 inherit;
  text-align: left;
  cursor: pointer;
}
.intent-response-suggestions button:hover {
  border-color: var(--da-accent);
  color: var(--da-accent);
}
.intent-response-suggestions i { font-style: normal; }
@media (max-width: 620px) {
  .intent-response-card { padding: 14px; }
  .intent-response-heading { align-items: flex-start; }
  .intent-response-heading span { display: none; }
}
</style>
