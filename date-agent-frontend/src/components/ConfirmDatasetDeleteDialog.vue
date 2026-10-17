<template>
  <div class="modal-mask" @click.self="cancel" @keydown.esc.stop="cancel">
    <section
      ref="dialogElement"
      class="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dataset-delete-title"
      aria-describedby="dataset-delete-description"
      tabindex="-1"
    >
      <h2 id="dataset-delete-title">删除上传版本？</h2>
      <p id="dataset-delete-description">
        删除后，该版本将立即退出可查询范围；正在引用它的数据上下文也会在下一次校验时失效。
      </p>
      <div class="dataset-summary">
        <span class="dataset-icon" aria-hidden="true">▦</span>
        <span>
          <strong>{{ dataset?.filename || "上传数据" }}</strong>
          <small>
            v{{ dataset?.version_no || 1 }} ·
            {{ dataset?.row_count || 0 }} 行 ·
            {{ dataset?.column_count || 0 }} 列
          </small>
        </span>
      </div>
      <footer>
        <button ref="cancelButton" type="button" :disabled="busy" @click="cancel">取消</button>
        <button type="button" class="danger" :disabled="busy" @click="$emit('confirm')">
          {{ busy ? "删除中…" : "确认删除" }}
        </button>
      </footer>
    </section>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref } from "vue";

const props = defineProps({
  dataset: { type: Object, default: null },
  busy: { type: Boolean, default: false },
});
const emit = defineEmits(["cancel", "confirm"]);
const dialogElement = ref(null);
const cancelButton = ref(null);

function cancel() {
  if (!props.busy) emit("cancel");
}

onMounted(() => nextTick(() => cancelButton.value?.focus() || dialogElement.value?.focus()));
</script>

<style scoped>
.modal-mask{position:fixed;z-index:1000;inset:0;display:grid;place-items:center;padding:16px;background:rgba(31,35,41,.4);backdrop-filter:blur(4px)}
.modal{width:min(430px,100%);overflow:hidden;border:1px solid rgba(255,255,255,.7);border-radius:17px;background:var(--da-bg);box-shadow:var(--da-shadow-lg);outline:0}
h2{margin:0;padding:18px 20px 5px;color:var(--da-text);font-size:17px}
p{margin:0;padding:0 20px 14px;color:var(--da-text-muted);font-size:12px;line-height:1.65}
.dataset-summary{display:flex;align-items:center;gap:11px;margin:0 20px 18px;padding:13px 14px;border:1px solid var(--da-border-light);border-radius:11px;background:var(--da-bg-soft)}
.dataset-icon{display:grid;width:30px;height:30px;place-items:center;border-radius:9px;background:#fff0ee;color:var(--da-danger);font-size:17px}
.dataset-summary>span:last-child{min-width:0}
strong,small{display:block}
strong{overflow:hidden;color:var(--da-text);font-size:13px;text-overflow:ellipsis;white-space:nowrap}
small{margin-top:4px;color:var(--da-text-muted);font-size:11px}
footer{display:flex;justify-content:flex-end;gap:8px;padding:12px 20px;border-top:1px solid var(--da-border-light)}
button{min-height:34px;padding:7px 12px;border:1px solid var(--da-border);border-radius:8px;background:#fff;color:var(--da-text-secondary);font:12px inherit;cursor:pointer}
button:focus-visible{outline:3px solid var(--da-accent-soft);outline-offset:2px}
button:disabled{cursor:wait;opacity:.65}
button.danger{border-color:var(--da-danger);background:var(--da-danger);color:#fff}
</style>
