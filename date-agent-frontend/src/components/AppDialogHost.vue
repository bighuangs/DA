<template>
  <Teleport to="body">
    <div
      v-if="state.open"
      class="app-dialog-mask"
      data-testid="app-dialog-mask"
      @click.self="cancel"
      @keydown.esc.stop.prevent="cancel"
    >
      <section
        ref="dialogElement"
        class="app-dialog"
        :class="`tone-${state.tone}`"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="titleId"
        :aria-describedby="state.message ? descriptionId : undefined"
        tabindex="-1"
        @keydown.tab="trapFocus"
      >
        <div class="app-dialog-copy">
          <span class="app-dialog-icon" aria-hidden="true">{{ state.tone === "danger" ? "!" : "✓" }}</span>
          <div>
            <h2 :id="titleId">{{ state.title }}</h2>
            <p v-if="state.message" :id="descriptionId">{{ state.message }}</p>
          </div>
        </div>

        <form v-if="state.mode === 'prompt'" @submit.prevent="submit">
          <label :for="inputId">{{ state.label }}</label>
          <input
            :id="inputId"
            ref="inputElement"
            v-model="state.inputValue"
            data-testid="app-dialog-input"
            :placeholder="state.placeholder"
            autocomplete="off"
          />
        </form>

        <footer>
          <button
            v-if="state.mode !== 'alert'"
            ref="cancelButton"
            type="button"
            class="secondary"
            @click="cancel"
          >
            {{ state.cancelLabel }}
          </button>
          <button
            ref="confirmButton"
            type="button"
            class="primary"
            :class="{ danger: state.tone === 'danger' }"
            @click="submit"
          >
            {{ state.confirmLabel }}
          </button>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import {
  appDialogState as state,
  cancelAppDialog,
  resolveAppDialog,
} from "../composables/appDialog.js";

const dialogElement = ref(null);
const inputElement = ref(null);
const cancelButton = ref(null);
const confirmButton = ref(null);
const titleId = computed(() => `app-dialog-title-${state.requestId}`);
const descriptionId = computed(() => `app-dialog-description-${state.requestId}`);
const inputId = computed(() => `app-dialog-input-${state.requestId}`);

let returnFocusTo = null;
let previousBodyOverflow = "";

function submit() {
  if (state.mode === "prompt") resolveAppDialog(state.inputValue);
  else if (state.mode === "confirm") resolveAppDialog(true);
  else resolveAppDialog(undefined);
}

function cancel() {
  cancelAppDialog();
}

function focusableElements() {
  if (!dialogElement.value) return [];
  return [...dialogElement.value.querySelectorAll(
    'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
  )].filter((element) => !element.hidden && element.getClientRects().length > 0);
}

function trapFocus(event) {
  const elements = focusableElements();
  if (!elements.length) {
    event.preventDefault();
    dialogElement.value?.focus();
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

watch(() => state.open, async (open) => {
  if (open) {
    returnFocusTo = document.activeElement;
    previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    await nextTick();
    if (state.mode === "prompt") {
      inputElement.value?.focus();
      inputElement.value?.select();
    } else if (state.mode === "alert") {
      confirmButton.value?.focus();
    } else {
      cancelButton.value?.focus();
    }
    return;
  }

  document.body.style.overflow = previousBodyOverflow;
  await nextTick();
  if (returnFocusTo?.isConnected) returnFocusTo.focus();
  returnFocusTo = null;
}, { flush: "post" });

onBeforeUnmount(() => {
  document.body.style.overflow = previousBodyOverflow;
});
</script>

<style scoped>
.app-dialog-mask{position:fixed;z-index:1200;inset:0;display:grid;place-items:center;padding:18px;background:rgba(25,31,41,.46);backdrop-filter:blur(5px)}
.app-dialog{width:min(460px,100%);overflow:hidden;border:1px solid rgba(255,255,255,.78);border-radius:18px;background:var(--da-bg,#fff);box-shadow:0 24px 72px rgba(31,43,61,.28);outline:0}
.app-dialog-copy{display:flex;align-items:flex-start;gap:12px;padding:21px 22px 12px}
.app-dialog-icon{display:grid;flex:0 0 28px;width:28px;height:28px;place-items:center;border-radius:9px;background:var(--da-accent-soft,#edf4ff);color:var(--da-accent,#3478f6);font-size:14px;font-weight:800}
.tone-danger .app-dialog-icon{background:#fff0ee;color:var(--da-danger,#d84a3a)}
h2{margin:2px 0 0;color:var(--da-text,#1f2329);font-size:17px;line-height:1.35}
p{margin:7px 0 0;color:var(--da-text-muted,#7b8494);font-size:12px;line-height:1.65;white-space:pre-line}
form{padding:2px 22px 18px}
label{display:block;color:var(--da-text-secondary,#4d5665);font-size:12px}
input{display:block;box-sizing:border-box;width:100%;min-height:40px;margin-top:7px;padding:9px 11px;border:1px solid var(--da-border,#cfd5df);border-radius:10px;background:#fbfcfe;color:var(--da-text,#1f2329);font:13px inherit;outline:0}
input:focus{border-color:var(--da-accent,#3478f6);background:#fff;box-shadow:0 0 0 3px var(--da-accent-soft,#edf4ff)}
footer{display:flex;justify-content:flex-end;gap:9px;padding:13px 22px;border-top:1px solid var(--da-border-light,#e8ebf0);background:var(--da-bg-soft,#f7f8fa)}
button{min-width:76px;min-height:36px;padding:8px 14px;border:1px solid var(--da-border,#cfd5df);border-radius:9px;background:#fff;color:var(--da-text-secondary,#4d5665);font:12px inherit;cursor:pointer}
button:focus-visible{outline:3px solid var(--da-accent-soft,#edf4ff);outline-offset:2px}
button.primary{border-color:var(--da-accent,#3478f6);background:var(--da-accent,#3478f6);color:#fff}
button.primary.danger{border-color:var(--da-danger,#d84a3a);background:var(--da-danger,#d84a3a)}
@media(max-width:520px){.app-dialog-mask{align-items:end;padding:10px}.app-dialog{border-radius:18px 18px 12px 12px}.app-dialog-copy{padding:19px 18px 11px}form{padding:2px 18px 16px}footer{padding:12px 18px}}
</style>
