import { reactive } from "vue";

const initialState = () => ({
  open: false,
  requestId: 0,
  mode: "confirm",
  title: "",
  message: "",
  label: "",
  inputValue: "",
  placeholder: "",
  confirmLabel: "确认",
  cancelLabel: "取消",
  tone: "default",
});

export const appDialogState = reactive(initialState());

let nextRequestId = 0;
let activeRequest = null;
let closeTransitionPending = false;
const pendingRequests = [];

function showNextRequest() {
  if (closeTransitionPending || appDialogState.open || activeRequest || !pendingRequests.length) return;
  activeRequest = pendingRequests.shift();
  Object.assign(appDialogState, initialState(), activeRequest.options, {
    open: true,
    requestId: ++nextRequestId,
  });
}

function enqueue(options) {
  return new Promise((resolve) => {
    pendingRequests.push({ options, resolve });
    showNextRequest();
  });
}

export function confirmDialog({
  title = "请确认",
  message = "",
  confirmLabel = "确认",
  cancelLabel = "取消",
  tone = "default",
} = {}) {
  return enqueue({ mode: "confirm", title, message, confirmLabel, cancelLabel, tone });
}

export function promptDialog({
  title = "请输入内容",
  message = "",
  label = "内容",
  value = "",
  placeholder = "",
  confirmLabel = "确认",
  cancelLabel = "取消",
} = {}) {
  return enqueue({
    mode: "prompt",
    title,
    message,
    label,
    inputValue: String(value ?? ""),
    placeholder,
    confirmLabel,
    cancelLabel,
    tone: "default",
  });
}

export function alertDialog({
  title = "提示",
  message = "",
  confirmLabel = "知道了",
  tone = "default",
} = {}) {
  return enqueue({ mode: "alert", title, message, confirmLabel, tone });
}

export function resolveAppDialog(value) {
  if (!activeRequest) return;
  const request = activeRequest;
  activeRequest = null;
  closeTransitionPending = true;
  appDialogState.open = false;
  request.resolve(value);
  // await continuation 可能立刻 enqueue 下一个请求；关闭门闩确保它不能在
  // Vue 完成卸载与焦点恢复前同步重开，从而保留完整的 false -> true 周期。
  window.setTimeout(() => {
    closeTransitionPending = false;
    showNextRequest();
  }, 0);
}

export function cancelAppDialog() {
  if (appDialogState.mode === "confirm") resolveAppDialog(false);
  else if (appDialogState.mode === "prompt") resolveAppDialog(null);
  else resolveAppDialog(undefined);
}
