function normalizePath(loc) {
  if (!Array.isArray(loc)) return "";
  const parts = loc.filter((part) => !["body", "query", "path"].includes(String(part)));
  return parts.length ? parts.join(".") : "";
}

function objectMessage(value) {
  const preferred = ["message", "msg", "error", "reason", "title"];
  for (const key of preferred) {
    if (typeof value?.[key] === "string" && value[key].trim()) {
      const path = normalizePath(value.loc);
      return path ? `${path}：${value[key].trim()}` : value[key].trim();
    }
  }
  const entries = Object.entries(value || {})
    .filter(([key]) => !["loc", "type", "ctx", "input", "url"].includes(key))
    .map(([key, item]) => {
      const message = apiErrorMessage(item, "");
      return message ? `${key}：${message}` : "";
    })
    .filter(Boolean);
  return entries.join("；");
}

export function apiErrorMessage(detail, fallback = "请求失败") {
  if (detail === null || detail === undefined || detail === "") return fallback;
  if (typeof detail === "string") return detail.trim() || fallback;
  if (typeof detail === "number" || typeof detail === "boolean") return String(detail);
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => apiErrorMessage(item, "")).filter(Boolean);
    return messages.length ? messages.join("；") : fallback;
  }
  if (typeof detail === "object") {
    const nested = detail.detail ?? detail.errors;
    if (nested !== undefined && nested !== detail) {
      return apiErrorMessage(nested, fallback);
    }
    const message = objectMessage(detail);
    if (message) return message;
    try {
      const serialized = JSON.stringify(detail);
      return serialized && serialized !== "{}" ? serialized : fallback;
    } catch {
      return fallback;
    }
  }
  return fallback;
}

export async function responseError(response, fallback = "请求失败") {
  const statusFallback = `${fallback}（${response.status}）`;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const body = await response.json().catch(() => null);
    return apiErrorMessage(body, statusFallback);
  }
  const text = await response.text().catch(() => "");
  return apiErrorMessage(text, statusFallback);
}

export async function requestJson(url, options = {}, fallback = "请求失败") {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) throw new Error(await responseError(response, fallback));
  if (response.status === 204) return null;
  return response.json();
}
