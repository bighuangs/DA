<template>
  <section class="workspace-page">
    <header class="workspace-head">
      <div>
        <div class="eyebrow">ADMIN / WORKSPACE</div>
        <h1>工作空间与成员</h1>
        <p>隔离数据、元知识和运营证据，并为每位成员分配最小必要角色。</p>
      </div>
      <button class="btn with-icon" @click="load"><RefreshCw :size="15" aria-hidden="true" />刷新</button>
    </header>

    <div v-if="error" class="alert error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-else-if="notice" class="alert notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>

    <section class="current-workspace" aria-label="当前工作空间">
      <span class="current-icon"><Building2 :size="20" aria-hidden="true" /></span>
      <span>
        <small>当前工作空间</small>
        <strong>{{ session?.workspace?.name || "—" }}</strong>
      </span>
      <span class="current-role">{{ roleLabel(session?.workspace?.role) }}</span>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-head">
          <div><h2>我的工作空间</h2><p>{{ workspaces.length }} 个可访问空间</p></div>
        </div>
        <div class="workspace-list">
          <article
            v-for="item in workspaces"
            :key="item.id"
            class="workspace-row"
            :class="{ active: item.id === session?.workspace?.id }"
          >
            <span class="row-icon"><Building2 :size="16" aria-hidden="true" /></span>
            <span class="row-main"><strong>{{ item.name }}</strong><small>{{ roleLabel(item.role) }}</small></span>
            <span v-if="item.id === session?.workspace?.id" class="active-badge">当前</span>
            <button v-else class="btn small" @click="switchWorkspace(item.id)">切换</button>
          </article>
        </div>
        <details class="inline-action">
          <summary><Plus :size="15" aria-hidden="true" />创建新的工作空间</summary>
          <div class="create">
            <input v-model.trim="newWorkspace" aria-label="新工作空间名称" placeholder="例如：销售分析组" />
            <button class="btn primary" :disabled="!newWorkspace" @click="createWorkspace">创建</button>
          </div>
        </details>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><h2>成员</h2><p>{{ members.length }} 位成员 · 角色变更对新请求立即生效</p></div>
        </div>
        <div class="role-guide" aria-label="角色能力说明">
          <span><Eye :size="14" aria-hidden="true" /><strong>查看者</strong><small>查询与查看</small></span>
          <span><ChartColumn :size="14" aria-hidden="true" /><strong>分析员</strong><small>可编辑看板</small></span>
          <span><ShieldCheck :size="14" aria-hidden="true" /><strong>管理员</strong><small>治理与成员</small></span>
        </div>
        <details v-if="canManage" class="inline-action invite-action">
          <summary><UserPlus :size="15" aria-hidden="true" />邀请已注册成员</summary>
          <div class="invite">
            <input v-model.trim="invite.username" aria-label="已注册用户名" placeholder="用户名，例如 bob" />
            <select v-model="invite.role" aria-label="成员角色">
              <option value="viewer">查看者</option>
              <option value="analyst">分析员</option>
              <option value="admin">管理员</option>
            </select>
            <button class="btn primary" @click="addMember">添加成员</button>
          </div>
        </details>
        <div class="member-list">
          <article v-for="item in members" :key="item.user_id" class="member-row">
            <span class="member-avatar">{{ String(item.display_name || item.username || "?").slice(0, 1).toUpperCase() }}</span>
            <span class="row-main">
              <strong>{{ item.display_name || item.username }}</strong>
              <small>@{{ item.username }}</small>
            </span>
            <select
              v-if="canManage && item.role !== 'owner' && item.user_id !== session?.user?.id"
              :value="item.role"
              :aria-label="`设置 ${item.username} 的角色`"
              @change="updateRole(item.user_id, $event.target.value)"
            >
              <option value="viewer">查看者</option>
              <option value="analyst">分析员</option>
              <option value="admin">管理员</option>
            </select>
            <span v-else class="role">{{ roleLabel(item.role) }}</span>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import {
  Building2,
  ChartColumn,
  CircleAlert,
  CircleCheck,
  Eye,
  Plus,
  RefreshCw,
  ShieldCheck,
  UserPlus,
} from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const props = defineProps({ session: Object });
const emit = defineEmits(["session-changed"]);
const workspaces = ref([]);
const members = ref([]);
const error = ref("");
const notice = ref("");
const newWorkspace = ref("");
const invite = reactive({ username: "", role: "viewer" });
const canManage = computed(() => ["owner", "admin"].includes(props.session?.workspace?.role));
let noticeTimer = null;

async function request(path, options = {}) {
  return requestJson(path, options, "工作空间请求失败");
}

function showNotice(message) {
  if (noticeTimer) window.clearTimeout(noticeTimer);
  notice.value = message;
  noticeTimer = window.setTimeout(() => {
    notice.value = "";
    noticeTimer = null;
  }, 3000);
}

async function load() {
  error.value = "";
  try {
    [workspaces.value, members.value] = await Promise.all([
      request("/api/workspaces"),
      request("/api/workspaces/current/members"),
    ]);
  } catch (requestError) {
    error.value = requestError.message || "工作空间加载失败";
  }
}

async function createWorkspace() {
  if (!newWorkspace.value) return;
  error.value = "";
  try {
    const name = newWorkspace.value;
    await request("/api/workspaces", { method: "POST", body: JSON.stringify({ name }) });
    newWorkspace.value = "";
    await load();
    showNotice(`工作空间“${name}”已创建`);
  } catch (requestError) {
    error.value = requestError.message || "创建失败";
  }
}

async function switchWorkspace(id) {
  error.value = "";
  try {
    const data = await request("/api/auth/switch-workspace", {
      method: "POST",
      body: JSON.stringify({ workspace_id: id }),
    });
    localStorage.setItem("da_access_token", data.access_token);
    emit("session-changed", data);
  } catch (requestError) {
    error.value = requestError.message || "切换失败";
  }
}

async function addMember() {
  if (!invite.username) {
    error.value = "请输入已注册用户名，例如 bob";
    return;
  }
  if (invite.username.length < 3) {
    error.value = "用户名至少需要 3 个字符";
    return;
  }
  error.value = "";
  try {
    const username = invite.username;
    await request("/api/workspaces/current/members", {
      method: "POST",
      body: JSON.stringify(invite),
    });
    invite.username = "";
    await load();
    showNotice(`成员 @${username} 已添加`);
  } catch (requestError) {
    error.value = requestError.message || "添加成员失败";
  }
}

async function updateRole(id, role) {
  error.value = "";
  try {
    await request(`/api/workspaces/current/members/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    });
    await load();
    showNotice("成员角色已更新");
  } catch (requestError) {
    error.value = requestError.message || "角色更新失败";
  }
}

function roleLabel(role) {
  return ({ owner: "所有者", admin: "管理员", analyst: "分析员", viewer: "查看者" })[role] || role;
}

onMounted(load);
</script>

<style scoped>
.workspace-page { flex: 1; overflow: auto; padding: 38px 34px 64px; color: var(--da-text); }
.workspace-head,
.alert,
.current-workspace,
.grid { max-width: 1180px; margin-left: auto; margin-right: auto; }
.workspace-head { margin-bottom: 18px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; }
.eyebrow { color: var(--da-accent); font-size: 10px; font-weight: 750; letter-spacing: .17em; }
h1 { margin: 8px 0 6px; font-size: 29px; letter-spacing: -.04em; }
p { margin: 0; color: var(--da-text-muted); font-size: 12px; }
.btn {
  min-height: 36px;
  padding: 8px 13px;
  border: 1px solid var(--da-border);
  border-radius: 9px;
  background: #fff;
  color: var(--da-text-secondary);
  transition: .15s;
}
.btn.with-icon { display: inline-flex; align-items: center; gap: 6px; }
.btn:hover:not(:disabled) { border-color: var(--da-accent); background: var(--da-accent-soft); color: var(--da-accent); }
.btn.primary { border-color: var(--da-accent); background: var(--da-accent); color: #fff; box-shadow: 0 6px 14px rgba(52,120,246,.16); }
.btn.primary:hover:not(:disabled) { background: var(--da-accent-hover); color: #fff; }
.btn:disabled { opacity: .55; }
.btn.small { min-height: 30px; padding: 5px 9px; font-size: 11px; }
.alert { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; padding: 11px 13px; border: 1px solid; border-radius: 10px; font-size: 12px; }
.alert.error { border-color: rgba(214,69,69,.22); background: rgba(214,69,69,.07); color: #b83d3d; }
.alert.notice { border-color: #b9e2c8; background: #f0fbf4; color: #287b4a; }
.current-workspace {
  display: flex;
  align-items: center;
  gap: 11px;
  margin-bottom: 18px;
  padding: 13px 15px;
  border: 1px solid rgba(36,99,212,.13);
  border-radius: 13px;
  background: rgba(238,245,255,.75);
}
.current-icon { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 11px; background: #fff; color: var(--da-accent); }
.current-workspace > span:nth-child(2) { min-width: 0; flex: 1; }
.current-workspace small, .current-workspace strong { display: block; }
.current-workspace small { margin-bottom: 3px; color: var(--da-text-muted); font-size: 9px; }
.current-workspace strong { font-size: 13px; }
.current-role { padding: 4px 8px; border-radius: 999px; background: #fff; color: var(--da-accent-strong); font-size: 10px; }
.grid { display: grid; grid-template-columns: .9fr 1.35fr; gap: 18px; }
.panel { padding: 21px; border: 1px solid var(--da-border-light); border-radius: 16px; background: rgba(255,255,255,.94); box-shadow: var(--da-shadow-sm); }
.panel-head { display: flex; justify-content: space-between; margin-bottom: 12px; }
.panel h2 { margin: 0 0 4px; font-size: 16px; }
.workspace-list, .member-list { display: grid; }
.workspace-row, .member-row { min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 10px; border-bottom: 1px solid var(--da-border-light); border-radius: 9px; }
.workspace-row.active { border-bottom-color: transparent; background: var(--da-accent-soft); }
.row-icon { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 8px; background: var(--da-bg-soft); color: var(--da-text-muted); }
.row-main { min-width: 0; flex: 1; }
.row-main strong, .row-main small { display: block; }
.row-main strong { overflow: hidden; font-size: 13px; text-overflow: ellipsis; white-space: nowrap; }
.row-main small { margin-top: 3px; color: var(--da-text-muted); font-size: 10px; }
.active-badge, .role { padding: 4px 8px; border-radius: 999px; background: #fff; color: var(--da-text-muted); font-size: 10px; }
.active-badge { color: var(--da-accent-strong); }
.inline-action { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--da-border-light); }
.inline-action summary { display: inline-flex; align-items: center; gap: 6px; color: var(--da-accent); cursor: pointer; font-size: 12px; font-weight: 600; list-style: none; }
.inline-action summary::-webkit-details-marker { display: none; }
.create, .invite { display: flex; gap: 8px; margin-top: 12px; }
.create input, .invite input, .invite select, .member-row select {
  min-width: 0;
  min-height: 38px;
  flex: 1;
  padding: 8px 10px;
  border: 1px solid var(--da-border);
  border-radius: 9px;
  background: #fbfcfe;
  color: var(--da-text);
}
.invite select, .member-row select { flex: 0 0 108px; }
.role-guide { display: grid; grid-template-columns: repeat(3,1fr); gap: 7px; margin-bottom: 8px; }
.role-guide > span { min-width: 0; display: grid; grid-template-columns: 18px 1fr; gap: 2px 5px; padding: 9px; border-radius: 9px; background: var(--da-bg-soft); color: var(--da-accent); }
.role-guide strong { color: var(--da-text-secondary); font-size: 10px; }
.role-guide small { grid-column: 2; color: var(--da-text-muted); font-size: 9px; }
.invite-action { margin-bottom: 8px; }
.member-avatar { width: 34px; height: 34px; flex: 0 0 34px; display: grid; place-items: center; border-radius: 10px; background: #eaf2ff; color: var(--da-accent-strong); font-size: 12px; font-weight: 750; }

@media (max-width: 800px) {
  .workspace-page { padding: 25px 15px 44px; }
  .grid { grid-template-columns: 1fr; }
  .workspace-head { align-items: flex-start; }
}
@media (max-width: 520px) {
  .workspace-head { display: block; }
  .workspace-head .btn { margin-top: 13px; }
  .panel { padding: 17px; }
  .role-guide { grid-template-columns: 1fr; }
  .invite, .create { flex-wrap: wrap; }
  .invite input, .create input { flex-basis: 100%; }
  .invite select { flex: 1; }
}
</style>
