<template>
  <section class="access-page">
    <div class="access-head">
      <div>
        <div class="eyebrow">ADMIN / DATA ACCESS</div>
        <h1>数据权限</h1>
        <p>按角色或成员配置表访问、允许列和行过滤条件；查询生成与实际执行都会再次校验。</p>
      </div>
      <button class="btn primary with-icon" @click="openCreate"><Plus :size="15" aria-hidden="true" />新增策略</button>
    </div>
    <div v-if="error" class="error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-if="notice" class="notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <div class="policy-note">
      <strong>默认行为</strong><span>没有匹配策略时保持原有工作空间表白名单。</span>
      <strong>行过滤占位符</strong><code v-for="placeholder in placeholders" :key="placeholder">{{ placeholder }}</code>
      <span>允许列留空表示该表全部列；拒绝表策略优先生效。</span>
    </div>
    <div v-if="loading" class="empty">正在加载数据权限策略…</div>
    <div v-else class="policy-list">
      <article v-for="item in policies" :key="item.id" class="policy-card">
        <div class="policy-main">
          <div class="policy-title"><strong>{{ item.table_pattern }}</strong><span :class="['badge', item.access ? 'allow' : 'deny']">{{ item.access ? '允许' : '拒绝' }}</span><span v-if="!item.enabled" class="badge disabled">已停用</span></div>
          <div class="policy-scope">作用于：{{ scopeLabel(item) }} · 优先级 {{ item.priority }}</div>
          <div v-if="item.allowed_columns?.length" class="policy-row"><label>允许列</label><code v-for="column in item.allowed_columns" :key="column">{{ column }}</code></div>
          <div v-if="item.row_filter" class="policy-row"><label>行过滤</label><code class="filter-code">{{ item.row_filter }}</code></div>
          <p v-if="item.description">{{ item.description }}</p>
        </div>
        <div class="policy-actions"><button class="status" @click="toggle(item)">{{ item.enabled ? '停用' : '启用' }}</button><button class="status" @click="openEdit(item)">编辑</button><button class="status danger" @click="remove(item)">删除</button></div>
      </article>
      <div v-if="!policies.length" class="empty">
        <ShieldCheck :size="30" aria-hidden="true" />
        <strong>当前工作空间还没有细粒度策略</strong>
        <p>现有查询仍受工作空间表白名单、只读 SQL 和脱敏规则保护。需要限制某类角色的表、列或数据行时再添加策略。</p>
        <button class="btn primary with-icon" @click="openCreate"><Plus :size="15" aria-hidden="true" />创建第一条策略</button>
      </div>
    </div>

    <div v-if="modal.open" class="modal-mask" @click.self="closeModal">
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="access-policy-dialog-title">
        <div class="modal-head"><div><h2 id="access-policy-dialog-title">{{ modal.id ? '编辑' : '新增' }}数据权限策略</h2><p>策略保存后立即作用于新的查询请求。</p></div><button class="close" aria-label="关闭数据权限策略弹窗" @click="closeModal"><X :size="20" aria-hidden="true" /></button></div>
        <form @submit.prevent="save">
          <div class="form-grid"><label>角色<select v-model="form.role"><option value="all">全部角色</option><option value="viewer">查看者</option><option value="analyst">分析员</option><option value="admin">管理员</option><option value="owner">所有者</option></select></label><label>成员 ID（可选）<input v-model.trim="form.user_id" placeholder="留空表示按角色" /></label></div>
          <label>表匹配规则<input v-model.trim="form.table_pattern" required placeholder="例如：fact_order 或 fact_*" /></label>
          <label>允许列（可选）<input v-model="form.allowed_columns_text" placeholder="例如：id, region, amount；留空表示全部列" /></label>
          <label>行过滤条件（可选）<textarea v-model.trim="form.row_filter" rows="3" placeholder="例如：tenant_id = {{user_id}} 或 region IN ('华东','华南')"></textarea></label>
          <label>说明<textarea v-model.trim="form.description" rows="2" placeholder="说明该策略的业务范围和原因"></textarea></label>
          <div class="form-grid"><label>优先级<input v-model.number="form.priority" type="number" min="0" max="10000" /></label><label class="check"><input v-model="form.access" type="checkbox" /> 允许访问匹配表</label><label class="check"><input v-model="form.enabled" type="checkbox" /> 保存后启用</label></div>
          <div class="modal-foot"><span>策略由后端 AST 校验，不依赖模型是否遵守提示。</span><button type="button" class="btn" @click="closeModal">取消</button><button class="btn primary" type="submit" :disabled="saving">{{ saving ? '保存中…' : '保存' }}</button></div>
        </form>
      </div>
    </div>
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from "vue";
import { CircleAlert, CircleCheck, Plus, ShieldCheck, X } from "@lucide/vue";
import { responseError } from "../utils/http.js";
import { confirmDialog } from "../composables/appDialog.js";

const API = "/api/admin/meta/access-policies";
const placeholders = ["{{user_id}}", "{{username}}", "{{workspace_id}}", "{{role}}"];
const policies = ref([]), loading = ref(false), saving = ref(false), error = ref(""), notice = ref("");
const modal = reactive({ open: false, id: null });
const form = reactive({ role: "all", user_id: "", table_pattern: "*", allowed_columns_text: "", row_filter: "", access: true, enabled: true, priority: 100, description: "" });

async function request(path = "", options = {}) { const response = await fetch(`${API}${path}`, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options }); if (!response.ok) throw new Error(await responseError(response, "数据权限请求失败")); return response.status === 204 ? null : response.json(); }
async function load() { loading.value = true; error.value = ""; try { policies.value = await request(); } catch (e) { error.value = e.message || "数据权限加载失败"; } finally { loading.value = false; } }
function showNotice(value) { notice.value = value; window.clearTimeout(showNotice.timer); showNotice.timer = window.setTimeout(() => { notice.value = ""; }, 3600); }
function reset() { Object.assign(form, { role: "all", user_id: "", table_pattern: "*", allowed_columns_text: "", row_filter: "", access: true, enabled: true, priority: 100, description: "" }); }
function openCreate() { reset(); modal.id = null; modal.open = true; }
function openEdit(item) { Object.assign(form, { role: item.role, user_id: item.user_id || "", table_pattern: item.table_pattern, allowed_columns_text: (item.allowed_columns || []).join(","), row_filter: item.row_filter || "", access: item.access, enabled: item.enabled, priority: item.priority, description: item.description || "" }); modal.id = item.id; modal.open = true; }
function closeModal() { modal.open = false; }
function columns() { return form.allowed_columns_text.split(",").map((item) => item.trim()).filter(Boolean); }
function payload() { return { role: form.role, user_id: form.user_id || null, table_pattern: form.table_pattern, allowed_columns: columns().length ? columns() : null, row_filter: form.row_filter || null, access: form.access, enabled: form.enabled, priority: Number(form.priority || 100), description: form.description }; }
async function save() { saving.value = true; error.value = ""; try { await request(modal.id ? `/${modal.id}` : "", { method: modal.id ? "PATCH" : "POST", body: JSON.stringify(payload()) }); closeModal(); await load(); showNotice("数据权限策略已保存，新查询将立即生效。"); } catch (e) { error.value = e.message || "保存失败"; } finally { saving.value = false; } }
async function toggle(item) { try { await request(`/${item.id}`, { method: "PATCH", body: JSON.stringify({ ...item, allowed_columns: item.allowed_columns?.length ? item.allowed_columns : null, row_filter: item.row_filter || null, enabled: !item.enabled }) }); await load(); } catch (e) { error.value = e.message || "状态更新失败"; } }
async function remove(item) {
  const confirmed = await confirmDialog({
    title: "删除数据权限策略？",
    message: `表匹配规则“${item.table_pattern}”将不再约束后续查询。`,
    confirmLabel: "删除策略",
    tone: "danger",
  });
  if (!confirmed) return;
  try {
    await request(`/${item.id}`, { method: "DELETE" });
    await load();
    showNotice("策略已删除。");
  } catch (e) {
    error.value = e.message || "删除失败";
  }
}
function scopeLabel(item) { return item.user_id ? `成员 ${item.user_id}` : ({ all: "全部角色", owner: "所有者", admin: "管理员", analyst: "分析员", viewer: "查看者" }[item.role] || item.role); }
onMounted(load);
</script>

<style scoped>
.access-page{flex:1 1 auto;min-height:0;overflow:auto;padding:30px 34px 55px;background:#f7f8fa;color:var(--da-text)}.access-head{max-width:1160px;margin:0 auto 22px;display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.eyebrow{color:var(--da-accent);font-size:11px;font-weight:700;letter-spacing:.14em}.access-head h1{margin:5px 0 0;font-size:27px}.access-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.btn{padding:9px 14px;border:1px solid var(--da-border);border-radius:8px;background:#fff;color:var(--da-text-secondary);font:inherit;font-size:13px}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.error,.notice,.policy-note,.policy-list{max-width:1160px;margin-left:auto;margin-right:auto}.error,.notice{margin-bottom:14px;padding:10px 13px;border-radius:8px;font-size:13px}.error{border:1px solid #f0c5bf;background:#fff3f1;color:#bd3b2c}.notice{border:1px solid #b9e2c8;background:#f0fbf4;color:#287b4a}.policy-note{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:18px;padding:12px 14px;border:1px solid #cfe0f4;border-radius:9px;background:#f4f8fd;color:var(--da-text-secondary);font-size:12px}.policy-note strong{color:var(--da-text)}.policy-note code,.policy-row code{padding:2px 6px;border-radius:4px;background:#e7f0fb;color:#5379a6;font:11px Consolas,monospace}.policy-list{display:grid;gap:10px}.policy-card{display:flex;justify-content:space-between;gap:20px;padding:17px 18px;border:1px solid var(--da-border-light);border-radius:10px;background:#fff;box-shadow:0 2px 9px rgba(31,35,41,.025)}.policy-main{min-width:0;flex:1}.policy-title{display:flex;align-items:center;gap:8px}.policy-title strong{font-size:15px}.badge{padding:2px 7px;border-radius:5px;font-size:11px}.badge.allow{background:#eaf9f0;color:#2e9a60}.badge.deny{background:#fff0ee;color:#bd3b2c}.badge.disabled{background:var(--da-bg-soft);color:var(--da-text-muted)}.policy-scope{margin-top:7px;color:var(--da-text-muted);font-size:12px}.policy-row{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:10px}.policy-row label{width:55px;color:var(--da-text-muted);font-size:12px}.filter-code{white-space:pre-wrap}.policy-main p{margin:11px 0 0;color:var(--da-text-secondary);font-size:12px}.policy-actions{display:flex;align-items:flex-start;gap:9px}.status{border:0;background:transparent;color:var(--da-accent);font:inherit;font-size:12px;white-space:nowrap;cursor:pointer}.status.danger{color:var(--da-danger)}.empty{padding:65px 20px;text-align:center;color:var(--da-text-muted);font-size:13px}.modal-mask{position:fixed;z-index:30;inset:0;display:grid;place-items:center;padding:20px;background:rgba(31,35,41,.35)}.modal{width:min(650px,100%);max-height:calc(100vh - 40px);overflow:auto;padding:23px;border-radius:13px;background:#fff;box-shadow:0 18px 60px rgba(31,35,41,.22)}.modal-head,.modal-foot{display:flex;align-items:center;justify-content:space-between;gap:15px}.modal-head{align-items:flex-start;margin-bottom:17px}.modal h2{margin:0;font-size:18px}.modal p{margin:6px 0 0;color:var(--da-text-muted);font-size:12px}.close{border:0;background:transparent;color:var(--da-text-muted);font-size:25px;cursor:pointer}.modal label{display:block;margin:13px 0;color:var(--da-text-secondary);font-size:12px}.modal input:not([type=checkbox]),.modal select,.modal textarea{display:block;width:100%;margin-top:6px;padding:9px 10px;border:1px solid var(--da-border);border-radius:7px;color:var(--da-text);font:inherit;font-size:13px}.modal textarea{resize:vertical}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.check{display:flex!important;align-items:center;gap:7px;padding-top:28px;color:var(--da-text)!important}.check input{width:15px;height:15px}.modal-foot{margin-top:20px;padding-top:15px;border-top:1px solid var(--da-border-light)}.modal-foot span{flex:1;color:var(--da-text-muted);font-size:11px;line-height:1.5}@media(max-width:750px){.access-page{padding:22px 15px 40px}.access-head{display:block}.access-head .btn{margin-top:14px}.policy-card{display:block}.policy-actions{margin-top:14px}.form-grid{grid-template-columns:1fr}.check{padding-top:0}.modal-foot{align-items:flex-end}}
/* visual refresh */
.access-page{padding:38px 34px 64px;background:transparent}.access-head{max-width:1240px;margin-bottom:25px}.access-head h1{margin-top:7px;font-size:29px;letter-spacing:-.04em}.eyebrow{font-size:10px;letter-spacing:.17em}.btn{min-height:38px;border-radius:10px;font-weight:550;transition:.15s}.btn:hover:not(:disabled){border-color:var(--da-accent);color:var(--da-accent);background:var(--da-accent-soft)}.btn.primary{box-shadow:0 7px 16px rgba(52,120,246,.17)}.btn.primary:hover:not(:disabled){color:#fff;background:var(--da-accent-hover)}.error,.notice,.policy-note,.policy-list{max-width:1240px}.error,.notice,.policy-note{border-radius:11px}.policy-note{padding:14px 16px;background:rgba(244,248,253,.88)}.policy-list{gap:12px}.policy-card{padding:20px;border-color:var(--da-border-light);border-radius:15px;box-shadow:var(--da-shadow-sm);transition:.15s}.policy-card:hover{box-shadow:var(--da-shadow-md);transform:translateY(-1px)}.policy-title strong{font-size:15px}.badge{padding:3px 8px;border-radius:999px}.policy-actions{padding-left:15px;border-left:1px solid var(--da-border-light)}.status{min-height:30px;padding:5px 7px;border-radius:7px}.status:hover{background:var(--da-accent-soft)}.empty{min-height:280px;display:grid;place-items:center;border:1px dashed var(--da-border);border-radius:15px;background:rgba(255,255,255,.6)}.modal-mask{backdrop-filter:blur(4px)}.modal{border:1px solid rgba(255,255,255,.75);border-radius:18px;box-shadow:var(--da-shadow-lg)}.modal input:not([type=checkbox]),.modal select,.modal textarea{border-radius:9px;background:#fbfcfe}.modal input:focus,.modal select:focus,.modal textarea:focus{border-color:var(--da-accent);background:#fff;box-shadow:0 0 0 3px var(--da-accent-soft)}@media(max-width:750px){.access-page{padding:25px 15px 44px}.policy-actions{padding-left:0;border-left:0}.modal{padding:19px}}
.btn.with-icon{display:inline-flex;align-items:center;justify-content:center;gap:6px}.error,.notice{display:flex;align-items:center;gap:8px}.empty{align-content:center;gap:9px}.empty strong{color:var(--da-text);font-size:15px}.empty p{max-width:580px;margin:0;color:var(--da-text-muted);font-size:12px;line-height:1.7}.empty .btn{margin-top:4px}.close{width:34px;height:34px;display:grid;place-items:center;border-radius:9px;background:var(--da-bg-soft);font-size:inherit}
</style>
