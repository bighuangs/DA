<template>
  <section class="repair-page">
    <header class="page-head">
      <div>
        <span class="eyebrow">EVALUATION / SEMANTIC GUARD</span>
        <h1>语义守卫与分层纠错</h1>
        <p>用稳定错误码、版本化规则和有界修复，把“为什么拦截、改了什么、能否回退”放在同一份证据里。</p>
      </div>
      <div class="head-actions">
        <button class="btn" type="button" :disabled="loading" @click="load">
          <RefreshCw :size="15" aria-hidden="true" />{{ loading ? "加载中…" : "刷新" }}
        </button>
        <button class="btn primary" type="button" @click="copyCommand">
          <Copy :size="15" aria-hidden="true" />复制评测命令
        </button>
      </div>
    </header>

    <p v-if="error" class="alert error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</p>
    <p v-if="notice" class="alert notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</p>
    <div v-if="loading && !report" class="loading">正在读取语义修复门禁证据…</div>

    <template v-else-if="report?.available">
      <section class="evidence-banner" data-testid="semantic-repair-evidence-boundary">
        <div class="evidence-icon"><ShieldCheck :size="21" aria-hidden="true" /></div>
        <div>
          <strong>P2 只完成治理迁移，不扩大线上自动修复范围</strong>
          <span>{{ report.decision.selection_reason }}</span>
        </div>
        <code>{{ report.evidence_level }}</code>
      </section>

      <section class="profile-hero">
        <article class="profile-card active-card" data-testid="semantic-active-profile">
          <header><span>当前线上 Active</span><b>P2 · 同规则注册表</b></header>
          <div class="profile-title"><span><BadgeCheck :size="22" /></span><div><strong>{{ profileLabel(runtimeProfile) }}</strong><code>{{ runtimeProfile }}</code></div></div>
          <p>既有连接修复与阈值投影规则已纳入版本合同；每条请求最多执行一轮并重新经过完整校验链。</p>
          <footer><span>Precision {{ percent(active.repair_precision) }}</span><span>误触发 {{ active.false_trigger_count || 0 }}</span><span>P95 {{ ms(active.repair_latency_ms?.p95) }}</span></footer>
        </article>
        <article class="profile-card fallback-card">
          <header><span>安全回退</span><b>P0 · 即时可用</b></header>
          <div class="profile-title"><span><RotateCcw :size="22" /></span><div><strong>{{ profileLabel(fallbackProfile) }}</strong><code>{{ fallbackProfile }}</code></div></div>
          <p>仅切换当前进程 profile，不迁移数据库、不重建索引；下一次请求立即按旧入口执行。</p>
          <button class="btn danger" type="button" :disabled="rollingBack || runtimeAtFallback" @click="rollback">
            <RotateCcw :size="14" aria-hidden="true" />{{ runtimeAtFallback ? "当前已在 P0" : rollingBack ? "回退中…" : "立即回退 P0" }}
          </button>
        </article>
      </section>

      <section class="summary-grid" aria-label="语义修复评测摘要">
        <article><span>受审样本</span><strong>{{ report.contract.total }}</strong><small>train / dev / test</small></article>
        <article><span>注册规则</span><strong>{{ report.registry.length }}</strong><small>active / candidate / shadow</small></article>
        <article class="good"><span>P2 修复精度</span><strong>{{ percent(active.repair_precision) }}</strong><small>{{ active.correct_repairs }}/{{ active.changed_cases }} 次命中正确</small></article>
        <article class="good"><span>近似样本误触发</span><strong>{{ active.false_trigger_count || 0 }}</strong><small>目标必须为 0</small></article>
        <article class="good"><span>幂等性</span><strong>{{ percent(active.idempotence_rate) }}</strong><small>重复执行不再改写</small></article>
        <article :class="gates.gate_passed ? 'good' : 'bad'"><span>实现门禁</span><strong>{{ gates.gate_passed ? "通过" : "阻断" }}</strong><small>不等于 P3–P6 发布批准</small></article>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><h2>四层纠错治理</h2><p>先拒绝不可自动修复的风险，再依次使用确定性规则、定向模型纠错和上游再生成。</p></div>
          <span class="pill neutral">单次有界</span>
        </div>
        <div class="governance-flow">
          <article v-for="(item, index) in report.governance_layers" :key="item.level">
            <span class="level">{{ item.level }}</span>
            <div><strong>{{ item.name }}</strong><small>{{ item.scope }}</small></div>
            <ArrowRight v-if="index < report.governance_layers.length - 1" :size="17" aria-hidden="true" />
          </article>
        </div>
      </section>

      <div class="two-column">
        <section class="panel registry-panel">
          <div class="panel-head"><div><h2>规则注册表</h2><p>规则码、版本、变更 AST 节点和运行模式均可追溯。</p></div><span class="pill active-pill">{{ report.runtime.rule_set_version }}</span></div>
          <div class="registry-list">
            <article v-for="rule in report.registry" :key="rule.rule_code">
              <span class="rule-mode" :class="rule.mode">{{ modeLabel(rule.mode) }}</span>
              <div><strong>{{ ruleLabel(rule.rule_code) }}</strong><code>{{ rule.rule_code }}@{{ rule.rule_version }}</code></div>
              <small>{{ rule.changed_ast_nodes.join(" · ") }}</small>
            </article>
          </div>
        </section>

        <section class="panel taxonomy-panel">
          <div class="panel-head"><div><h2>Violation 分类合同</h2><p>错误不再统一归为“SQL 校验失败”，路由只消费稳定 code。</p></div><span class="pill neutral">V1</span></div>
          <div class="taxonomy-grid">
            <article v-for="item in report.taxonomy" :key="item.layer">
              <span>{{ taxonomyLabel(item.layer) }}</span><code>{{ item.prefix }}.*</code><b :class="{ reject: item.default === 'reject' }">{{ item.default }}</b>
            </article>
          </div>
        </section>
      </div>

      <section class="panel">
        <div class="panel-head">
          <div><h2>P0–P6 分层消融</h2><p>P3–P6 即使离线更高召回，也必须经过真实 Full Run、影子流量和差评复核后才可发布。</p></div>
          <span class="pill warn">{{ report.decision.offline_candidates.length }} 个离线候选</span>
        </div>
        <div class="table-wrap" tabindex="0" aria-label="语义修复 P0 到 P6 消融表">
          <table aria-label="语义修复 P0 到 P6 消融表">
            <thead><tr><th>层级</th><th>Profile</th><th>规则数</th><th>Precision</th><th>Rescue</th><th>误触发</th><th>有害修复</th><th>幂等</th><th>安全</th><th>P95</th><th>状态</th></tr></thead>
            <tbody>
              <tr v-for="(item, key) in report.profiles" :key="key" :class="{ active: key === 'P2', fallback: key === 'P0' }">
                <td><span class="profile-key">{{ key }}</span></td>
                <td><strong>{{ profileLabel(item.profile) }}</strong><code>{{ item.profile }}</code></td>
                <td>{{ item.rules.length }}</td><td>{{ percent(item.repair_precision) }}</td><td>{{ percent(item.targeted_rescue_rate) }}</td>
                <td>{{ item.false_trigger_count }}</td><td>{{ item.harmful_repair_count }}</td><td>{{ percent(item.idempotence_rate) }}</td>
                <td>{{ percent(item.unsafe_unchanged_rate) }}</td><td>{{ ms(item.repair_latency_ms?.p95) }}</td>
                <td><span v-if="key === 'P2'" class="pill active-pill">线上 active</span><span v-else-if="key === 'P0'" class="pill fallback-pill">fallback</span><span v-else class="pill neutral">离线 / 影子</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="panel release-panel">
        <div class="panel-head"><div><h2>发布边界与证据缺口</h2><p>自动修复必须证明“修对了”，不能只证明 SQL 最终能执行。</p></div><span class="pill warn">P3–P6 BLOCKED</span></div>
        <div class="release-grid">
          <article v-for="item in report.decision.release_blockers" :key="item"><CircleDashed :size="16" aria-hidden="true" /><div><strong>{{ blockerLabel(item) }}</strong><small>{{ blockerHelp(item) }}</small></div></article>
        </div>
      </section>

      <section class="command-panel">
        <div><strong>重跑确定性语义修复消融</strong><small>运行后刷新本页；不会调用 SQLBot / STT，也不会自动切换 P3–P6。</small></div>
        <code>{{ command }}</code>
        <button type="button" aria-label="复制语义修复评测命令" @click="copyCommand"><Copy :size="15" /></button>
      </section>
    </template>

    <section v-else-if="report" class="empty">
      <Wrench :size="35" aria-hidden="true" /><h2>还没有语义修复评测报告</h2>
      <p>先在后端目录执行评测命令，再返回本页刷新。</p><code>{{ command }}</code>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import {
  ArrowRight, BadgeCheck, CircleAlert, CircleCheck, CircleDashed, Copy,
  RefreshCw, RotateCcw, ShieldCheck, Wrench,
} from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const command = "python -m app.eval.semantic_repair_ablation --enforce";
const report = ref(null);
const loading = ref(false);
const rollingBack = ref(false);
const error = ref("");
const notice = ref("");
const active = computed(() => report.value?.profiles?.P2 || {});
const gates = computed(() => report.value?.implementation_gates || {});
const runtimeProfile = computed(() => report.value?.runtime?.runtime_profile || report.value?.decision?.active_profile || "registry_same_rules_v1");
const fallbackProfile = computed(() => report.value?.runtime?.fallback_profile || report.value?.decision?.fallback_profile || "inline_v1");
const runtimeAtFallback = computed(() => runtimeProfile.value === fallbackProfile.value);

async function load() {
  loading.value = true;
  error.value = "";
  try {
    report.value = await requestJson("/api/admin/evaluation/semantic-repair/latest", {}, "语义修复报告加载失败");
  } catch (exception) {
    error.value = exception.message || "语义修复报告加载失败";
  } finally {
    loading.value = false;
  }
}
async function rollback() {
  rollingBack.value = true;
  error.value = "";
  try {
    const result = await requestJson("/api/admin/evaluation/semantic-repair/rollback", { method: "POST" }, "语义修复回退失败");
    report.value.runtime = result.runtime;
    report.value.decision.active_profile = result.runtime.runtime_profile;
    notice.value = result.message || "已回退 P0";
    window.setTimeout(() => { notice.value = ""; }, 3200);
  } catch (exception) {
    error.value = exception.message || "语义修复回退失败";
  } finally {
    rollingBack.value = false;
  }
}
async function copyCommand() {
  try {
    await navigator.clipboard.writeText(command);
    notice.value = "评测命令已复制";
    window.setTimeout(() => { notice.value = ""; }, 2200);
  } catch {
    error.value = "剪贴板不可用，请手动复制页面底部命令。";
  }
}
function profileLabel(value) {
  return ({
    inline_v1: "内联安全基线", violation_only_v1: "只诊断同规则",
    registry_same_rules_v1: "版本化同规则注册表", registry_extended_v1: "扩展确定性注册表",
    targeted_correction_v1: "定向模型纠错", upstream_first_v1: "上游优先再生成", cascade_v1: "分层级联候选",
  })[value] || value || "—";
}
function ruleLabel(value) {
  return ({
    "join.unique_metadata_edge": "唯一元数据连接边",
    "projection.threshold_measure": "阈值指标投影闭合",
    "ir.unique_limit_order": "唯一排序与条数约束",
    "ir.unique_group_by": "唯一分组维度（影子）",
  })[value] || value;
}
function taxonomyLabel(value) { return ({ syntax: "语法", semantic: "业务语义", semantic_ir: "语义 IR", metric: "指标口径", security: "安全权限", cost: "成本" })[value] || value; }
function modeLabel(value) { return ({ active: "ACTIVE", candidate: "CANDIDATE", shadow: "SHADOW" })[value] || value; }
function blockerLabel(value) {
  return ({
    missing_live_llm_targeted_correction_pairs: "缺少真实模型定向纠错对",
    missing_cold_warm_warm_full_run: "缺少 cold / warm / warm 三次 Full Run",
    missing_shadow_canary_and_negative_feedback_review: "缺少影子灰度与负反馈复核",
  })[value] || value;
}
function blockerHelp(value) {
  return ({
    missing_live_llm_targeted_correction_pairs: "需逐题比较修前、修后 SQL 与真实执行结果，不能只看语法成功。",
    missing_cold_warm_warm_full_run: "需同时守住逐题非回退、TTSQL、TTResult、Token 和失败调用。",
    missing_shadow_canary_and_negative_feedback_review: "先观察 would_apply 与误触发，再由人工复核差评样本。",
  })[value] || "完成对应证据后才可申请发布。";
}
function percent(value) { const number = Number(value); return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—"; }
function ms(value) { const number = Number(value); return Number.isFinite(number) ? `${number.toFixed(number < 1 ? 3 : 1)} ms` : "—"; }
onMounted(load);
</script>

<style scoped>
.repair-page{flex:1 1 auto;min-height:0;overflow:auto;padding:36px 34px 70px;color:var(--da-text)}.page-head,.evidence-banner,.profile-hero,.summary-grid,.panel,.command-panel,.alert,.empty{max-width:1320px;margin-left:auto;margin-right:auto}.page-head{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:20px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:780;letter-spacing:.17em}.page-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.page-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.head-actions{display:flex;gap:8px}.btn{min-height:38px;display:inline-flex;align-items:center;justify-content:center;gap:7px;padding:8px 13px;border:1px solid var(--da-border);border-radius:10px;background:#fff;color:var(--da-text-secondary);font:12px inherit;cursor:pointer}.btn:hover:not(:disabled){border-color:var(--da-accent);color:var(--da-accent)}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.btn.danger{border-color:#e4b8b2;color:#ac4134}.btn:disabled{cursor:not-allowed;opacity:.55}.alert{display:flex;align-items:center;gap:8px;margin-bottom:13px;padding:11px 14px;border-radius:10px;font-size:12px}.alert.error{background:#fff2ef;color:#b73f31}.alert.notice{background:#effaf3;color:#287b4a}.loading,.empty{min-height:380px;display:grid;place-content:center;text-align:center;color:var(--da-text-muted)}.empty h2{margin:13px 0 5px;color:var(--da-text);font-size:18px}.empty code{margin-top:10px;padding:9px 12px;border-radius:8px;background:#202634;color:#e8edf5}.evidence-banner{display:flex;align-items:center;gap:12px;margin-bottom:13px;padding:12px 14px;border:1px solid #cfe1f5;border-radius:13px;background:#f5f9ff}.evidence-icon{width:36px;height:36px;flex:0 0 36px;display:grid;place-items:center;border-radius:10px;background:#e4efff;color:#2869c5}.evidence-banner div:nth-child(2){min-width:0;flex:1}.evidence-banner strong,.evidence-banner span{display:block}.evidence-banner strong{font-size:12px}.evidence-banner span{margin-top:4px;color:var(--da-text-muted);font-size:10px;line-height:1.5}.evidence-banner code{padding:5px 8px;border-radius:7px;background:#e8f1ff;color:#315f9d;font-size:9px}.profile-hero{display:grid;grid-template-columns:1.35fr 1fr;gap:13px;margin-bottom:13px}.profile-card{padding:19px;border:1px solid var(--da-border-light);border-radius:16px;background:#fff;box-shadow:var(--da-shadow-sm)}.profile-card header,.profile-card footer{display:flex;align-items:center;justify-content:space-between;gap:10px}.profile-card header span,.profile-card p,.profile-card footer{color:var(--da-text-muted);font-size:10px}.profile-card header b{padding:4px 8px;border-radius:99px;font-size:9px}.active-card{border-color:#bed9f5;background:linear-gradient(145deg,#fff 35%,#f4f9ff)}.active-card header b{background:#e4efff;color:#2b65b5}.fallback-card{border-color:#ead9cf}.fallback-card header b{background:#fff0e6;color:#9a5a2e}.profile-title{display:flex;align-items:center;gap:11px;margin:14px 0 10px}.profile-title>span{width:44px;height:44px;display:grid;place-items:center;border-radius:13px;background:#e9f2ff;color:#2f6bc2}.fallback-card .profile-title>span{background:#fff1e9;color:#a66436}.profile-title strong,.profile-title code{display:block}.profile-title strong{font-size:17px}.profile-title code{margin-top:4px;color:var(--da-text-muted);font-size:9px}.profile-card p{min-height:31px;line-height:1.55}.profile-card footer{justify-content:flex-start;flex-wrap:wrap;padding-top:12px;border-top:1px solid var(--da-border-light)}.fallback-card .btn{margin-top:9px}.summary-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-bottom:16px}.summary-grid article{position:relative;overflow:hidden;padding:16px;border:1px solid var(--da-border-light);border-radius:13px;background:#fff;box-shadow:var(--da-shadow-sm)}.summary-grid article:before{position:absolute;inset:0 auto 0 0;width:3px;background:#96a3b5;content:""}.summary-grid article.good:before{background:#35a56a}.summary-grid article.bad:before{background:#dc6656}.summary-grid span,.summary-grid small{display:block;color:var(--da-text-muted);font-size:9px}.summary-grid strong{display:block;margin:7px 0 4px;font-size:20px;letter-spacing:-.03em}.panel{margin-bottom:16px;padding:21px;border:1px solid var(--da-border-light);border-radius:16px;background:rgba(255,255,255,.94);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:15px}.panel-head h2{margin:0;font-size:16px}.pill{display:inline-flex;align-items:center;padding:4px 7px;border-radius:99px;font-size:9px;white-space:nowrap}.pill.neutral{background:#eef1f5;color:#667287}.pill.active-pill{background:#e9f2ff;color:#2d65b7}.pill.fallback-pill{background:#fff0e7;color:#92582e}.pill.warn{background:#fff3d8;color:#8a641c}.governance-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.governance-flow article{position:relative;min-height:91px;display:flex;align-items:flex-start;gap:10px;padding:13px;border:1px solid var(--da-border-light);border-radius:12px;background:#fafbfd}.governance-flow article>svg{position:absolute;right:-14px;top:38px;z-index:2;color:#9aa8ba}.level{width:32px;height:32px;flex:0 0 32px;display:grid;place-items:center;border-radius:9px;background:#e9f2ff;color:#2e67b9;font-size:10px;font-weight:800}.governance-flow strong,.governance-flow small{display:block}.governance-flow strong{font-size:11px}.governance-flow small{margin-top:6px;color:var(--da-text-muted);font-size:9px;line-height:1.45}.two-column{max-width:1320px;display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin:0 auto}.registry-list{display:grid;gap:8px}.registry-list article{display:grid;grid-template-columns:74px minmax(0,1fr) auto;align-items:center;gap:10px;padding:10px 11px;border-radius:10px;background:#f8fafc}.rule-mode{padding:4px 6px;border-radius:6px;text-align:center;font-size:8px;font-weight:800}.rule-mode.active{background:#e7f5ed;color:#29774b}.rule-mode.candidate{background:#e8f1ff;color:#2f65af}.rule-mode.shadow{background:#f2ecfb;color:#7453a0}.registry-list strong,.registry-list code{display:block}.registry-list strong{font-size:10px}.registry-list code{margin-top:3px;color:var(--da-text-muted);font-size:8px}.registry-list small{color:var(--da-text-muted);font-size:8px}.taxonomy-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.taxonomy-grid article{display:grid;gap:5px;padding:11px;border:1px solid var(--da-border-light);border-radius:10px;background:#fafbfd}.taxonomy-grid span{font-size:10px;font-weight:700}.taxonomy-grid code{color:#596b82;font-size:8px}.taxonomy-grid b{justify-self:start;padding:3px 6px;border-radius:6px;background:#e9f5ed;color:#2d7550;font-size:8px}.taxonomy-grid b.reject{background:#fff0ed;color:#a64a3d}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:10px}th,td{padding:10px 8px;border-bottom:1px solid var(--da-border-light);text-align:left;white-space:nowrap}th{color:var(--da-text-muted);font-weight:620}tbody tr:last-child td{border-bottom:0}tbody tr.active{background:#f3f8ff}tbody tr.fallback{background:#fffaf6}.profile-key{width:29px;height:29px;display:grid;place-items:center;border-radius:8px;background:var(--da-bg-soft);font-weight:800}td strong,td code{display:block}td code{margin-top:3px;color:var(--da-text-muted);font:8px Consolas,monospace}.release-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.release-grid article{display:flex;align-items:flex-start;gap:9px;padding:12px;border:1px solid #eee2c8;border-radius:11px;background:#fffbf3;color:#946314}.release-grid strong,.release-grid small{display:block}.release-grid strong{font-size:10px}.release-grid small{margin-top:5px;color:#8b7653;font-size:9px;line-height:1.45}.command-panel{display:grid;grid-template-columns:auto minmax(0,1fr) 38px;align-items:center;gap:17px;padding:15px 17px;border:1px solid #cfe0f4;border-radius:13px;background:#f4f8fd}.command-panel strong,.command-panel small{display:block}.command-panel strong{font-size:11px}.command-panel small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.command-panel code{overflow:auto;padding:9px 11px;border-radius:8px;background:#202634;color:#e8edf5;font:10px Consolas,monospace;white-space:nowrap}.command-panel button{height:36px;display:grid;place-items:center;border:1px solid #cddbf0;border-radius:9px;background:#fff;color:var(--da-accent);cursor:pointer}@media(max-width:1150px){.summary-grid{grid-template-columns:repeat(3,1fr)}.governance-flow{grid-template-columns:1fr 1fr}.governance-flow article>svg{display:none}}@media(max-width:900px){.page-head{display:block}.head-actions{margin-top:13px}.profile-hero,.two-column{grid-template-columns:1fr}.release-grid{grid-template-columns:1fr}}@media(max-width:700px){.repair-page{padding:25px 14px 50px}.summary-grid,.governance-flow,.taxonomy-grid{grid-template-columns:1fr}.head-actions{display:grid;grid-template-columns:1fr 1fr}.evidence-banner{align-items:flex-start}.evidence-banner code{display:none}.registry-list article{grid-template-columns:70px 1fr}.registry-list small{grid-column:2}.command-panel{grid-template-columns:1fr}.command-panel button{display:none}}
</style>
