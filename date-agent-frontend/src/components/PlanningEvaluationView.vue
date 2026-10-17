<template>
  <section class="planning-page">
    <header class="page-head">
      <div>
        <span class="eyebrow">EVALUATION / SQL PLANNING</span>
        <h1>SQL 规划复杂度路由</h1>
        <p>比较 Direct 与 Plan 的质量、延迟和成本，并把“离线候选”与“线上生效”明确分开。</p>
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
    <div v-if="loading && !report" class="loading">正在读取规划路由消融证据…</div>

    <template v-else-if="report?.available">
      <section class="evidence-banner" data-testid="planning-evidence-boundary">
        <div class="evidence-icon"><ShieldCheck :size="21" aria-hidden="true" /></div>
        <div>
          <strong>评测证据边界：受审离线反事实，不是 Full Run 发布结论</strong>
          <span>{{ report.decision.selection_reason }}</span>
        </div>
        <code>{{ report.evidence_level }}</code>
      </section>

      <section class="route-hero">
        <article class="route-card active-card" data-testid="active-profile">
          <header><span>当前线上 Active</span><b>保持不变</b></header>
          <div class="route-name"><span>C1</span><strong>{{ profileLabel(activeProfile) }}</strong></div>
          <p>兼容既有“关键词 + 长度”合同；发布门禁完成前继续承接真实流量。</p>
          <footer><code>{{ activeProfile }}</code><span>可立即回退</span></footer>
        </article>
        <div class="route-arrow" aria-hidden="true"><ArrowRight :size="23" /><span>离线推荐</span></div>
        <article class="route-card candidate-card" data-testid="recommended-profile">
          <header><span>挑战者候选</span><b>尚未发布</b></header>
          <div class="route-name"><span>{{ report.decision.offline_recommendation_label }}</span><strong>{{ profileLabel(report.decision.offline_recommendation) }}</strong></div>
          <p>在受审测试集上平衡漏规划与过度规划；需真实 Direct/Plan 对、三次 Full Run 和影子灰度。</p>
          <footer><code>{{ report.decision.offline_recommendation }}</code><span>仅离线推荐</span></footer>
        </article>
      </section>

      <section class="summary-grid" aria-label="规划路由评测摘要">
        <article><span>受审样本</span><strong>{{ report.contract.total }}</strong><small>{{ report.contract.categories }} 类 · train/dev/test</small></article>
        <article><span>可比较测试对</span><strong>{{ report.evaluation_scope.comparable_cases }}</strong><small>排除 {{ excludedCount }} 个不可归因样本</small></article>
        <article class="good"><span>C4 Plan Recall</span><strong>{{ percent(candidate.plan_recall) }}</strong><small>C1 为 {{ percent(active.plan_recall) }}</small></article>
        <article class="good"><span>C4 结果一致</span><strong>{{ percent(candidate.selected_path?.result_equivalence_rate) }}</strong><small>C1 为 {{ percent(active.selected_path?.result_equivalence_rate) }}</small></article>
        <article :class="candidate.decision_latency_ms?.p95 <= 5 ? 'good' : 'warn'"><span>路由决策 P95</span><strong>{{ ms(candidate.decision_latency_ms?.p95) }}</strong><small>门槛 ≤ 5 ms</small></article>
        <article :class="report.implementation_gates.gate_passed ? 'good' : 'bad'"><span>实现门禁</span><strong>{{ report.implementation_gates.gate_passed ? "通过" : "未通过" }}</strong><small>不代表发布批准</small></article>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><h2>C0–C5 候选对照</h2><p>固定 Prompt、语义 IR、候选策略与缓存配置，仅改变“是否调用规划模型”的路由决策。</p></div>
          <span class="pill neutral">{{ report.evaluation_scope.comparable_cases }} 对</span>
        </div>
        <div class="table-wrap" tabindex="0" aria-label="SQL 规划候选对照表">
          <table aria-label="SQL 规划候选对照表">
            <thead><tr><th>候选</th><th>策略</th><th>Precision</th><th>Recall</th><th>过度规划</th><th>漏规划</th><th>结果一致</th><th>TTResult P95</th><th>Token 均值</th><th>决策 P95</th><th>证据状态</th></tr></thead>
            <tbody>
              <tr v-for="(item, key) in report.profiles" :key="key" :class="{ selected: key === report.decision.offline_recommendation_label, active: key === 'C1' }">
                <td><span class="candidate-key">{{ key }}</span></td>
                <td><strong>{{ profileLabel(item.profile) }}</strong><code>{{ item.profile }}</code></td>
                <td>{{ percent(item.plan_precision) }}</td>
                <td>{{ percent(item.plan_recall) }}</td>
                <td>{{ percent(item.over_plan_rate) }}</td>
                <td>{{ percent(item.under_plan_rate) }}</td>
                <td>{{ percent(item.selected_path?.result_equivalence_rate) }}</td>
                <td>{{ integer(item.selected_path?.tt_result_p95_ms) }} ms</td>
                <td>{{ integer(item.selected_path?.tokens_mean) }}</td>
                <td>{{ ms(item.decision_latency_ms?.p95) }}</td>
                <td><span v-if="key === 'C1'" class="pill active-pill">线上 active</span><span v-else-if="key === report.decision.offline_recommendation_label" class="pill recommend-pill">离线推荐</span><span v-else class="pill neutral">消融</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <div class="two-column">
        <section class="panel ablation-panel">
          <div class="panel-head"><div><h2>A0–A4 特征增量</h2><p>嵌套消融展示哪组结构证据真正改善 Plan 判定。</p></div></div>
          <div class="ablation-list">
            <article v-for="(item, key) in report.feature_ablations" :key="key">
              <div class="ablation-label"><b>{{ key }}</b><span>{{ featureLabel(key) }}</span></div>
              <div class="bar-track" aria-hidden="true"><i :style="{ width: `${item.plan_recall * 100}%` }"></i></div>
              <strong>{{ percent(item.plan_recall) }}</strong>
              <small>F1 {{ decimal(item.f1) }}</small>
            </article>
          </div>
        </section>

        <section class="panel blockers-panel">
          <div class="panel-head"><div><h2>发布前仍需完成</h2><p>这些证据缺口会阻止 C4 自动切换为 active。</p></div><span class="pill warn">{{ report.decision.release_blockers.length }} 项</span></div>
          <ul class="blocker-list">
            <li v-for="item in report.decision.release_blockers" :key="item"><CircleDashed :size="16" aria-hidden="true" /><span><strong>{{ blockerLabel(item) }}</strong><small>{{ blockerHelp(item) }}</small></span></li>
          </ul>
          <div class="rollback-box">
            <div><strong>运行时安全回退</strong><small>当前 {{ runtimeProfile }}；回退目标 {{ report.runtime?.fallback_profile || report.decision.fallback_profile }}</small></div>
            <button class="btn danger" type="button" :disabled="rollingBack || runtimeAtFallback" @click="rollback">
              <RotateCcw :size="14" aria-hidden="true" />{{ runtimeAtFallback ? "已在安全基线" : rollingBack ? "回退中…" : "立即回退 C1" }}
            </button>
          </div>
        </section>
      </div>

      <section class="panel timing-panel">
        <div class="panel-head"><div><h2>时间口径不混用</h2><p>TODO-63 不改任何 TTL；路由耗时、规划模型耗时与缓存存活时间分别记录。</p></div><span class="pill active-pill">TTL FROZEN</span></div>
        <div class="timing-grid">
          <article><TimerReset :size="18" aria-hidden="true" /><div><strong>路由 Decision</strong><span>纯本地特征提取与候选判定</span></div><b>{{ ms(candidate.decision_latency_ms?.p95) }} P95</b></article>
          <article><Clock3 :size="18" aria-hidden="true" /><div><strong>Planning Provider</strong><span>仅实际进入 Plan 时记录供应商耗时</span></div><b>独立 Trace</b></article>
          <article><Gauge :size="18" aria-hidden="true" /><div><strong>Time To Last Token</strong><span>只在流式模型回调可得时写入</span></div><b>不可伪造</b></article>
          <article><DatabaseZap :size="18" aria-hidden="true" /><div><strong>Cache TTL</strong><span>查询缓存 {{ report.frozen_variables.cache_ttl_seconds }}s · 召回缓存 {{ report.frozen_variables.retrieval_cache_ttl_seconds }}s</span></div><b>本轮未改</b></article>
        </div>
      </section>

      <section class="command-panel">
        <div><strong>重跑确定性规划消融</strong><small>运行后刷新本页；它不会切换 active，也不调用 SQLBot / STT 代码。</small></div>
        <code>{{ command }}</code>
        <button type="button" aria-label="复制规划路由评测命令" @click="copyCommand"><Copy :size="15" /></button>
      </section>
    </template>

    <section v-else-if="report" class="empty">
      <Route :size="35" aria-hidden="true" /><h2>还没有规划路由评测报告</h2>
      <p>先在后端目录执行评测命令，再返回此页刷新。</p><code>{{ command }}</code>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import {
  ArrowRight,
  CircleAlert,
  CircleCheck,
  CircleDashed,
  Clock3,
  Copy,
  DatabaseZap,
  Gauge,
  RefreshCw,
  RotateCcw,
  Route,
  ShieldCheck,
  TimerReset,
} from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const command = "python -m app.eval.planning_router_ablation --enforce";
const report = ref(null);
const loading = ref(false);
const rollingBack = ref(false);
const error = ref("");
const notice = ref("");
const active = computed(() => report.value?.profiles?.C1 || {});
const candidate = computed(() => report.value?.profiles?.[report.value?.decision?.offline_recommendation_label] || {});
const activeProfile = computed(() => report.value?.decision?.active_profile || "keyword_length_v1");
const runtimeProfile = computed(() => report.value?.runtime?.runtime_profile || activeProfile.value);
const runtimeAtFallback = computed(() => runtimeProfile.value === (report.value?.runtime?.fallback_profile || report.value?.decision?.fallback_profile));
const excludedCount = computed(() => Object.values(report.value?.evaluation_scope?.excluded || {}).reduce((sum, value) => sum + Number(value || 0), 0));

async function load() {
  loading.value = true;
  error.value = "";
  try {
    report.value = await requestJson("/api/admin/evaluation/planning/latest", {}, "规划路由报告加载失败");
  } catch (exception) {
    error.value = exception.message || "规划路由报告加载失败";
  } finally {
    loading.value = false;
  }
}

async function rollback() {
  rollingBack.value = true;
  error.value = "";
  try {
    const result = await requestJson("/api/admin/evaluation/planning/rollback", { method: "POST" }, "规划路由回退失败");
    report.value.runtime = result.runtime;
    report.value.decision.active_profile = result.runtime.runtime_profile;
    notice.value = result.message || "已回退到安全基线";
    window.setTimeout(() => { notice.value = ""; }, 3000);
  } catch (exception) {
    error.value = exception.message || "规划路由回退失败";
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
    never_plan: "从不规划",
    keyword_length_v1: "关键词 + 长度基线",
    always_plan: "始终规划",
    structured_rules_v1: "结构化规则",
    weighted_scorecard_v1: "加权评分卡",
    calibrated_linear_v1: "校准线性候选",
  })[value] || value || "—";
}
function featureLabel(value) {
  return ({ A0: "关键词 / 长度", A1: "+ 查询信号", A2: "+ 语义操作", A3: "+ 表与连接结构", A4: "+ 多轮与预算" })[value] || value;
}
function blockerLabel(value) {
  return ({
    missing_live_direct_plan_pairs: "缺少真实 Direct / Plan 配对",
    missing_cold_warm_warm_full_run: "缺少 cold / warm / warm 三次 Full Run",
    missing_production_shadow_canary: "缺少生产影子与小流量灰度",
  })[value] || value;
}
function blockerHelp(value) {
  return ({
    missing_live_direct_plan_pairs: "离线受审结果不能替代真实模型输出和数据库执行。",
    missing_cold_warm_warm_full_run: "需要同时守住逐题正确性、TTSQL、TTResult、Token 与成本。",
    missing_production_shadow_canary: "先比较影子决策，再灰度并保留无需重启的 C1 回退。",
  })[value] || "完成对应证据后才可申请发布。";
}
function percent(value) { const number = Number(value); return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—"; }
function decimal(value) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(3) : "—"; }
function integer(value) { const number = Number(value); return Number.isFinite(number) ? Math.round(number).toLocaleString("zh-CN") : "—"; }
function ms(value) { const number = Number(value); return Number.isFinite(number) ? `${number.toFixed(number < 1 ? 3 : 1)} ms` : "—"; }

onMounted(load);
</script>

<style scoped>
.planning-page{flex:1 1 auto;min-height:0;overflow:auto;padding:36px 34px 70px;color:var(--da-text)}.page-head,.evidence-banner,.route-hero,.summary-grid,.panel,.command-panel,.alert,.empty{max-width:1320px;margin-left:auto;margin-right:auto}.page-head{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:20px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:780;letter-spacing:.17em}.page-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.page-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.head-actions{display:flex;gap:8px}.btn{min-height:38px;display:inline-flex;align-items:center;justify-content:center;gap:7px;padding:8px 13px;border:1px solid var(--da-border);border-radius:10px;background:#fff;color:var(--da-text-secondary);font:12px inherit;cursor:pointer}.btn:hover:not(:disabled){border-color:var(--da-accent);color:var(--da-accent)}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.btn.danger{border-color:#e4b8b2;color:#ac4134}.btn:disabled{cursor:not-allowed;opacity:.55}.alert{display:flex;align-items:center;gap:8px;margin-bottom:13px;padding:11px 14px;border-radius:10px;font-size:12px}.alert.error{background:#fff2ef;color:#b73f31}.alert.notice{background:#effaf3;color:#287b4a}.loading,.empty{min-height:380px;display:grid;place-content:center;text-align:center;color:var(--da-text-muted)}.empty h2{margin:13px 0 5px;color:var(--da-text);font-size:18px}.empty p{font-size:12px}.empty code{margin-top:10px;padding:9px 12px;border-radius:8px;background:#202634;color:#e8edf5}.evidence-banner{display:flex;align-items:center;gap:12px;margin-bottom:13px;padding:12px 14px;border:1px solid #e5d7b6;border-radius:13px;background:#fffaf0}.evidence-icon{width:36px;height:36px;flex:0 0 36px;display:grid;place-items:center;border-radius:10px;background:#fff0c9;color:#9a6b15}.evidence-banner div:nth-child(2){min-width:0;flex:1}.evidence-banner strong,.evidence-banner span{display:block}.evidence-banner strong{font-size:12px}.evidence-banner span{margin-top:4px;color:var(--da-text-muted);font-size:10px;line-height:1.5}.evidence-banner code{padding:5px 8px;border-radius:7px;background:#fff0d2;color:#815d17;font-size:9px}.route-hero{display:grid;grid-template-columns:1fr 92px 1fr;align-items:stretch;gap:13px;margin-bottom:13px}.route-card{padding:19px;border:1px solid var(--da-border-light);border-radius:16px;background:#fff;box-shadow:var(--da-shadow-sm)}.route-card header,.route-card footer,.route-name{display:flex;align-items:center;justify-content:space-between;gap:12px}.route-card header span,.route-card footer,.route-card p{color:var(--da-text-muted);font-size:10px}.route-card header b{padding:4px 7px;border-radius:99px;font-size:9px}.active-card{border-color:#cfe0f4}.active-card header b{background:#e9f2ff;color:#2d65b7}.candidate-card{border-color:#c9e6d4}.candidate-card header b{background:#eaf8ef;color:#277849}.route-name{justify-content:flex-start;margin-top:13px}.route-name>span{width:43px;height:43px;display:grid;place-items:center;border-radius:12px;background:#edf4ff;color:#2a67c1;font-size:15px;font-weight:800}.candidate-card .route-name>span{background:#edf9f1;color:#2e8252}.route-name strong{font-size:18px}.route-card p{min-height:30px;margin:11px 0 15px;line-height:1.55}.route-card footer{padding-top:12px;border-top:1px solid var(--da-border-light)}.route-card footer code{color:var(--da-text-secondary)}.route-arrow{display:grid;place-content:center;justify-items:center;gap:5px;color:#8b98aa}.route-arrow span{font-size:9px}.summary-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin-bottom:16px}.summary-grid article{position:relative;overflow:hidden;padding:16px;border:1px solid var(--da-border-light);border-radius:13px;background:#fff;box-shadow:var(--da-shadow-sm)}.summary-grid article:before{position:absolute;inset:0 auto 0 0;width:3px;background:#96a3b5;content:""}.summary-grid article.good:before{background:#35a56a}.summary-grid article.warn:before{background:#dfa33c}.summary-grid article.bad:before{background:#dc6656}.summary-grid span,.summary-grid small{display:block;color:var(--da-text-muted);font-size:9px}.summary-grid strong{display:block;margin:7px 0 4px;font-size:20px;letter-spacing:-.03em}.panel{margin-bottom:16px;padding:21px;border:1px solid var(--da-border-light);border-radius:16px;background:rgba(255,255,255,.94);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:15px}.panel-head h2{margin:0;font-size:16px}.pill{display:inline-flex;align-items:center;padding:4px 7px;border-radius:99px;font-size:9px;white-space:nowrap}.pill.neutral{background:#eef1f5;color:#667287}.pill.active-pill{background:#e9f2ff;color:#2d65b7}.pill.recommend-pill{background:#e9f8ef;color:#277849}.pill.warn{background:#fff3d8;color:#8a641c}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:10px}th,td{padding:10px 8px;border-bottom:1px solid var(--da-border-light);text-align:left;white-space:nowrap}th{color:var(--da-text-muted);font-weight:620}tbody tr:last-child td{border-bottom:0}tbody tr.active{background:#f7faff}tbody tr.selected{background:#f3fbf6}.candidate-key{width:29px;height:29px;display:grid;place-items:center;border-radius:8px;background:var(--da-bg-soft);font-weight:800}td strong,td code{display:block}td code{margin-top:3px;color:var(--da-text-muted);font:8px Consolas,monospace}.two-column{max-width:1320px;display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0 auto}.ablation-list{display:grid;gap:9px}.ablation-list article{display:grid;grid-template-columns:168px 1fr 52px 57px;align-items:center;gap:10px;padding:9px 11px;border-radius:10px;background:#f8fafc}.ablation-label{display:flex;align-items:center;gap:9px}.ablation-label b{width:31px;height:27px;display:grid;place-items:center;border-radius:7px;background:#fff;color:var(--da-accent);font-size:10px}.ablation-label span,.ablation-list small{color:var(--da-text-muted);font-size:9px}.bar-track{height:7px;overflow:hidden;border-radius:99px;background:#e8edf4}.bar-track i{display:block;height:100%;border-radius:99px;background:#4b8df5}.ablation-list article:nth-child(n+4) .bar-track i{background:#35a56a}.ablation-list strong{font-size:11px}.blocker-list{display:grid;gap:8px;margin:0;padding:0;list-style:none}.blocker-list li{display:flex;align-items:flex-start;gap:9px;padding:10px;border:1px solid #eee2c8;border-radius:10px;background:#fffbf3;color:#946314}.blocker-list li span{display:grid;gap:3px}.blocker-list strong{font-size:10px}.blocker-list small{color:#8b7653;font-size:9px;line-height:1.45}.rollback-box{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;padding-top:12px;border-top:1px solid var(--da-border-light)}.rollback-box strong,.rollback-box small{display:block}.rollback-box strong{font-size:11px}.rollback-box small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.timing-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.timing-grid article{display:grid;grid-template-columns:34px 1fr;gap:4px 9px;padding:13px;border:1px solid var(--da-border-light);border-radius:11px;background:#fafbfd;color:#68778d}.timing-grid article>svg{grid-row:1/3;align-self:start;margin-top:2px;color:var(--da-accent)}.timing-grid strong,.timing-grid span{display:block}.timing-grid strong{color:var(--da-text);font-size:10px}.timing-grid span{margin-top:4px;color:var(--da-text-muted);font-size:8px;line-height:1.45}.timing-grid b{grid-column:2;color:var(--da-text-secondary);font-size:10px}.command-panel{display:grid;grid-template-columns:auto minmax(0,1fr) 38px;align-items:center;gap:17px;padding:15px 17px;border:1px solid #cfe0f4;border-radius:13px;background:#f4f8fd}.command-panel strong,.command-panel small{display:block}.command-panel strong{font-size:11px}.command-panel small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.command-panel code{overflow:auto;padding:9px 11px;border-radius:8px;background:#202634;color:#e8edf5;font:10px Consolas,monospace;white-space:nowrap}.command-panel button{height:36px;display:grid;place-items:center;border:1px solid #cddbf0;border-radius:9px;background:#fff;color:var(--da-accent);cursor:pointer}@media(max-width:1150px){.summary-grid{grid-template-columns:repeat(3,1fr)}.timing-grid{grid-template-columns:1fr 1fr}}@media(max-width:900px){.page-head{display:block}.head-actions{margin-top:13px}.route-hero{grid-template-columns:1fr}.route-arrow{display:none}.two-column{grid-template-columns:1fr}}@media(max-width:700px){.planning-page{padding:25px 14px 50px}.summary-grid,.timing-grid{grid-template-columns:1fr}.head-actions{display:grid;grid-template-columns:1fr 1fr}.evidence-banner{align-items:flex-start}.evidence-banner code{display:none}.ablation-list article{grid-template-columns:130px 1fr 48px}.ablation-list small{display:none}.rollback-box{align-items:flex-start;flex-direction:column}.command-panel{grid-template-columns:1fr}.command-panel button{display:none}}
</style>
