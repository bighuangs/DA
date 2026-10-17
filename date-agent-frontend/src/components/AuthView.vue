<template>
  <section class="auth-page">
    <div class="auth-shell">
      <aside class="auth-intro">
        <div class="intro-brand"><span class="brand-mark"><ChartNoAxesCombined :size="20" aria-hidden="true" /></span><strong>DA 数据助手</strong></div>
        <div class="intro-content">
          <span class="intro-kicker">ENTERPRISE DATA COPILOT</span>
          <h2>让每一次数据提问，<br />都更快抵达答案。</h2>
          <p>连接业务口径、数据权限与可视化分析，在一个安全工作空间里完成从提问到决策。</p>
          <ul>
            <li><CircleCheck :size="18" aria-hidden="true" />自然语言查询与结果解读</li>
            <li><CircleCheck :size="18" aria-hidden="true" />企业级权限与数据脱敏</li>
            <li><CircleCheck :size="18" aria-hidden="true" />可沉淀的指标、图表与仪表盘</li>
          </ul>
        </div>
        <small>READ-ONLY · TRACEABLE · WORKSPACE ISOLATED</small>
      </aside>
      <main class="auth-card">
        <div class="mobile-brand"><span class="brand-mark"><ChartNoAxesCombined :size="20" aria-hidden="true" /></span>DA 数据助手</div>
        <div class="eyebrow">{{ mode === 'login' ? 'WELCOME BACK' : 'GET STARTED' }}</div>
        <h1>{{ mode === 'login' ? '登录数据助手' : '创建工作空间' }}</h1>
        <p>{{ mode === 'login' ? '继续访问你的查询、仪表盘和元知识资源。' : '创建账号后，会自动生成一个独立工作空间。' }}</p>
        <div v-if="error" class="error" role="alert"><CircleAlert :size="17" aria-hidden="true" />{{ error }}</div>
        <form @submit.prevent="submit">
          <label><span>用户名</span><input v-model.trim="form.username" required autocomplete="username" placeholder="例如：alice" /></label>
          <label v-if="mode === 'register'"><span>显示名称</span><input v-model.trim="form.display_name" autocomplete="name" placeholder="例如：Alice" /></label>
          <label><span>密码</span><input v-model="form.password" required type="password" :autocomplete="mode === 'login' ? 'current-password' : 'new-password'" placeholder="至少 8 个字符" /></label>
          <label v-if="mode === 'register'"><span>工作空间名称</span><input v-model.trim="form.workspace_name" required placeholder="例如：销售分析组" /></label>
          <button class="submit" :disabled="loading"><span v-if="loading" class="spinner"></span>{{ loading ? '处理中…' : mode === 'login' ? '登录并进入' : '注册并进入' }}<ArrowRight v-if="!loading" :size="17" aria-hidden="true" /></button>
        </form>
        <div class="divider"><span>或</span></div>
        <button class="switch" @click="mode = mode === 'login' ? 'register' : 'login'; error = ''">{{ mode === 'login' ? '首次使用？创建账号与工作空间' : '已有账号？返回登录' }}</button>
        <small class="security-tip">仅在受信任设备上登录；开发验证请使用测试账号。</small>
      </main>
    </div>
  </section>
</template>
<script setup>
import { reactive, ref } from "vue";
import { ArrowRight, ChartNoAxesCombined, CircleAlert, CircleCheck } from "@lucide/vue";
import { requestJson } from "../utils/http.js";
const emit = defineEmits(["authenticated"]);
const mode = ref("login"), loading = ref(false), error = ref("");
const form = reactive({ username: "", display_name: "", password: "", workspace_name: "我的工作空间" });
async function submit() { loading.value = true; error.value = ""; try { if (mode.value === "register" && form.password.length < 8) throw new Error("密码至少需要 8 个字符，例如 bob12345"); const path = mode.value === "login" ? "/api/auth/login" : "/api/auth/register"; const data = await requestJson(path, { method: "POST", body: JSON.stringify(form) }, "认证失败"); localStorage.setItem("da_access_token", data.access_token); emit("authenticated", data); } catch (e) { error.value = e.message || "认证失败"; } finally { loading.value = false; } }
</script>
<style scoped>
.auth-page{min-height:100%;display:grid;place-items:center;padding:34px;background:radial-gradient(circle at 12% 15%,rgba(52,120,246,.13),transparent 30%),radial-gradient(circle at 88% 88%,rgba(117,91,246,.09),transparent 25%),#f4f7fb;color:var(--da-text);overflow:auto}.auth-shell{width:min(980px,100%);min-height:610px;display:grid;grid-template-columns:1.05fr .95fr;border:1px solid rgba(255,255,255,.85);border-radius:26px;background:#fff;box-shadow:0 26px 80px rgba(31,55,96,.16);overflow:hidden}.auth-intro{position:relative;display:flex;flex-direction:column;padding:38px;color:#fff;background:linear-gradient(145deg,#10264f,#1e57bb 58%,#337bf3);overflow:hidden}.auth-intro:before,.auth-intro:after{content:'';position:absolute;border:1px solid rgba(255,255,255,.1);border-radius:50%}.auth-intro:before{width:380px;height:380px;right:-210px;top:-170px}.auth-intro:after{width:280px;height:280px;left:-170px;bottom:-130px}.intro-brand,.mobile-brand{position:relative;z-index:1;display:flex;align-items:center;gap:10px;font-size:15px}.brand-mark{width:34px;height:34px;display:flex;align-items:flex-end;justify-content:center;gap:3px;padding:8px;border-radius:10px;background:rgba(255,255,255,.15);box-shadow:inset 0 0 0 1px rgba(255,255,255,.2)}.brand-mark i{width:4px;border-radius:4px;background:#fff}.brand-mark i:nth-child(1){height:8px;opacity:.7}.brand-mark i:nth-child(2){height:13px}.brand-mark i:nth-child(3){height:17px;opacity:.85}.intro-content{position:relative;z-index:1;margin:auto 0}.intro-kicker{font-size:10px;font-weight:700;letter-spacing:.2em;opacity:.65}.intro-content h2{margin:17px 0 16px;font-size:34px;line-height:1.35;letter-spacing:-.04em}.intro-content p{max-width:390px;margin:0;color:rgba(255,255,255,.72);font-size:14px;line-height:1.9}.intro-content ul{display:grid;gap:12px;margin:28px 0 0;padding:0;list-style:none;color:rgba(255,255,255,.85);font-size:13px}.intro-content li{display:flex;align-items:center;gap:9px}.intro-content li i{width:19px;height:19px;display:grid;place-items:center;border-radius:50%;background:rgba(255,255,255,.14);font-size:10px;font-style:normal}.auth-intro>small{position:relative;z-index:1;color:rgba(255,255,255,.48);font-size:9px;letter-spacing:.13em}.auth-card{display:flex;flex-direction:column;justify-content:center;padding:48px 52px}.mobile-brand{display:none}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:750;letter-spacing:.17em}h1{margin:9px 0 7px;font-size:28px;letter-spacing:-.035em}p{margin:0 0 22px;color:var(--da-text-muted);font-size:13px;line-height:1.7}.auth-card label{display:block;margin:13px 0;color:var(--da-text-secondary);font-size:12px;font-weight:600}.auth-card label span{display:block;margin-bottom:7px}.auth-card input{display:block;width:100%;height:44px;padding:0 13px;border:1px solid var(--da-border);border-radius:10px;background:#fbfcfe;color:var(--da-text);transition:.15s}.auth-card input:hover{border-color:#bdc8d7}.auth-card input:focus{border-color:var(--da-accent);background:#fff;box-shadow:0 0 0 3px var(--da-accent-soft)}.submit{width:100%;height:46px;margin-top:12px;display:flex;align-items:center;justify-content:center;gap:8px;border:0;border-radius:11px;background:linear-gradient(135deg,var(--da-accent),var(--da-accent-strong));color:#fff;font-size:14px;font-weight:650;box-shadow:0 10px 22px rgba(52,120,246,.2);transition:.15s}.submit:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 13px 27px rgba(52,120,246,.26)}.submit b{font-size:17px}.submit:disabled{opacity:.65}.spinner{width:15px;height:15px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}.divider{display:flex;align-items:center;gap:10px;margin:18px 0;color:var(--da-text-muted);font-size:10px}.divider:before,.divider:after{content:'';height:1px;flex:1;background:var(--da-border-light)}.switch{width:100%;border:0;background:transparent;color:var(--da-accent);font-size:12px}.switch:hover{color:var(--da-accent-strong)}.security-tip{display:block;margin-top:22px;color:var(--da-text-muted);font-size:10px;text-align:center}.error{display:flex;align-items:flex-start;gap:8px;margin:0 0 8px;padding:10px 12px;border:1px solid rgba(214,69,69,.22);border-radius:9px;background:rgba(214,69,69,.07);color:#b83d3d;font-size:12px;line-height:1.5}.error span{width:17px;height:17px;flex:0 0 17px;display:grid;place-items:center;border-radius:50%;background:#d64545;color:#fff;font-size:10px;font-weight:700}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:760px){.auth-page{padding:18px}.auth-shell{min-height:0;display:block;max-width:450px}.auth-intro{display:none}.auth-card{padding:35px 28px}.mobile-brand{display:flex;margin-bottom:30px;color:var(--da-text);font-weight:700}.mobile-brand .brand-mark{background:linear-gradient(145deg,#4a8eff,#225ad0)}h1{font-size:26px}}@media(max-width:420px){.auth-page{padding:0;background:#fff}.auth-shell{min-height:100%;border:0;border-radius:0;box-shadow:none}.auth-card{min-height:100%;padding:26px 20px}}
.brand-mark{display:grid;place-items:center;padding:7px;color:#fff}.intro-content li svg{flex:0 0 18px;opacity:.9}.error svg{flex:0 0 17px;margin-top:1px}.mobile-brand .brand-mark{color:#fff}
</style>
