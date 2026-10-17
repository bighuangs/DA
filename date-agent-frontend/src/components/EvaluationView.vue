<template>
  <section class="evaluation-page">
    <header class="evaluation-head">
      <div>
        <div class="eyebrow">ADMIN / NL2SQL QUALITY</div>
        <h1>正确性归因</h1>
        <p>把失败定位到召回、筛表、规划、生成或校验阶段，并给出可直接运行的定向回归切片。</p>
      </div>
      <div class="head-actions">
        <select v-if="runs.length" v-model="selectedRun" class="run-select" aria-label="选择评测批次" @change="load">
          <option v-for="item in runs" :key="item.run_id" :value="item.run_id">{{ item.run_id }} · {{ item.total }}题 · {{ suiteLabel(item.dataset_suite, item.suite_label) }} · {{ profileLabel(item.evaluation_profile) }}{{ item.experiment_id ? ` · R${item.repeat_index}` : "" }}</option>
        </select>
        <label class="switch"><input v-model="includePassed" type="checkbox" @change="load" /> 显示通过用例</label>
        <button class="btn" :disabled="loading" @click="load">{{ loading ? "刷新中…" : "刷新报告" }}</button>
      </div>
    </header>

    <nav class="difficulty-tabs" role="tablist" aria-label="评测难度">
      <button
        v-for="item in difficultyOptions"
        :key="item.value"
        type="button"
        role="tab"
        :aria-selected="difficulty === item.value"
        :class="{ active: difficulty === item.value }"
        :disabled="loading"
        @click="selectDifficulty(item.value)"
      >
        <span>{{ item.label }}</span>
        <small>{{ baselineTabLabel(baselineFor(item.value)) }}</small>
      </button>
    </nav>

    <section class="baseline-track-grid" aria-label="三种难度观测基线">
      <article
        v-for="item in baselineCards"
        :key="item.difficulty"
        class="baseline-track-card"
        :class="[{ active: difficulty === item.difficulty }, item.resolution_error ? 'unresolvable' : item.available ? 'available' : 'missing']"
      >
        <header>
          <div><span>{{ item.difficulty_label }}观测基线</span><strong>{{ baselineStateLabel(item) }}</strong></div>
          <button type="button" :aria-label="`查看${item.difficulty_label}评测`" :disabled="loading || difficulty === item.difficulty" @click="selectDifficulty(item.difficulty)">查看</button>
        </header>
        <div class="baseline-track-run">
          <b class="mono" :title="item.run_id || ''">{{ item.run_id || "等待首个正式 run" }}</b>
          <small>{{ suiteLabel(item.dataset_suite, item.suite_label) }}</small>
        </div>
        <div class="baseline-track-metrics">
          <span>结果<strong>{{ percent(item.metrics?.result_rate) }}</strong></span>
          <span>表选择<strong>{{ percent(item.metrics?.table_exact_rate) }}</strong></span>
          <span>执行<strong>{{ percent(item.metrics?.execution_rate) }}</strong></span>
          <span>P95<strong>{{ optionalNumber(item.metrics?.latency_p95) }} ms</strong></span>
        </div>
        <footer>
          <span>LLM {{ optionalNumber(item.metrics?.llm_attempts) }}</span>
          <span>Token 均值 {{ optionalNumber(item.metrics?.tokens_mean) }}</span>
          <span>成本 {{ cost(item.metrics?.estimated_cost_total, item.metrics?.cost_currency, item.metrics?.cost_estimate_available) }}</span>
          <time v-if="item.reviewed_at">{{ item.reviewed_at }}</time>
        </footer>
        <p v-if="item.resolution_error" class="baseline-resolution-error">{{ item.resolution_error }}</p>
      </article>
    </section>

    <div v-if="error" class="alert error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-if="notice" class="alert notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <div v-if="loading && !report" class="loading" role="status" aria-live="polite">
      正在建立评测批次索引并读取当前正式实验…
    </div>

    <template v-else-if="report && !report.available">
      <div class="empty-state">
        <div class="empty-icon"><ClipboardCheck :size="25" aria-hidden="true" /></div>
        <h2>还没有可分析的评测结果</h2>
        <p>{{ report.message }}</p>
        <div class="command"><code>{{ report.run_command }}</code><button @click="copy(report.run_command)">复制</button></div>
        <small>在 data-agent 后端目录运行命令；完成后回到此页点击“刷新报告”。</small>
      </div>
    </template>

    <template v-else-if="report?.available">
      <div class="run-strip">
        <div><span>评测批次</span><strong>{{ report.run.run_id }}</strong></div>
        <div><span>难度</span><strong>{{ activeDifficultyLabel }}</strong></div>
        <div><span>生成时间</span><strong>{{ report.run.timestamp || "—" }}</strong></div>
        <div><span>评测套件</span><strong>{{ suiteLabel(report.run.dataset_suite, report.run.suite_label) }}</strong></div>
        <div><span>评测口径</span><strong>{{ profileLabel(report.run.evaluation_profile) }}</strong></div>
        <div><span>候选</span><strong>{{ sqlIdentity.candidateLabel }}</strong></div>
        <div><span>SQL 模型</span><strong>{{ sqlIdentity.modelName }}</strong></div>
        <div><span>Provider</span><strong>{{ sqlIdentity.provider }}</strong></div>
        <div><span>证据制品</span><strong :title="report.run.artifact || ''">{{ artifactName(report.run.artifact) }}</strong></div>
        <div><span>样本</span><strong>{{ report.run.total }} 条</strong></div>
        <div><span>TTComplete P95</span><strong>{{ number(report.metrics.latency_p95) }} ms</strong><small>{{ report.performance.timing_protocol_version }}</small></div>
      </div>

      <div class="metric-grid">
        <article v-for="metric in metricCards" :key="metric.label" class="metric-card" :class="[metric.tone, { primary: metric.primary }]">
          <span>{{ metric.label }}</span><strong>{{ metric.value }}</strong><small>{{ metric.note }}</small>
        </article>
      </div>

      <section class="panel ablation-panel" aria-label="SQL 模型消融指标">
        <div class="panel-head">
          <div><h2>First → Correction → Final</h2><p>分开呈现首版 SQL、纠错归因和最终 SQL，避免把纠错收益误记为模型首轮能力。</p></div>
          <span>{{ sqlIdentity.candidateLabel }} · {{ sqlIdentity.modelName }}</span>
        </div>
        <div class="ablation-grid">
          <article class="ablation-card" :class="strictMetricTone(firstSqlAggregate.rate, firstSqlAggregate.available)">
            <span>First SQL 严格结果</span>
            <strong>{{ firstSqlAggregate.available ? percent(firstSqlAggregate.rate) : "—" }}</strong>
            <small>{{ firstSqlAggregate.available ? strictAggregateCountLabel(firstSqlAggregate) : "旧制品未记录" }}</small>
          </article>
          <article class="ablation-card correction" :class="correctionAggregate.available ? 'warn' : 'neutral'">
            <span>Correction 归因</span>
            <strong>{{ correctionAggregate.available ? correctionCountLabel(correctionAggregate) : "—" }}</strong>
            <small v-if="correctionAggregate.available">修复结果 {{ correctionAggregate.fixedResult }} · 修复执行 {{ correctionAggregate.fixedExecution }}</small>
            <small v-if="correctionAggregate.available">有害修改 {{ correctionAggregate.harmful }} · 不可归因 {{ correctionAggregate.unattributable }}</small>
            <small v-else>旧制品未记录</small>
          </article>
          <article class="ablation-card primary-result" :class="finalResultTone">
            <span>Final SQL 最终结果（主要指标）</span>
            <strong>{{ finalSqlAggregate.available ? percent(finalSqlAggregate.rate) : "—" }}</strong>
            <small>{{ finalSqlAggregate.available ? strictAggregateCountLabel(finalSqlAggregate) : "旧制品未记录" }}</small>
            <small>{{ finalResultPolicy.note }}</small>
          </article>
        </div>
      </section>

      <section v-if="governance?.available" class="panel governance-panel">
        <div class="panel-head">
          <div><h2>样本、版本指纹与{{ activeDifficultyLabel }}观测基线</h2><p>只有样本已审核、环境指纹完整、证据链齐全且无基础设施异常的全量 run 才能登记为当前能力观测。</p></div>
          <span class="baseline-status" :class="governance.selected.status">{{ statusLabel(governance.selected.status) }}</span>
        </div>
        <div class="governance-grid">
          <article>
            <span>黄金集契约</span>
            <strong>{{ governance.sample_contract.complete_count }}/{{ governance.sample_contract.total }} 完整</strong>
            <small>{{ governance.sample_contract.reviewed_count }}/{{ governance.sample_contract.total }} 已审核 · 重复 ID {{ governance.sample_contract.duplicate_ids.length }}</small>
          </article>
          <article>
            <span>运行指纹</span>
            <strong>{{ governance.selected.fingerprint.complete ? "完整" : "不完整" }}</strong>
            <small class="mono">{{ shortHash(governance.selected.fingerprint.run_fingerprint) }}</small>
          </article>
          <article>
            <span>{{ activeDifficultyLabel }}观测基线</span>
            <strong>{{ governance.approved?.run_id || "尚未批准" }}</strong>
            <small>{{ baselineSourceLabel(governance.baseline_resolution) }}</small>
            <small v-if="governance.comparison">{{ governance.comparison.comparable ? "与当前 run 可直接比较" : `不可直接比较：${governance.comparison.differences.join("、")}` }}</small>
            <small v-else>批准后，后续 run 将按指纹判定可比性</small>
          </article>
        </div>
        <div class="checklist" aria-label="基线准入清单">
          <div v-for="(passed, key) in governance.selected.checklist" :key="key" :class="passed ? 'pass' : 'fail'">
            <span aria-hidden="true"><Check v-if="passed" :size="13" /><X v-else :size="13" /></span>{{ checklistLabel(key) }}
          </div>
        </div>
        <details class="fingerprints">
          <summary>查看版本指纹</summary>
          <dl>
            <template v-for="item in fingerprintRows" :key="item.label"><dt>{{ item.label }}</dt><dd>{{ item.value }}</dd></template>
          </dl>
        </details>
        <section v-if="governance.experiment?.available" class="experiment-box" aria-label="三次可比实验">
          <div class="experiment-head">
            <div><span>当前难度三次重复 Experiment</span><strong>{{ governance.experiment.experiment_id }}</strong><small>{{ governance.experiment.candidate_label }} · {{ profileLabel(governance.experiment.evaluation_profile) }} · {{ suiteLabel(governance.experiment.baseline?.candidate_suite) }}</small></div>
            <span class="experiment-status" :class="governance.experiment.status">{{ experimentStatusLabel(governance.experiment) }}</span>
          </div>
          <div class="experiment-metrics">
            <div><span>重复进度</span><strong>{{ governance.experiment.repeat_count }}/{{ governance.experiment.required_repeats }}</strong><small>cold → warm → warm</small></div>
            <div><span>最差表路径 / Final 结果</span><strong>{{ governance.experiment.worst.tables_correct }} / {{ governance.experiment.worst.result_equivalent }}</strong><small>表路径仅观察，Final 结果为主</small></div>
            <div><span>最差范围安全 / 证据完整</span><strong>{{ experimentScopeLabel(governance.experiment) }}</strong><small>两者均为阻断证据；缺失不按安全处理</small></div>
            <div><span>最差 TTResult</span><strong>{{ number(governance.experiment.worst.tt_result_p95) }} ms</strong><small>工程参考目标 ≤ 20,000 ms</small></div>
            <div><span>最差 TTComplete</span><strong>{{ number(governance.experiment.worst.tt_complete_p95) }} ms</strong><small>工程参考目标 ≤ 30,000 ms</small></div>
            <div><span>最差 Prompt 压缩</span><strong>{{ percent(governance.experiment.worst.prompt_compression_rate) }}</strong><small>按三次最低值</small></div>
            <div><span>最高候选触发率</span><strong>{{ percent(governance.experiment.worst.adaptive_candidate_rate) }}</strong><small>工程参考目标 ≤ 20%</small></div>
            <div><span>候选 B 最低有效数</span><strong>{{ optionalNumber(governance.experiment.worst.candidate_valid_b) }}</strong><small>失败最多 {{ optionalNumber(governance.experiment.worst.candidate_failed) }} 次</small></div>
            <div><span>候选最高等待</span><strong>{{ number(governance.experiment.worst.candidate_wait_total_ms) }} ms</strong><small>慢路径累计等待</small></div>
            <div><span>核心 / 可选失败</span><strong>{{ optionalNumber(governance.experiment.worst.failed_core_llm_calls) }} / {{ optionalNumber(governance.experiment.worst.failed_optional_llm_calls) }}</strong><small>默认发布只允许核心为 0</small></div>
          </div>
          <section v-if="governance.experiment.stability_summary" class="stability-box" :aria-label="`${governance.experiment.stability_summary.total}题跨轮稳定性`">
            <div class="stability-head">
              <div><strong>{{ governance.experiment.stability_summary.total }} 题逐题稳定性</strong><small>同一指纹下按三轮结果一致判定；波动题不能被平均通过率掩盖。</small></div>
              <span>{{ governance.experiment.stability_summary.total }} 题</span>
            </div>
            <div class="stability-metrics">
              <article class="stable"><span>稳定通过</span><strong>{{ governance.experiment.stability_summary.stable_pass }}</strong></article>
              <article class="failed"><span>稳定失败</span><strong>{{ governance.experiment.stability_summary.stable_fail }}</strong></article>
              <article class="variable"><span>跨轮波动</span><strong>{{ governance.experiment.stability_summary.variable }}</strong></article>
            </div>
            <div class="family-table" tabindex="0" aria-label="语义族三轮通过率">
              <div class="family-row family-head"><span>语义族</span><span>样本</span><span>最差轮</span><span>稳定通过 / 失败 / 波动</span><span>R1 / R2 / R3</span></div>
              <div v-for="family in governance.experiment.semantic_family_stability || []" :key="family.semantic_family" class="family-row">
                <strong>{{ semanticFamilyLabel(family.semantic_family) }}</strong>
                <span>{{ family.total }}</span>
                <span>{{ percent(family.worst_pass_rate) }}</span>
                <span>{{ family.stable_pass }} / {{ family.stable_fail }} / {{ family.variable }}</span>
                <span>{{ (family.repeats || []).map(item => percent(item.pass_rate)).join(" / ") }}</span>
              </div>
            </div>
            <div v-if="governance.experiment.stability?.some(item => item.status !== 'stable_pass')" class="stability-cases">
              <span>需复核</span>
              <button v-for="item in governance.experiment.stability.filter(item => item.status !== 'stable_pass')" :key="item.case_id" :class="item.status" @click="focusCase(item.case_id)">
                {{ item.case_id }} · {{ item.status === "stable_fail" ? "稳定失败" : "波动" }}
              </button>
            </div>
          </section>
          <div class="experiment-runs">
            <div class="experiment-run experiment-run-head"><span>Repeat</span><span>Run</span><span>Profile</span><span>缓存</span><span>表 Exact</span><span>范围安全</span><span>范围证据</span><span>结果</span><span>TTSQL</span><span>TTResult</span><span>候选有效/失败</span><span>候选等待</span></div>
            <div v-for="item in governance.experiment.runs" :key="item.run_id" class="experiment-run">
              <strong>R{{ item.repeat_index }}</strong><span class="mono">{{ item.run_id }}</span><span>{{ profileLabel(item.evaluation_profile) }}</span><span>{{ item.cache_state }}</span><span>{{ countFraction(item.tables_correct, item.total) }}</span><span>{{ countFraction(item.table_scope_safe, item.total) }}</span><span>{{ countFraction(item.table_scope_evidence_complete, item.total) }}</span><span>{{ countFraction(item.result_equivalent, item.total) }}</span><span>{{ number(item.tt_sql_p95) }} ms</span><span>{{ number(item.tt_result_p95) }} ms</span><span>{{ optionalNumber(item.candidate_valid_b) }}/{{ optionalNumber(item.candidate_failed) }}</span><span>{{ number(item.candidate_wait_total_ms) }} ms</span>
            </div>
          </div>
          <div class="checklist experiment-checklist" aria-label="实验准入清单">
            <div v-for="(passed, key) in governance.experiment.checks" :key="key" :class="passed ? 'pass' : isExperimentObservation(key) ? 'observe' : 'fail'">
              <span aria-hidden="true"><Check v-if="passed" :size="13" /><CircleAlert v-else-if="isExperimentObservation(key)" :size="13" /><X v-else :size="13" /></span>{{ experimentCheckLabel(key) }}
            </div>
          </div>
          <p v-if="governance.baseline_policy === 'observational_three_repeat'" class="experiment-note">
            该难度采用观测基线：以观测最差一次记录当前能力，工程参考目标仅用于诊断；低于目标仍会原样登记，不代表达到发布质量门槛。
          </p>
          <p v-else-if="governance.experiment.regressed_case_ids?.length" class="blocked-tip">回归用例：{{ governance.experiment.regressed_case_ids.join("、") }}</p>
          <p v-else-if="governance.experiment.baseline?.comparison_applicable" class="experiment-ok">当前未发现 approved baseline 已通过样本回退。</p>
          <p v-else class="experiment-note">
            {{ suiteLabel(governance.experiment.baseline?.candidate_suite) }} 与
            {{ suiteLabel(governance.experiment.baseline?.suite) }} 不跨套件逐题比较；
            当前按挑战集绝对门槛准入，approved 套件需独立运行验证。
          </p>
        </section>
        <section v-else class="experiment-setup" aria-label="建立三轮可比实验">
          <div><strong>尚未建立当前难度的三次 v2 实验</strong><p>依次执行下面三条命令；每次完成后刷新页面。单次 run 只能作为候选证据，不能单独登记为观测基线。</p></div>
          <div v-for="(command, index) in governance.experiment_setup?.commands || []" :key="command" class="command compact">
            <span>R{{ index + 1 }}</span><code>{{ command }}</code><button @click="copy(command)">复制</button>
          </div>
        </section>
        <div class="review-bar">
          <label for="baseline-review-note">审核说明</label>
          <input id="baseline-review-note" v-model="reviewNote" maxlength="500" placeholder="例如：数据负责人和分析负责人已复核" />
          <button class="btn reject" :disabled="reviewing" @click="review('reject')">驳回</button>
          <button class="btn approve" :disabled="reviewing || !governance.selected.eligible" @click="review('approve')">登记为{{ activeDifficultyLabel }}观测基线</button>
        </div>
        <p v-if="!governance.selected.eligible" class="blocked-tip">
          {{ governance.baseline_policy === "observational_three_repeat"
            ? "当前尚未形成三次同指纹、全量且证据完整的观测；稳定失败可以记录，但缺失证据不能批准。"
            : "当前 run 或所属 experiment 未通过全部准入项；必须完成三次同指纹评测并以最差一次满足发布门禁。" }}
        </p>
      </section>

      <section class="panel performance-panel">
        <div class="panel-head"><div><h2>节点性能与调用预算</h2><p>区分 SQL 可用、结果可用和增强内容完成时间；Attempt 状态与降级证据必须可守恒、可追溯。</p></div><span>{{ report.performance.degraded_cases }} 个降级用例</span></div>
        <div class="performance-grid">
          <div><span>TTSQL 平均 / P95</span><strong>{{ number(report.performance.tt_sql_mean) }} / {{ number(report.performance.tt_sql_p95) }} ms</strong></div>
          <div><span>TTResult 平均 / P95</span><strong>{{ number(report.performance.tt_result_mean) }} / {{ number(report.performance.tt_result_p95) }} ms</strong></div>
          <div><span>TTComplete 平均 / P95</span><strong>{{ number(report.performance.tt_complete_mean) }} / {{ number(report.performance.tt_complete_p95) }} ms</strong><small>时序异常 {{ report.performance.timing_invariant_violations?.length || 0 }}</small></div>
          <div><span>LLM 调用平均 / P95</span><strong>{{ number(report.performance.llm_calls_mean) }} / {{ number(report.performance.llm_calls_p95) }}</strong></div>
          <div><span>Token 平均 / P95</span><strong>{{ number(report.performance.tokens_mean) }} / {{ number(report.performance.tokens_p95) }}</strong></div>
          <div><span>批次估算成本</span><strong>{{ cost(report.performance.estimated_cost_total, report.performance.cost_currency, report.performance.cost_estimate_available) }}</strong></div>
          <div><span>Prompt payload</span><strong>{{ number(report.performance.prompt_tokens_before) }} → {{ number(report.performance.prompt_tokens_after) }}</strong><small>压缩 {{ percent(report.performance.prompt_compression_rate) }}</small></div>
          <div><span>LLM Attempt 对账</span><strong>{{ optionalNumber(report.performance.llm_attempts) }} 次</strong><small>成功 {{ optionalNumber(report.performance.llm_attempts_succeeded) }} · 失败 {{ optionalNumber(report.performance.llm_attempts_failed) }} · 超时 {{ optionalNumber(report.performance.llm_attempts_timed_out) }} · 取消 {{ optionalNumber(report.performance.llm_attempts_cancelled) }}</small></div>
          <div :class="report.performance.llm_attempt_accounting_complete === true ? 'evidence-pass' : 'evidence-fail'"><span>Attempt 证据</span><strong>{{ evidenceLabel(report.performance.llm_attempt_accounting_complete) }}</strong><small>{{ report.performance.llm_attempt_schema }} · Token 未知 {{ optionalNumber(report.performance.llm_attempt_unknown_tokens) }} · callback 缺失 {{ optionalNumber(report.performance.llm_callback_missing) }}</small></div>
          <div :class="Number(report.performance.llm_attempt_coverage_missing_cases || 0) === 0 ? 'evidence-pass' : 'evidence-fail'"><span>调用覆盖</span><strong>{{ Number(report.performance.llm_attempt_coverage_missing_cases || 0) === 0 ? "完整" : "有缺口" }}</strong><small>缺证据用例 {{ optionalNumber(report.performance.llm_attempt_coverage_missing_cases) }} · 调用前失败 {{ optionalNumber(report.performance.llm_pre_invoke_failures) }}</small></div>
          <div :class="Number(report.performance.semantic_coverage_cases || 0) === Number(report.metrics.total || 0) && Number(report.performance.semantic_unresolved_cases || 0) === 0 ? 'evidence-pass' : 'evidence-fail'"><span>语义槽位覆盖</span><strong>{{ optionalNumber(report.performance.semantic_coverage_cases) }} / {{ optionalNumber(report.metrics.total) }}</strong><small>平均 {{ percent(report.performance.semantic_coverage_mean) }} · 未闭合 {{ optionalNumber(report.performance.semantic_unresolved_cases) }} · 不安全编译 {{ optionalNumber(report.performance.semantic_unsafe_compile_cases) }}</small></div>
          <div :class="report.performance.request_deadline_contract_complete === true && Number(report.performance.request_deadline_exhausted_cases || 0) === 0 ? 'evidence-pass' : 'evidence-fail'"><span>共享截止时间</span><strong>{{ evidenceLabel(report.performance.request_deadline_contract_complete) }}</strong><small>耗尽 {{ optionalNumber(report.performance.request_deadline_exhausted_cases) }} 题 · 仅保存相对预算</small></div>
          <div><span>排队 / Provider P95</span><strong>{{ number(report.performance.llm_queue_wait_p95_ms) }} / {{ number(report.performance.llm_provider_p95_ms) }} ms</strong><small>恢复 {{ optionalNumber(report.performance.llm_recoveries) }} · Provider ID 不可用 {{ optionalNumber(report.performance.llm_provider_request_id_unavailable) }}</small></div>
          <div><span>调用结果分类</span><strong>{{ outcomeSummary(report.performance.llm_outcome_classes) }}</strong><small>Token 时序不可用 {{ optionalNumber(report.performance.llm_token_timing_unavailable) }} 次，明确展示而不猜测</small></div>
          <div><span>核心 / 可选模型失败</span><strong>{{ optionalNumber(report.performance.failed_core_llm_calls) }} / {{ optionalNumber(report.performance.failed_optional_llm_calls) }}</strong><small>核心失败阻断发布；可选失败按 profile 判定</small></div>
          <div :class="report.performance.degradation_evidence_complete === true ? 'evidence-pass' : 'evidence-fail'"><span>核心 / 可选降级</span><strong>{{ optionalNumber(report.performance.core_degradation_count) }} / {{ optionalNumber(report.performance.optional_degradation_count) }}</strong><small>证据{{ evidenceLabel(report.performance.degradation_evidence_complete) }}</small></div>
          <div><span>自适应候选</span><strong>{{ percent(report.performance.adaptive_candidate_rate) }}</strong><small>选择 B {{ report.performance.candidate_selected_b }} · 有效 B {{ report.performance.candidate_valid_b }} · 修复 {{ report.performance.candidate_repaired }}</small></div>
          <div><span>候选失败 / 等待 P95</span><strong>{{ report.performance.candidate_failed }} / {{ number(report.performance.candidate_wait_p95_ms) }} ms</strong><small>累计等待 {{ number(report.performance.candidate_wait_total_ms) }} ms</small></div>
          <div><span>结构化输出降级</span><strong>{{ report.performance.structured_output_fallbacks }}</strong><small>核心 SQL 路径要求 0</small></div>
        </div>
        <div v-if="report.performance.llm_route_performance?.length" class="node-table route-table" aria-label="模型路由性能">
          <div class="route-row node-head"><span>节点</span><span>Route / 模型</span><span>调用 / 成功</span><span>失败 / 超时 / 取消</span><span>未知 Token</span><span>模型 P95</span></div>
          <div v-for="node in report.performance.llm_route_performance" :key="`${node.node_id}-${node.route}`" class="route-row">
            <strong>{{ node.node_id }}</strong><span>{{ node.route }} · {{ node.models?.join(' / ') }}</span><span>{{ node.calls }} / {{ optionalNumber(node.succeeded) }}</span><span>{{ optionalNumber(node.errors) }} / {{ optionalNumber(node.timed_out) }} / {{ optionalNumber(node.cancelled) }}</span><span>{{ optionalNumber(node.unknown_tokens) }}</span><span>{{ number(node.latency_p95_ms) }} ms</span>
          </div>
        </div>
        <div v-if="report.performance.node_performance?.length" class="node-table">
          <div class="node-row node-head"><span>节点</span><span>调用</span><span>平均</span><span>P95</span><span>累计</span></div>
          <div v-for="node in report.performance.node_performance" :key="node.node_id" class="node-row">
            <strong>{{ node.node_id }}</strong><span>{{ node.calls }}</span><span>{{ number(node.mean_ms) }} ms</span><span>{{ number(node.p95_ms) }} ms</span><span>{{ number(node.total_ms) }} ms</span>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><div><h2>根因分布</h2><p>仅统计结果不一致用例；一题只分配一个首要根因，避免重复计数。</p></div><span class="failure-total">{{ report.summary.failed_count }} 个失败</span></div>
        <div v-if="report.summary.root_causes.length" class="cause-grid">
          <article v-for="cause in report.summary.root_causes" :key="cause.code" class="cause-card" :class="cause.code">
            <div class="cause-top"><span>{{ cause.label }}</span><strong>{{ cause.count }}</strong></div>
            <div class="stage">阶段 · {{ cause.stage }}</div>
            <div class="case-chips"><button v-for="id in cause.case_ids" :key="id" @click="focusCase(id)">{{ id }}</button></div>
            <p>{{ cause.recommendation }}</p>
          </article>
        </div>
        <div v-else class="success-state">全部用例结果一致，当前没有待处理根因。</div>
      </section>

      <section v-if="report.summary.targeted_actions.length" class="panel">
        <div class="panel-head"><div><h2>定向优化与回归</h2><p>先按根因修改对应节点，再只跑受影响切片；通过后必须执行全量防回归。</p></div></div>
        <div class="action-list">
          <article v-for="action in report.summary.targeted_actions" :key="action.cause" class="action-card">
            <div class="action-index">{{ action.case_ids.length }}</div>
            <div class="action-main"><h3>{{ action.title }}</h3><p>{{ action.action }}</p><span>责任阶段：{{ action.owner_stage }} · 用例：{{ action.case_ids.join("、") }}</span><div class="command compact"><code>{{ action.regression_command }}</code><button @click="copy(action.regression_command)">复制</button></div></div>
          </article>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><div><h2>逐题证据</h2><p>对照生成 SQL、期望 SQL和阶段快照，确认归因是否符合业务事实。</p></div><span>{{ report.cases.length }} 条</span></div>
        <div class="case-list">
          <details v-for="item in report.cases" :id="`quality-${item.id}`" :key="item.id" class="case-row">
            <summary>
              <span class="status-dot" :class="item.result_equivalent ? 'pass' : 'fail'"></span>
              <strong>{{ item.id }}</strong><span class="question">{{ item.question }}</span>
              <span class="cause-badge">{{ item.attribution.label }}</span><span class="confidence">{{ percent(item.attribution.confidence) }}</span>
            </summary>
            <div class="case-body">
              <div class="evidence"><strong>归因证据</strong><ul><li v-for="line in item.attribution.evidence || []" :key="line">{{ line }}</li><li v-if="!item.attribution.evidence?.length">暂无额外阶段证据</li></ul><p>{{ item.attribution.recommendation }}</p></div>
              <section class="case-stage-grid" :aria-label="`${item.id} SQL 阶段指标`">
                <article class="case-stage-card" :class="caseSqlTone(item, 'first')">
                  <span>First SQL 严格结果</span>
                  <strong>{{ caseSqlResultLabel(item, "first") }}</strong>
                  <small>{{ caseSqlDetail(item, "first") }}</small>
                </article>
                <article class="case-stage-card correction" :class="caseCorrectionTone(item.correction_metrics)">
                  <span>Correction 归因</span>
                  <strong>{{ correctionOutcomeLabel(item.correction_metrics) }}</strong>
                  <small>{{ correctionDetail(item.correction_metrics) }}</small>
                </article>
                <article class="case-stage-card" :class="caseSqlTone(item, 'final')">
                  <span>Final SQL 严格结果</span>
                  <strong>{{ caseSqlResultLabel(item, "final") }}</strong>
                  <small>{{ caseSqlDetail(item, "final") }}</small>
                </article>
              </section>
              <div class="sql-grid"><div><label>生成 SQL</label><pre>{{ item.gen_sql || item.error || "—" }}</pre></div><div><label>期望 SQL</label><pre>{{ item.expected_sql || "—" }}</pre></div></div>
              <div v-if="item.stage_evidence?.semantic_coverage?.schema_version || item.stage_evidence?.request_deadline?.contract_version" class="runtime-evidence-grid">
                <article v-if="item.stage_evidence?.semantic_coverage?.schema_version" :class="coverageTone(item.stage_evidence.semantic_coverage)">
                  <header><div><span>SemanticCoverageV1</span><strong>语义槽位覆盖 {{ percent(item.stage_evidence.semantic_coverage.coverage_score) }}</strong></div><b>{{ item.stage_evidence.semantic_coverage.compiler_supported ? "允许确定性编译" : "安全回退" }}</b></header>
                  <dl>
                    <div><dt>检测</dt><dd>{{ slotText(item.stage_evidence.semantic_coverage.detected_slots) }}</dd></div>
                    <div><dt>已表达</dt><dd>{{ slotText(item.stage_evidence.semantic_coverage.represented_slots) }}</dd></div>
                    <div><dt>未闭合</dt><dd>{{ slotText(item.stage_evidence.semantic_coverage.unresolved_slots) }}</dd></div>
                  </dl>
                  <small>问题明文 {{ item.stage_evidence.semantic_coverage.raw_question_persisted ? "异常保存" : "未保存" }} · 筛选值 {{ item.stage_evidence.semantic_coverage.filter_values_persisted ? "异常保存" : "未保存" }}</small>
                </article>
                <article v-if="item.stage_evidence?.request_deadline?.contract_version" :class="deadlineTone(item.stage_evidence.request_deadline)">
                  <header><div><span>RequestDeadlineV1</span><strong>剩余 {{ optionalNumber(item.stage_evidence.request_deadline.remaining_ms) }} ms</strong></div><b>{{ item.stage_evidence.request_deadline.contract_version }}</b></header>
                  <dl>
                    <div><dt>总预算</dt><dd>{{ optionalNumber(item.stage_evidence.request_deadline.budget_ms) }} ms</dd></div>
                    <div><dt>执行保留</dt><dd>{{ optionalNumber(item.stage_evidence.request_deadline.execution_reserve_ms) }} ms</dd></div>
                    <div><dt>阶段预算</dt><dd>{{ phaseBudgetText(item.stage_evidence.request_deadline.phase_budgets_ms) }}</dd></div>
                  </dl>
                  <small>绝对时间戳 {{ item.stage_evidence.request_deadline.absolute_timestamps_persisted ? "异常保存" : "未保存" }}</small>
                </article>
              </div>
              <details v-if="item.stage_evidence?.semantic_ir?.schema_version" class="semantic-ir">
                <summary>
                  <span>语义 IR</span>
                  <b>{{ item.stage_evidence.semantic_ir.intent }}</b>
                  <em :class="item.stage_evidence.semantic_ir.compiler?.supported ? 'compiled' : 'fallback'">
                    {{ item.stage_evidence.semantic_ir.compiler?.supported ? "AST 编译" : "LLM 回退" }}
                  </em>
                </summary>
                <pre>{{ formatIr(item.stage_evidence.semantic_ir) }}</pre>
              </details>
              <div v-if="item.stage_evidence?.plan" class="plan"><label>查询规划</label><p>{{ item.stage_evidence.plan }}</p></div>
              <div v-if="item.stage_evidence?.first_sql" class="evidence-chain">
                <div><span>召回</span><strong>{{ retrievalCount(item.stage_evidence) }} 项</strong></div>
                <div><span>首版 SQL</span><strong>已记录</strong></div>
                <div><span>纠错</span><strong>{{ item.stage_evidence.correction_sqls?.length || 0 }} 次</strong></div>
                <div><span>授权 SQL</span><strong>{{ item.stage_evidence.authorized_sql ? "已记录" : "未生成" }}</strong></div>
                <div><span>脱敏结果摘要</span><strong>{{ item.stage_evidence.result_summary?.row_count ?? 0 }} 行</strong></div>
              </div>
              <details v-if="item.llm_attempts?.length" class="attempt-details">
                <summary>LLM Attempt 证据 · {{ item.llm_attempts.length }} 次 · {{ profileLabel(item.evaluation_profile) }}</summary>
                <div class="attempt-row attempt-head"><span>节点 / Route</span><span>模型</span><span>状态 / 结果类</span><span>总耗时 / 节点预算</span><span>物理预算 / 恢复预留</span><span>排队 / Provider</span><span>截止前 / 后</span><span>调用序号 / 恢复</span><span>服务商请求 ID</span><span>Token / 路径</span></div>
                <div v-for="attempt in item.llm_attempts" :key="attempt.attempt_id" class="attempt-row">
                  <strong>{{ attempt.node_id }} · {{ attempt.route }}</strong>
                  <span>{{ attempt.model || "—" }}</span>
                  <span :class="`attempt-${attempt.status}`">{{ attemptStatusLabel(attempt.status) }} · {{ outcomeClassLabel(attempt.outcome_class) }}<small>{{ attemptReasonLabel(attempt.reason_code) }}</small></span>
                  <span>{{ number(attempt.latency_ms) }} / {{ optionalNumber(attempt.timeout_budget_ms) }} ms</span>
                  <span>{{ optionalNumber(attempt.physical_attempt_budget_ms) }} / {{ optionalNumber(attempt.recovery_reserve_ms) }} ms</span>
                  <span>{{ optionalNumber(attempt.queue_wait_ms) }} / {{ optionalNumber(attempt.provider_ms) }} ms</span>
                  <span>{{ optionalNumber(attempt.deadline_before_ms) }} / {{ optionalNumber(attempt.deadline_after_ms) }} ms</span>
                  <span>#{{ Number(attempt.retry_index || 0) + 1 }} · {{ attempt.recovery ? "恢复调用" : "首次调用" }}</span>
                  <span>{{ providerRequestLabel(attempt) }}<small v-if="!attempt.provider_request_id_available">{{ unavailableReasonLabel(attempt.provider_request_id_unavailable_reason) }}</small></span>
                  <span>{{ attempt.token_known === false ? "Token 未知" : `${optionalNumber(attempt.total_tokens)} tokens` }} · {{ attempt.affects_core_path ? "核心" : "可选" }}<small v-if="attempt.token_known === false">{{ unavailableReasonLabel(attempt.token_timing_unavailable_reason) }}</small></span>
                </div>
                <p class="attempt-coverage" :class="item.llm_attempt_coverage?.complete ? 'complete' : 'incomplete'">
                  覆盖{{ evidenceLabel(item.llm_attempt_coverage?.complete) }}
                  <template v-if="item.llm_attempt_coverage?.missing_failure_attempts?.length"> · 失败节点缺失：{{ item.llm_attempt_coverage.missing_failure_attempts.join("、") }}</template>
                  <template v-if="item.llm_attempt_coverage?.missing_success_attempts?.length"> · 成功节点缺失：{{ item.llm_attempt_coverage.missing_success_attempts.join("、") }}</template>
                </p>
              </details>
              <div v-if="item.degradations?.length" class="degradation-list">
                <strong>降级证据</strong>
                <div v-for="(degradation, index) in item.degradations" :key="`${degradation.node_id}-${index}`">
                  <span>{{ degradation.affects_core_path ? "核心" : "可选" }} · {{ degradation.node_id || "unknown" }} · {{ degradation.reason_code || "unknown" }}</span>
                  <p>{{ degradation.message || degradation.reason || "未记录说明" }}</p>
                </div>
              </div>
            </div>
          </details>
        </div>
      </section>

      <div class="full-run"><span>定向切片通过后执行全量评测：</span><div class="command compact"><code>{{ report.run_command }}</code><button @click="copy(report.run_command)">复制</button></div></div>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import { Check, CircleAlert, CircleCheck, ClipboardCheck, X } from "@lucide/vue";
import { requestJson } from "../utils/http.js";

const difficultyOptions = Object.freeze([
  { value: "easy", label: "简单" },
  { value: "medium", label: "中等" },
  { value: "hard", label: "困难" },
]);
const fallbackResultPolicies = Object.freeze({
  easy_sql_v1: Object.freeze({ caseCount: 50, resultRateExclusive: .95 }),
  medium_sql_v1: Object.freeze({ caseCount: 50, resultRateExclusive: .85 }),
  hard_sql_v1: Object.freeze({ caseCount: 50, resultRateExclusive: null }),
});
const difficultySuites = Object.freeze({
  easy: "easy_sql_v1",
  medium: "medium_sql_v1",
  hard: "hard_sql_v1",
});
const report = ref(null), governance = ref(null), loading = ref(false), reviewing = ref(false), error = ref(""), notice = ref("");
const baselines = ref([]), runs = ref([]), selectedRun = ref(""), difficulty = ref("easy");
const includePassed = ref(false);
const reviewNote = ref("");

const activeDifficultyLabel = computed(() => difficultyOptions.find((item) => item.value === difficulty.value)?.label || difficulty.value);
const baselineCards = computed(() => difficultyOptions.map((option) => {
  const baseline = baselines.value.find((item) => item.difficulty === option.value) || {};
  return {
    difficulty: option.value,
    difficulty_label: baseline.difficulty_label || option.label,
    dataset_suite: baseline.dataset_suite || "",
    suite_label: baseline.suite_label || "",
    available: baseline.available === true,
    status: baseline.status || "unreviewed",
    run_id: baseline.run_id || null,
    reviewed_at: baseline.reviewed_at || null,
    resolution_status: baseline.resolution_status || "not_approved",
    resolution_error: baseline.resolution_error || null,
    metrics: baseline.metrics || {},
  };
}));

function baselineFor(value) {
  return baselineCards.value.find((item) => item.difficulty === value) || { available: false };
}
function baselineTabLabel(item) {
  if (item?.resolution_error) return "基线制品异常";
  return item?.available ? "已有观测基线" : "尚未建立";
}
function baselineStateLabel(item) {
  if (item?.resolution_error) return "已审批但制品不可解析";
  return item?.available ? statusLabel(item.status) : "尚未建立";
}

const sqlIdentity = computed(() => {
  const currentReport = report.value || {};
  const run = currentReport.run || {};
  const caseWithProfile = (currentReport.cases || []).find((item) => hasObjectData(item?.model_profile));
  const caseWithManifest = (currentReport.cases || []).find((item) => hasObjectData(item?.stage_evidence?.llm_route_manifest?.sql));
  const sources = [
    run.model_profile,
    run.sql_route,
    currentReport.sql_route,
    run.provider_parameters?.sql_route,
    run.experiment?.provider_parameters?.sql_route,
    run.llm_route_manifest?.sql,
    currentReport.stage_evidence?.llm_route_manifest?.sql,
    caseWithProfile?.model_profile,
    caseWithManifest?.stage_evidence?.llm_route_manifest?.sql,
  ].filter(hasObjectData);
  const routeValue = (...keys) => firstDefined(...sources.flatMap((source) => keys.map((key) => source[key])));
  return {
    candidateLabel: firstDefined(
      run.candidate_label,
      run.experiment?.candidate_label,
      governance.value?.experiment?.candidate_label,
      routeValue("candidate_label", "candidate_id"),
    ) || "未记录",
    modelName: firstDefined(run.model_name, routeValue("model_name", "model_id", "deployment_id")) || "未记录",
    provider: firstDefined(run.provider, routeValue("provider", "model_provider"), run.model_provider) || "未记录",
  };
});

const firstSqlAggregate = computed(() => {
  const metrics = report.value?.metrics || {};
  const stage = hasObjectData(metrics.first_sql) ? metrics.first_sql : {};
  const count = numericOrNull(firstDefined(stage.result_equivalent, metrics.first_result_equivalent));
  const denominator = numericOrNull(firstDefined(stage.denominator, metrics.first_result_evaluated, metrics.total));
  const rate = numericOrNull(firstDefined(
    metrics.first_result_rate,
    metrics.first_sql_result_rate,
    ratio(count, denominator),
  ));
  const available = stage.evidence_complete === false
    ? false
    : stage.evidence_complete === true || rate !== null;
  return { available, count, denominator, rate };
});

const finalSqlAggregate = computed(() => {
  const metrics = report.value?.metrics || {};
  const stage = hasObjectData(metrics.final_sql) ? metrics.final_sql : {};
  const count = numericOrNull(firstDefined(stage.result_equivalent, metrics.final_result_equivalent, metrics.result_equivalent));
  const denominator = numericOrNull(firstDefined(stage.result_evaluated, metrics.final_result_evaluated, metrics.result_evaluated, stage.denominator, metrics.total));
  const rate = numericOrNull(firstDefined(
    metrics.final_result_rate,
    metrics.final_sql_result_rate,
    metrics.result_rate,
    ratio(count, denominator),
  ));
  return { available: rate !== null, count, denominator, rate };
});

const finalResultPolicy = computed(() => {
  const backend = governance.value?.experiment?.promotion_policy;
  const backendHasRate = hasObjectData(backend)
    && Object.prototype.hasOwnProperty.call(backend, "result_rate_exclusive");
  const suite = String(firstDefined(
    backend?.dataset_suite,
    report.value?.run?.dataset_suite,
    governance.value?.dataset_suite,
    difficultySuites[difficulty.value],
  ) || "");
  const fallback = fallbackResultPolicies[suite]
    || fallbackResultPolicies[difficultySuites[difficulty.value]]
    || fallbackResultPolicies.hard_sql_v1;
  const configuredRate = backendHasRate
    ? numericOrNull(backend.result_rate_exclusive)
    : fallback.resultRateExclusive;
  const caseCount = numericOrNull(backendHasRate ? backend.case_count : fallback.caseCount)
    || fallback.caseCount;
  const perRun = Array.isArray(backend?.per_run) ? backend.per_run : [];
  const currentRunId = report.value?.run?.run_id;
  const currentRunPolicy = perRun.find((item) => item?.run_id === currentRunId)
    || perRun.find((item) => numericOrNull(item?.minimum) !== null)
    || {};
  const minimum = configuredRate === null
    ? null
    : (numericOrNull(currentRunPolicy.minimum)
      ?? Math.floor(caseCount * configuredRate) + 1);
  const source = backendHasRate ? "后端治理策略" : "套件回退策略";

  if (configuredRate === null) {
    return {
      suite, caseCount, resultRateExclusive: null, minimum: null, source,
      note: `主要指标 · 仅观测，不设强制正确率门槛 · ${source}`,
    };
  }

  const minimumRate = caseCount > 0 ? minimum / caseCount : null;
  return {
    suite, caseCount, resultRateExclusive: configuredRate, minimum, source,
    note: `主要指标 · 门槛 ≥ ${minimum}/${caseCount}（${percent(minimumRate)}，严格 >${formatWholePercent(configuredRate)}） · ${source}`,
  };
});

const finalResultTone = computed(() => {
  const policy = finalResultPolicy.value;
  const aggregate = finalSqlAggregate.value;
  if (policy.resultRateExclusive === null) return "observe";
  if (!aggregate.available) return "neutral";
  if (aggregate.count !== null && aggregate.denominator !== null) {
    if (aggregate.denominator !== policy.caseCount) return "warn";
    return aggregate.count >= policy.minimum ? "good" : "bad";
  }
  return aggregate.rate > policy.resultRateExclusive ? "good" : "bad";
});

const correctionAggregate = computed(() => {
  const metrics = report.value?.metrics || {};
  const nested = hasObjectData(metrics.correction)
    ? metrics.correction
    : hasObjectData(report.value?.performance?.correction)
      ? report.value.performance.correction
      : {};
  const flatKeys = [
    "correction_triggered", "correction_fixed_execution", "correction_fixed_result",
    "correction_harmful_modification", "correction_unattributable",
  ];
  const hasFlatEvidence = flatKeys.some((key) => Object.prototype.hasOwnProperty.call(metrics, key));
  const explicitAvailability = firstDefined(nested.evidence_complete, metrics.correction_evidence_complete);
  const available = explicitAvailability === false
    ? false
    : explicitAvailability === true
      || hasFlatEvidence
      || (hasObjectData(nested) && firstSqlAggregate.value.available);
  return {
    available,
    denominator: numericOrNull(firstDefined(nested.denominator, metrics.total)),
    triggered: numericOrZero(firstDefined(nested.triggered, metrics.correction_triggered)),
    fixedExecution: numericOrZero(firstDefined(nested.fixed_execution, metrics.correction_fixed_execution)),
    fixedResult: numericOrZero(firstDefined(nested.fixed_result, metrics.correction_fixed_result)),
    harmful: numericOrZero(firstDefined(nested.harmful_modification, metrics.correction_harmful_modification)),
    unattributable: numericOrZero(firstDefined(nested.unattributable, metrics.correction_unattributable)),
  };
});

const metricCards = computed(() => {
  const metrics = report.value?.metrics || {};
  return [
    { label: "语法正确率", value: percent(metrics.syntax_rate), note: "工程参考目标 ≥ 99%", tone: metricTone(metrics.syntax_rate, .99, "warn") },
    { label: "执行成功率", value: percent(metrics.execution_rate), note: "工程参考目标 ≥ 98%", tone: metricTone(metrics.execution_rate, .98, "warn") },
    { label: "表路径 Exact（观察）", value: percent(metrics.table_exact_rate), note: "效率与可解释性诊断，不阻断 Final 正确性", tone: "neutral" },
    { label: "First SQL 严格结果", value: firstSqlAggregate.value.available ? percent(firstSqlAggregate.value.rate) : "—", note: firstSqlAggregate.value.available ? "首版生成，未计入纠错" : "旧制品未记录", tone: strictMetricTone(firstSqlAggregate.value.rate, firstSqlAggregate.value.available) },
    { label: "Final SQL 最终结果正确率", value: finalSqlAggregate.value.available ? percent(finalSqlAggregate.value.rate) : "—", note: finalResultPolicy.value.note, tone: finalResultTone.value, primary: true },
    { label: "平均表 F1", value: percent(metrics.table_f1_mean), note: "诊断指标", tone: "neutral" },
  ];
});
const fingerprintRows = computed(() => {
  const item = governance.value?.selected?.fingerprint || {};
  return [
    ["数据集", item.dataset_version], ["DW 快照", item.dw_snapshot], ["Schema", item.schema_version],
    ["元知识", item.knowledge_version], ["索引", item.index_version], ["配置", item.config_hash],
    ["Git commit", item.git_commit], ["代码内容", item.code_version], ["模型", item.model?.model_name],
  ].map(([label, value]) => ({ label, value: value || "缺失" }));
});

async function load() {
  loading.value = true; error.value = "";
  try {
    const [baselineBody, runBody] = await Promise.all([
      requestJson("/api/admin/evaluation/nl2sql/baselines", {}, "三难度基线加载失败"),
      requestJson(`/api/admin/evaluation/nl2sql/runs?limit=100&difficulty=${encodeURIComponent(difficulty.value)}`, {}, "评测批次加载失败"),
    ]);
    baselines.value = Array.isArray(baselineBody?.baselines) ? baselineBody.baselines : [];
    runs.value = Array.isArray(runBody) ? runBody : (runBody?.runs || []);
    if (selectedRun.value && !runs.value.some((item) => item.run_id === selectedRun.value)) selectedRun.value = "";
    const query = new URLSearchParams({ include_passed: String(includePassed.value) });
    query.set("difficulty", difficulty.value);
    if (selectedRun.value) query.set("run_id", selectedRun.value);
    const body = await requestJson(`/api/admin/evaluation/nl2sql/latest?${query}`, {}, "评测归因加载失败");
    report.value = body;
    if (body.available && !selectedRun.value) selectedRun.value = body.run.run_id;
    if (body.available) await loadGovernance(body.run.run_id);
    else governance.value = null;
  } catch (e) { error.value = e.message || "评测归因加载失败"; }
  finally { loading.value = false; }
}
async function loadGovernance(runId) {
  governance.value = await requestJson(
    `/api/admin/evaluation/nl2sql/governance?run_id=${encodeURIComponent(runId)}&difficulty=${encodeURIComponent(difficulty.value)}`,
    {},
    "治理信息请求失败",
  );
}
async function selectDifficulty(value) {
  if (!difficultyOptions.some((item) => item.value === value) || value === difficulty.value) return;
  difficulty.value = value;
  selectedRun.value = "";
  report.value = null;
  governance.value = null;
  notice.value = "";
  await load();
}
async function review(decision) {
  if (!governance.value?.selected?.run?.run_id) return;
  reviewing.value = true; error.value = "";
  try {
    await requestJson(`/api/admin/evaluation/nl2sql/baselines/${encodeURIComponent(governance.value.selected.run.run_id)}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, note: reviewNote.value }),
    }, "基线审核失败");
    notice.value = decision === "approve" ? `已登记为当前工作空间${activeDifficultyLabel.value}观测基线` : "已驳回该候选 run";
    await load();
  } catch (e) { error.value = e.message || "基线审核失败"; }
  finally { reviewing.value = false; }
}
async function initialize() {
  await load();
}
function firstDefined(...values) { return values.find((value) => value !== null && value !== undefined && value !== ""); }
function hasObjectData(value) { return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length); }
function numericOrNull(value) { if (value === null || value === undefined || value === "") return null; const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; }
function numericOrZero(value) { return numericOrNull(value) ?? 0; }
function ratio(count, denominator) { const numeratorValue = numericOrNull(count); const denominatorValue = numericOrNull(denominator); return numeratorValue !== null && denominatorValue > 0 ? numeratorValue / denominatorValue : null; }
function metricTone(value, target, failureTone = "bad") { const parsed = numericOrNull(value); return parsed === null ? "neutral" : parsed >= target ? "good" : failureTone; }
function strictMetricTone(value, available = true) { return available ? metricTone(value, .85) : "neutral"; }
function strictAggregateCountLabel(value) { return value.count !== null && value.denominator !== null ? `${number(value.count)} / ${number(value.denominator)} 题` : "严格结果分母未记录"; }
function correctionCountLabel(value) { return value.denominator !== null ? `${number(value.triggered)} / ${number(value.denominator)} 题触发` : `${number(value.triggered)} 题触发`; }
function metricObject(item, stage) { const value = item?.[stage === "first" ? "first_sql_metrics" : "final_sql_metrics"]; return hasObjectData(value) ? value : null; }
function caseSqlResult(item, stage) {
  const metrics = metricObject(item, stage);
  if (metrics) {
    if (metrics.result_evaluated === false) return null;
    if (typeof metrics.result_equivalent === "boolean") return metrics.result_equivalent;
  }
  if (stage === "final" && typeof item?.result_equivalent === "boolean") return item.result_equivalent;
  return undefined;
}
function caseSqlResultLabel(item, stage) {
  const result = caseSqlResult(item, stage);
  if (result === true) return "结果一致";
  if (result === false) return "结果不一致";
  return result === null ? "未评估" : "旧制品未记录";
}
function caseSqlTone(item, stage) {
  const result = caseSqlResult(item, stage);
  return result === true ? "good" : result === false ? "bad" : result === null ? "warn" : "neutral";
}
function booleanEvidenceLabel(value, trueLabel, falseLabel) { return value === true ? trueLabel : value === false ? falseLabel : "未记录"; }
function caseSqlDetail(item, stage) {
  const metrics = metricObject(item, stage);
  if (!metrics) return stage === "final" && typeof item?.result_equivalent === "boolean" ? "沿用历史最终结果字段" : "旧制品未记录";
  const execution = metrics.execution_attempted === false
    ? "未执行"
    : booleanEvidenceLabel(metrics.exec_success, "成功", "失败");
  return `闸门 ${booleanEvidenceLabel(metrics.guard_passed, "通过", "未通过")} · 执行 ${execution}`;
}
function correctionOutcomeLabel(value) {
  if (!hasObjectData(value)) return "旧制品未记录";
  if (value.triggered === false || value.outcome === "not_triggered") return "未触发";
  if (value.attribution_valid === false || value.outcome === "not_attributable") return "不可直接归因";
  return ({ fixed_result: "修复严格结果", fixed_execution: "修复执行", harmful: "有害修改", no_strict_gain: "无严格收益", pipeline_failed: "纠错链路失败" })[value.outcome]
    || (value.fixed_result ? "修复严格结果" : value.fixed_execution ? "修复执行" : value.harmful_modification ? "有害修改" : "已触发");
}
function correctionAttributionLabel(value) { return ({ direct: "直接归因", not_triggered: "未触发", downstream_sql_revision_present: "存在下游 SQL 改写" })[value] || value || "归因状态未记录"; }
function correctionDetail(value) {
  if (!hasObjectData(value)) return "旧制品未记录";
  if (value.triggered === false || value.outcome === "not_triggered") return "未产生纠错调用";
  const reasons = Array.isArray(value.trigger_reason_codes) && value.trigger_reason_codes.length ? value.trigger_reason_codes.join("、") : "原因未记录";
  return `${correctionAttributionLabel(value.attribution_status)} · ${reasons} · LLM ${optionalNumber(value.llm_attempts)} 次 · +${optionalNumber(value.additional_tokens)} Token · +${optionalNumber(value.additional_latency_ms)} ms`;
}
function caseCorrectionTone(value) {
  if (!hasObjectData(value) || value.triggered === false || value.outcome === "not_triggered") return "neutral";
  if (value.outcome === "fixed_result" || value.outcome === "fixed_execution" || value.fixed_result || value.fixed_execution) return "good";
  if (value.outcome === "harmful" || value.harmful_modification) return "bad";
  return "warn";
}
function percent(value) { const numberValue = numericOrNull(value); return numberValue !== null ? `${(numberValue * 100).toFixed(1)}%` : "—"; }
function formatWholePercent(value) { const numberValue = numericOrNull(value); return numberValue !== null ? `${(numberValue * 100).toLocaleString("zh-CN", { maximumFractionDigits: 2 })}%` : "—"; }
function number(value) { const n = Number(value); return Number.isFinite(n) ? n.toLocaleString("zh-CN", { maximumFractionDigits: 1 }) : "—"; }
function optionalNumber(value) { return value === null || value === undefined ? "—" : number(value); }
function profileLabel(value) { return ({ "core-default": "核心默认", "enhanced-analysis": "增强解读", "legacy/unknown": "历史未知" })[value] || value || "历史未知"; }
function suiteLabel(value, providedLabel = "") {
  return providedLabel || ({
    easy_sql_v1: "简单 SQL 基线集",
    medium_sql_v1: "中等 SQL 基线集",
    hard_sql_v1: "困难 SQL 基线集",
  })[value] || value || "尚未绑定套件";
}
function artifactName(value) { return value ? String(value).split(/[\\/]/).pop() : "未绑定"; }
function evidenceLabel(value) { return value === true ? "完整" : value === false ? "不完整" : "未知"; }
function attemptStatusLabel(value) { return ({ success: "成功", error: "错误", timeout: "超时", cancelled: "取消", empty_output: "空输出" })[value] || value || "未知"; }
function attemptReasonLabel(value) { return ({ "llm.success": "完成", "llm.timeout": "预算超时", "llm.deadline_exhausted": "共享预算耗尽", "llm.queue_timeout": "排队超时", "llm.cancelled": "请求取消", "llm.connection_error": "连接失败", "llm.rate_limited": "服务限流", "llm.server_error": "服务端异常", "llm.stream_interrupted": "流式响应中断", "llm.truncated_output": "输出截断", "llm.invalid_structured_output": "结构化输出无效", "llm.refusal": "模型拒答", "llm.empty_output": "空输出", "llm.provider_error": "服务错误", "llm.pre_invoke_failure": "调用前失败", "llm.unknown_failure": "未知失败" })[value] || value || "未知原因"; }
function outcomeClassLabel(value) { return ({ success: "完成", deadline_exhausted: "截止时间耗尽", queue_timeout: "排队超时", timeout: "Provider 超时", rate_limited: "限流", server_error: "服务异常", connection_error: "连接异常", stream_interrupted: "流中断", empty_output: "空输出", truncated_output: "截断", invalid_structured_output: "格式无效", refusal: "拒答", cancelled: "取消" })[value] || value || "未分类"; }
function outcomeSummary(value) { const rows = Object.entries(value || {}).filter(([, count]) => Number(count) > 0); return rows.length ? rows.map(([key, count]) => `${outcomeClassLabel(key)} ${count}`).join(" · ") : "无模型调用"; }
function providerRequestLabel(attempt) { return attempt?.provider_request_id_available ? attempt.provider_request_id : "服务商请求 ID 不可用"; }
function unavailableReasonLabel(value) { return ({ provider_response_unavailable_after_timeout: "超时后无服务商响应", provider_request_id_not_exposed: "服务商 SDK 未提供请求 ID", callback_not_observed: "未观测到模型回调", token_timing_not_exposed: "服务商未提供 Token 时序", no_token_usage_callback: "未收到 Token 用量回调" })[value] || value || "未记录原因"; }
function slotText(value) { return (value || []).join(" · ") || "无"; }
function phaseBudgetText(value) { return Object.entries(value || {}).map(([key, budget]) => `${key} ${number(budget)} ms`).join(" · ") || "未记录"; }
function coverageTone(value) { return !value?.unresolved_slots?.length && Number(value?.coverage_score) === 1 ? "evidence-ok" : "evidence-warn"; }
function deadlineTone(value) { return Number(value?.remaining_ms) > 0 && value?.absolute_timestamps_persisted === false ? "evidence-ok" : "evidence-warn"; }
function semanticFamilyLabel(value) { return ({
  aggregate: "聚合 / 排名",
  above_global_average: "高于全局均值",
  above_group_average: "高于组内均值",
  above_average_customer: "客户均值子查询",
  above_average_brand: "品牌均值子查询",
  activation_rate: "激活率",
  active_both_periods: "跨期活跃",
  anti_join: "反连接",
  average_order_value: "客单价",
  cohort_active: "Cohort 活跃",
  cohort_new: "Cohort 新增",
  cohort_split: "Cohort 分层",
  cumulative_share_threshold: "累计贡献阈值",
  customer_value: "客户价值",
  dual_rank: "双重排名",
  first_event: "首次事件",
  group_share: "分组占比",
  having_threshold: "聚合阈值",
  inventory_status: "库存状态",
  last_event_cohort: "最近事件 Cohort",
  multi_turn: "多轮闭合",
  multiple_metrics: "多指标聚合",
  per_entity_ratio: "实体级比率",
  period_compare: "周期比较",
  period_inactive: "期间未活跃",
  period_presence: "期间存在性",
  qualified_entity_count: "达标实体数",
  quantile_segmentation: "分位数分层",
  repeat_customer_count: "复购客户数",
  repeat_rate: "复购率",
  segmentation: "分段统计",
  set_coverage: "集合覆盖",
  set_intersection: "集合交集",
  top_per_group: "组内 Top-N",
  top_share: "Top-N 贡献率",
  window_average: "窗口均值",
  window_metric: "窗口指标",
  window_mom: "环比窗口",
  window_running_total: "累计窗口",
  unclassified: "未分类",
})[value] || value || "未分类"; }
function shortHash(value) { return value ? `${value.slice(0, 18)}…${value.slice(-8)}` : "尚未生成"; }
function statusLabel(value) { return ({ unreviewed: "未登记", candidate: "候选", approved: "已登记", rejected: "已驳回", superseded: "已被替代" })[value] || value || "未登记"; }
function cost(value, currency, available) {
  const parsed = numericOrNull(value);
  if (available === false || (available !== true && parsed === 0)) return "未知（未配置价格）";
  if (parsed === null) return available === true ? "—" : "未知（未配置价格）";
  return `${number(parsed)} ${currency || "USD"}`;
}
function checklistLabel(value) { return ({ valid_run: "有效运行", full_dataset: "全量样本", fingerprint_complete: "指纹完整", sample_contract_complete: "样本契约完整", samples_reviewed: "样本已审核", infrastructure_clean: "基础设施无异常", evidence_chain_complete: "证据链完整", no_raw_result_rows: "不含原始结果行", evaluation_profile_complete: "评测 Profile 明确", llm_attempt_accounting_complete: "Attempt 状态守恒", degradation_evidence_complete: "降级证据完整" })[value] || value; }
function experimentStatusLabel(value) {
  const experiment = hasObjectData(value) ? value : { status: value };
  const status = experiment.status;
  const repeatCount = numericOrNull(experiment.repeat_count);
  const requiredRepeats = numericOrNull(experiment.required_repeats);
  if (status === "incomplete") {
    return repeatCount !== null && requiredRepeats !== null && repeatCount >= requiredRepeats
      ? "三轮已完成，阻断证据未闭合"
      : "等待其余 Run";
  }
  return ({ qualified: "满足工程参考目标", recordable: "观测完整，可登记", blocked: "观测证据不完整", not_started: "尚未开始" })[status] || status || "未知";
}
function countFraction(value, total) {
  const count = numericOrNull(value);
  const denominator = numericOrNull(total);
  if (count === null && denominator === null) return "—";
  if (count === null) return `—/${number(denominator)}`;
  if (denominator === null) return number(count);
  return `${number(count)}/${number(denominator)}`;
}
function experimentScopeLabel(experiment) {
  const total = numericOrNull(experiment?.promotion_policy?.case_count)
    ?? numericOrNull(experiment?.runs?.[0]?.total);
  const safe = numericOrNull(experiment?.worst?.table_scope_safe);
  const evidence = numericOrNull(experiment?.worst?.table_scope_evidence_complete);
  if (safe === null && evidence === null) return "未记录";
  return `${countFraction(safe, total)} / ${countFraction(evidence, total)}`;
}
function isExperimentObservation(value) { return Object.prototype.hasOwnProperty.call(governance.value?.experiment?.observational_checks || {}, value); }
function experimentCheckLabel(value) { return ({ metadata_complete: "协议元数据", three_repeats: "三次完整运行", repeat_indexes: "重复序号 1/2/3", cache_protocol: "cold/warm/warm", same_workspace: "同一工作空间", same_candidate: "同一候选参数", same_evaluation_profile: "同一评测 Profile", full_dataset: "全量当前套件", valid_runs: "运行有效", infrastructure_clean: "无基础设施错误", fingerprint_v2: "evaluation-v2", fingerprints_complete: "指纹完整", fingerprints_comparable: "三次指纹一致", result_denominator_complete: "结果分母完整", run_checklists_eligible: "单次证据链完整", approved_baseline_available: "已绑定 approved baseline", baseline_non_regression: "基线保护策略", table_exact_non_regression: "表路径 Exact（观察）", table_scope_safety: "表范围安全 100%", table_scope_evidence_complete: "表范围证据完整", security_guard_integrity: "安全守卫无绕过", syntax_floor_gate: "语法正确率下限", execution_floor_gate: "执行成功率下限", result_floor_non_regression: "结果一致准入", ttsql_gate: "TTSQL ≤15s", ttresult_gate: "TTResult ≤20s", ttcomplete_gate: "TTComplete ≤30s", timing_protocol_complete: "timing-v2 与时序不变量", routing_evidence_complete: "模型路由证据", prompt_compression_evidence: "Prompt 压缩证据", llm_attempt_accounting_complete: "Attempt 状态守恒", llm_attempt_coverage_complete: "LLM 节点覆盖完整", semantic_coverage_evidence_complete: "语义覆盖证据完整", request_deadline_evidence_complete: "共享截止时间证据", llm_outcome_taxonomy_complete: "调用结果分类完整", llm_recovery_bounded: "恢复调用有界", degradation_evidence_complete: "降级证据完整", failed_llm_calls_zero: "核心模型调用无失败", optional_postprocess_clean: "增强解读无失败", structured_output_clean: "结构化输出无降级", candidate_trigger_gate: "第二候选 ≤20%", candidate_safety_gate: "候选安全", semantic_review_first_correct_final_incorrect_zero: "复核不损失首版正确结果", semantic_review_same_sql_execution_loss_zero: "同 SQL 不重复执行损失", semantic_review_selected_b_no_sql_delta_zero: "候选 B 必须有 SQL 差异", token_non_regression: "Token 不回退", token_max_gate: "单题 Token 峰值" })[value] || value; }
function baselineSourceLabel(value) {
  if (!value) return "未解析来源";
  const label = ({ meta_mysql: "Meta MySQL 已审批", explicit: "显式选择", none: "尚未建立" })[value.source] || value.source;
  return `${label} · ${value.schema || "unknown"}`;
}
function retrievalCount(evidence) { return Object.values(evidence?.retrieval || {}).reduce((total, items) => total + (Array.isArray(items) ? items.length : 0), 0); }
function formatIr(value) { return JSON.stringify(value, null, 2); }
async function copy(value) { try { await navigator.clipboard.writeText(value); notice.value = "命令已复制"; window.setTimeout(() => { notice.value = ""; }, 2200); } catch { error.value = "浏览器未授予剪贴板权限，请手动复制。"; } }
function focusCase(id) { const element = document.getElementById(`quality-${id}`); if (!element) return; element.open = true; element.scrollIntoView({ behavior: "smooth", block: "center" }); }
onMounted(initialize);
</script>

<style scoped>
.difficulty-tabs,.baseline-track-grid{max-width:1240px;margin-left:auto;margin-right:auto}.difficulty-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;padding:5px;border:1px solid var(--da-border-light);border-radius:14px;background:rgba(255,255,255,.72)}.difficulty-tabs button{display:flex;align-items:center;justify-content:center;gap:9px;min-height:45px;padding:8px 14px;border:1px solid transparent;border-radius:10px;background:transparent;color:var(--da-text-secondary);font:inherit;cursor:pointer}.difficulty-tabs button span{font-size:13px;font-weight:700}.difficulty-tabs button small{color:var(--da-text-muted);font-size:10px}.difficulty-tabs button:hover:not(:disabled){border-color:#c8d9f2;background:#f5f9ff}.difficulty-tabs button.active{border-color:#a8c5ed;background:#edf5ff;color:var(--da-accent-strong);box-shadow:var(--da-shadow-sm)}.difficulty-tabs button.active small{color:#5a78a0}.difficulty-tabs button:focus-visible,.baseline-track-card header button:focus-visible{outline:3px solid rgba(55,126,241,.25);outline-offset:2px}.difficulty-tabs button:disabled{cursor:default}.baseline-track-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:16px}.baseline-track-card{min-width:0;padding:15px;border:1px solid var(--da-border-light);border-radius:14px;background:rgba(255,255,255,.9);box-shadow:var(--da-shadow-sm);transition:border-color .15s ease,box-shadow .15s ease}.baseline-track-card.active{border-color:#91b8ed;box-shadow:0 0 0 2px rgba(55,126,241,.08),var(--da-shadow-sm)}.baseline-track-card.missing{background:rgba(250,251,253,.82)}.baseline-track-card>header{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.baseline-track-card>header span,.baseline-track-card>header strong{display:block}.baseline-track-card>header span{color:var(--da-text-muted);font-size:10px}.baseline-track-card>header strong{margin-top:3px;color:#247848;font-size:12px}.baseline-track-card.missing>header strong{color:#956514}.baseline-track-card>header button{padding:4px 8px;border:1px solid var(--da-border);border-radius:7px;background:#fff;color:var(--da-accent);font:10px inherit;cursor:pointer}.baseline-track-card>header button:disabled{border-color:transparent;background:var(--da-accent-soft);color:var(--da-text-muted);cursor:default}.baseline-track-run{min-width:0;margin-top:12px}.baseline-track-run b,.baseline-track-run small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.baseline-track-run b{font-size:11px}.baseline-track-run small{margin-top:4px;color:var(--da-text-muted);font-size:10px}.baseline-track-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px}.baseline-track-metrics span{padding:7px;border-radius:8px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:9px}.baseline-track-metrics strong{display:block;margin-top:3px;color:var(--da-text);font-size:11px}.baseline-track-card>footer{display:flex;align-items:center;flex-wrap:wrap;gap:4px 10px;margin-top:10px;color:var(--da-text-muted);font-size:9px}.baseline-track-card>footer time{margin-left:auto}
.evaluation-page{flex:1 1 auto;min-height:0;overflow:auto;padding:38px 34px 70px;background:transparent;color:var(--da-text)}.evaluation-head,.panel,.metric-grid,.run-strip,.alert,.empty-state,.full-run{max-width:1240px;margin-left:auto;margin-right:auto}.evaluation-head{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:24px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:750;letter-spacing:.17em}.evaluation-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.evaluation-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.head-actions{display:flex;align-items:center;gap:10px}.switch{display:flex;align-items:center;gap:7px;color:var(--da-text-muted);font-size:12px;white-space:nowrap}.btn{min-height:38px;padding:9px 14px;border:1px solid var(--da-border);border-radius:10px;background:#fff;color:var(--da-text-secondary);font:inherit;font-size:13px;cursor:pointer}.btn:hover{border-color:var(--da-accent);color:var(--da-accent)}.alert{margin-bottom:14px;padding:11px 14px;border-radius:11px;font-size:13px}.alert.error{border:1px solid #f0c5bf;background:#fff3f1;color:#bd3b2c}.alert.notice{border:1px solid #b9e2c8;background:#f0fbf4;color:#287b4a}.loading,.empty-state{min-height:360px;display:grid;place-items:center;color:var(--da-text-muted)}.empty-state{display:flex;flex-direction:column;justify-content:center;padding:45px;border:1px dashed var(--da-border);border-radius:17px;background:rgba(255,255,255,.65);text-align:center}.empty-icon{width:48px;height:48px;display:grid;place-items:center;border-radius:15px;background:var(--da-accent-soft);color:var(--da-accent);font-size:25px}.empty-state h2{margin:16px 0 0;font-size:19px}.empty-state p{margin:8px 0 18px;color:var(--da-text-muted);font-size:13px}.empty-state small{margin-top:12px;color:var(--da-text-muted)}.run-strip{display:flex;flex-wrap:wrap;gap:0;margin-bottom:15px;padding:13px 17px;border:1px solid var(--da-border-light);border-radius:13px;background:rgba(255,255,255,.76)}.run-strip div{display:flex;align-items:baseline;gap:8px;padding:2px 22px;border-right:1px solid var(--da-border-light)}.run-strip div:first-child{padding-left:0}.run-strip div:last-child{border:0}.run-strip span{color:var(--da-text-muted);font-size:11px}.run-strip strong{font-size:12px}.metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}.metric-card{position:relative;overflow:hidden;padding:19px;border:1px solid var(--da-border-light);border-radius:15px;background:#fff;box-shadow:var(--da-shadow-sm)}.metric-card:before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:#aab4c2}.metric-card.good:before{background:#35a56a}.metric-card.warn:before{background:#e2a63b}.metric-card.bad:before{background:#dc6656}.metric-card span,.metric-card small{display:block;color:var(--da-text-muted);font-size:11px}.metric-card strong{display:block;margin:7px 0 4px;font-size:25px;letter-spacing:-.04em}.panel{margin-bottom:18px;padding:22px;border:1px solid var(--da-border-light);border-radius:17px;background:rgba(255,255,255,.9);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:17px}.panel-head h2{margin:0;font-size:17px}.panel-head>span,.failure-total{padding:5px 9px;border-radius:999px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:11px}.cause-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.cause-card{padding:16px;border:1px solid var(--da-border-light);border-radius:13px;background:#fbfcfe}.cause-top{display:flex;justify-content:space-between;gap:10px}.cause-top span{font-weight:650;font-size:14px}.cause-top strong{color:var(--da-danger);font-size:20px}.stage{margin-top:5px;color:var(--da-text-muted);font-size:11px}.case-chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px}.case-chips button{padding:3px 7px;border:0;border-radius:6px;background:var(--da-accent-soft);color:var(--da-accent);font:11px Consolas,monospace;cursor:pointer}.cause-card p{margin:12px 0 0;color:var(--da-text-secondary);font-size:12px;line-height:1.55}.action-list{display:grid;gap:10px}.action-card{display:flex;gap:14px;padding:15px;border:1px solid var(--da-border-light);border-radius:13px;background:#fbfcfe}.action-index{flex:0 0 34px;height:34px;display:grid;place-items:center;border-radius:10px;background:#fff0ee;color:#bd3b2c;font-weight:750}.action-main{min-width:0;flex:1}.action-main h3{margin:0;font-size:14px}.action-main p{margin:6px 0;color:var(--da-text-secondary);font-size:12px}.action-main>span{color:var(--da-text-muted);font-size:11px}.command{display:flex;align-items:center;max-width:820px;border:1px solid #dce3ec;border-radius:10px;background:#202634;text-align:left}.command.compact{margin-top:10px;max-width:none}.command code{min-width:0;flex:1;overflow:auto;padding:10px 12px;color:#e8edf5;font:11px Consolas,monospace;white-space:nowrap}.command button{align-self:stretch;padding:0 12px;border:0;border-left:1px solid #3b4353;background:transparent;color:#9bc1ff;cursor:pointer}.case-list{display:grid;gap:8px}.case-row{border:1px solid var(--da-border-light);border-radius:11px;background:#fff}.case-row[open]{border-color:#cbdcf3}.case-row summary{display:flex;align-items:center;gap:10px;padding:13px 14px;cursor:pointer;list-style:none}.case-row summary::-webkit-details-marker{display:none}.status-dot{width:8px;height:8px;border-radius:50%;background:#dc6656}.status-dot.pass{background:#35a56a}.case-row summary strong{font:12px Consolas,monospace}.question{min-width:0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}.cause-badge{padding:4px 8px;border-radius:999px;background:#fff2e5;color:#b76b17;font-size:11px}.confidence{color:var(--da-text-muted);font-size:11px}.case-body{padding:0 14px 15px;border-top:1px solid var(--da-border-light)}.evidence{padding:14px 2px}.evidence strong,.sql-grid label,.plan label{font-size:12px}.evidence ul{margin:8px 0;padding-left:18px;color:var(--da-text-secondary);font-size:12px}.evidence p{margin:8px 0 0;color:var(--da-accent-strong);font-size:12px}.sql-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.sql-grid pre{min-height:92px;max-height:220px;overflow:auto;margin:6px 0 0;padding:12px;border-radius:9px;background:#202634;color:#e8edf5;font:11px/1.55 Consolas,monospace;white-space:pre-wrap}.plan{margin-top:10px;padding:11px 12px;border-radius:9px;background:var(--da-bg-soft)}.plan p{margin:6px 0 0;color:var(--da-text-secondary);font-size:12px;white-space:pre-wrap}.success-state{padding:30px;text-align:center;color:#2e8b57}.full-run{display:flex;align-items:center;gap:12px;padding:13px 16px;border:1px solid #cfe0f4;border-radius:13px;background:#f4f8fd}.full-run>span{font-size:12px;font-weight:600;white-space:nowrap}.full-run .command{flex:1;margin:0}@media(max-width:1000px){.metric-grid{grid-template-columns:repeat(3,1fr)}.cause-grid{grid-template-columns:1fr 1fr}}@media(max-width:700px){.evaluation-page{padding:25px 15px 45px}.evaluation-head{display:block}.head-actions{margin-top:14px}.metric-grid,.cause-grid,.sql-grid{grid-template-columns:1fr}.run-strip div{width:50%;padding:5px 0;border:0}.full-run{display:block}.full-run .command{margin-top:9px}.cause-badge{display:none}}
.run-select{min-height:38px;max-width:220px;padding:7px 10px;border:1px solid var(--da-border);border-radius:10px;background:#fff;color:var(--da-text-secondary);font:12px inherit}
.performance-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.performance-grid>div{padding:13px;border-radius:11px;background:var(--da-bg-soft)}.performance-grid>div.evidence-pass{box-shadow:inset 3px 0 #35a56a}.performance-grid>div.evidence-fail{box-shadow:inset 3px 0 #dc6656}.performance-grid span,.performance-grid strong,.performance-grid small{display:block}.performance-grid span,.performance-grid small{color:var(--da-text-muted);font-size:11px}.performance-grid strong{margin-top:6px;font-size:13px}.node-table{margin-top:15px;border:1px solid var(--da-border-light);border-radius:11px;overflow:hidden}.node-row{display:grid;grid-template-columns:minmax(180px,2fr) repeat(4,1fr);gap:10px;padding:9px 12px;border-top:1px solid var(--da-border-light);font-size:12px}.node-row:first-child{border-top:0}.node-head{background:var(--da-bg-soft);color:var(--da-text-muted)}.route-row{display:grid;grid-template-columns:1.05fr 1.55fr .75fr 1fr .65fr .75fr;gap:10px;padding:9px 12px;border-top:1px solid var(--da-border-light);font-size:11px}.route-row:first-child{border-top:0}.route-table{margin-top:12px}
.baseline-status.approved{background:#e9f8ef;color:#247848}.baseline-status.rejected{background:#fff0ee;color:#b53e31}.governance-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.governance-grid article{padding:15px;border:1px solid var(--da-border-light);border-radius:12px;background:var(--da-bg-soft)}.governance-grid span,.governance-grid strong,.governance-grid small{display:block}.governance-grid span,.governance-grid small{color:var(--da-text-muted);font-size:11px}.governance-grid strong{margin:6px 0;font-size:15px}.mono{font-family:Consolas,monospace}.checklist{display:flex;flex-wrap:wrap;gap:7px;margin-top:13px}.checklist div{padding:6px 9px;border-radius:999px;font-size:11px}.checklist .pass{background:#eaf8ef;color:#27794a}.checklist .observe{background:#edf4ff;color:#41658e}.checklist .fail{background:#fff0ee;color:#b23d30}.checklist span{margin-right:4px;font-weight:800}.fingerprints{margin-top:13px;border-top:1px solid var(--da-border-light);padding-top:11px}.fingerprints summary{cursor:pointer;color:var(--da-accent);font-size:12px}.fingerprints dl{display:grid;grid-template-columns:100px minmax(0,1fr);gap:7px 12px;margin:12px 0 0;font-size:11px}.fingerprints dt{color:var(--da-text-muted)}.fingerprints dd{overflow-wrap:anywhere;margin:0;font-family:Consolas,monospace}.experiment-box,.experiment-setup{margin-top:16px;padding:16px;border:1px solid #cfdcf0;border-radius:13px;background:linear-gradient(145deg,#f8fbff,#f4f7fc)}.experiment-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.experiment-head>div span,.experiment-head>div strong,.experiment-head>div small{display:block}.experiment-head>div span,.experiment-head>div small{color:var(--da-text-muted);font-size:11px}.experiment-head>div strong{margin:4px 0;font:600 14px Consolas,monospace}.experiment-status{padding:5px 9px;border-radius:999px;font-size:11px}.experiment-status.qualified,.experiment-status.recordable{background:#e8f8ee;color:#247848}.experiment-status.blocked{background:#fff0ee;color:#b23d30}.experiment-status.incomplete{background:#fff7e7;color:#956514}.experiment-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}.experiment-metrics div{padding:11px;border:1px solid var(--da-border-light);border-radius:10px;background:#fff}.experiment-metrics span,.experiment-metrics strong,.experiment-metrics small{display:block}.experiment-metrics span,.experiment-metrics small{color:var(--da-text-muted);font-size:10px}.experiment-metrics strong{margin:5px 0;font-size:15px}.experiment-runs{margin-top:12px;overflow:auto;border:1px solid var(--da-border-light);border-radius:10px;background:#fff}.experiment-run{min-width:900px;display:grid;grid-template-columns:60px 1.4fr .7fr .6fr .6fr 1fr .8fr .8fr;gap:8px;padding:8px 11px;border-top:1px solid var(--da-border-light);font-size:11px}.experiment-run:first-child{border-top:0}.experiment-run-head{color:var(--da-text-muted);background:var(--da-bg-soft)}.experiment-checklist{margin-top:12px}.experiment-ok{margin:10px 0 0;color:#27794a;font-size:11px}.experiment-note{margin:10px 0 0;padding:9px 11px;border-left:3px solid #6d8fbd;border-radius:6px;background:#edf4ff;color:#41658e;font-size:11px;line-height:1.6}.experiment-setup>div:first-child strong{font-size:13px}.experiment-setup>div:first-child p{margin:5px 0 12px;color:var(--da-text-muted);font-size:11px}.experiment-setup .command{margin-top:7px;grid-template-columns:28px minmax(0,1fr) auto}.experiment-setup .command>span{color:var(--da-accent);font-size:11px;font-weight:700}.review-bar{display:grid;grid-template-columns:auto minmax(220px,1fr) auto auto;align-items:center;gap:8px;margin-top:15px}.review-bar label{font-size:12px}.review-bar input{min-height:38px;padding:8px 11px;border:1px solid var(--da-border);border-radius:9px;font:inherit}.btn.approve{border-color:#78c49a;background:#effaf3;color:#247848}.btn.reject{color:#b23d30}.blocked-tip{margin:10px 0 0;color:#956514;font-size:11px}
.evidence-chain{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-top:10px}.evidence-chain div{padding:9px;border-radius:8px;background:var(--da-bg-soft)}.evidence-chain span,.evidence-chain strong{display:block;font-size:10px}.evidence-chain span{color:var(--da-text-muted)}.evidence-chain strong{margin-top:4px}
.runtime-evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.runtime-evidence-grid article{display:grid;gap:9px;padding:13px;border:1px solid #d7e2ef;border-radius:11px;background:linear-gradient(145deg,#fbfdff,#f5f9fe);box-shadow:inset 3px 0 #e2a63b}.runtime-evidence-grid article.evidence-ok{box-shadow:inset 3px 0 #35a56a}.runtime-evidence-grid header{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.runtime-evidence-grid header span,.runtime-evidence-grid header strong{display:block}.runtime-evidence-grid header span{color:var(--da-text-muted);font:9px Consolas,monospace}.runtime-evidence-grid header strong{margin-top:4px;font-size:12px}.runtime-evidence-grid header b{padding:4px 7px;border-radius:99px;background:#fff;color:var(--da-accent-strong);font-size:9px}.runtime-evidence-grid dl{display:grid;gap:6px;margin:0}.runtime-evidence-grid dl div{display:grid;grid-template-columns:56px minmax(0,1fr);gap:7px}.runtime-evidence-grid dt{color:var(--da-text-muted);font-size:9px}.runtime-evidence-grid dd{overflow-wrap:anywhere;margin:0;color:var(--da-text-secondary);font:9px/1.45 Consolas,monospace}.runtime-evidence-grid small{color:var(--da-text-muted);font-size:9px}
.semantic-ir{margin-top:10px;border:1px solid #cfdcf0;border-radius:9px;background:#f7faff}.semantic-ir summary{display:flex;align-items:center;gap:8px;padding:10px 12px;cursor:pointer;font-size:12px}.semantic-ir summary span{font-weight:650}.semantic-ir summary b{color:var(--da-accent);font:11px Consolas,monospace}.semantic-ir summary em{margin-left:auto;padding:3px 7px;border-radius:999px;font-style:normal;font-size:10px}.semantic-ir summary em.compiled{background:#e8f8ee;color:#247848}.semantic-ir summary em.fallback{background:#fff7e7;color:#956514}.semantic-ir pre{max-height:280px;overflow:auto;margin:0;padding:12px;border-top:1px solid #dce5f2;background:#202634;color:#e8edf5;font:11px/1.55 Consolas,monospace;white-space:pre-wrap}
.experiment-run{min-width:1320px;grid-template-columns:55px 1.25fr .8fr .55fr .55fr .6fr .6fr .55fr .8fr .85fr .9fr .8fr}
.stability-box{margin-top:12px;padding:14px;border:1px solid #d4dfed;border-radius:11px;background:rgba(255,255,255,.88)}.stability-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.stability-head strong,.stability-head small{display:block}.stability-head small{margin-top:4px;color:var(--da-text-muted);font-size:10px}.stability-head>span{padding:4px 8px;border-radius:999px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:10px}.stability-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:11px}.stability-metrics article{padding:10px 12px;border-radius:9px;background:#f5f8fc;box-shadow:inset 3px 0 #92a2b6}.stability-metrics article.stable{box-shadow:inset 3px 0 #35a56a}.stability-metrics article.failed{box-shadow:inset 3px 0 #dc6656}.stability-metrics article.variable{box-shadow:inset 3px 0 #e2a63b}.stability-metrics span,.stability-metrics strong{display:block}.stability-metrics span{color:var(--da-text-muted);font-size:10px}.stability-metrics strong{margin-top:4px;font-size:18px}.family-table{margin-top:11px;overflow:auto;border:1px solid var(--da-border-light);border-radius:9px}.family-row{min-width:720px;display:grid;grid-template-columns:1.35fr .45fr .7fr 1.2fr 1.2fr;gap:9px;padding:8px 10px;border-top:1px solid var(--da-border-light);font-size:10px}.family-row:first-child{border-top:0}.family-head{background:var(--da-bg-soft);color:var(--da-text-muted)}.stability-cases{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:10px}.stability-cases>span{color:var(--da-text-muted);font-size:10px}.stability-cases button{padding:4px 7px;border:0;border-radius:7px;font:10px Consolas,monospace;cursor:pointer}.stability-cases button.stable_fail{background:#fff0ee;color:#b23d30}.stability-cases button.flaky{background:#fff7e7;color:#956514}
.attempt-details{margin-top:10px;border:1px solid var(--da-border-light);border-radius:9px;overflow:auto}.attempt-details>summary{position:sticky;left:0;padding:10px 12px;background:#f7faff;color:var(--da-accent-strong);font-size:12px;cursor:pointer}.attempt-row{min-width:1580px;display:grid;grid-template-columns:1.25fr 1fr 1.05fr .72fr .82fr .75fr .75fr .72fr 1.12fr .9fr;gap:8px;padding:8px 11px;border-top:1px solid var(--da-border-light);font-size:10px}.attempt-row strong{font-family:Consolas,monospace}.attempt-row span{overflow-wrap:anywhere}.attempt-row span small{display:block;margin-top:2px;color:var(--da-text-muted)}.attempt-head{color:var(--da-text-muted);background:var(--da-bg-soft)}.attempt-success{color:#247848}.attempt-error,.attempt-timeout,.attempt-empty_output{color:#b23d30}.attempt-cancelled{color:#956514}.attempt-coverage{position:sticky;left:0;margin:0;padding:8px 11px;border-top:1px solid var(--da-border-light);font-size:10px}.attempt-coverage.complete{background:#effaf3;color:#247848}.attempt-coverage.incomplete{background:#fff0ee;color:#b23d30}.degradation-list{margin-top:10px;padding:11px 12px;border-left:3px solid #e2a63b;border-radius:7px;background:#fff8eb}.degradation-list>strong{font-size:12px}.degradation-list>div{margin-top:7px}.degradation-list span{color:#956514;font:10px Consolas,monospace}.degradation-list p{margin:2px 0 0;color:var(--da-text-secondary);font-size:11px}
.baseline-track-card.unresolvable{border-color:#e5b0a9;background:#fff8f6}.baseline-track-card.unresolvable>header strong{color:#b23d30}.baseline-resolution-error{margin:10px 0 0;color:#b23d30;font-size:10px;line-height:1.45}
.ablation-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.ablation-card,.case-stage-card{padding:14px;border:1px solid var(--da-border-light);border-radius:11px;background:var(--da-bg-soft);box-shadow:inset 3px 0 #aab4c2}.ablation-card.good,.case-stage-card.good{box-shadow:inset 3px 0 #35a56a}.ablation-card.warn,.case-stage-card.warn{box-shadow:inset 3px 0 #e2a63b}.ablation-card.bad,.case-stage-card.bad{box-shadow:inset 3px 0 #dc6656}.ablation-card span,.ablation-card strong,.ablation-card small,.case-stage-card span,.case-stage-card strong,.case-stage-card small{display:block}.ablation-card span,.ablation-card small,.case-stage-card span,.case-stage-card small{color:var(--da-text-muted);font-size:10px}.ablation-card strong{margin:6px 0 4px;font-size:20px}.ablation-card small+small{margin-top:3px}.case-stage-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-bottom:10px}.case-stage-card strong{margin:5px 0 4px;font-size:14px}.case-stage-card small{line-height:1.5}.ablation-card.correction,.case-stage-card.correction{background:#f7faff}
@media(max-width:700px){.head-actions{flex-wrap:wrap}}
@media(max-width:900px){.performance-grid,.experiment-metrics{grid-template-columns:1fr 1fr}.node-row{grid-template-columns:1.5fr repeat(4,1fr);overflow:auto}.governance-grid{grid-template-columns:1fr}.review-bar{grid-template-columns:1fr 1fr}.review-bar label,.review-bar input{grid-column:1/-1}}
@media(max-width:700px){.stability-metrics,.runtime-evidence-grid{grid-template-columns:1fr}.family-table,.attempt-details{overflow:auto}}
@media(max-width:900px){.baseline-track-grid{grid-template-columns:1fr}.baseline-track-metrics{grid-template-columns:repeat(4,1fr)}}
@media(max-width:700px){.difficulty-tabs button{display:grid;gap:2px}.baseline-track-metrics,.ablation-grid,.case-stage-grid{grid-template-columns:1fr}.baseline-track-card>footer time{width:100%;margin-left:0}}
.metric-card.primary{border-color:#afc9eb;background:linear-gradient(145deg,#fff,#f4f8ff);box-shadow:0 0 0 2px rgba(55,126,241,.07),var(--da-shadow-sm)}
.metric-card.primary span{color:var(--da-accent-strong);font-weight:750}
.metric-card.observe:before{background:#4f83c2}
.ablation-card.primary-result{border-color:#afc9eb;background:linear-gradient(145deg,#fbfdff,#f2f7ff)}
.ablation-card.primary-result>span{color:var(--da-accent-strong);font-weight:750}
.ablation-card.observe{box-shadow:inset 3px 0 #4f83c2}
</style>
