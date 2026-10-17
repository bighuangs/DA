<template>
  <section class="ops-page">
    <header class="ops-head">
      <div>
        <div class="eyebrow">ADMIN / QUALITY OPERATIONS</div>
        <h1>质量运营</h1>
        <p>把线上 Trace、用户反馈、版本指纹与人工改进候选串成可审计闭环。</p>
      </div>
      <form class="filters" aria-label="质量运营筛选" @submit.prevent="load">
        <label>时间范围<select v-model.number="filters.days"><option :value="1">近 1 天</option><option :value="7">近 7 天</option><option :value="30">近 30 天</option><option :value="90">近 90 天</option></select></label>
        <label>版本<select v-model="filters.version"><option value="">全部版本</option><option v-for="item in data?.versions || []" :key="item.run_fingerprint" :value="item.run_fingerprint">{{ item.version_run_id || "未版本化" }}</option></select></label>
        <label>问题类型<select v-model="filters.category"><option value="">全部类型</option><option v-for="item in categories" :key="item" :value="item">{{ categoryLabel(item) }}</option></select></label>
        <label>状态<select v-model="filters.status"><option value="">全部状态</option><option value="success">成功</option><option value="failed">失败</option><option value="running">运行中</option></select></label>
        <button class="btn primary" :disabled="loading">{{ loading ? "加载中…" : "应用筛选" }}</button>
      </form>
    </header>

    <div v-if="error" class="alert error" role="alert"><CircleAlert :size="16" aria-hidden="true" />{{ error }}</div>
    <div v-if="notice" class="alert notice" role="status"><CircleCheck :size="16" aria-hidden="true" />{{ notice }}</div>
    <div v-if="data?.coverage?.truncated" class="alert warning" role="status">
      <TriangleAlert :size="16" aria-hidden="true" />当前指标使用最近 {{ data.coverage.trace.sampled }}/{{ data.coverage.trace.matched }} 条 Trace、
      {{ data.coverage.feedback.sampled }}/{{ data.coverage.feedback.matched }} 条反馈；
      覆盖 {{ dateTime(data.coverage.coverage_from) }} 至 {{ dateTime(data.coverage.coverage_to) }}，不是所选周期全量。
    </div>
    <div v-if="loading && !data" class="loading">正在聚合当前工作空间的质量数据…</div>

    <template v-else-if="data">
      <div v-if="!data.available" class="empty">
        <Activity :size="34" aria-hidden="true" /><h2>还没有线上质量数据</h2>
        <p>先在“对话”中完成一次查询并提交反馈；新 Trace 会在查询完成后写入。</p>
      </div>
      <template v-else>
        <section class="metric-grid" aria-label="核心质量指标">
          <article v-for="metric in metricCards" :key="metric.label" class="metric" :class="metric.tone" :data-testid="metric.testId">
            <span>{{ metric.label }}</span><strong>{{ metric.value }}</strong><small>{{ metric.note }}</small>
          </article>
        </section>

        <section v-if="data.runtime_governance?.semantic_coverage_observed || data.runtime_governance?.deadline_observed || data.runtime_governance?.llm_attempt_accounting?.attempts" class="panel runtime-ops">
          <div class="panel-head"><div><h2>运行态语义与调用治理</h2><p>汇总结构槽位闭合、请求级共享预算和每次物理模型调用；证据仅含指纹、原因码与相对耗时。</p></div><span>{{ data.runtime_governance.llm_attempt_accounting?.schema || "llm-attempt-v2" }}</span></div>
          <div class="policy-grid runtime-policy-grid" aria-label="运行态语义与调用治理指标">
            <article :class="data.runtime_governance.semantic_unresolved_traces === 0 ? 'runtime-ok' : 'runtime-warn'"><span>语义覆盖 Trace</span><strong>{{ data.runtime_governance.semantic_coverage_observed }} / {{ data.summary.traces }}</strong><small>平均 {{ percent(data.runtime_governance.semantic_coverage_mean) }} · 未闭合 {{ data.runtime_governance.semantic_unresolved_traces }}</small></article>
            <article :class="data.runtime_governance.semantic_unsafe_compile_traces === 0 ? 'runtime-ok' : 'runtime-warn'"><span>不安全确定性编译</span><strong>{{ data.runtime_governance.semantic_unsafe_compile_traces }}</strong><small>存在未闭合槽位时必须回退或拒绝</small></article>
            <article :class="data.runtime_governance.deadline_contract_complete && data.runtime_governance.deadline_exhausted_traces === 0 ? 'runtime-ok' : 'runtime-warn'"><span>共享截止时间</span><strong>{{ data.runtime_governance.deadline_observed }} / {{ data.summary.traces }}</strong><small>耗尽 {{ data.runtime_governance.deadline_exhausted_traces }} · 相对证据{{ data.runtime_governance.deadline_contract_complete ? "完整" : "不完整" }}</small></article>
            <article><span>排队 / Provider P95</span><strong>{{ number(data.runtime_governance.llm_queue_wait_p95_ms) }} / {{ number(data.runtime_governance.llm_provider_p95_ms) }} ms</strong><small>拆分排队拥塞与模型服务耗时</small></article>
            <article :class="data.runtime_governance.llm_attempt_accounting?.evidence_complete ? 'runtime-ok' : 'runtime-warn'"><span>物理调用证据</span><strong>{{ data.runtime_governance.llm_attempt_accounting?.attempts || 0 }} 次</strong><small>恢复 {{ data.runtime_governance.llm_attempt_accounting?.recoveries || 0 }} · ID 不可用 {{ data.runtime_governance.llm_attempt_accounting?.provider_request_id_unavailable || 0 }}</small></article>
            <article><span>调用结果分类</span><strong>{{ outcomeSummary(data.runtime_governance.llm_outcome_classes) }}</strong><small>错误、超时、限流、空响应和截断分别归类</small></article>
          </div>
        </section>

        <section v-if="data.candidate_policy?.available" class="panel">
          <div class="panel-head"><div><h2>候选判优策略门禁</h2><p>确定性策略矩阵验证触发、预算与安全淘汰；不等同于真实模型正确率。</p></div><span :class="data.candidate_policy.summary.gate_passed ? 'badge ok' : 'badge failed'">{{ data.candidate_policy.summary.gate_passed ? "门禁通过" : "门禁失败" }}</span></div>
          <div class="policy-grid" aria-label="候选判优策略指标"><article><span>单候选 → 自适应</span><strong>{{ percent(data.candidate_policy.summary.baseline_success_rate) }} → {{ percent(data.candidate_policy.summary.adaptive_success_rate) }}</strong></article><article><span>第二候选触发率</span><strong>{{ percent(data.candidate_policy.summary.second_candidate_rate) }}</strong><small>目标 ≤20%</small></article><article><span>低置信修复率</span><strong>{{ percent(data.candidate_policy.summary.targeted_repair_rate) }}</strong><small>目标 ≥80%</small></article><article><span>语义合同正反例</span><strong>{{ data.candidate_policy.summary.semantic_contract_passed ?? 0 }} / {{ data.candidate_policy.summary.semantic_contract_cases ?? 0 }}</strong><small>误触发 {{ data.candidate_policy.summary.semantic_false_positive_count ?? 0 }}</small></article><article><span>安全 / 高置信额外调用</span><strong>{{ data.candidate_policy.summary.unsafe_selected }} / {{ data.candidate_policy.summary.high_confidence_extra_calls }}</strong><small>均须为 0</small></article></div>
        </section>

        <section v-if="data.semantic_repair?.observed_traces" class="panel semantic-ops">
          <div class="panel-head"><div><h2>语义守卫运行态</h2><p>按版本化 Trace 汇总规则命中、错误码与 P0 回退；不保存原始问题或 SQL。</p></div><span>{{ data.semantic_repair.observed_traces }} 条证据</span></div>
          <div class="policy-grid" aria-label="语义守卫运行态指标">
            <article><span>确定性改写率</span><strong>{{ percent(data.semantic_repair.change_rate) }}</strong><small>{{ data.semantic_repair.changed_traces }} 条发生安全改写</small></article>
            <article><span>P0 回退率</span><strong>{{ percent(data.semantic_repair.fallback_rate) }}</strong><small>{{ data.semantic_repair.fallback_count }} 次注册表异常</small></article>
            <article><span>修复耗时 P95</span><strong>{{ number(data.semantic_repair.repair_p95_ms) }} ms</strong><small>仅规则执行，不含 LLM</small></article>
            <article><span>首要命中规则</span><strong>{{ semanticRuleLabel(data.semantic_repair.rules?.[0]?.rule_code) }}</strong><small>{{ data.semantic_repair.rules?.[0]?.count || 0 }} 次</small></article>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head"><div><h2>趋势与版本</h2><p>只有携带完整指纹的版本才显示为可比较；已绑定的观测基线另行标记。</p></div><span>{{ data.summary.traces }} 条 Trace</span></div>
          <div class="split">
            <div class="trend" aria-label="每日成功率趋势">
              <div v-for="item in data.trends" :key="item.date" class="trend-row">
                <time>{{ item.date }}</time><div class="bar"><i :style="{ width: `${item.success_rate * 100}%` }"></i></div><strong>{{ percent(item.success_rate) }}</strong><span>P95 {{ number(item.p95_ms) }} ms</span>
              </div>
              <p v-if="!data.trends.length" class="muted">当前筛选没有趋势数据。</p>
            </div>
            <div class="table-wrap" tabindex="0" aria-label="线上版本质量表格">
              <table><caption class="sr-only">线上版本质量</caption><thead><tr><th>绑定评测</th><th>状态</th><th>Trace</th><th>成功率</th><th>P95</th></tr></thead>
                <tbody><tr v-for="item in data.versions" :key="item.run_fingerprint"><td class="mono">{{ item.version_run_id || "未版本化" }}</td><td><span class="badge" :class="item.version_complete ? 'ok' : 'warn'">{{ item.baseline_bound ? "已绑定观测" : item.version_complete ? "指纹完整" : "不可比较" }}</span></td><td>{{ item.traces }}</td><td>{{ percent(item.success_rate) }}</td><td>{{ number(item.p95_ms) }} ms</td></tr></tbody>
              </table>
            </div>
          </div>
        </section>

        <div class="two-columns">
          <section class="panel">
            <div class="panel-head"><div><h2>失败漏斗</h2><p>使用稳定错误码归并，不暴露异常原文。</p></div></div>
            <div v-if="data.failure_funnel.length" class="funnel"><div v-for="item in data.failure_funnel" :key="item.failure_code"><span>{{ failureLabel(item.failure_code) }}</span><div><i :style="{ width: `${Math.max(7, item.count / maxFailure * 100)}%` }"></i></div><strong>{{ item.count }}</strong></div></div>
            <p v-else class="success">当前筛选没有失败 Trace。</p>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>慢节点与 Token</h2><p>按累计耗时排序，优先发现线上预算热点。</p></div></div>
            <div class="table-wrap" tabindex="0" aria-label="慢节点与 Token 表格"><table><caption class="sr-only">慢节点与 Token</caption><thead><tr><th>节点</th><th>调用</th><th>平均</th><th>P95</th><th>Token</th></tr></thead><tbody><tr v-for="item in data.slow_nodes" :key="item.node_id"><td><strong class="node-name">{{ nodeLabel(item.node_id) }}</strong><small class="node-id">{{ item.node_id }}</small></td><td>{{ item.calls }}</td><td>{{ number(item.mean_ms) }} ms</td><td>{{ number(item.p95_ms) }} ms</td><td>{{ number(item.tokens) }}</td></tr></tbody></table></div>
          </section>
        </div>

        <section class="panel feedback-panel">
          <div class="panel-head"><div><h2>差评审核与改进候选</h2><p>先分类审核，再生成候选，最后二次确认；训练样例确认后仍保持停用。</p></div><span>{{ pendingCount }} 条待审核</span></div>
          <div v-if="data.feedback_queue.length" class="feedback-list">
            <article v-for="item in data.feedback_queue" :key="item.id" class="feedback-card">
              <div class="feedback-main">
                <div class="feedback-title"><strong>{{ item.query_summary || "已脱敏问题" }}</strong><span class="badge" :class="item.review_status">{{ reviewLabel(item.review_status) }}</span><span v-if="item.review_category" class="badge category">{{ reviewCategoryLabel(item.review_category) }}</span></div>
                <p>{{ item.comment_summary || "用户未填写差评说明" }}</p>
                <small>反馈 #{{ item.id }} · {{ dateTime(item.created_at) }}<template v-if="item.trace"> · {{ failureLabel(item.trace.failure_code) }} · {{ number(item.trace.tt_complete_ms) }} ms</template></small>
              </div>
              <div v-if="item.review_status !== 'reviewed'" class="review-form">
                <label>归因分类<select v-model="item._category"><option value="">请选择</option><option v-for="category in reviewCategories" :key="category" :value="category">{{ reviewCategoryLabel(category) }}</option></select></label>
                <label>审核说明<input v-model="item._note" maxlength="1000" placeholder="记录判断依据" /></label>
                <button class="btn" @click="review(item, 'dismissed')">无效反馈</button><button class="btn primary" :disabled="!item._category" @click="review(item, 'reviewed')">完成审核</button>
              </div>
              <div v-else class="candidate-box">
                <div v-if="item.candidate_status === 'none'" class="candidate-form">
                  <label>候选类型<select v-model="item._candidateType"><option value="training">训练样例</option><option value="golden">黄金集候选</option></select></label>
                  <label>标准问题<input v-model="item._question" maxlength="500" /></label>
                  <label class="sql-field">只读 SQL<textarea v-model="item._sql" rows="3" spellcheck="false" placeholder="SELECT ..."></textarea></label>
                  <button class="btn primary" :disabled="!item._question || !item._sql" @click="draft(item)">保存候选草稿</button>
                </div>
                <div v-else class="candidate-summary">
                  <div><span>{{ item.candidate_type === 'training' ? "训练样例" : "黄金集" }} · {{ candidateLabel(item.candidate_status) }}</span><strong>{{ item.candidate_question }}</strong><code>{{ item.candidate_sql_summary }}</code></div>
                  <button v-if="item.candidate_status === 'draft'" class="btn confirm" @click="confirmCandidate(item)">二次确认</button>
                  <span v-else class="confirmed"><CircleCheck :size="14" aria-hidden="true" />已确认<template v-if="item.candidate_training_id">，停用样例 #{{ item.candidate_training_id }}</template></span>
                </div>
              </div>
            </article>
          </div>
          <p v-else class="success">当前工作空间没有待运营差评。</p>
        </section>

        <section class="panel">
          <div class="panel-head"><div><h2>最近 Trace</h2><p>列表仅返回脱敏问题摘要；详情只包含脱敏 SQL、节点状态与聚合耗时。</p></div></div>
          <div class="table-wrap" tabindex="0" aria-label="最近 Trace 表格"><table><caption class="sr-only">最近 Trace</caption><thead><tr><th>时间</th><th>问题摘要</th><th>类型</th><th>状态</th><th>置信度</th><th>耗时</th><th>Token</th><th></th></tr></thead><tbody><tr v-for="item in data.recent_traces" :key="item.trace_id"><td>{{ dateTime(item.started_at) }}</td><td>{{ item.query_summary }}</td><td>{{ categoryLabel(item.query_category) }}</td><td><span class="badge" :class="item.status === 'success' ? 'ok' : 'failed'">{{ item.status === 'success' ? '成功' : failureLabel(item.failure_code) }}</span></td><td><span class="badge" :class="confidenceTone(item.candidate_band)">{{ confidenceLabel(item.candidate_band) }} {{ score(item.candidate_confidence) }}</span><small v-if="item.adaptive_candidate_triggered"> · 双候选</small></td><td>{{ number(item.tt_complete_ms) }} ms</td><td>{{ number(item.total_tokens) }}</td><td><button class="link" @click="openTrace(item.trace_id)">查看详情</button></td></tr></tbody></table></div>
        </section>
      </template>
    </template>

    <div v-if="trace" class="modal-mask" @click.self="trace = null">
      <section class="trace-dialog" role="dialog" aria-modal="true" aria-labelledby="trace-title">
        <div class="dialog-head"><div><div class="eyebrow">PRIVACY-SAFE TRACE</div><h2 id="trace-title">Trace 详情</h2></div><button class="close" aria-label="关闭 Trace 详情" @click="trace = null"><X :size="19" aria-hidden="true" /></button></div>
        <dl class="trace-meta"><div><dt>问题摘要</dt><dd>{{ trace.query_summary }}</dd></div><div><dt>SQL 摘要</dt><dd><code>{{ trace.sql_summary || "—" }}</code></dd></div><div><dt>版本</dt><dd>{{ trace.version_run_id || "未版本化" }}</dd></div><div><dt>结果</dt><dd>{{ trace.status }} · {{ number(trace.tt_complete_ms) }} ms · {{ number(trace.total_tokens) }} tokens</dd></div><div v-if="trace.follow_up_action"><dt>多轮闭合</dt><dd>{{ trace.follow_up_action }} · {{ trace.follow_up_summary }}</dd></div><div v-if="trace.candidate_decision?.candidate_count"><dt>候选判优</dt><dd>{{ trace.candidate_decision.adaptive_triggered ? "自适应双候选" : "单候选快路径" }} · 选择 {{ trace.candidate_decision.selected }} · {{ trace.candidate_decision.selection_reason }}</dd></div></dl>
        <div v-if="trace.candidate_decision?.candidates?.length" class="candidate-trace"><article v-for="item in trace.candidate_decision.candidates" :key="item.label"><strong>候选 {{ item.label }} · {{ confidenceLabel(item.band) }} {{ score(item.score) }}</strong><span>{{ item.valid ? "确定性校验通过" : `已淘汰：${item.error_code}` }}</span><small>{{ (item.weaknesses || []).join(" · ") || "未发现结构化弱点" }}</small></article></div>
        <section v-if="trace.planning_route" class="planning-trace" aria-label="规划路由 Trace 证据">
          <header><div><strong>SQL 规划路由</strong><small>仅保留版本、决策、原因码、数值特征与影子候选；不保存原始问题。</small></div><span>{{ planningActionLabel(trace.planning_route.actual_action) }}</span></header>
          <div class="planning-proof">
            <span>Active <code>{{ trace.planning_route.profile }}</code></span>
            <span>Score {{ score(trace.planning_route.score) }}</span>
            <span>Confidence {{ score(trace.planning_route.confidence) }}</span>
            <span>Decision {{ number(trace.planning_route.decision_ms) }} ms</span>
            <span>Raw text {{ trace.planning_route.raw_text_logged ? "异常开启" : "关闭" }}</span>
          </div>
          <p>{{ (trace.planning_route.reason_codes || []).map(planningReasonLabel).join(" · ") || "无额外触发原因" }}</p>
          <div v-if="trace.planning_route.shadow_decisions?.length" class="planning-shadow-proof"><span v-for="item in trace.planning_route.shadow_decisions" :key="item.profile"><code>{{ item.profile }}</code> → {{ planningActionLabel(item.decision) }}</span></div>
        </section>
        <section v-if="trace.semantic_repair?.contract_version" class="semantic-trace" aria-label="语义修复 Trace 证据">
          <header><div><strong>语义守卫与自动修复</strong><small>只保留 violation code、规则码、SQL hash、AST 节点和不变量；原始 SQL 日志固定关闭。</small></div><span>{{ semanticProfileLabel(trace.semantic_repair.profile) }}</span></header>
          <div class="semantic-proof">
            <span>{{ trace.semantic_repair.changed ? "已改写" : "未改写" }}</span>
            <span>Repair {{ number(trace.semantic_repair.repair_ms) }} ms</span>
            <span>Fallback {{ trace.semantic_repair.fallback_used ? "是" : "否" }}</span>
            <span>Raw SQL {{ trace.semantic_repair.raw_sql_logged ? "异常开启" : "关闭" }}</span>
          </div>
          <p v-if="trace.semantic_repair.violation"><code>{{ trace.semantic_repair.violation.violation_code }}</code> · {{ repairabilityLabel(trace.semantic_repair.violation.repairability) }}</p>
          <div v-if="trace.semantic_repair.applied_rules?.length" class="semantic-rule-proof"><span v-for="rule in trace.semantic_repair.applied_rules" :key="rule.rule_code"><code>{{ rule.rule_code }}@{{ rule.rule_version }}</code> · {{ listText(rule.changed_ast_nodes) }}</span></div>
          <small class="semantic-hash">SQL hash {{ shortHash(trace.semantic_repair.original_sql_hash) }} → {{ shortHash(trace.semantic_repair.repaired_sql_hash) }}</small>
        </section>
        <section v-if="trace.semantic_coverage?.schema_version" class="coverage-trace" aria-label="语义槽位覆盖 Trace 证据">
          <header><div><strong>语义槽位覆盖</strong><small>检测到的运算、维度和过滤必须在 IR 中逐项表示；这里只保存结构标签与问题指纹。</small></div><span :class="coverageTone(trace.semantic_coverage)">{{ percent(trace.semantic_coverage.coverage_score) }}</span></header>
          <div class="coverage-proof">
            <span>编译 {{ trace.semantic_coverage.compiler_supported ? "允许" : "安全回退" }}</span>
            <span>模式 <code>{{ trace.semantic_coverage.compiler_mode || "—" }}</code></span>
            <span>问题明文 {{ trace.semantic_coverage.raw_question_persisted ? "异常保存" : "未保存" }}</span>
            <span>筛选值 {{ trace.semantic_coverage.filter_values_persisted ? "异常保存" : "未保存" }}</span>
          </div>
          <dl class="coverage-slots"><div><dt>检测</dt><dd>{{ listText(trace.semantic_coverage.detected_slots) }}</dd></div><div><dt>已表达</dt><dd>{{ listText(trace.semantic_coverage.represented_slots) }}</dd></div><div><dt>未闭合</dt><dd>{{ listText(trace.semantic_coverage.unresolved_slots) }}</dd></div></dl>
        </section>
        <section v-if="trace.semantic_coverage?.query_contract" class="contract-trace" aria-label="统一语义合同与关系闭包 Trace 证据">
          <header><div><strong>统一语义合同</strong><small>召回、规划、生成和守卫共享同一最终粒度、指标公式、分母作用域、阶段顺序与元数据关系路径。</small></div><span :class="trace.semantic_coverage.query_contract.authority === 'closed' ? 'complete' : 'incomplete'">{{ trace.semantic_coverage.query_contract.authority === "closed" ? "已闭合" : "引导" }}</span></header>
          <div class="contract-proof">
            <span>语义族 <code>{{ trace.semantic_coverage.query_contract.semantic_family || "unknown" }}</code></span>
            <span>最终粒度 {{ listText((trace.semantic_coverage.query_contract.output_grain || []).map((item) => item.column || item.logical)) }}</span>
            <span>运算 {{ listText(trace.semantic_coverage.query_contract.operations) }}</span>
            <span>阶段 {{ listText((trace.semantic_coverage.query_contract.stage_plan || []).map((item) => item.stage)) }}</span>
            <span>合同 {{ shortHash(trace.semantic_coverage.query_contract.contract_hash) }}</span>
          </div>
          <div v-if="trace.semantic_coverage.relation_closure || trace.semantic_coverage.query_contract.relation_plan" class="relation-proof">
            <b>关系闭包 {{ (trace.semantic_coverage.relation_closure || trace.semantic_coverage.query_contract.relation_plan).status }}</b>
            <span>终点 {{ listText((trace.semantic_coverage.relation_closure || trace.semantic_coverage.query_contract.relation_plan).terminal_tables) }}</span>
            <span>补桥 {{ listText((trace.semantic_coverage.relation_closure || trace.semantic_coverage.query_contract.relation_plan).added_tables) }}</span>
            <span>连接 {{ (trace.semantic_coverage.relation_closure || trace.semantic_coverage.query_contract.relation_plan).joins?.length || 0 }} 条</span>
          </div>
        </section>
        <section v-if="trace.request_deadline?.contract_version" class="deadline-trace" aria-label="共享截止时间 Trace 证据">
          <header><div><strong>请求级共享截止时间</strong><small>模型排队、Provider 调用、纠错与 SQL 执行共享同一总预算；不保存绝对单调时间。</small></div><span :class="deadlineTone(trace.request_deadline)">{{ number(trace.request_deadline.remaining_ms) }} ms 剩余</span></header>
          <div class="deadline-proof">
            <span>总预算 {{ number(trace.request_deadline.budget_ms) }} ms</span>
            <span>执行保留 {{ number(trace.request_deadline.execution_reserve_ms) }} ms</span>
            <span>{{ phaseBudgetText(trace.request_deadline.phase_budgets_ms) }}</span>
            <span>绝对时间戳 {{ trace.request_deadline.absolute_timestamps_persisted ? "异常保存" : "未保存" }}</span>
          </div>
        </section>
        <section v-if="trace.retrieval_evidence?.length" class="retrieval-trace" aria-label="检索 Trace 证据"><header><div><strong>检索 Trace 证据</strong><small>只保留计数、排序合同、Point 策略、查询信号指纹与角色，不保存问题、实体、向量或投影原文。</small></div></header><article v-for="(route, routeIndex) in trace.retrieval_evidence" :key="`${route.route}-${routeIndex}`"><div class="retrieval-route"><b>{{ retrievalRouteLabel(route.route) }}</b><code>{{ route.version || "无精排版本" }}</code><span>动态 K {{ route.dynamic_k ?? "—" }} · {{ route.fusion_method || "—" }}</span></div><div v-if="route.query_signals?.contract_version" class="signal-proof" :aria-label="`${retrievalRouteLabel(route.route)}查询信号证据`"><span>Profile <code>{{ route.query_signals.extractor_profile }}</code></span><span>路由 <code>{{ route.query_signals.route_policy }}</code></span><span>Local / Remote / Rerank {{ route.query_signals.local_signal_count }} / {{ route.query_signals.remote_query_count }} / {{ route.query_signals.rerank_signal_count }}</span><span>编码 <code>{{ route.query_signals.query_encode_mode }}</code></span><span>远端类型 {{ signalKinds(route.query_signals.actual_remote_signals) }}</span><span>明文 {{ route.query_signals.raw_text_logged ? "异常开启" : "关闭" }}</span></div><div v-for="(query, queryIndex) in route.hybrid_queries" :key="queryIndex" class="hybrid-proof"><span>Dense {{ query.dense?.raw_points ?? 0 }} Point / {{ query.dense?.unique_entities ?? 0 }} 实体</span><span>Sparse {{ query.sparse?.raw_points ?? 0 }} Point / {{ query.sparse?.unique_entities ?? 0 }} 实体</span><span>Fusion {{ query.fusion_method }} · {{ query.fused_entities ?? 0 }} 实体</span><span>Point {{ listText(query.point_strategies) }}</span><span>角色 {{ listText(query.matched_point_roles) }}</span></div></article></section>
        <section v-if="trace.llm_attempts?.length" class="llm-trace" aria-label="LLM Attempt 证据">
          <header><div><strong>模型调用证据</strong><small>每次物理调用均保留预算、重试和稳定原因码；不展示 Prompt 或服务商响应正文。</small></div><span :class="trace.llm_attempt_accounting?.evidence_complete ? 'complete' : 'incomplete'">{{ trace.llm_attempt_accounting?.evidence_complete ? "证据完整" : "证据不完整" }}</span></header>
          <div class="llm-accounting">
            <span>调用 {{ trace.llm_attempt_accounting?.attempts ?? 0 }}</span>
            <span>成功 {{ trace.llm_attempt_accounting?.succeeded ?? 0 }}</span>
            <span>核心失败 {{ trace.llm_attempt_accounting?.core_failures ?? 0 }}</span>
            <span>恢复 {{ trace.llm_attempt_accounting?.recoveries ?? 0 }}</span>
            <span>Provider ID 不可用 {{ trace.llm_attempt_accounting?.provider_request_id_unavailable ?? 0 }}</span>
            <span>Callback 缺失 {{ trace.llm_attempt_accounting?.callback_missing ?? 0 }}</span>
          </div>
          <div class="llm-attempt-table">
            <div class="llm-attempt-row llm-attempt-head"><span>节点 / Route</span><span>状态 / 结果类</span><span>总耗时 / 节点预算</span><span>物理预算 / 恢复预留</span><span>排队 / Provider</span><span>截止前 / 后</span><span>调用序号 / 恢复</span><span>服务商请求 ID</span><span>Token / 路径</span></div>
            <div v-for="attempt in trace.llm_attempts" :key="attempt.attempt_id" class="llm-attempt-row">
              <strong>{{ attempt.node_id }} · {{ attempt.route }}</strong>
              <span :class="attempt.status === 'success' ? 'ok' : 'failed'">{{ attemptStatusLabel(attempt.status) }} · {{ outcomeClassLabel(attempt.outcome_class) }}<small>{{ attemptReasonLabel(attempt.reason_code) }}</small></span>
              <span>{{ number(attempt.latency_ms) }} / {{ number(attempt.timeout_budget_ms) }} ms</span>
              <span>{{ number(attempt.physical_attempt_budget_ms) }} / {{ number(attempt.recovery_reserve_ms) }} ms</span>
              <span>{{ number(attempt.queue_wait_ms) }} / {{ number(attempt.provider_ms) }} ms</span>
              <span>{{ number(attempt.deadline_before_ms) }} / {{ number(attempt.deadline_after_ms) }} ms</span>
              <span>#{{ Number(attempt.retry_index || 0) + 1 }} · {{ attempt.recovery ? "恢复调用" : "首次调用" }}</span>
              <span>{{ providerRequestLabel(attempt) }}<small v-if="!attempt.provider_request_id_available">{{ unavailableReasonLabel(attempt.provider_request_id_unavailable_reason) }}</small></span>
              <span>{{ attempt.token_known ? `${number(attempt.total_tokens)} tokens` : "Token 未知" }} · {{ attempt.affects_core_path ? "核心" : "可选" }}<small v-if="!attempt.token_known">{{ unavailableReasonLabel(attempt.token_timing_unavailable_reason) }}</small></span>
            </div>
          </div>
        </section>
        <div class="step-list"><div v-for="step in trace.steps" :key="`${step.step_name}-${step.attempt}`"><span class="dot" :class="step.status"></span><strong>{{ step.step_name }}</strong><span>{{ number(step.duration_ms) }} ms</span><small>{{ step.error_summary }}</small></div></div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { Activity, CircleAlert, CircleCheck, TriangleAlert, X } from "@lucide/vue";
import { responseError } from "../utils/http.js";

const API = "/api/admin/evaluation/quality";
const data = ref(null), trace = ref(null), loading = ref(false), error = ref(""), notice = ref("");
const filters = reactive({ days: 7, version: "", category: "", status: "" });
const reviewCategories = ["invalid_feedback", "data_issue", "retrieval_issue", "sql_issue", "metric_issue", "presentation_issue"];
const categories = ["general", "aggregation", "metric", "ranking", "time_series", "adhoc"];
const metricCards = computed(() => { const s = data.value?.summary || {}; return [
  { label: "成功率", value: percent(s.success_rate), note: `${s.traces || 0} 条 Trace`, tone: s.success_rate >= .95 ? "good" : "warn", testId: "metric-success" },
  { label: "结果代理率", value: percent(s.result_proxy_rate), note: "成功且未收到差评", tone: s.result_proxy_rate >= .9 ? "good" : "warn", testId: "metric-result-proxy" },
  { label: "差评率", value: percent(s.negative_rate), note: `${s.negative_feedback || 0}/${s.feedback_total || 0} 条反馈`, tone: s.negative_rate > .1 ? "bad" : "good", testId: "metric-negative" },
  { label: "P50 / P95", value: `${number(s.p50_ms)} / ${number(s.p95_ms)}`, note: latencyNote(s.p95_ms), tone: latencyTone(s.p95_ms), testId: "metric-latency" },
  { label: "平均 Token", value: number(s.tokens_mean), note: `估算成本 ${number(s.estimated_cost)}`, tone: "neutral", testId: "metric-token" },
  { label: "纠错 / 缓存", value: `${percent(s.correction_rate)} / ${percent(s.cache_hit_rate)}`, note: "纠错率 / 缓存命中率", tone: "neutral", testId: "metric-cache" },
  { label: "低置信 / 备选", value: `${percent(s.low_confidence_rate)} / ${percent(s.second_candidate_rate)}`, note: "备选触发目标 ≤20%", tone: s.second_candidate_rate <= .2 ? "good" : "warn", testId: "metric-candidate" },
  { label: "备选修复率", value: percent(s.adaptive_repair_rate), note: "选择 B 且查询成功", tone: s.adaptive_repair_rate >= .8 || !s.second_candidate_rate ? "good" : "warn", testId: "metric-repair" },
]; });
const maxFailure = computed(() => Math.max(1, ...(data.value?.failure_funnel || []).map(item => item.count)));
const pendingCount = computed(() => (data.value?.feedback_queue || []).filter(item => item.review_status === "pending").length);

async function request(path = "", options = {}) { const response = await fetch(`${API}${path}`, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options }); if (!response.ok) throw new Error(await responseError(response, "质量运营请求失败")); return response.json(); }
function prepare(items) { items.forEach(item => { item._category = item.review_category || ""; item._note = item.review_note || ""; item._candidateType = item.candidate_type || "training"; item._question = item.candidate_question || item.query_summary || ""; item._sql = ""; }); }
async function load() { loading.value = true; error.value = ""; try { const query = new URLSearchParams({ days: String(filters.days) }); for (const key of ["version", "category", "status"]) if (filters[key]) query.set(key, filters[key]); const body = await request(`/overview?${query}`); prepare(body.feedback_queue || []); data.value = body; } catch (e) { error.value = e.message || "质量运营数据加载失败"; } finally { loading.value = false; } }
async function mutate(item, path, options, message) { error.value = ""; try { const updated = await request(path, options); Object.assign(item, updated); prepare([item]); notice.value = message; window.setTimeout(() => { notice.value = ""; }, 2500); } catch (e) { error.value = e.message || "操作失败"; } }
function review(item, status) { return mutate(item, `/feedback/${item.id}/review`, { method: "PATCH", body: JSON.stringify({ status, category: status === "reviewed" ? item._category : "invalid_feedback", note: item._note }) }, status === "reviewed" ? "差评已完成分类审核" : "反馈已标记为无效"); }
function draft(item) { return mutate(item, `/feedback/${item.id}/candidate`, { method: "POST", body: JSON.stringify({ candidate_type: item._candidateType, question: item._question, sql: item._sql }) }, "候选草稿已保存，需二次确认才会生效"); }
function confirmCandidate(item) { return mutate(item, `/feedback/${item.id}/candidate/confirm`, { method: "POST" }, "候选已确认；训练样例保持停用，需在元知识中复核启用"); }
async function openTrace(id) { error.value = ""; try { trace.value = await request(`/traces/${encodeURIComponent(id)}`); } catch (e) { error.value = e.message || "Trace 详情加载失败"; } }
function percent(value) { const n = Number(value); return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—"; }
function number(value) { const n = Number(value); return Number.isFinite(n) ? n.toLocaleString("zh-CN", { maximumFractionDigits: 2 }) : "—"; }
function latencyTone(value) { const n = Number(value); if (!Number.isFinite(n)) return "neutral"; return n >= 30000 ? "bad" : n >= 15000 ? "warn" : "good"; }
function latencyNote(value) { const tone = latencyTone(value); return tone === "bad" ? "端到端毫秒 · 严重（≥30 秒）" : tone === "warn" ? "端到端毫秒 · 需关注（≥15 秒）" : tone === "good" ? "端到端毫秒 · 正常" : "端到端毫秒"; }
function dateTime(value) { return value ? String(value).replace("T", " ").slice(0, 16) : "—"; }
function categoryLabel(value) { return ({ general: "一般查询", aggregation: "聚合", metric: "指标", ranking: "排名", time_series: "时间序列", adhoc: "临时数据" })[value] || value || "未分类"; }
function failureLabel(value) { return ({ none: "无", timeout: "超时", security_policy: "安全策略", cost_guard: "成本守卫", dependency_connection: "依赖连接", sql_validation: "SQL 校验", sql_execution: "SQL 执行", pipeline_unknown: "管线异常" })[value] || value || "未知"; }
function attemptStatusLabel(value) { return ({ success: "成功", error: "错误", timeout: "超时", cancelled: "取消", empty_output: "空输出" })[value] || value || "未知"; }
function attemptReasonLabel(value) { return ({ "llm.success": "完成", "llm.timeout": "预算超时", "llm.deadline_exhausted": "共享预算耗尽", "llm.queue_timeout": "排队超时", "llm.cancelled": "请求取消", "llm.connection_error": "连接失败", "llm.rate_limited": "服务限流", "llm.server_error": "服务端异常", "llm.stream_interrupted": "流式响应中断", "llm.truncated_output": "输出截断", "llm.invalid_structured_output": "结构化输出无效", "llm.refusal": "模型拒答", "llm.empty_output": "空输出", "llm.provider_error": "服务错误", "llm.pre_invoke_failure": "调用前失败", "llm.unknown_failure": "未知失败" })[value] || value || "未知原因"; }
function outcomeClassLabel(value) { return ({ success: "完成", deadline_exhausted: "截止耗尽", queue_timeout: "排队超时", timeout: "Provider 超时", rate_limited: "限流", server_error: "服务异常", connection_error: "连接异常", stream_interrupted: "流中断", empty_output: "空输出", truncated_output: "截断", invalid_structured_output: "格式无效", refusal: "拒答", cancelled: "取消" })[value] || value || "未分类"; }
function outcomeSummary(value) { const rows = Object.entries(value || {}).filter(([, count]) => Number(count) > 0); return rows.length ? rows.map(([key, count]) => `${outcomeClassLabel(key)} ${count}`).join(" · ") : "无模型调用"; }
function providerRequestLabel(attempt) { return attempt?.provider_request_id_available ? attempt.provider_request_id : "服务商请求 ID 不可用"; }
function unavailableReasonLabel(value) { return ({ provider_response_unavailable_after_timeout: "超时后无服务商响应", provider_request_id_not_exposed: "服务商 SDK 未提供请求 ID", callback_not_observed: "未观测到模型回调", token_timing_not_exposed: "服务商未提供 Token 时序", no_token_usage_callback: "未收到 Token 用量回调" })[value] || value || "未记录原因"; }
function phaseBudgetText(value) { return Object.entries(value || {}).map(([key, budget]) => `${key} ${number(budget)} ms`).join(" · ") || "阶段预算未记录"; }
function coverageTone(value) { return !value?.unresolved_slots?.length && Number(value?.coverage_score) === 1 ? "complete" : "incomplete"; }
function deadlineTone(value) { return Number(value?.remaining_ms) > 0 && value?.absolute_timestamps_persisted === false ? "complete" : "incomplete"; }
function nodeLabel(value) { return ({ extract_keywords: "理解问题", rewrite_follow_up: "补全追问", recall_column: "字段召回", recall_metric: "指标召回", recall_training: "SQL 样例召回", recall_value: "字段值检索", merge_retrieved_info: "合并上下文", build_semantic_ir: "构建语义约束", filter_metric: "筛选指标", filter_table: "筛选数据表", add_extra_context: "补充元数据", decide_plan_route: "评估规划必要性", plan_sql: "规划查询", generate_sql: "生成 SQL", validate_sql: "校验 SQL", correct_sql: "修正 SQL", execute_sql: "执行查询", generate_chart: "生成图表", generate_analysis: "生成解读", generate_recommend: "生成追问建议" })[value] || "执行节点"; }
function reviewLabel(value) { return ({ pending: "待审核", reviewed: "已审核", dismissed: "已忽略" })[value] || value; }
function reviewCategoryLabel(value) { return ({ invalid_feedback: "无效反馈", data_issue: "数据问题", retrieval_issue: "召回问题", sql_issue: "SQL 问题", metric_issue: "指标口径", presentation_issue: "展示问题" })[value] || value; }
function retrievalRouteLabel(value) { return ({ column: "字段召回", metric: "指标/术语召回", training: "SQL 样例召回", value: "字段值召回" })[value] || value || "未知路由"; }
function listText(value) { return (value || []).join(" / ") || "—"; }
function signalKinds(value) { return [...new Set((value || []).map(item => item.kind).filter(Boolean))].join(" / ") || "—"; }
function candidateLabel(value) { return ({ none: "无候选", draft: "待二次确认", confirmed: "已确认" })[value] || value; }
function confidenceLabel(value) { return ({ high: "高", medium: "中", low: "低" })[value] || "未记录"; }
function confidenceTone(value) { return value === "high" ? "ok" : value === "low" ? "failed" : "warn"; }
function score(value) { const n = Number(value); return Number.isFinite(n) ? n.toFixed(2) : "—"; }
function planningActionLabel(value) { return ({ plan_then_generate: "先规划再生成", direct_generate: "直接生成", deterministic_ir_bypass: "语义 IR 直达", abstain: "灰区回退" })[value] || value || "未知决策"; }
function planningReasonLabel(value) { const text = String(value || ""); return text.replace(/^score:/, "评分：").replace(/^linear:/, "线性特征：").replace(/^structure:/, "结构：").replaceAll("_", " "); }
function semanticProfileLabel(value) { return ({ inline_v1: "P0 内联基线", registry_same_rules_v1: "P2 同规则注册表", registry_extended_v1: "P3 扩展注册表", targeted_correction_v1: "P4 定向纠错", upstream_first_v1: "P5 上游再生成", cascade_v1: "P6 分层级联" })[value] || value || "未记录"; }
function semanticRuleLabel(value) { return ({ "join.unique_metadata_edge": "唯一连接边", "projection.threshold_measure": "阈值指标投影", "ir.unique_limit_order": "唯一排序 / Limit", "ir.unique_group_by": "唯一分组（影子）" })[value] || value || "暂无命中"; }
function repairabilityLabel(value) { return ({ deterministic: "确定性修复", llm: "有界模型纠错", upstream: "上游再生成", reject: "拒绝自动修复" })[value] || value || "待分类"; }
function shortHash(value) { return String(value || "—").replace("sha256:", "").slice(0, 12); }
onMounted(load);
</script>

<style scoped>
.ops-page{flex:1 1 auto;min-height:0;overflow:auto;padding:36px 34px 70px;color:var(--da-text)}.ops-head,.panel,.metric-grid,.two-columns,.alert,.empty{max-width:1320px;margin-left:auto;margin-right:auto}.ops-head{display:flex;justify-content:space-between;gap:24px;margin-bottom:22px}.eyebrow{color:var(--da-accent);font-size:10px;font-weight:750;letter-spacing:.17em}.ops-head h1{margin:7px 0 0;font-size:29px;letter-spacing:-.04em}.ops-head p,.panel-head p{margin:7px 0 0;color:var(--da-text-muted);font-size:13px}.filters{display:flex;align-items:end;justify-content:flex-end;gap:8px;flex-wrap:wrap}.filters label,.review-form label,.candidate-form label{display:grid;gap:5px;color:var(--da-text-muted);font-size:10px}.filters select,.review-form select,.review-form input,.candidate-form select,.candidate-form input,.candidate-form textarea{min-height:38px;padding:8px 10px;border:1px solid var(--da-border);border-radius:9px;background:#fff;color:var(--da-text);font:12px inherit}.btn{min-height:38px;padding:8px 13px;border:1px solid var(--da-border);border-radius:9px;background:#fff;color:var(--da-text-secondary);font:12px inherit}.btn.primary{border-color:var(--da-accent);background:var(--da-accent);color:#fff}.btn.confirm{border-color:#75c091;background:#eefaf3;color:#237848}.btn:disabled{opacity:.55}.alert{display:flex;align-items:center;gap:8px;margin-bottom:13px;padding:11px 14px;border-radius:10px;font-size:12px}.alert.error{border:1px solid #f0c5bf;background:#fff3f1;color:#bd3b2c}.alert.notice{border:1px solid #b9e2c8;background:#f0fbf4;color:#287b4a}.loading,.empty{min-height:360px;display:grid;place-items:center}.empty{display:flex;flex-direction:column;justify-content:center;border:1px dashed var(--da-border);border-radius:16px;background:#fff;color:var(--da-text-muted)}.empty h2{margin:12px 0 4px;color:var(--da-text);font-size:18px}.empty p{font-size:12px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:17px}.metric{position:relative;min-width:0;overflow:hidden;padding:17px;border:1px solid var(--da-border-light);border-radius:14px;background:#fff;box-shadow:var(--da-shadow-sm)}.metric:before{position:absolute;inset:0 auto 0 0;width:3px;background:#9aa6b6;content:""}.metric.good:before{background:#35a56a}.metric.warn:before{background:#e0a33a}.metric.bad:before{background:#dc6656}.metric span,.metric small{display:block;color:var(--da-text-muted);font-size:10px}.metric strong{display:block;margin:7px 0 4px;overflow-wrap:anywhere;font-size:20px;letter-spacing:-.03em}.panel{margin-bottom:17px;padding:21px;border:1px solid var(--da-border-light);border-radius:16px;background:rgba(255,255,255,.92);box-shadow:var(--da-shadow-sm)}.panel-head{display:flex;justify-content:space-between;gap:15px;margin-bottom:15px}.panel-head h2{margin:0;font-size:16px}.panel-head>span{align-self:start;padding:5px 9px;border-radius:99px;background:var(--da-bg-soft);color:var(--da-text-muted);font-size:10px}.split,.two-columns{display:grid;grid-template-columns:1fr 1.25fr;gap:17px}.two-columns{grid-template-columns:1fr 1fr}.trend{display:grid;align-content:start;gap:8px}.trend-row{display:grid;grid-template-columns:84px minmax(80px,1fr) 48px 92px;align-items:center;gap:8px;font-size:10px}.trend-row time,.trend-row span{color:var(--da-text-muted)}.bar,.funnel>div>div{height:7px;overflow:hidden;border-radius:99px;background:#edf1f6}.bar i,.funnel i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#377ef1,#62a1ff)}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:11px}th,td{padding:9px 8px;border-bottom:1px solid var(--da-border-light);text-align:left;white-space:nowrap}th{color:var(--da-text-muted);font-weight:600}.mono,code{font-family:Consolas,monospace}.node-name,.node-id{display:block}.node-name{font-size:11px}.node-id{margin-top:2px;color:var(--da-text-muted);font:9px Consolas,monospace}.badge{display:inline-block;padding:4px 7px;border-radius:99px;background:#eef1f5;color:#667287;font-size:9px}.badge.ok,.badge.reviewed{background:#e9f8ef;color:#257949}.badge.warn{background:#fff6e4;color:#946314}.badge.failed{background:#fff0ee;color:#b63e31}.badge.pending{background:#fff6e4;color:#946314}.badge.dismissed{background:#eef1f5;color:#667287}.badge.category{background:#eef5ff;color:#2b65bd}.funnel{display:grid;gap:10px}.funnel>div{display:grid;grid-template-columns:105px 1fr 28px;align-items:center;gap:9px;font-size:11px}.funnel i{background:#de7061}.success{padding:22px;text-align:center;color:#2d8554;font-size:12px}.muted{color:var(--da-text-muted);font-size:11px}.feedback-list{display:grid;gap:10px}.feedback-card{padding:15px;border:1px solid var(--da-border-light);border-radius:12px;background:#fbfcfe}.feedback-title{display:flex;align-items:center;gap:7px}.feedback-title strong{min-width:0;flex:1;font-size:13px}.feedback-main p{margin:7px 0;color:var(--da-text-secondary);font-size:11px}.feedback-main small{color:var(--da-text-muted);font-size:10px}.review-form{display:grid;grid-template-columns:160px minmax(220px,1fr) auto auto;align-items:end;gap:8px;margin-top:12px;padding-top:11px;border-top:1px solid var(--da-border-light)}.candidate-box{margin-top:12px;padding-top:11px;border-top:1px solid var(--da-border-light)}.candidate-form{display:grid;grid-template-columns:130px minmax(200px,.8fr) minmax(280px,1.3fr) auto;align-items:end;gap:8px}.candidate-form textarea{resize:vertical}.candidate-summary{display:flex;align-items:end;justify-content:space-between;gap:13px}.candidate-summary>div{min-width:0;display:grid;gap:5px}.candidate-summary span{color:var(--da-text-muted);font-size:10px}.candidate-summary strong{font-size:12px}.candidate-summary code{overflow:hidden;text-overflow:ellipsis;color:var(--da-text-secondary);font-size:10px;white-space:nowrap}.confirmed{display:inline-flex;align-items:center;gap:5px;color:#27794a!important}.link{border:0;background:transparent;color:var(--da-accent);font:11px inherit}.modal-mask{position:fixed;z-index:120;inset:0;display:grid;place-items:center;padding:20px;background:rgba(18,28,45,.42)}.trace-dialog{width:min(780px,100%);max-height:85vh;overflow:auto;padding:22px;border-radius:17px;background:#fff;box-shadow:var(--da-shadow-lg)}.dialog-head{display:flex;justify-content:space-between}.dialog-head h2{margin:6px 0 0;font-size:19px}.close{width:34px;height:34px;display:grid;place-items:center;border:0;border-radius:9px;background:var(--da-bg-soft);color:var(--da-text-secondary)}.trace-meta{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:18px 0}.trace-meta div{padding:11px;border-radius:9px;background:var(--da-bg-soft)}.trace-meta dt{color:var(--da-text-muted);font-size:9px}.trace-meta dd{overflow-wrap:anywhere;margin:5px 0 0;font-size:11px}.step-list{display:grid;gap:7px}.step-list>div{display:grid;grid-template-columns:9px minmax(140px,1fr) 90px 2fr;align-items:center;gap:7px;padding:9px;border:1px solid var(--da-border-light);border-radius:8px;font-size:10px}.dot{width:7px;height:7px;border-radius:50%;background:#9aa6b6}.dot.success{background:#35a56a}.dot.failed{background:#dc6656}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
  @media(max-width:1100px){.ops-head{display:block}.filters{justify-content:flex-start;margin-top:14px}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.split,.two-columns{grid-template-columns:1fr}.candidate-form{grid-template-columns:1fr 1fr}.sql-field{grid-column:1/-1}}
  @media(max-width:700px){.ops-page{padding:25px 14px 50px}.metric-grid{grid-template-columns:1fr}.review-form,.candidate-form{grid-template-columns:1fr}.trace-meta{grid-template-columns:1fr}.step-list>div{grid-template-columns:9px 1fr auto}.step-list small{grid-column:2/-1}.candidate-summary{display:grid}.trend-row{grid-template-columns:76px 1fr 45px}.trend-row span{display:none}.metric strong{font-size:22px}.filters{display:grid;grid-template-columns:1fr 1fr}.filters .btn{grid-column:1/-1}.filters select{width:100%}}
.alert.warning{border:1px solid #e9cf91;background:#fff8e8;color:#815d17}
.candidate-trace{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:0 0 14px}.candidate-trace article{display:grid;gap:5px;padding:11px;border:1px solid var(--da-border-light);border-radius:9px;background:var(--da-bg-soft);font-size:10px}.candidate-trace span,.candidate-trace small{color:var(--da-text-muted)}
.retrieval-trace{display:grid;gap:8px;margin:0 0 14px;padding:12px;border:1px solid #dce8f5;border-radius:11px;background:#f8fbff}.retrieval-trace header strong,.retrieval-trace header small{display:block}.retrieval-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.retrieval-trace article{padding:9px;border-radius:8px;background:#fff}.retrieval-route,.hybrid-proof,.signal-proof{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.retrieval-route code,.retrieval-route span,.hybrid-proof span,.signal-proof span{padding:4px 6px;border-radius:6px;background:var(--da-bg-soft);color:var(--da-text-secondary);font-size:9px}.hybrid-proof,.signal-proof{margin-top:7px}.signal-proof{padding:7px;border:1px dashed #d8e2ef;border-radius:8px}.signal-proof code{font-size:9px}
.planning-trace{display:grid;gap:8px;margin:0 0 14px;padding:12px;border:1px solid #ddd5f1;border-radius:11px;background:#faf8ff}.planning-trace header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.planning-trace header strong,.planning-trace header small{display:block}.planning-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.planning-trace header>span{padding:5px 8px;border-radius:99px;background:#eee9fb;color:#67538f;font-size:9px}.planning-proof,.planning-shadow-proof{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.planning-proof span,.planning-shadow-proof span{padding:5px 7px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:9px}.planning-trace p{margin:0;color:#6c5b88;font-size:9px}
.semantic-trace{display:grid;gap:8px;margin:0 0 14px;padding:12px;border:1px solid #cfe0f4;border-radius:11px;background:#f7fbff}.semantic-trace header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.semantic-trace header strong,.semantic-trace header small{display:block}.semantic-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.semantic-trace header>span{padding:5px 8px;border-radius:99px;background:#e7f1ff;color:#2d65b7;font-size:9px}.semantic-proof,.semantic-rule-proof{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.semantic-proof span,.semantic-rule-proof span{padding:5px 7px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:9px}.semantic-trace p{margin:0;color:#315f9d;font-size:9px}.semantic-hash{color:var(--da-text-muted);font:8px Consolas,monospace}.semantic-ops{border-color:#d9e6f5}
.runtime-ops{border-color:#d8e2f0;background:linear-gradient(145deg,rgba(255,255,255,.96),rgba(247,250,255,.96))}.runtime-policy-grid{grid-template-columns:repeat(3,1fr)}.runtime-policy-grid article{position:relative;overflow:hidden}.runtime-policy-grid article:before{position:absolute;inset:0 auto 0 0;width:3px;background:#94a2b5;content:""}.runtime-policy-grid article.runtime-ok:before{background:#35a56a}.runtime-policy-grid article.runtime-warn:before{background:#e2a63b}
.coverage-trace,.deadline-trace{display:grid;gap:9px;margin:0 0 14px;padding:12px;border:1px solid #cde3d5;border-radius:11px;background:#f6fcf8}.deadline-trace{border-color:#ded8f1;background:#fbf9ff}.coverage-trace header,.deadline-trace header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.coverage-trace header strong,.coverage-trace header small,.deadline-trace header strong,.deadline-trace header small{display:block}.coverage-trace header small,.deadline-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.coverage-trace header>span,.deadline-trace header>span{padding:5px 8px;border-radius:99px;font-size:9px}.coverage-trace header>span.complete,.deadline-trace header>span.complete{background:#e3f5e9;color:#247848}.coverage-trace header>span.incomplete,.deadline-trace header>span.incomplete{background:#fff0ee;color:#b63e31}.coverage-proof,.deadline-proof{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.coverage-proof span,.deadline-proof span{padding:5px 7px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:9px}.coverage-slots{display:grid;gap:6px;margin:0}.coverage-slots div{display:grid;grid-template-columns:48px minmax(0,1fr);gap:7px}.coverage-slots dt{color:var(--da-text-muted);font-size:9px}.coverage-slots dd{overflow-wrap:anywhere;margin:0;font:9px/1.45 Consolas,monospace}
.contract-trace{display:grid;gap:9px;margin:0 0 14px;padding:12px;border:1px solid #d5def4;border-radius:11px;background:#f7f9ff}.contract-trace header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.contract-trace header strong,.contract-trace header small{display:block}.contract-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.contract-trace header>span{padding:5px 8px;border-radius:99px;font-size:9px}.contract-trace header>span.complete{background:#e5f0ff;color:#315f9d}.contract-trace header>span.incomplete{background:#fff1e7;color:#a45f22}.contract-proof,.relation-proof{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.contract-proof span,.relation-proof span,.relation-proof b{padding:5px 7px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:9px}.relation-proof{padding-top:7px;border-top:1px dashed #d8e0f0}.relation-proof b{color:#315f9d}
.llm-trace{display:grid;gap:9px;margin:0 0 14px;padding:12px;border:1px solid #d9d8f3;border-radius:11px;background:#fafaff}.llm-trace header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.llm-trace header strong,.llm-trace header small{display:block}.llm-trace header small{margin-top:4px;color:var(--da-text-muted);font-size:9px}.llm-trace header>span{padding:5px 8px;border-radius:99px;font-size:9px}.llm-trace header>span.complete{background:#e9f8ef;color:#257949}.llm-trace header>span.incomplete{background:#fff0ee;color:#b63e31}.llm-accounting{display:flex;flex-wrap:wrap;gap:7px}.llm-accounting span{padding:5px 7px;border-radius:7px;background:#fff;color:var(--da-text-secondary);font-size:9px}.llm-attempt-table{overflow:auto;border:1px solid var(--da-border-light);border-radius:8px;background:#fff}.llm-attempt-row{min-width:680px;display:grid;grid-template-columns:1.25fr 1.15fr .8fr .4fr .65fr .45fr;gap:7px;padding:7px 9px;border-top:1px solid var(--da-border-light);font-size:9px}.llm-attempt-row:first-child{border-top:0}.llm-attempt-head{background:var(--da-bg-soft);color:var(--da-text-muted)}.llm-attempt-row strong{font-family:Consolas,monospace}.llm-attempt-row .ok{color:#257949}.llm-attempt-row .failed{color:#b63e31}
.llm-attempt-row{min-width:1430px;grid-template-columns:1.15fr 1.1fr .72fr .82fr .72fr .72fr .75fr 1.2fr .85fr}.llm-attempt-row span{overflow-wrap:anywhere}.llm-attempt-row span small{display:block;margin-top:2px;color:var(--da-text-muted)}
.policy-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.policy-grid article{padding:14px;border:1px solid var(--da-border-light);border-radius:10px;background:var(--da-bg-soft)}.policy-grid span,.policy-grid small{display:block;color:var(--da-text-muted);font-size:10px}.policy-grid strong{display:block;margin:7px 0 4px;font-size:17px}
@media(max-width:900px){.runtime-policy-grid{grid-template-columns:1fr 1fr}}
@media(max-width:700px){.candidate-trace,.policy-grid,.runtime-policy-grid{grid-template-columns:1fr}}
</style>
