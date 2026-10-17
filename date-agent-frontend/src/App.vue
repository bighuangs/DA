<template>
  <div class="chat-page">
    <AuthView v-if="authReady && !currentUser" @authenticated="handleAuthenticated" />
    <template v-else-if="authReady">
      <div class="app-shell">
        <AppNavigation
          :mode="mode"
          :current-user="currentUser"
          :conversations="conversations"
          :active-chat-id="chatId || ''"
          :history-status="conversationListStatus"
          :history-loading="conversationListLoading"
          :history-error="conversationListError"
          :history-has-more="conversationListHasMore"
          :history-retention-days="conversationRetentionDays"
          @update:mode="mode = $event"
          @new-chat="newConversation"
          @logout="logout"
          @open-chat="openConversation"
          @refresh-history="loadConversations({ reset: true })"
          @load-more-history="loadConversations({ reset: false })"
          @archive-chat="setConversationStatus($event, 'archived')"
          @restore-chat="setConversationStatus($event, 'active')"
          @delete-chat="deleteConversation"
          @history-status-change="changeConversationListStatus"
        />

        <main class="app-main">
          <DashboardView
            v-if="mode === 'dashboard'"
            :select-id="pendingDashboardId"
            :can-edit="canEditDashboard"
            class="dash-slot"
            @open-chat="mode = 'chat'"
          />
          <MetaKnowledgeView v-else-if="mode === 'meta'" class="meta-slot" />
          <AdhocView
            v-else-if="mode === 'adhoc'"
            class="adhoc-slot"
            @use-dataset="useAdhocDataset"
          />
          <MaskingRuleView v-else-if="mode === 'masking'" class="masking-slot" />
          <WorkspaceView v-else-if="mode === 'workspace'" class="workspace-slot" :session="currentUser" @session-changed="handleAuthenticated" />
          <AccessPolicyView v-else-if="mode === 'access'" class="access-slot" />
          <EvaluationView v-else-if="mode === 'evaluation'" class="evaluation-slot" />
          <QualityOperationsView v-else-if="mode === 'quality-operations'" class="evaluation-slot" />
          <ScenarioMatrixView v-else-if="mode === 'scenario-matrix'" class="evaluation-slot" />
          <ResilienceView v-else-if="mode === 'resilience'" class="evaluation-slot" />
          <AdhocIsolationEvaluationView v-else-if="mode === 'adhoc-isolation'" class="evaluation-slot" />
          <RAGEvaluationView v-else-if="mode === 'rag-evaluation'" class="evaluation-slot" />
          <PlanningEvaluationView v-else-if="mode === 'planning-evaluation'" class="evaluation-slot" />
          <SemanticRepairEvaluationView v-else-if="mode === 'semantic-repair'" class="evaluation-slot" />

          <template v-else>
            <section
              v-if="activeConversation || historyHydrating"
              class="conversation-context"
              :class="{ archived: conversationReadOnly }"
              aria-live="polite"
            >
              <div class="conversation-context-icon">
                <LoaderCircle v-if="historyHydrating" :size="16" class="spinning" aria-hidden="true" />
                <History v-else :size="16" aria-hidden="true" />
              </div>
              <div class="conversation-context-copy">
                <strong>{{ historyHydrating ? "正在恢复会话…" : activeConversation?.title }}</strong>
                <span v-if="!historyHydrating">
                  {{ conversationReadOnly ? "已归档 · 只读浏览" : `已恢复 ${activeConversation?.turn_count || 0} 轮对话` }}
                  <template v-if="activeConversation?.context_reset_at"> · 上下文曾重置，历史仍完整保留</template>
                </span>
              </div>
              <button
                v-if="transcriptHasMore && !historyHydrating"
                type="button"
                class="conversation-context-action"
                :disabled="transcriptLoadingMore"
                @click="loadOlderTurns"
              >
                <LoaderCircle v-if="transcriptLoadingMore" :size="13" class="spinning" aria-hidden="true" />
                {{ transcriptLoadingMore ? "加载中…" : "更早记录" }}
              </button>
              <button
                v-if="conversationReadOnly && !historyHydrating"
                type="button"
                class="conversation-context-action primary"
                @click="setConversationStatus(activeConversation, 'active')"
              >
                      <History :size="13" aria-hidden="true" />恢复并继续
              </button>
            </section>

            <div ref="messagesEl" class="messages">
              <div v-if="messages.length === 0" class="welcome">
                <div class="welcome-eyebrow"><i aria-hidden="true"></i> AI DATA COPILOT</div>
                <div class="welcome-logo" aria-hidden="true"><span></span><span></span><span></span></div>
                <div class="welcome-title">从一个业务问题开始</div>
                <div class="welcome-sub">
                  描述你想了解的指标、范围或对比，我会完成取数、校验和可视化；技术执行证据随时可以展开核对。
                </div>
                <div class="welcome-capabilities">
                  <span><b aria-hidden="true">✓</b>只读安全查询</span>
                  <span><b aria-hidden="true">▥</b>表格与图表</span>
                  <span><b aria-hidden="true">◎</b>统一指标口径</span>
                </div>
                <div class="welcome-suggestions">
                  <button
                    v-for="(suggestion, index) in suggestions"
                    :key="suggestion.text"
                    type="button"
                    class="suggestion-card"
                    @click="askRecommend(suggestion.text)"
                  >
                    <span class="suggestion-icon">
                      <span aria-hidden="true">{{ suggestion.glyph }}</span>
                    </span>
                    <span class="suggestion-text">{{ suggestion.text }}</span>
                    <span class="suggestion-arrow" aria-hidden="true">↗</span>
                  </button>
                </div>
              </div>

              <div v-else class="chat-stream">
                <div
                  v-for="(msg, index) in messages"
                  :key="index"
                  class="row"
                  :class="[
                    msg.role,
                    { wide: ['result', 'recommend', 'steps', 'analysis', 'sql', 'predict', 'anomaly', 'predict_failed', 'history-status', 'intent-response'].includes(msg.type) },
                  ]"
                >
                  <div v-if="msg.role === 'assistant'" class="avatar ai" aria-hidden="true">
                    <Bot :size="19" />
                  </div>

                  <div class="content">
                    <div v-if="msg.type === 'text' && msg.role === 'user'" class="user-bubble">
                      {{ msg.content }}
                    </div>

                    <details
                      v-else-if="msg.type === 'steps'"
                      class="steps-card"
                      :open="thinkingState(msg) === 'running'"
                    >
                      <summary class="steps-header" :aria-label="`${thinkingLabel(msg)}，查看执行详情`">
                        <span class="thinking-dot" :class="thinkingState(msg)"></span>
                        <span class="steps-title">{{ thinkingLabel(msg) }}</span>
                        <span class="steps-progress">{{ completedStepCount(msg.steps) }}/{{ msg.steps.length }} 步</span>
                        <span v-if="msg.trace" class="steps-latency" :class="latencyTone(msg.trace.tt_result_ms)">
                          结果 {{ formatDuration(msg.trace.tt_result_ms) }}
                        </span>
                        <span class="steps-detail-label">查看执行详情</span>
                        <span class="steps-chevron" aria-hidden="true">⌄</span>
                      </summary>
                      <div class="steps-detail">
                        <div class="steps-list">
                          <div v-for="(step, stepIndex) in msg.steps" :key="stepIndex" class="step">
                            <span class="step-dot" :class="step.status"></span>
                            <span class="step-text">{{ step.text }}</span>
                          </div>
                        </div>
                        <div v-if="msg.followUp" class="follow-up-audit">
                          <strong>多轮闭合 · {{ msg.followUp.action }}</strong>
                          <span>{{ msg.followUp.standalone_question }}</span>
                        </div>
                        <div v-if="msg.candidateDecision" class="follow-up-audit candidate-audit">
                          <strong>候选判优 · {{ msg.candidateDecision.selected }}</strong>
                          <span>{{ msg.candidateDecision.adaptive_triggered ? `已比较 ${msg.candidateDecision.candidate_count} 个候选` : "高置信单候选快路径" }} · {{ msg.candidateDecision.selection_reason }}</span>
                        </div>
                        <div v-if="msg.planningRoute" class="planning-audit">
                          <div>
                            <strong>规划路由 · {{ planningActionLabel(msg.planningRoute.actual_action) }}</strong>
                            <span>{{ profileLabel(msg.planningRoute.profile) }} · 决策 {{ formatDecisionMs(msg.planningRoute.decision_ms) }}</span>
                          </div>
                          <small>{{ (msg.planningRoute.reason_codes || []).map(planningReasonLabel).join(" · ") || "无额外触发原因" }}</small>
                          <div v-if="msg.planningRoute.shadow_decisions?.length" class="planning-shadows">
                            <span v-for="shadow in msg.planningRoute.shadow_decisions" :key="shadow.profile">
                              {{ profileLabel(shadow.profile) }} → {{ planningActionLabel(shadow.decision) }}
                            </span>
                          </div>
                        </div>
                        <div v-if="msg.semanticRepair || msg.semanticViolation" class="semantic-repair-audit">
                          <div>
                            <strong>语义守卫 · {{ semanticRepairStatus(msg.semanticRepair, msg.semanticViolation) }}</strong>
                            <span>{{ semanticProfileLabel(msg.semanticRepair?.profile) }} · {{ formatDecisionMs(msg.semanticRepair?.repair_ms) }}</span>
                          </div>
                          <small v-if="msg.semanticViolation">
                            {{ msg.semanticViolation.violation_code }} · {{ repairabilityLabel(msg.semanticViolation.repairability) }}
                          </small>
                          <div v-if="msg.semanticRepair?.applied_rules?.length" class="semantic-rule-chips">
                            <span v-for="rule in msg.semanticRepair.applied_rules" :key="rule.rule_code">{{ semanticRuleLabel(rule.rule_code) }}</span>
                          </div>
                          <small v-else>未触发确定性改写；SQL 继续经过权限、语义、IR、成本与指标口径复核。</small>
                          <em v-if="msg.semanticRepair?.fallback_used">已使用 P0 fallback · {{ msg.semanticRepair.fallback_reason || "注册表异常" }}</em>
                        </div>
                        <div v-if="msg.memoryContext" class="memory-audit">
                          <strong>分层会话记忆 · {{ msg.memoryContext.schema_version || "memory-context-v2" }} · {{ msg.memoryContext.working_source || "MySQL" }}</strong>
                          <span>
                            近期精确 {{ msg.memoryContext.recent_selected || 0 }} ·
                            召回摘要 {{ msg.memoryContext.summary_count || 0 }} ·
                            身份事实 {{ msg.memoryContext.profile_fact_count || 0 }} ·
                            长期目标 {{ msg.memoryContext.long_term_goal_count || 0 }}
                          </span>
                          <small>
                            历史片段 {{ msg.memoryContext.episode_count || 0 }} ·
                            长期 ES {{ msg.memoryContext.es_candidates || 0 }} / Qdrant
                            {{ msg.memoryContext.qdrant_candidates || 0 }} ·
                            {{ msg.memoryContext.authoritative_recheck ? "已回 MySQL 复核" : "未复核" }}
                          </small>
                          <small v-if="msg.memoryContext.summary_status">
                            摘要 {{ msg.memoryContext.summary_status }} · {{ msg.memoryContext.summary_trigger || "未触发远距召回" }} ·
                            近期原文 {{ msg.memoryContext.recent_messages_in_prompt === false ? "未注入" : "已纳入预算" }}
                          </small>
                        </div>
                        <div v-if="msg.sourceContext" class="source-context-audit">
                          <strong>数据上下文 · {{ sourceModeLabel(msg.sourceContext.source_mode) }}</strong>
                          <span>
                            {{ msg.sourceContext.datasets?.length || 0 }} 个上传版本 ·
                            revision {{ msg.sourceContext.revision || 1 }}
                          </span>
                          <small>每轮冻结 · 服务端资产授权 · {{ String(msg.sourceContext.context_fingerprint || "").slice(0, 19) }}</small>
                        </div>
                        <div v-if="msg.sourceComparison" class="source-comparison-audit">
                          <strong>确定性来源对比 · {{ msg.sourceComparison.status }}</strong>
                          <template v-if="msg.sourceComparison.status === 'compared'">
                            <template v-if="msg.sourceComparison.mode === 'grouped'">
                              <span>
                                按 {{ (msg.sourceComparison.dimensions || []).join("、") || "分组键" }}
                                对齐 {{ msg.sourceComparison.compared_key_count }} 组 ·
                                匹配 {{ msg.sourceComparison.matched_key_count }} 组
                              </span>
                              <small>
                                {{ comparisonSourceLabel(msg.sourceComparison.baseline_source) }}
                                {{ formatComparisonValue(msg.sourceComparison.baseline_value) }}
                                →
                                {{ comparisonSourceLabel(msg.sourceComparison.comparison_source) }}
                                {{ formatComparisonValue(msg.sourceComparison.comparison_value) }}；
                                总差值 {{ formatComparisonValue(msg.sourceComparison.delta) }}
                                <template v-if="msg.sourceComparison.delta_pct !== null">
                                  （{{ msg.sourceComparison.delta_pct }}%）
                                </template>
                              </small>
                              <details class="source-comparison-details">
                                <summary>
                                  查看逐组差异（{{ msg.sourceComparison.comparisons?.length || 0 }}）
                                </summary>
                                <div class="source-comparison-table-wrap">
                                  <table>
                                    <thead>
                                      <tr>
                                        <th>分组</th>
                                        <th>{{ comparisonSourceLabel(msg.sourceComparison.baseline_source) }}</th>
                                        <th>{{ comparisonSourceLabel(msg.sourceComparison.comparison_source) }}</th>
                                        <th>差值</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      <tr
                                        v-for="(item, comparisonIndex) in msg.sourceComparison.comparisons"
                                        :key="comparisonIndex"
                                      >
                                        <td>{{ comparisonDimensionText(item.dimensions) }}</td>
                                        <td>{{ formatComparisonValue(item.baseline_value) }}</td>
                                        <td>{{ formatComparisonValue(item.comparison_value) }}</td>
                                        <td>
                                          {{ formatComparisonValue(item.delta) }}
                                          <small v-if="item.delta_pct !== null">
                                            {{ item.delta_pct }}%
                                          </small>
                                        </td>
                                      </tr>
                                    </tbody>
                                  </table>
                                </div>
                              </details>
                            </template>
                            <span v-else>
                              {{ msg.sourceComparison.measure }} 差值
                              {{ formatComparisonValue(msg.sourceComparison.delta) }}
                              <template v-if="msg.sourceComparison.delta_pct !== null">
                                （{{ msg.sourceComparison.delta_pct }}%）
                              </template>
                            </span>
                          </template>
                          <small v-else>{{ msg.sourceComparison.reason }}</small>
                        </div>
                        <div v-if="msg.trace" class="trace-summary">
                          <span>链路 {{ String(msg.trace.trace_id || "—").slice(0, 8) }}</span>
                          <span>结果 {{ msg.trace.tt_result_ms ?? "—" }} ms</span>
                          <span>完成 {{ msg.trace.tt_complete_ms ?? msg.trace.duration_ms }} ms</span>
                          <span>{{ msg.trace.llm_calls ?? 0 }} 次 LLM</span>
                          <span>{{ msg.trace.total_tokens }} tokens</span>
                          <span v-if="Number(msg.trace.estimated_cost) > 0">≈ {{ msg.trace.estimated_cost }} {{ msg.trace.cost_currency }}</span>
                          <span v-if="msg.trace.degraded_count" class="trace-degraded">降级 {{ msg.trace.degraded_count }} 项</span>
                          <span v-if="msg.trace.follow_up_action" class="trace-follow-up">多轮 {{ msg.trace.follow_up_action }}</span>
                          <span v-if="msg.trace.candidate_band" class="trace-follow-up">置信度 {{ msg.trace.candidate_band }} {{ Number(msg.trace.candidate_confidence || 0).toFixed(2) }}</span>
                          <span v-if="msg.trace.candidate_count > 1" class="trace-degraded">候选 {{ msg.trace.candidate_count }} · 选择 {{ msg.trace.candidate_decision?.selected }}</span>
                          <details v-if="msg.trace.slowest_nodes?.length" class="trace-nodes">
                            <summary>最慢节点</summary>
                            <div v-for="node in msg.trace.slowest_nodes" :key="`${node.node_id}-${node.duration_ms}`">
                              <span>{{ nodeLabel(node.node_id) }} <code>{{ node.node_id }}</code></span><strong>{{ node.duration_ms }} ms</strong>
                            </div>
                          </details>
                          <details v-if="msg.trace.retrieval_metrics?.length" class="trace-retrieval">
                            <summary>召回职责与版本</summary>
                            <section v-for="(metric, metricIndex) in msg.trace.retrieval_metrics" :key="`${metric.route || metric.responsibility}-${metricIndex}`">
                              <strong>{{ metric.responsibility === "es_field_value" ? "ES · 真实字段值" : metric.responsibility === "conversation_memory" ? "会话记忆 · Redis/MySQL/ES/Qdrant" : `Qdrant · ${metric.route || "元知识"}` }}</strong>
                              <span v-if="metric.responsibility === 'es_field_value'">active {{ String(metric.index_version || "未发布").replace("sha256:", "").slice(0, 12) }} · {{ (metric.query_modes || []).join(" / ") || "full_text" }} · 表/列权限 {{ metric.workspace_scope_applied ? "已下推" : "缺失" }} · Sparse 推断 literal {{ metric.sparse_literal_inference ? "是" : "否" }}</span>
                              <span v-else-if="metric.responsibility === 'conversation_memory'">{{ metric.working_source || "MySQL" }} · Redis {{ metric.redis_hit ? "命中" : "重建/降级" }} · 近期 {{ metric.recent_selected || 0 }} · 摘要 {{ metric.summary?.selected_count || metric.summary_count || 0 }} · 长期 ES {{ metric.es_candidates || 0 }} · Qdrant {{ metric.qdrant_candidates || 0 }} · 权威复核 {{ metric.authoritative_recheck ? "通过" : "缺失" }}</span>
                              <span v-else>{{ metric.embedding_mode || "dense_sparse" }} · {{ metric.fusion_method || "entity_rrf" }} · rule-rerank-v6 · 选中 {{ metric.selected_count ?? "—" }}</span>
                              <small v-for="(audit, auditIndex) in metric.value_search_audits || []" :key="`${audit.query_fingerprint}-${auditIndex}`">{{ audit.query_mode }} · 命中 {{ audit.hit_count || 0 }} · {{ audit.latency_ms || 0 }} ms · 原值日志 {{ audit.raw_value_logged ? "是" : "否" }}</small>
                            </section>
                          </details>
                          <details v-if="msg.trace.semantic_coverage?.query_contract" class="trace-contract">
                            <summary>语义合同与关系闭包</summary>
                            <section>
                              <strong>{{ msg.trace.semantic_coverage.query_contract.authority === "closed" ? "已闭合" : "引导模式" }} · {{ msg.trace.semantic_coverage.query_contract.semantic_family || "unknown" }}</strong>
                              <span>最终粒度 {{ (msg.trace.semantic_coverage.query_contract.output_grain || []).map((item) => item.column || item.logical).join(" / ") || "标量" }}</span>
                              <span>运算 {{ (msg.trace.semantic_coverage.query_contract.operations || []).join(" / ") || "基础聚合" }}</span>
                              <span>关系 {{ msg.trace.semantic_coverage.relation_closure?.status || msg.trace.semantic_coverage.query_contract.relation_plan?.status || "—" }} · 补桥 {{ (msg.trace.semantic_coverage.relation_closure?.added_tables || msg.trace.semantic_coverage.query_contract.relation_plan?.added_tables || []).join(" / ") || "无需" }}</span>
                            </section>
                          </details>
                        </div>
                      </div>
                    </details>

                    <IntentResponseCard
                      v-else-if="msg.type === 'intent-response'"
                      :response="msg.response"
                      @suggest="askRecommend"
                    />

                    <div v-else-if="msg.type === 'sql'" class="sql-card">
                      <div class="sql-head">
                        <span class="sql-label"><span aria-hidden="true">{ }</span>生成 SQL</span>
                        <div class="sql-head-actions">
                          <button class="copy-btn" @click="copyText(msg.content)">复制</button>
                          <button
                            v-if="!msg._editing && !msg._historical && !conversationReadOnly"
                            class="copy-btn"
                            @click="startEditSql(msg)"
                          >
                            编辑
                          </button>
                          <template v-else>
                            <button class="copy-btn" aria-label="运行 SQL" :disabled="msg._running" @click="runEditedSql(msg)">
                              <span aria-hidden="true">▶</span>{{ msg._running ? "运行中…" : "运行" }}
                            </button>
                            <button class="copy-btn" @click="cancelEditSql(msg)">取消</button>
                          </template>
                        </div>
                      </div>
                      <pre v-if="!msg._editing" class="sql-code">{{ msg.content }}</pre>
                      <textarea
                        v-else
                        v-model="msg._editText"
                        class="sql-edit"
                        aria-label="编辑 SQL"
                        spellcheck="false"
                      ></textarea>
                    </div>

                    <ResultWorkbench
                      v-else-if="msg.type === 'result'"
                      :msg="msg"
                      :can-edit-dashboard="canEditDashboard"
                      :loading="loading"
                      @add-dashboard="addToDashboard"
                      @export="exportResult"
                      @predict="runPredict"
                      @anomaly="runAnomaly"
                      @point-click="handlePointClick"
                      @drill-down="runDrillDown"
                      @linked-filter="runLinkedFilter"
                      @period-compare="runPeriodCompare"
                      @clear-point="clearPoint"
                    />

                    <div v-else-if="msg.type === 'analysis'" class="analysis-card">
                      <div class="card-head">
                        <span class="card-title"><i class="inline-glyph" aria-hidden="true">◇</i>数据解读</span>
                      </div>
                      <div class="analysis-text">{{ msg.content }}</div>
                      <div class="fb-bar">
                        <template v-if="!msg._fbSubmitted">
                          <span class="fb-label">这段解读有帮助吗？</span>
                          <button class="fb-btn" aria-label="有帮助" :class="{ active: msg._fbScore === 1 }" @click="msg._fbScore = 1">
                            <span aria-hidden="true">↑</span>
                          </button>
                          <button class="fb-btn" aria-label="无帮助" :class="{ active: msg._fbScore === -1 }" @click="msg._fbScore = -1">
                            <span aria-hidden="true">↓</span>
                          </button>
                          <input
                            v-if="msg._fbScore !== null"
                            v-model="msg._fbComment"
                            class="fb-input"
                            aria-label="反馈说明"
                            placeholder="可选：补充原因（差评会优先用于改进）"
                          />
                          <button
                            v-if="msg._fbScore !== null"
                            class="mini-btn primary"
                            :disabled="msg._fbSending"
                            @click="submitFeedback(msg)"
                          >
                            {{ msg._fbSending ? "提交中…" : "提交反馈" }}
                          </button>
                        </template>
                        <span v-else class="fb-done"><span aria-hidden="true">✓</span>反馈已提交</span>
                        <span v-if="msg._fbError" class="fb-error" role="alert">{{ msg._fbError }}</span>
                      </div>
                    </div>

                    <div v-else-if="msg.type === 'recommend'" class="recommend-card">
                      <div class="recommend-title">继续分析</div>
                      <div class="recommend-grid">
                        <button
                          v-for="(recommendation, recommendationIndex) in msg.questions"
                          :key="recommendationKey(recommendation, recommendationIndex)"
                          class="recommend-chip"
                          :disabled="loading"
                          @click="askRecommend(recommendation)"
                        >
                          <span>{{ recommendationLabel(recommendation) }}</span><i class="inline-glyph arrow" aria-hidden="true">→</i>
                        </button>
                      </div>
                    </div>

                    <div v-else-if="msg.type === 'predict'" class="chart-card predict-card">
                      <div class="card-head">
                        <span class="card-title"><i class="inline-glyph" aria-hidden="true">↗</i>数据预测（含预测段）</span>
                        <span class="card-badge">{{ chartTypeLabel(msg.chart.type) }}</span>
                      </div>
                      <div class="card-body">
                        <ChartView :chart="msg.chart" :rows="msg.rows" :predict-start="msg.predictStart" />
                      </div>
                    </div>

                    <div v-else-if="msg.type === 'anomaly'" class="analysis-card anomaly-card">
                      <div class="card-head">
                        <span class="card-title"><i class="inline-glyph warning" aria-hidden="true">!</i>数据异常检测</span>
                      </div>
                      <div class="analysis-text">{{ msg.content }}</div>
                    </div>

                    <div v-else-if="msg.type === 'predict_failed'" class="error-card" role="alert">
                      <i class="inline-glyph" aria-hidden="true">∅</i>
                      <span class="error-text">{{ msg.content }}</span>
                    </div>

                    <div v-else-if="msg.type === 'error'" class="error-card" role="alert">
                      <span aria-hidden="true">!</span>
                      <div class="error-text">
                        <span>{{ msg.content }}</span>
                        <div v-if="msg.clarification" class="clarification-help">
                          <strong>建议补充后重试</strong>
                          <ul>
                            <li v-for="item in msg.clarification.suggestions || []" :key="item">{{ item }}</li>
                          </ul>
                          <div v-if="msg.clarification.unresolved_slots?.length" class="clarification-slots">
                            <code v-for="slot in msg.clarification.unresolved_slots" :key="slot">{{ slot }}</code>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div
                      v-else-if="msg.type === 'history-status'"
                      class="history-status-card"
                      :class="msg.status"
                      role="status"
                    >
                      <History :size="16" aria-hidden="true" />
                      <span>
                        <strong>{{ historyTurnStatusLabel(msg.status) }}</strong>
                        <small>{{ msg.content }}</small>
                        <em v-if="msg.sourceContext?.source_mode">
                          数据上下文：{{ sourceModeLabel(msg.sourceContext.source_mode) }}
                          · revision {{ msg.sourceContext.revision || 1 }}
                        </em>
                        <em v-if="msg.unavailable?.length">
                          未保存：{{ msg.unavailable.map(historyPartLabel).join("、") }}
                        </em>
                      </span>
                    </div>
                  </div>

                  <div v-if="msg.role === 'user'" class="avatar user" aria-hidden="true">
                    <UserRound :size="18" />
                  </div>
                </div>
                <div class="stream-bottom-spacer"></div>
              </div>
            </div>

            <ConversationMemoryPanel
              v-if="memoryPanelOpen && chatId"
              :chat-id="chatId"
              @close="memoryPanelOpen = false"
              @reset-complete="handleMemoryReset"
            />
            <DataContextDrawer
              :chat-id="chatId || ''"
              @notice="showAppNotice($event.text, $event.tone)"
            />

            <div class="input-wrapper" :class="{ readonly: conversationReadOnly }">
              <div class="input-box">
                <label
                  class="attach-btn"
                  :class="{ disabled: loading || historyHydrating || conversationReadOnly }"
                  title="在当前会话上传 CSV/XLSX"
                >
                  <Paperclip :size="18" aria-hidden="true" />
                  <input
                    type="file"
                    aria-label="上传会话数据文件"
                    accept=".csv,.xlsx"
                    :disabled="loading || historyHydrating || conversationReadOnly"
                    @change="uploadAdhocFromChat"
                  />
                </label>
                <textarea
                  ref="textareaEl"
                  v-model="question"
                  rows="1"
                  aria-label="输入问题"
                  :disabled="historyHydrating || conversationReadOnly"
                  @keydown.enter.exact.prevent="sendQuestion"
                  :placeholder="conversationReadOnly ? '该会话已归档，恢复后可继续提问' : historyHydrating ? '正在恢复会话记录…' : '描述指标、范围或对比，回车发送…'"
                ></textarea>
                <button
                  class="send-btn"
                  aria-label="发送问题"
                  @click="sendQuestion"
                  :disabled="loading || historyHydrating || conversationReadOnly"
                  :title="conversationReadOnly ? '请先恢复归档会话' : loading ? '执行中…' : '发送'"
                >
                  <span v-if="loading" class="loader"></span>
                  <Send v-else :size="17" aria-hidden="true" />
                </button>
              </div>
              <div class="input-hint">
                <div class="input-options">
                  <button
                    type="button"
                    class="data-context-pill"
                    :class="{ upload: dataContextStore.state.applied.sourceMode !== 'dw_only', dirty: dataContextStore.state.dirty }"
                    data-testid="data-context-trigger"
                    @click="openDataContext"
                  >
                    <Database v-if="dataContextStore.state.applied.sourceMode === 'dw_only'" :size="14" aria-hidden="true" />
                    <Paperclip v-else :size="14" aria-hidden="true" />
                    <span>{{ dataContextStore.contextLabel.value }}</span>
                    <em>{{ sourceModeLabel(dataContextStore.state.applied.sourceMode) }}</em>
                    <span v-if="dataContextStore.state.dirty" class="context-dirty-dot" aria-label="有未应用更改"></span>
                  </button>
                  <label class="analysis-toggle" title="开启后会生成 AI 数据解读，复杂问题可能增加等待时间">
                    <input v-model="analysisEnabled" type="checkbox" @change="saveAnalysisPreference" />
                    <span>AI 数据解读</span>
                    <em>{{ analysisEnabled ? "已开启" : "默认关闭" }}</em>
                  </label>
                  <button v-if="chatId" type="button" class="memory-toggle" data-testid="memory-toggle" @click="memoryPanelOpen = true">
                    <i class="inline-glyph" aria-hidden="true">◉</i>会话记忆
                  </button>
                </div>
                <span>只读查询 · SQL 与执行证据可审计</span>
              </div>
            </div>
          </template>

          <div v-if="appNotice.text" class="app-toast" :class="appNotice.tone" :role="appNotice.tone === 'error' ? 'alert' : 'status'">
            <span aria-hidden="true">{{ appNotice.tone === "error" ? "!" : "✓" }}</span>
            {{ appNotice.text }}
          </div>
        </main>
      </div>

      <ConfirmContextSwitchDialog
        v-if="adhocSwitch.open"
        :dataset="adhocSwitch.dataset"
        @cancel="closeAdhocSwitchModal"
        @confirm="confirmAdhocSwitch"
      />

      <div v-if="addToDash.open" class="modal-mask" @click.self="closeDashboardModal" @keydown.esc.stop="closeDashboardModal">
        <div ref="dashboardModalEl" class="modal" role="dialog" aria-modal="true" aria-labelledby="add-dashboard-title" tabindex="-1">
          <div id="add-dashboard-title" class="modal-title">加入仪表盘</div>
          <div class="modal-sub">选择目标仪表盘，当前结果会作为一个图表块加入画布。</div>
          <div v-if="addToDash.error" class="modal-error" role="alert">{{ addToDash.error }}</div>
          <div class="modal-list">
            <div v-if="!addToDash.dashboards.length" class="modal-empty">暂无仪表盘，可在下方直接创建。</div>
            <button
              v-for="dashboard in addToDash.dashboards"
              :key="dashboard.id"
              type="button"
              class="modal-row"
              @click="confirmAddToDashboard(dashboard.id)"
            >
              <span aria-hidden="true">▦</span>{{ dashboard.name }}
            </button>
          </div>
          <div class="modal-footer">
            <button class="mini-btn" @click="closeDashboardModal">取消</button>
            <button class="mini-btn primary" @click="confirmAddToDashboard(null)">
              <span aria-hidden="true">＋</span>新建仪表盘并加入
            </button>
          </div>
        </div>
      </div>
    </template>
    <div v-else class="auth-loading">正在检查登录状态…</div>
    <AppDialogHost />
  </div>
</template>

<script setup>
import {
  nextTick,
  ref,
  computed,
  reactive,
  onBeforeUnmount,
  onMounted,
  defineAsyncComponent,
} from "vue";
import {
  Bot,
  Database,
  LoaderCircle,
  History,
  Paperclip,
  Send,
  UserRound,
} from "@lucide/vue";
import { apiErrorMessage, responseError } from "./utils/http.js";
import { useDataContextStore } from "./stores/dataContext.js";
import AppDialogHost from "./components/AppDialogHost.vue";
import IntentResponseCard from "./components/IntentResponseCard.vue";
import { confirmDialog, promptDialog } from "./composables/appDialog.js";

const ChartView = defineAsyncComponent(() => import("./components/ChartView.vue"));
const AppNavigation = defineAsyncComponent(() => import("./components/AppNavigation.vue"));
const AuthView = defineAsyncComponent(() => import("./components/AuthView.vue"));
const ResultWorkbench = defineAsyncComponent(() => import("./components/ResultWorkbench.vue"));
const DashboardView = defineAsyncComponent(() => import("./components/DashboardView.vue"));
const MetaKnowledgeView = defineAsyncComponent(() => import("./components/MetaKnowledgeView.vue"));
const AdhocView = defineAsyncComponent(() => import("./components/AdhocView.vue"));
const MaskingRuleView = defineAsyncComponent(() => import("./components/MaskingRuleView.vue"));
const WorkspaceView = defineAsyncComponent(() => import("./components/WorkspaceView.vue"));
const AccessPolicyView = defineAsyncComponent(() => import("./components/AccessPolicyView.vue"));
const EvaluationView = defineAsyncComponent(() => import("./components/EvaluationView.vue"));
const QualityOperationsView = defineAsyncComponent(() => import("./components/QualityOperationsView.vue"));
const ScenarioMatrixView = defineAsyncComponent(() => import("./components/ScenarioMatrixView.vue"));
const ResilienceView = defineAsyncComponent(() => import("./components/ResilienceView.vue"));
const AdhocIsolationEvaluationView = defineAsyncComponent(() => import("./components/AdhocIsolationEvaluationView.vue"));
const RAGEvaluationView = defineAsyncComponent(() => import("./components/RAGEvaluationView.vue"));
const PlanningEvaluationView = defineAsyncComponent(() => import("./components/PlanningEvaluationView.vue"));
const SemanticRepairEvaluationView = defineAsyncComponent(() => import("./components/SemanticRepairEvaluationView.vue"));
const ConversationMemoryPanel = defineAsyncComponent(() => import("./components/ConversationMemoryPanel.vue"));
const DataContextDrawer = defineAsyncComponent(() => import("./components/DataContextDrawer.vue"));
const ConfirmContextSwitchDialog = defineAsyncComponent(() => import("./components/ConfirmContextSwitchDialog.vue"));

const nativeFetch = window.__daNativeFetch || window.fetch.bind(window);
window.__daNativeFetch = nativeFetch;
if (!window.__daAuthFetchWrapped) {
  window.fetch = async (input, init = {}) => {
    const headers = new Headers(init.headers || {});
    const token = localStorage.getItem("da_access_token");
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await nativeFetch(input, { ...init, headers });
    const url = typeof input === "string" ? input : input?.url || "";
    const isCredentialRequest = /\/api\/auth\/(login|register)$/.test(url);
    if (response.status === 401 && !isCredentialRequest) {
      window.dispatchEvent(new CustomEvent("da:unauthorized"));
    }
    return response;
  };
  window.__daAuthFetchWrapped = true;
}

const authReady = ref(false);
const currentUser = ref(null);
let identityGeneration = 0;
let conversationGeneration = 0;
let queryAbortController = null;
let directoryAbortController = null;
let transcriptAbortController = null;
const canEditDashboard = computed(() => ["owner", "admin", "analyst"].includes(currentUser.value?.workspace?.role));
const dataContextStore = useDataContextStore();
let dataContextPersistedChatId = "";

function normalizedSession(data) {
  return {
    user: data?.user || null,
    workspace: data?.user?.workspace || data?.workspace || null,
  };
}

function sessionIdentity(value = currentUser.value) {
  const userId = value?.user?.id;
  const workspaceId = value?.workspace?.id;
  return userId && workspaceId ? `${userId}:${workspaceId}` : "";
}

async function handleAuthenticated(data) {
  const previousIdentity = sessionIdentity();
  const nextSession = normalizedSession(data);
  const nextIdentity = sessionIdentity(nextSession);
  currentUser.value = nextSession;
  dataContextStore.setIdentity(nextIdentity);
  authReady.value = true;
  mode.value = "chat";
  if (!nextIdentity) return;
  if (previousIdentity !== nextIdentity) {
    identityGeneration += 1;
    resetConversationState({ clearDirectory: true });
    await loadConversations({ reset: true, autoRestore: true });
  } else {
    await loadConversations({ reset: true });
  }
}

function handleUnauthorized() {
  if (!currentUser.value && !localStorage.getItem("da_access_token")) return;
  localStorage.removeItem("da_access_token");
  dataContextStore.setIdentity("");
  identityGeneration += 1;
  resetConversationState({ clearDirectory: true });
  currentUser.value = null;
  authReady.value = true;
  mode.value = "chat";
}

function logout() {
  localStorage.removeItem("da_access_token");
  dataContextStore.setIdentity("");
  identityGeneration += 1;
  resetConversationState({ clearDirectory: true });
  currentUser.value = null;
  authReady.value = true;
  mode.value = "chat";
}

async function loadAuth() {
  if (!localStorage.getItem("da_access_token")) { authReady.value = true; return; }
  try {
    const response = await fetch("/api/auth/me");
    if (!response.ok) throw new Error("expired");
    await handleAuthenticated(await response.json());
  } catch {
    handleUnauthorized();
  } finally {
    authReady.value = true;
  }
}

onMounted(() => {
  window.addEventListener("da:unauthorized", handleUnauthorized);
  loadAuth();
});

onBeforeUnmount(() => {
  window.removeEventListener("da:unauthorized", handleUnauthorized);
  queryAbortController?.abort();
  directoryAbortController?.abort();
  transcriptAbortController?.abort();
});

const API_URL = "/api/query";
const PREDICT_URL = "/api/predict";
const ANOMALY_URL = "/api/anomaly";

const question = ref("");
const loading = ref(false);
const messages = ref([]);
const messagesEl = ref(null);
const textareaEl = ref(null);
// 当前会话 chat_id：首轮由后端首个 SSE chunk 下发，后续轮次需回传以走历史继承逻辑
const chatId = ref(null);
// 最近一次查询结果行：data.chart 到达时与之配对渲染图表（chart 在 result 之后下发）
const lastResult = ref([]);
// 最近一次提问与 SQL：供"数据预测/异常检测"按需端点回传上下文
const lastQuery = ref("");
const lastSql = ref("");
const STREAM_QUERY_ARTIFACT_TYPES = new Set([
  "sql",
  "result",
  "analysis",
  "recommend",
]);

function clearCurrentStreamArtifacts(stepIndex) {
  for (let index = messages.value.length - 1; index > stepIndex; index -= 1) {
    const message = messages.value[index];
    if (
      message?._streamStepIndex === stepIndex
      && STREAM_QUERY_ARTIFACT_TYPES.has(message.type)
    ) {
      messages.value.splice(index, 1);
    }
  }
  // handled 回合没有可复用的查询制品，避免后续操作误用早到的 SQL/结果。
  lastSql.value = "";
  lastResult.value = [];
}
const conversations = ref([]);
const conversationListStatus = ref("active");
const conversationListLoading = ref(false);
const conversationListError = ref("");
const conversationListCursor = ref(null);
const conversationListHasMore = ref(false);
const conversationRetentionDays = ref(30);
const activeConversation = ref(null);
const historyHydrating = ref(false);
const transcriptCursor = ref(null);
const transcriptHasMore = ref(false);
const transcriptLoadingMore = ref(false);
const conversationReadOnly = computed(() => activeConversation.value?.status === "archived");

// 顶栏模式：对话 / 仪表盘
const mode = ref("chat");
const analysisEnabled = ref(localStorage.getItem("da_analysis_enabled") === "true");
const memoryPanelOpen = ref(false);
function saveAnalysisPreference() {
  localStorage.setItem("da_analysis_enabled", analysisEnabled.value ? "true" : "false");
}
// 跳转到仪表盘模式时指定要加载的仪表盘ID（"加入仪表盘"成功后跳转用）
const pendingDashboardId = ref("");

// 加入仪表盘弹层
const addToDash = reactive({
  open: false,
  dashboards: [],   // 仅 dashboard 节点
  payload: null,    // 待加入的 {chart, sql, query}
  error: "",
});
const adhocSwitch = reactive({ open: false, dataset: null });
const appNotice = reactive({ text: "", tone: "success", timer: null });
const dashboardModalEl = ref(null);
let dashboardModalTrigger = null;
let adhocSwitchTrigger = null;

// 欢迎屏快捷建议（点击即发送）
const suggestions = [
  { glyph: "Σ", text: "统计各地区的销售总额" },
  { glyph: "%", text: "用饼图展示各品类销量占比" },
  { glyph: "↗", text: "用折线图展示月度销售额趋势" },
  { glyph: "10", text: "销量前十的商品有哪些" },
];

function showAppNotice(text, tone = "success", duration = 2600) {
  if (appNotice.timer) window.clearTimeout(appNotice.timer);
  appNotice.text = text;
  appNotice.tone = tone;
  appNotice.timer = window.setTimeout(() => {
    appNotice.text = "";
    appNotice.timer = null;
  }, duration);
}

function setMessageStatus(msg, text, tone = "success") {
  msg._actionNotice = tone === "success" ? text : "";
  msg._actionError = tone === "error" ? text : "";
  window.setTimeout(() => {
    if (tone === "success" && msg._actionNotice === text) msg._actionNotice = "";
  }, 2800);
}

function completedStepCount(steps = []) {
  return steps.filter((step) => ["success", "done"].includes(step.status)).length;
}

function formatDuration(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds)) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(milliseconds >= 10000 ? 1 : 2)} s`;
}

function formatComparisonValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 4,
  }).format(number);
}

function comparisonSourceLabel(source) {
  const key = String(source || "").toLowerCase();
  if (key === "dw" || key === "official" || key === "official_dw") return "正式 DW";
  if (key === "upload" || key === "uploaded") return "上传文件";
  return String(source || "未知来源");
}

function comparisonDimensionText(dimensions) {
  const entries = Object.entries(dimensions || {});
  if (!entries.length) return "全部";
  return entries
    .map(([key, value]) => `${key}：${value ?? "空值"}`)
    .join(" · ");
}

function latencyTone(value) {
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds)) return "";
  if (milliseconds >= 30000) return "bad";
  if (milliseconds >= 15000) return "warn";
  return "good";
}

function nodeLabel(nodeId) {
  return ({
    resolve_data_source_context: "解析数据源上下文",
    profile_upload_schema: "剖析上传 Schema",
    propose_source_mapping: "核对关联映射",
    extract_keywords: "理解问题",
    rewrite_follow_up: "补全追问",
    recall_column: "召回字段",
    recall_metric: "召回指标",
    recall_training: "召回 SQL 样例",
    recall_value: "召回字段值",
    merge_retrieved_info: "合并上下文",
    build_semantic_ir: "构建语义约束",
    filter_metric: "筛选指标",
    filter_table: "筛选数据表",
    add_extra_context: "补充元数据",
    decide_plan_route: "评估规划必要性",
    plan_sql: "规划查询",
    generate_sql: "生成 SQL",
    validate_sql: "校验 SQL",
    correct_sql: "修正 SQL",
    execute_sql: "执行查询",
    generate_chart: "生成图表",
    generate_analysis: "生成解读",
    generate_recommend: "生成追问建议",
  })[nodeId] || "执行节点";
}

function profileLabel(value) {
  return ({
    never_plan: "从不规划",
    keyword_length_v1: "关键词 + 长度",
    always_plan: "始终规划",
    structured_rules_v1: "结构化规则",
    weighted_scorecard_v1: "加权评分卡",
    calibrated_linear_v1: "校准线性候选",
  })[value] || value || "未知策略";
}

function planningActionLabel(value) {
  return ({
    plan_then_generate: "先规划再生成",
    direct_generate: "直接生成",
    deterministic_ir_bypass: "语义 IR 直达",
    abstain: "灰区回退",
  })[value] || value || "未知决策";
}

function planningReasonLabel(value) {
  const reason = String(value || "");
  const exact = {
    deterministic_ir_supported: "语义 IR 已完整支持",
    legacy_complex_keyword: "命中既有复杂关键词",
    legacy_long_query: "问题长度超过既有阈值",
    legacy_no_trigger: "既有规则未触发",
    abstain_fallback: "灰区使用安全基线",
    structured_no_trigger: "结构规则未触发",
  };
  if (exact[reason]) return exact[reason];
  return reason
    .replace(/^score:/, "评分：")
    .replace(/^linear:/, "线性特征：")
    .replace(/^structure:/, "结构：")
    .replaceAll("_", " ");
}

function semanticProfileLabel(value) {
  return ({
    inline_v1: "P0 内联基线",
    violation_only_v1: "P1 只诊断",
    registry_same_rules_v1: "P2 同规则注册表",
    registry_extended_v1: "P3 扩展注册表",
    targeted_correction_v1: "P4 定向纠错",
    upstream_first_v1: "P5 上游再生成",
    cascade_v1: "P6 分层级联",
  })[value] || value || "语义修复注册表";
}

function semanticRuleLabel(value) {
  return ({
    "join.unique_metadata_edge": "唯一连接边",
    "projection.threshold_measure": "阈值投影",
    "ir.unique_limit_order": "排序 / Limit",
    "ir.unique_group_by": "分组影子",
  })[value] || value;
}

function repairabilityLabel(value) {
  return ({
    deterministic: "确定性修复",
    llm: "有界模型纠错",
    upstream: "上游再生成",
    reject: "拒绝自动修复",
    none: "无需修复",
  })[value] || value || "待分类";
}

function semanticRepairStatus(repair, violation) {
  if (repair?.fallback_used) return "P0 安全回退";
  if (repair?.changed) return `已应用 ${repair.applied_rules?.length || 0} 条规则`;
  if (violation?.repairability === "reject") return "已拒绝风险请求";
  if (violation) return "已生成分层诊断";
  return "校验通过";
}

function sourceModeLabel(value) {
  return ({
    dw_only: "正式 DW",
    upload_only: "仅上传表",
    upload_compare_dw: "独立对比",
    upload_join_dw: "受控关联",
    multi_upload: "双上传表",
  })[value] || value || "正式 DW";
}

function openDataContext() {
  dataContextStore.state.drawerOpen = true;
  dataContextStore.loadCatalog();
}

function formatDecisionMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(number < 1 ? 3 : 1)} ms` : "—";
}

function lastChatStorageKey() {
  const identity = sessionIdentity();
  return identity ? `da_last_chat:${identity}` : "";
}

function rememberCurrentChat(value) {
  const key = lastChatStorageKey();
  if (!key) return;
  if (value) localStorage.setItem(key, String(value));
  else localStorage.removeItem(key);
}

function resetConversationState({ clearDirectory = false } = {}) {
  conversationGeneration += 1;
  queryAbortController?.abort();
  directoryAbortController?.abort();
  transcriptAbortController?.abort();
  queryAbortController = null;
  directoryAbortController = null;
  transcriptAbortController = null;
  loading.value = false;
  historyHydrating.value = false;
  transcriptLoadingMore.value = false;
  chatId.value = null;
  activeConversation.value = null;
  messages.value = [];
  lastResult.value = [];
  lastQuery.value = "";
  lastSql.value = "";
  dataContextPersistedChatId = "";
  dataContextStore.resetLocal();
  memoryPanelOpen.value = false;
  transcriptCursor.value = null;
  transcriptHasMore.value = false;
  if (clearDirectory) {
    conversations.value = [];
    conversationListStatus.value = "active";
    conversationListLoading.value = false;
    conversationListError.value = "";
    conversationListCursor.value = null;
    conversationListHasMore.value = false;
  }
}

function historyTurnStatusLabel(status) {
  return ({
    succeeded: "历史结果快照",
    failed: "本轮未完成",
    cancelled: "本轮已取消",
    pending: "本轮仍在处理中",
  })[status] || "历史回合";
}

function historyPartLabel(part) {
  return ({
    sql: "SQL",
    result: "结果快照",
    chart: "图表",
    analysis: "AI 解读",
    recommendation: "追问建议",
  })[part] || part;
}

function messagesFromHistoryTurn(turn) {
  const turnId = turn?.turn_id || crypto.randomUUID();
  const items = [{
    role: "user",
    type: "text",
    content: turn?.user_text || "（历史问题不可用）",
    _turnId: turnId,
    _historical: true,
  }];
  if (turn?.sql) {
    items.push({
      role: "assistant",
      type: "sql",
      content: turn.sql,
      _editing: false,
      _editText: "",
      _running: false,
      _historical: true,
      _turnId: turnId,
      chart: turn?.chart || null,
      _datasetId: null,
      _compareDw: false,
    });
  }
  if (turn?.result) {
    const rows = Array.isArray(turn.result.rows) ? turn.result.rows : [];
    items.push({
      role: "assistant",
      type: "result",
      columns: Array.isArray(turn.result.columns) ? turn.result.columns : Object.keys(rows[0] || {}),
      rows,
      chart: turn?.chart || null,
      _query: turn?.user_text || "",
      _sql: turn?.sql || "",
      _historical: true,
      _turnId: turnId,
      _resultTruncated: Boolean(turn.result.truncated),
      _rowCount: turn.result.row_count,
      _predicting: false,
      _anomaly: false,
      _actionNotice: "",
      _actionError: "",
    });
  }
  items.push({
    role: "assistant",
    type: "history-status",
    status: turn?.status || "failed",
    content: turn?.assistant_summary || "该回合仅保留了可安全恢复的内容。",
    unavailable: Array.isArray(turn?.unavailable_parts) ? turn.unavailable_parts : [],
    fidelity: turn?.replay_fidelity,
    sourceContext: turn?.data_source_context || null,
    _turnId: turnId,
    _historical: true,
  });
  return items;
}

function transcriptMessages(turns) {
  return (Array.isArray(turns) ? turns : []).flatMap(messagesFromHistoryTurn);
}

function restoreDerivedQueryState() {
  const userMessage = [...messages.value].reverse().find((item) => item.role === "user");
  const sqlMessage = [...messages.value].reverse().find((item) => item.type === "sql");
  const resultMessage = [...messages.value].reverse().find((item) => item.type === "result");
  lastQuery.value = userMessage?.content || "";
  lastSql.value = sqlMessage?.content || "";
  lastResult.value = resultMessage?.rows || [];
}

async function loadConversations({ reset = true, autoRestore = false } = {}) {
  if (!sessionIdentity() || conversationListLoading.value && !reset) return;
  const requestIdentity = identityGeneration;
  const requestedStatus = conversationListStatus.value;
  if (reset) {
    directoryAbortController?.abort();
    directoryAbortController = new AbortController();
    conversationListCursor.value = null;
    conversationListHasMore.value = false;
  } else if (!conversationListCursor.value) {
    return;
  }
  const controller = directoryAbortController || new AbortController();
  directoryAbortController = controller;
  conversationListLoading.value = true;
  conversationListError.value = "";
  try {
    const params = new URLSearchParams({ status: requestedStatus, limit: "20" });
    if (!reset && conversationListCursor.value) params.set("cursor", conversationListCursor.value);
    const response = await fetch(`/api/chats?${params}`, { signal: controller.signal });
    if (!response.ok) throw new Error(await responseError(response, "会话历史加载失败"));
    const data = await response.json();
    if (requestIdentity !== identityGeneration || requestedStatus !== conversationListStatus.value) return;
    const incoming = Array.isArray(data.items) ? data.items : [];
    if (reset) {
      conversations.value = incoming;
    } else {
      const seen = new Set(conversations.value.map((item) => item.chat_id));
      conversations.value = [...conversations.value, ...incoming.filter((item) => !seen.has(item.chat_id))];
    }
    const refreshedActive = incoming.find((item) => item.chat_id === chatId.value);
    if (refreshedActive) activeConversation.value = refreshedActive;
    conversationListCursor.value = data.next_cursor || null;
    conversationListHasMore.value = Boolean(data.has_more && data.next_cursor);
    conversationRetentionDays.value = Number(data.retention_days || 30);
    if (autoRestore && requestedStatus === "active" && incoming.length) {
      const pointer = localStorage.getItem(lastChatStorageKey());
      const target = incoming.find((item) => item.chat_id === pointer) || incoming[0];
      await openConversation(target, { preserveDirectoryRequest: true });
    }
  } catch (error) {
    if (error?.name !== "AbortError" && requestIdentity === identityGeneration) {
      conversationListError.value = error?.message || "会话历史加载失败";
    }
  } finally {
    if (requestIdentity === identityGeneration && controller === directoryAbortController) {
      conversationListLoading.value = false;
    }
  }
}

async function openConversation(item, { preserveDirectoryRequest = false } = {}) {
  const target = item?.chat_id ? item : { chat_id: String(item || "") };
  if (!target.chat_id || !sessionIdentity()) return;
  queryAbortController?.abort();
  transcriptAbortController?.abort();
  conversationGeneration += 1;
  const requestConversation = conversationGeneration;
  const requestIdentity = identityGeneration;
  const controller = new AbortController();
  transcriptAbortController = controller;
  loading.value = false;
  historyHydrating.value = true;
  transcriptLoadingMore.value = false;
  activeConversation.value = target;
  chatId.value = target.chat_id;
  messages.value = [];
  lastResult.value = [];
  lastQuery.value = "";
  lastSql.value = "";
  dataContextStore.resetLocal();
  dataContextPersistedChatId = "";
  memoryPanelOpen.value = false;
  mode.value = "chat";
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(target.chat_id)}/turns?limit=30`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response, "会话恢复失败"));
    const data = await response.json();
    if (requestIdentity !== identityGeneration || requestConversation !== conversationGeneration) return;
    await dataContextStore.hydrate(target.chat_id);
    dataContextPersistedChatId = target.chat_id;
    activeConversation.value = data.conversation || target;
    chatId.value = target.chat_id;
    messages.value = transcriptMessages(data.items);
    transcriptCursor.value = data.next_cursor || null;
    transcriptHasMore.value = Boolean(data.has_more && data.next_cursor);
    conversationRetentionDays.value = Number(data.retention_days || conversationRetentionDays.value);
    rememberCurrentChat(target.chat_id);
    restoreDerivedQueryState();
    await nextTick();
    scrollToBottom();
  } catch (error) {
    if (error?.name !== "AbortError" && requestIdentity === identityGeneration && requestConversation === conversationGeneration) {
      activeConversation.value = null;
      chatId.value = null;
      messages.value = [];
      showAppNotice(error?.message || "会话恢复失败", "error", 4200);
      if (!preserveDirectoryRequest) await loadConversations({ reset: true });
    }
  } finally {
    if (requestIdentity === identityGeneration && requestConversation === conversationGeneration) {
      historyHydrating.value = false;
    }
  }
}

async function loadOlderTurns() {
  if (!chatId.value || !transcriptCursor.value || transcriptLoadingMore.value) return;
  const requestIdentity = identityGeneration;
  const requestConversation = conversationGeneration;
  const targetChat = chatId.value;
  transcriptLoadingMore.value = true;
  try {
    const params = new URLSearchParams({ limit: "30", before: transcriptCursor.value });
    const response = await fetch(`/api/chats/${encodeURIComponent(targetChat)}/turns?${params}`);
    if (!response.ok) throw new Error(await responseError(response, "更早记录加载失败"));
    const data = await response.json();
    if (
      requestIdentity !== identityGeneration
      || requestConversation !== conversationGeneration
      || targetChat !== chatId.value
    ) return;
    const existingTurns = new Set(messages.value.map((item) => item._turnId).filter(Boolean));
    const older = transcriptMessages(data.items).filter((item) => !existingTurns.has(item._turnId));
    const container = messagesEl.value;
    const previousHeight = container?.scrollHeight || 0;
    const previousTop = container?.scrollTop || 0;
    messages.value = [...older, ...messages.value];
    transcriptCursor.value = data.next_cursor || null;
    transcriptHasMore.value = Boolean(data.has_more && data.next_cursor);
    await nextTick();
    if (container) container.scrollTop = previousTop + (container.scrollHeight - previousHeight);
  } catch (error) {
    if (error?.name !== "AbortError") showAppNotice(error?.message || "更早记录加载失败", "error");
  } finally {
    if (requestIdentity === identityGeneration && requestConversation === conversationGeneration) {
      transcriptLoadingMore.value = false;
    }
  }
}

async function changeConversationListStatus(status) {
  if (!["active", "archived"].includes(status) || status === conversationListStatus.value) return;
  conversationListStatus.value = status;
  await loadConversations({ reset: true });
}

async function setConversationStatus(item, status) {
  if (!item?.chat_id || !["active", "archived"].includes(status)) return;
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(item.chat_id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!response.ok) throw new Error(await responseError(response, status === "active" ? "恢复失败" : "归档失败"));
    const updated = await response.json();
    if (chatId.value === item.chat_id) {
      activeConversation.value = updated;
      if (status === "active") rememberCurrentChat(item.chat_id);
    }
    await loadConversations({ reset: true });
    showAppNotice(status === "active" ? "会话已恢复，可继续提问" : "会话已归档，历史仍可只读查看");
  } catch (error) {
    showAppNotice(error?.message || "会话状态更新失败", "error");
  }
}

async function deleteConversation(item) {
  if (!item?.chat_id) return;
  const confirmed = await confirmDialog({
    title: "删除历史会话？",
    message: `会话“${item.title || "未命名会话"}”将从历史列表移除，此操作不可撤销。`,
    confirmLabel: "删除会话",
    tone: "danger",
  });
  if (!confirmed) return;
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(item.chat_id)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await responseError(response, "删除会话失败"));
    if (chatId.value === item.chat_id) {
      rememberCurrentChat(null);
      resetConversationState();
    }
    await loadConversations({ reset: true });
    showAppNotice("会话已删除");
  } catch (error) {
    showAppNotice(error?.message || "删除会话失败", "error");
  }
}

function handleMemoryReset() {
  memoryPanelOpen.value = false;
  if (activeConversation.value) {
    activeConversation.value = {
      ...activeConversation.value,
      context_reset_at: new Date().toISOString(),
    };
  }
  showAppNotice("当前上下文已重置；可见历史仍完整保留，下一问将从新上下文开始。", "success", 4200);
}

// 开始全新对话：清空消息与 chat_id
function newConversation() {
  rememberCurrentChat(null);
  resetConversationState();
  nextTick(() => textareaEl.value && textareaEl.value.focus());
}

function applyAdhocDataset(dataset) {
  rememberCurrentChat(null);
  resetConversationState();
  dataContextStore.selectOnly(dataset);
  dataContextStore.applyLocal();
  mode.value = "chat";
  dataContextStore.state.drawerOpen = true;
}

function useAdhocDataset(dataset) {
  if (!dataset) return;
  if (!messages.value.length) {
    applyAdhocDataset(dataset);
    return;
  }
  adhocSwitchTrigger = document.activeElement;
  adhocSwitch.dataset = dataset;
  adhocSwitch.open = true;
}

function closeAdhocSwitchModal() {
  adhocSwitch.open = false;
  adhocSwitch.dataset = null;
  nextTick(() => adhocSwitchTrigger?.focus?.());
}

function confirmAdhocSwitch() {
  const dataset = adhocSwitch.dataset;
  adhocSwitch.open = false;
  adhocSwitch.dataset = null;
  if (dataset) applyAdhocDataset(dataset);
}

async function uploadAdhocFromChat(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  try {
    const data = await dataContextStore.upload(file, "managed");
    messages.value.push({
      role: "assistant",
      type: "text",
      content: `已创建“${data.filename}”版本 v${data.version_no}，共 ${data.row_count} 行、${data.column_count} 列。当前默认仅查询上传表；可在“数据上下文”中切换独立对比或受控关联。`,
    });
    dataContextStore.state.drawerOpen = true;
  } catch (error) {
    messages.value.push({ role: "assistant", type: "error", content: error?.message || "文件上传失败" });
  }
}

function scrollToBottom() {
  const el = messagesEl.value;
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

// 步骤卡片：整体状态与标题
function thinkingState(message) {
  const steps = message?.steps || [];
  const last = steps.at(-1);
  if (last && last.status === "error") return "error";
  if (last && last.status === "running") return "running";
  if (message?._pending) return "running";
  return "done";
}
function thinkingLabel(message) {
  const state = thinkingState(message);
  if (state === "error") return "处理出错";
  if (state === "running") {
    return message?.steps?.length ? "正在思考…" : "正在准备…";
  }
  if (message?.intentResponse?.response_kind === "refusal") return "已安全拒绝";
  if (message?.intentResponse?.response_kind === "clarification") return "等待补充信息";
  if (message?.intentResponse) return "已引导回业务问数";
  return "处理完成";
}

function chartTypeLabel(t) {
  return { column: "柱状图", bar: "条形图", line: "折线图", pie: "饼图", table: "表格" }[t] || "图表";
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showAppNotice("SQL 已复制到剪贴板");
  } catch {
    showAppNotice("浏览器未允许访问剪贴板，请手动选择 SQL 复制。", "error", 4200);
  }
}

async function exportResult(msg) {
  if (msg._exporting) return;
  msg._exporting = true;
  try {
    const response = await fetch("/api/query/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        columns: msg.columns || [],
        rows: msg.rows || [],
        filename: `data-agent-result-${new Date().toISOString().slice(0, 10)}`,
      }),
    });
    if (!response.ok) {
      throw new Error(await responseError(response, "导出失败"));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `data-agent-result-${new Date().toISOString().slice(0, 10)}.xlsx`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setMessageStatus(msg, "Excel 已开始下载");
  } catch (error) {
    setMessageStatus(msg, error?.message || "导出失败", "error");
  } finally {
    msg._exporting = false;
  }
}

// ===== 手动 SQL 编辑 + 重跑（TODO-22）=====
function startEditSql(msg) {
  msg._editText = msg.content;
  msg._editing = true;
}
function cancelEditSql(msg) {
  msg._editing = false;
  msg._editText = "";
}
// 运行编辑后的 SQL：调 /api/query/exec（后端复用只读校验 + LIMIT 强制），
// 成功后追加新的表格 + 图表（复用原 chart 配置）消息
async function runEditedSql(msg) {
  const sql = String(msg._editText || "").trim();
  if (!sql) return;
  msg._running = true;
  try {
    const res = await fetch("/api/query/exec", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sql,
        ...(msg._dataContext || {}),
      }),
    });
    if (!res.ok) {
      messages.value.push({
        role: "assistant",
        type: "error",
        content: await responseError(res, "运行失败"),
      });
      return;
    }
    const data = await res.json();
    const rows = data.rows || [];
    const columns = data.columns || (rows.length ? Object.keys(rows[0]) : []);
    // 一个执行结果只生成一个工作台，表格和图表在工作台内切换。
    messages.value.push({
      role: "assistant",
      type: "result",
      columns,
      rows,
      chart: msg.chart || { type: "table" },
      _query: msg._query || lastQuery.value,
      _sql: sql,
      _predicting: false,
      _anomaly: false,
      _actionNotice: "",
      _actionError: "",
    });
    // 用编辑后的 SQL 更新本卡展示，并退出编辑态
    msg.content = sql;
    msg._editing = false;
    msg._editText = "";
    lastSql.value = sql;
    lastResult.value = rows;
  } catch (e) {
    messages.value.push({
      role: "assistant",
      type: "error",
      content: e?.message || "运行请求失败",
    });
  } finally {
    msg._running = false;
    await nextTick();
    scrollToBottom();
  }
}

// 提交反馈（TODO-23）：把 👍/👎 + 评论 + 本轮 query/sql 发后端入库
async function submitFeedback(msg) {
  if (msg._fbScore === null || msg._fbSending) return;
  msg._fbSending = true;
  msg._fbError = "";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId.value || null,
        query: msg._query || lastQuery.value,
        sql: msg._sql || lastSql.value || null,
        score: msg._fbScore,
        comment: msg._fbComment || null,
      }),
    });
    if (!response.ok) throw new Error(await responseError(response, "反馈提交失败"));
    msg._fbSubmitted = true;
  } catch (error) {
    msg._fbError = error?.message || "反馈提交失败，请稍后重试";
  } finally {
    msg._fbSending = false;
  }
}

async function sendQuestion() {
  if (loading.value || historyHydrating.value || conversationReadOnly.value) return;
  const q = String(question.value || "").trim();
  if (!q) return;
  const requestIdentity = identityGeneration;
  const requestConversation = conversationGeneration;
  queryAbortController?.abort();
  const controller = new AbortController();
  queryAbortController = controller;
  question.value = "";
  // 输入框高度复位
  if (textareaEl.value) textareaEl.value.style.height = "auto";
  loading.value = true;
  lastQuery.value = q;

  messages.value.push({ role: "user", type: "text", content: q });

  const stepIndex =
    messages.value.push({
      role: "assistant",
      type: "steps",
      steps: [],
      _pending: true,
      trace: null,
      followUp: null,
      memoryContext: null,
      memoryWrite: null,
      sourceContext: null,
      sourceComparison: null,
      planningRoute: null,
      planningTiming: null,
      semanticRepair: null,
      semanticViolation: null,
      clarification: null,
      intentResponse: null,
    }) - 1;

  await nextTick();
  scrollToBottom();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      // 携带当前会话 chat_id（首轮为 null，后端会生成并下发；非首轮带上以走历史继承）
      body: JSON.stringify({
        query: q,
        enable_analysis: analysisEnabled.value,
        ...(chatId.value ? { chat_id: chatId.value } : {}),
        ...dataContextStore.queryPayload(),
      }),
    });

    if (!response.ok) {
      throw new Error(await responseError(response, "查询请求失败"));
    }
    if (!response.body) throw new Error("服务器未返回流");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let streamComplete = false;
    let intentHandled = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const evt of events) {
        if (
          requestIdentity !== identityGeneration
          || requestConversation !== conversationGeneration
          || controller !== queryAbortController
          || !messages.value[stepIndex]
        ) return;
        const line = evt.trim();
        if (!line.startsWith("data:")) continue;

        let data;
        try {
          data = JSON.parse(line.replace(/^data:\s*/, ""));
        } catch {
          continue;
        }
        dataContextStore.handleStreamEvent(data);

        const steps = messages.value[stepIndex].steps;

        // 1) 首个 chunk：后端下发 chat_id（新会话时生成，已有会话时回传）
        if (data.done) {
          streamComplete = true;
          messages.value[stepIndex]._pending = false;
          const last = steps.at(-1);
          if (last && last.status === "running") last.status = "success";
        } else if (data.chat_id) {
          if (chatId.value && chatId.value !== data.chat_id) {
            throw new Error("服务端返回了不一致的会话标识，请刷新后重试");
          }
          chatId.value = data.chat_id;
          rememberCurrentChat(data.chat_id);
          activeConversation.value = {
            ...(activeConversation.value || {}),
            chat_id: data.chat_id,
            title: activeConversation.value?.title || q.slice(0, 80),
            preview: q.slice(0, 160),
            status: "active",
            last_turn_status: "pending",
            turn_count: Number(activeConversation.value?.turn_count || 0) + 1,
          };
        } else if (data.memory_context) {
          messages.value[stepIndex].memoryContext = data.memory_context;
        } else if (data.memory_write) {
          messages.value[stepIndex].memoryWrite = data.memory_write;
        } else if (data.source_context_resolved) {
          messages.value[stepIndex].sourceContext = data.source_context_resolved;
        } else if (data.source_comparison) {
          messages.value[stepIndex].sourceComparison = data.source_comparison;
        } else if (data.source_context_invalidated) {
          messages.value.push({
            role: "assistant",
            type: "error",
            content: data.source_context_invalidated.message || "数据上下文已失效并重置",
          });
        } else if (data.stage) {
          const last = steps.at(-1);
          if (last && last.status === "running") last.status = "success";
          steps.push({ text: data.stage, status: "running" });
        } else if (data.intent_response?.decision === "respond") {
          intentHandled = true;
          clearCurrentStreamArtifacts(stepIndex);
          messages.value[stepIndex].intentResponse = data.intent_response;
          const last = steps.at(-1);
          if (last && last.status === "running") last.status = "success";
          messages.value.push({
            role: "assistant",
            type: "intent-response",
            response: data.intent_response,
          });
        } else if (data.follow_up) {
          messages.value[stepIndex].followUp = data.follow_up;
        } else if (data.candidate_decision) {
          messages.value[stepIndex].candidateDecision = data.candidate_decision;
        } else if (data.planning_route) {
          messages.value[stepIndex].planningRoute = data.planning_route;
        } else if (data.planning_timing) {
          messages.value[stepIndex].planningTiming = data.planning_timing;
        } else if (data.semantic_repair) {
          messages.value[stepIndex].semanticRepair = data.semantic_repair;
        } else if (data.semantic_violation) {
          messages.value[stepIndex].semanticViolation = data.semantic_violation;
        } else if (data.clarification) {
          messages.value[stepIndex].clarification = data.clarification;
        } else if (intentHandled && ["sql", "result", "chart", "analysis", "recommend"]
          .some((key) => Object.prototype.hasOwnProperty.call(data, key))) {
          // 后端 Application 已执行同一终态隔离；浏览器继续关闭式防御，避免
          // 异常流在引导/拒绝卡之后渲染任何查询产物。
          continue;
        } else if (data.sql) {
          // execute_sql 节点下发的最终 SQL，单独展示便于核对多轮上下文
          lastSql.value = data.sql;
          messages.value.push({
            role: "assistant",
            type: "sql",
            content: data.sql,
            _streamStepIndex: stepIndex,
            // 编辑重跑相关状态（TODO-22）
            _editing: false,
            _editText: "",
            _running: false,
            chart: null,   // 由后续 data.chart chunk 回挂
            _dataContext: dataContextStore.queryPayload(),
          });
        } else if (data.trace) {
          messages.value[stepIndex].trace = data.trace;
          messages.value[stepIndex]._pending = false;
          const last = steps.at(-1);
          const terminalSuccess = ["success", "handled"].includes(data.trace.status);
          if (last && last.status === "running") {
            last.status = terminalSuccess ? "success" : "error";
          }
          // trace success 是后端查询协议的终态。部分开发代理会延迟关闭
          // text/event-stream 连接，主动结束读取可避免 UI 一直停在“执行中”。
          if (terminalSuccess) {
            streamComplete = true;
            if (
              chatId.value
              && dataContextPersistedChatId !== chatId.value
              && dataContextStore.state.applied.sourceMode !== "dw_only"
            ) {
              dataContextStore.apply(chatId.value)
                .then(() => { dataContextPersistedChatId = chatId.value; })
                .catch((error) => showAppNotice(error?.message || "会话数据上下文保存失败", "error"));
            }
          }
        } else if (data.error) {
          messages.value[stepIndex]._pending = false;
          const last = steps.at(-1);
          if (last) last.status = "error";
          else steps.push({ text: "请求失败", status: "error" });
          messages.value.push({
            role: "assistant",
            type: "error",
            content: apiErrorMessage(data.error, "查询失败"),
            clarification: messages.value[stepIndex].clarification || null,
          });
        } else if (Array.isArray(data.result)) {
          const last = steps.at(-1);
          if (last) last.status = "success";
          lastResult.value = data.result;
          messages.value.push({
            role: "assistant",
            type: "result",
            columns: Object.keys(data.result[0] || {}),
            rows: data.result,
            chart: null,
            _query: q,
            _sql: lastSql.value,
            _streamStepIndex: stepIndex,
            _predicting: false,
            _anomaly: false,
            _actionNotice: "",
            _actionError: "",
          });
        } else if (data.chart) {
          const resultMessage = [...messages.value]
            .reverse()
            .find((message) => (
              message.type === "result"
              && message._streamStepIndex === stepIndex
            ));
          if (resultMessage) {
            resultMessage.chart = data.chart;
          } else {
            messages.value.push({
              role: "assistant",
              type: "result",
              columns: Object.keys(lastResult.value[0] || {}),
              rows: lastResult.value,
              chart: data.chart,
              _query: q,
              _sql: lastSql.value,
              _streamStepIndex: stepIndex,
              _predicting: false,
              _anomaly: false,
              _actionNotice: "",
              _actionError: "",
            });
          }
          // 回挂 chart 到最近一条 sql 消息，供"编辑重跑"时复用图表配置（TODO-22）
          const lastSqlMsg = [...messages.value]
            .reverse()
            .find((message) => (
              message.type === "sql"
              && message._streamStepIndex === stepIndex
            ));
          if (lastSqlMsg) lastSqlMsg.chart = data.chart;
        } else if (data.analysis) {
          // generate_analysis 节点下发的自然语言解读
          const last = steps.at(-1);
          if (last && last.status === "running") last.status = "success";
          messages.value.push({
            role: "assistant",
            type: "analysis",
            content: data.analysis,
            // 反馈闭环（TODO-23）：携带本轮 query/sql 供提交反馈
            _query: q,
            _sql: lastSql.value,
            _streamStepIndex: stepIndex,
            _fbScore: null,      // null | 1 | -1
            _fbComment: "",
            _fbSubmitted: false,
            _fbSending: false,
            _fbError: "",
          });
        } else if (Array.isArray(data.recommendations) || Array.isArray(data.recommend)) {
          // v2 推荐项分离展示 label 与可独立执行的 query；仍兼容旧 string[]。
          const last = steps.at(-1);
          if (last && last.status === "running") last.status = "success";
          const recommendations = Array.isArray(data.recommendations)
            ? data.recommendations
            : data.recommend;
          if (recommendations.length) {
            messages.value.push({
              role: "assistant",
              type: "recommend",
              questions: recommendations,
              _streamStepIndex: stepIndex,
            });
          }
        }

        await nextTick();
        scrollToBottom();
      }
      if (streamComplete) {
        // 不等待代理层完成 TCP 取消握手；终态已经收到，UI 应立即进入可交互态。
        reader.cancel().catch(() => {});
        break;
      }
    }
  } catch (e) {
    if (
      e?.name !== "AbortError"
      && requestIdentity === identityGeneration
      && requestConversation === conversationGeneration
    ) {
      if (messages.value[stepIndex]) {
        messages.value[stepIndex]._pending = false;
        const last = messages.value[stepIndex].steps.at(-1);
        if (last) last.status = "error";
        else messages.value[stepIndex].steps.push({ text: "请求失败", status: "error" });
      }
      messages.value.push({
        role: "assistant",
        type: "error",
        content: e?.message || "请求失败",
      });
    }
  } finally {
    if (
      requestIdentity === identityGeneration
      && requestConversation === conversationGeneration
      && controller === queryAbortController
    ) {
      if (messages.value[stepIndex]) {
        messages.value[stepIndex]._pending = false;
      }
      queryAbortController = null;
      loading.value = false;
      await loadConversations({ reset: true });
      await nextTick();
      scrollToBottom();
    }
  }
}

function recommendationLabel(recommendation) {
  if (typeof recommendation === "string") return recommendation;
  return String(recommendation?.label || recommendation?.query || "").trim();
}

function recommendationQuery(recommendation) {
  if (typeof recommendation === "string") return recommendation;
  return String(recommendation?.query || recommendation?.label || "").trim();
}

function recommendationKey(recommendation, index) {
  if (typeof recommendation === "object" && recommendation?.id) return recommendation.id;
  return `${recommendationQuery(recommendation)}-${index}`;
}

// 点击推荐问题 chips / 欢迎屏建议：展示 label，提交后端已闭合上下文的 query。
function askRecommend(recommendation) {
  if (loading.value) return;
  const query = recommendationQuery(recommendation);
  if (!query) return;
  question.value = query;
  sendQuestion();
}

// ===== 图表下钻 / 联动 / 周期对比（TODO-26） =====
function handlePointClick(msg, point) {
  // 点击只选中数据点，不立即发起请求；用户可选择下钻、联动或周期对比动作。
  msg._selectedPoint = point;
}

function clearPoint(msg) {
  msg._selectedPoint = null;
}

function runChartInteraction(query, msg) {
  if (loading.value || !query) return;
  clearPoint(msg);
  question.value = query;
  sendQuestion();
}

function runDrillDown(msg) {
  const point = msg._selectedPoint;
  if (!point) return;
  const base = msg._query || lastQuery.value || "当前图表";
  runChartInteraction(
    `下钻分析：请在${point.label}为“${point.value}”的范围内，沿更细粒度维度拆分并分析“${base}”，保持原指标和时间范围，返回结果表和图表。`,
    msg,
  );
}

function runLinkedFilter(msg) {
  const point = msg._selectedPoint;
  if (!point) return;
  const base = msg._query || lastQuery.value || "当前图表";
  runChartInteraction(
    `联动筛选：后续查询限定在${point.label}为“${point.value}”，请继续回答“${base}”，保持原指标、分组和时间范围。`,
    msg,
  );
}

function runPeriodCompare(msg, period) {
  const base = msg._query || lastQuery.value || "当前图表";
  const point = msg._selectedPoint;
  const filter = point ? `，并限定${point.label}为“${point.value}”` : "";
  runChartInteraction(
    `周期对比：请将“${base}”按${period}进行对比${filter}。保持原指标口径和分组维度，结果必须增加“周期”字段（当前周期/对比周期），并尽量输出指标值、变化量和变化率，返回结果表和图表。`,
    msg,
  );
}

// 通用 SSE 请求：POST JSON body，逐 chunk 解析 data: {json}
async function streamSSE(url, body, onChunk) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "流式请求失败"));
  }
  if (!response.body) throw new Error("服务器未返回流");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith("data:")) continue;
      let data;
      try {
        data = JSON.parse(line.replace(/^data:\s*/, ""));
      } catch {
        continue;
      }
      onChunk(data);
      await nextTick();
      scrollToBottom();
    }
  }
}

// 数据预测：把当前回答的 query/sql/chart/result 回传 /api/predict，流式返回预测行
async function runPredict(msg) {
  if (msg._predicting) return;
  msg._predicting = true;
  try {
    await streamSSE(PREDICT_URL, {
      query: msg._query || lastQuery.value,
      sql: msg._sql || lastSql.value,
      chart: msg.chart || { type: "table" },
      result: msg.rows || [],
    }, (data) => {
      if (Array.isArray(data.predict)) {
        // 拼接原始行 + 预测行，predictStart 标记预测段起点供 ChartView 虚线渲染
        const original = msg.rows || [];
        messages.value.push({
          role: "assistant",
          type: "predict",
          chart: msg.chart || { type: "line" },
          rows: original.concat(data.predict),
          predictStart: original.length,
        });
      } else if (data.predict_failed) {
        messages.value.push({
          role: "assistant",
          type: "predict_failed",
          content: data.predict_failed,
        });
      }
    });
  } catch (e) {
    messages.value.push({
      role: "assistant",
      type: "predict_failed",
      content: e?.message || "数据预测请求失败",
    });
  } finally {
    msg._predicting = false;
  }
}

// 数据异常检测：把 query/sql/chart/result 回传 /api/anomaly，流式返回中文报告
async function runAnomaly(msg) {
  if (msg._anomaly) return;
  msg._anomaly = true;
  try {
    await streamSSE(ANOMALY_URL, {
      query: msg._query || lastQuery.value,
      sql: msg._sql || lastSql.value,
      chart: msg.chart || { type: "table" },
      result: msg.rows || [],
    }, (data) => {
      if (data.anomaly) {
        messages.value.push({
          role: "assistant",
          type: "anomaly",
          content: data.anomaly,
        });
      }
    });
  } catch (e) {
    messages.value.push({
      role: "assistant",
      type: "anomaly",
      content: e?.message || "数据异常检测请求失败",
    });
  } finally {
    msg._anomaly = false;
  }
}

// 加入仪表盘：打开弹层，拉取仪表盘列表供选择
async function addToDashboard(msg) {
  let chart = msg.chart;
  if (!chart) {
    chart = {
      type: "table",
      columns: (msg.columns || []).map((c) => ({ name: c, value: c })),
    };
  }
  addToDash.payload = {
    chart: chart || { type: "table" },
    sql: msg._sql || lastSql.value || "",
    query: msg._query || lastQuery.value || "",
    sourceMessage: msg,
  };
  dashboardModalTrigger = document.activeElement;
  addToDash.error = "";
  addToDash.open = true;
  await nextTick();
  dashboardModalEl.value?.focus();
  try {
    const res = await fetch("/api/dashboard/tree");
    if (!res.ok) throw new Error(await responseError(res, "加载仪表盘失败"));
    const tree = await res.json();
    addToDash.dashboards = (tree || []).filter((n) => n.node_type === "dashboard");
  } catch (error) {
    addToDash.dashboards = [];
    addToDash.error = error?.message || "仪表盘列表加载失败";
  }
}

function closeDashboardModal() {
  addToDash.open = false;
  addToDash.error = "";
  nextTick(() => dashboardModalTrigger?.focus?.());
}

// 确认加入：dashboardId 为空则先新建仪表盘再加入，成功后跳转仪表盘模式
async function confirmAddToDashboard(dashboardId) {
  if (!addToDash.payload) return;
  addToDash.error = "";
  try {
    let id = dashboardId;
    if (!id) {
      const name = (await promptDialog({
        title: "新建仪表盘并加入",
        message: "创建后，当前结果会立即作为图表块加入该仪表盘。",
        label: "仪表盘名称",
        value: "新仪表盘",
        confirmLabel: "创建并加入",
      }))?.trim();
      if (!name) return;
      const res = await fetch("/api/dashboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, node_type: "dashboard" }),
      });
      if (!res.ok) throw new Error(await responseError(res, "创建仪表盘失败"));
      const node = await res.json();
      id = node.id;
    }
    const { sourceMessage, ...payload } = addToDash.payload;
    const response = await fetch(`/api/dashboard/${id}/block`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await responseError(response, "加入仪表盘失败"));
    setMessageStatus(sourceMessage, "结果已加入仪表盘");
    addToDash.open = false;
    addToDash.payload = null;
    pendingDashboardId.value = id;
    mode.value = "dashboard";
    showAppNotice("结果已加入仪表盘");
  } catch (error) {
    addToDash.error = error?.message || "加入仪表盘失败";
  }
}
</script>

<style scoped>
.inline-glyph {
  min-width: 1em;
  display: inline-grid;
  place-items: center;
  color: currentColor;
  font-style: normal;
  font-weight: 760;
  line-height: 1;
}
.inline-glyph.warning {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: rgba(224, 129, 31, .14);
}
.inline-glyph.arrow { font-size: 14px; }
/* ============ 页面骨架 ============ */
.chat-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--da-bg);
  overflow: hidden;
}
.app-shell {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  display: flex;
  overflow: hidden;
}
.app-main {
  position: relative;
  min-width: 0;
  min-height: 0;
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 顶栏 */
.topbar {
  flex: 0 0 auto;
  border-bottom: 1px solid var(--da-border-light);
  background: var(--da-bg);
}
.topbar-inner {
  max-width: 900px;
  margin: 0 auto;
  height: 56px;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand-logo {
  font-size: 20px;
}
.brand-name {
  font-size: 16px;
  font-weight: 600;
  color: var(--da-text);
}
.new-chat-top {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid var(--da-border);
  background: var(--da-bg);
  color: var(--da-text-secondary);
  font-size: 13px;
  transition: all 0.15s;
}
.new-chat-top:hover {
  border-color: var(--da-accent);
  color: var(--da-accent);
  background: var(--da-accent-soft);
}
.new-chat-top .plus {
  font-size: 14px;
  line-height: 1;
}

/* 顶栏模式切换 tabs */
.topbar-tabs {
  display: flex;
  gap: 4px;
}
.tab {
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--da-text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}
.tab:hover {
  background: var(--da-bg-soft);
}
.tab.active {
  background: var(--da-accent-soft);
  color: var(--da-accent);
  border-color: var(--da-accent);
}

/* 仪表盘模式占位（撑满除顶栏外区域） */
.dash-slot {
  flex: 1 1 auto;
  min-height: 0;
}

/* 加入仪表盘弹层 */
.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(31, 35, 41, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal {
  width: 360px;
  max-height: 70vh;
  background: var(--da-bg);
  border-radius: 14px;
  box-shadow: 0 12px 40px rgba(31, 35, 41, 0.18);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.modal-title {
  padding: 16px 18px 4px;
  font-size: 16px;
  font-weight: 600;
  color: var(--da-text);
}
.modal-sub {
  padding: 0 18px 12px;
  font-size: 12px;
  color: var(--da-text-muted);
}
.modal-list {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 0 10px;
}
.modal-empty {
  padding: 20px;
  text-align: center;
  font-size: 13px;
  color: var(--da-text-muted);
}
.modal-row {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  font-size: 14px;
  color: var(--da-text);
  text-align: left;
  transition: background 0.12s;
}
.modal-row:hover {
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.modal-footer {
  padding: 12px 18px;
  border-top: 1px solid var(--da-border-light);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.modal-error {
  margin: 0 18px 10px;
  padding: 9px 11px;
  border: 1px solid #f0c5bf;
  border-radius: 9px;
  background: #fff3f1;
  color: #b83d31;
  font-size: 11px;
}
/* App 内 mini-btn（与 DashboardView 一致的视觉） */
.mini-btn {
  padding: 4px 10px;
  font-size: 12px;
  line-height: 18px;
  border-radius: 8px;
  border: 1px solid var(--da-border);
  background: var(--da-bg);
  color: var(--da-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
}
.mini-btn:hover:not(:disabled) {
  border-color: var(--da-accent);
  color: var(--da-accent);
  background: var(--da-accent-soft);
}
.mini-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.mini-btn.primary {
  background: var(--da-accent);
  color: #fff;
  border-color: var(--da-accent);
}
.mini-btn.primary:hover:not(:disabled) {
  background: var(--da-accent-hover);
  color: #fff;
}

/* ============ 消息区 ============ */
.messages {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 24px 20px 38px;
}

/* 欢迎屏 */
.welcome {
  max-width: 720px;
  margin: 8vh auto 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 0 20px;
}
.welcome-logo {
  font-size: 48px;
  margin-bottom: 16px;
  color: #fff;
}
.welcome-title {
  font-size: 26px;
  font-weight: 600;
  color: var(--da-text);
  margin-bottom: 8px;
}
.welcome-sub {
  font-size: 14px;
  color: var(--da-text-secondary);
  line-height: 24px;
  max-width: 460px;
  margin-bottom: 32px;
}
.welcome-suggestions {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
.suggestion-card {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border: 1px solid var(--da-border);
  border-radius: 12px;
  background: var(--da-bg);
  cursor: pointer;
  transition: all 0.15s;
  text-align: left;
}
.suggestion-card:hover {
  border-color: var(--da-accent);
  background: var(--da-accent-soft);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(31, 35, 41, 0.06);
}
.suggestion-icon {
  color: var(--da-accent);
}
.suggestion-text {
  font-size: 14px;
  color: var(--da-text);
  line-height: 20px;
}

/* 对话流：居中 800px */
.chat-stream {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
}
.stream-bottom-spacer {
  height: 40px;
}

/* 单行：row */
.row {
  display: flex;
  gap: 12px;
  padding: 16px 0;
  align-items: flex-start;
}
.row.user {
  flex-direction: row-reverse;
}
.row.wide .content {
  max-width: 100%;
  width: 100%;
}

/* 头像 */
.avatar {
  flex: 0 0 32px;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}
.avatar.ai {
  background: linear-gradient(135deg, #409eff, #66b1ff);
}
.avatar.user {
  background: var(--da-bg-soft);
}

/* 内容区：用户气泡限宽，assistant 宽内容撑满 */
.content {
  max-width: calc(100% - 44px);
}
.row:not(.wide) .content {
  max-width: min(680px, 100%);
}

/* 用户气泡 */
.user-bubble {
  display: inline-block;
  padding: 12px 16px;
  background: var(--da-bg-soft);
  color: var(--da-text);
  border-radius: 16px;
  font-size: 15px;
  line-height: 24px;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* ============ 步骤卡片（思考过程） ============ */
.steps-card {
  width: 100%;
  border: 1px solid var(--da-border-light);
  border-radius: 12px;
  background: var(--da-bg);
  overflow: hidden;
}
.steps-header {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 48px;
  padding: 10px 14px;
  cursor: pointer;
  list-style: none;
}
.steps-header::-webkit-details-marker { display: none; }
.steps-card[open] .steps-header { border-bottom: 1px solid var(--da-border-light); background: #fbfcfe; }
.steps-detail { padding: 12px 14px 14px; }
.steps-progress,
.steps-latency,
.steps-detail-label {
  color: var(--da-text-muted);
  font-size: 10px;
}
.steps-progress { margin-left: 3px; }
.steps-latency {
  padding: 3px 7px;
  border-radius: 999px;
  background: var(--da-bg-soft);
}
.steps-latency.good { background: #eaf8f0; color: #287b4a; }
.steps-latency.warn { background: #fff5e4; color: #956315; }
.steps-latency.bad { background: #fff0ee; color: #b83d31; }
.steps-detail-label { margin-left: auto; color: var(--da-accent); }
.steps-chevron { color: var(--da-text-muted); transition: transform .15s; }
.steps-card[open] .steps-chevron { transform: rotate(180deg); }
.thinking-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--da-text-muted);
}
.thinking-dot.running {
  background: var(--da-accent);
  animation: pulse 1s ease-in-out infinite;
}
.thinking-dot.done {
  background: var(--da-success);
}
.thinking-dot.error {
  background: var(--da-danger);
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}
.steps-title {
  font-size: 13px;
  font-weight: 650;
  color: var(--da-text);
}
.steps-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.step {
  display: flex;
  align-items: center;
  gap: 8px;
}
.step-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: 0 0 8px;
}
.step-dot.running {
  background: var(--da-warning);
  animation: pulse 1s ease-in-out infinite;
}
.step-dot.success {
  background: var(--da-success);
}
.step-dot.error {
  background: var(--da-danger);
}
.step-text {
  font-size: 13px;
  color: var(--da-text-secondary);
  line-height: 20px;
}
.trace-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--da-border-light);
  color: var(--da-text-muted);
  font-size: 11px;
}
.trace-degraded { color: var(--da-warning); }
.trace-follow-up { color: var(--da-accent); }
.follow-up-audit { display: grid; gap: 3px; margin: 2px 14px 11px; padding: 9px 11px; border: 1px solid rgba(52,120,246,.2); border-radius: 9px; background: rgba(52,120,246,.055); font-size: 10px; }
.follow-up-audit strong { color: var(--da-accent); }
.follow-up-audit span { color: var(--da-text-secondary); overflow-wrap: anywhere; }
.planning-audit { display:grid; gap:7px; margin:2px 14px 11px; padding:11px 12px; border:1px solid #d9d1f2; border-radius:10px; background:#faf8ff; font-size:10px; }
.planning-audit>div:first-child { display:flex; align-items:center; justify-content:space-between; gap:10px; }
.planning-audit strong { color:#6b55a0; }
.planning-audit span,.planning-audit small { color:var(--da-text-muted); overflow-wrap:anywhere; }
.planning-shadows { display:flex; flex-wrap:wrap; gap:6px; }
.planning-shadows span { padding:4px 7px; border-radius:7px; background:#fff; color:#62537f; }
.semantic-repair-audit {
  display: grid;
  gap: 7px;
  margin: 2px 14px 11px;
  padding: 11px 12px;
  border: 1px solid #cfe0f4;
  border-radius: 10px;
  background: linear-gradient(145deg, #f7fbff, #f2f7fd);
  font-size: 10px;
}
.semantic-repair-audit > div:first-child {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.semantic-repair-audit strong { color: #2d65b7; }
.semantic-repair-audit span,
.semantic-repair-audit small { color: var(--da-text-muted); overflow-wrap: anywhere; }
.semantic-repair-audit em {
  padding: 6px 8px;
  border-radius: 7px;
  background: #fff1e9;
  color: #9b5d34;
  font-style: normal;
}
.semantic-rule-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.semantic-rule-chips span {
  padding: 4px 7px;
  border-radius: 7px;
  background: #fff;
  color: #315f9d;
  box-shadow: inset 0 0 0 1px #dae7f6;
}
.memory-audit { display: grid; gap: 4px; margin: 2px 14px 11px; padding: 10px 12px; border: 1px solid #cfe0f7; border-radius: 10px; background: #f6faff; color: var(--da-text-secondary); font-size: 10px; }
.memory-audit strong { color: #245fae; }
.memory-audit span, .memory-audit small { color: var(--da-text-muted); }
.trace-nodes { flex-basis: 100%; }
.trace-nodes summary { cursor: pointer; color: var(--da-accent); }
.trace-nodes div { display: flex; justify-content: space-between; max-width: 360px; padding: 3px 0; }
.trace-nodes strong { color: var(--da-text-secondary); font-weight: 600; }
.trace-nodes code { margin-left: 4px; color: var(--da-text-muted); font-size: 9px; }
.trace-retrieval{flex-basis:100%;margin-top:2px}.trace-retrieval summary{cursor:pointer;color:var(--da-accent)}.trace-retrieval section{display:grid;grid-template-columns:minmax(130px,auto) 1fr;gap:3px 10px;max-width:760px;padding:7px 0;border-bottom:1px dashed var(--da-border)}.trace-retrieval section:last-child{border-bottom:0}.trace-retrieval strong{color:var(--da-text-secondary);font-weight:650}.trace-retrieval small{grid-column:2;color:var(--da-text-muted)}
.trace-contract{flex-basis:100%;margin-top:2px}.trace-contract summary{cursor:pointer;color:#5572b7}.trace-contract section{display:flex;flex-wrap:wrap;gap:6px 10px;max-width:840px;padding:8px 0}.trace-contract strong{color:#365b9b}.trace-contract span{padding:3px 6px;border-radius:6px;background:#f3f7ff;color:var(--da-text-secondary)}

/* ============ SQL 代码卡 ============ */
.sql-card {
  width: 100%;
  background: #1e1e2e;
  border-radius: 12px;
  overflow: hidden;
}
.sql-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: rgba(255, 255, 255, 0.06);
}
.sql-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #cdd6f4;
  letter-spacing: 1px;
  font-weight: 600;
}
.copy-btn {
  padding: 3px 10px;
  font-size: 12px;
  border: 1px solid rgba(205, 214, 244, 0.3);
  border-radius: 6px;
  background: transparent;
  color: #cdd6f4;
  transition: all 0.15s;
}
.copy-btn:hover {
  background: rgba(205, 214, 244, 0.12);
  border-color: #cdd6f4;
}
.sql-code {
  margin: 0;
  padding: 12px 14px;
  color: #cdd6f4;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12.5px;
  line-height: 20px;
  white-space: pre-wrap;
  word-break: break-all;
  overflow-x: auto;
}
/* SQL 卡头部操作区 + 编辑态 textarea（TODO-22） */
.sql-head-actions {
  display: flex;
  gap: 6px;
}
.sql-edit {
  margin: 0;
  padding: 12px 14px;
  width: 100%;
  min-height: 120px;
  background: #1e1e2e;
  color: #cdd6f4;
  border: none;
  outline: none;
  resize: vertical;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12.5px;
  line-height: 20px;
  white-space: pre-wrap;
  word-break: break-all;
  box-sizing: border-box;
}

/* ============ 通用卡片（图表/表格/解读） ============ */
.chart-card,
.table-card,
.analysis-card,
.recommend-card {
  width: 100%;
  border: 1px solid var(--da-border);
  border-radius: 12px;
  background: var(--da-bg);
  overflow: hidden;
}
.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--da-border-light);
}
.card-title {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 15px;
  font-weight: 600;
  color: var(--da-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.card-badge {
  flex: 0 0 auto;
  padding: 2px 10px;
  font-size: 12px;
  border-radius: 999px;
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.card-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.predict-card .card-badge {
  background: rgba(180, 120, 255, 0.12);
  color: #8a5cf6;
}
.anomaly-card .card-title {
  color: var(--da-warning);
}
.card-body {
  padding: 16px;
}
.chart-interaction {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 16px;
  border-top: 1px solid var(--da-border-light);
  background: var(--da-bg-soft);
}
.interaction-selection {
  margin-right: auto;
  color: var(--da-text-secondary);
  font-size: 12px;
}
.interaction-selection strong {
  color: var(--da-accent);
  font-weight: 600;
}
.interaction-divider {
  width: 1px;
  height: 18px;
  margin: 0 2px;
  background: var(--da-border);
}
.interaction-clear {
  padding: 3px 5px;
  border: none;
  background: transparent;
  color: var(--da-text-muted);
  font-size: 12px;
}
.interaction-clear:hover {
  color: var(--da-danger);
}

/* 解读卡 */
.analysis-card .analysis-text {
  padding: 14px 16px;
  font-size: 14px;
  line-height: 1.8;
  color: var(--da-text);
  white-space: pre-wrap;
}
/* 反馈条（TODO-23） */
.fb-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-top: 1px solid var(--da-border-light);
  flex-wrap: wrap;
}
.fb-label {
  font-size: 12px;
  color: var(--da-text-muted);
}
.fb-btn {
  padding: 3px 9px;
  font-size: 14px;
  border-radius: 8px;
  border: 1px solid var(--da-border);
  background: var(--da-bg);
  cursor: pointer;
  transition: all 0.15s;
}
.fb-btn:hover {
  border-color: var(--da-accent);
}
.fb-btn.active {
  background: var(--da-accent-soft);
  border-color: var(--da-accent);
}
.fb-input {
  flex: 1 1 200px;
  min-width: 160px;
  padding: 5px 10px;
  font-size: 12px;
  border: 1px solid var(--da-border);
  border-radius: 8px;
  background: var(--da-bg);
  color: var(--da-text);
  outline: none;
}
.fb-input:focus {
  border-color: var(--da-accent);
}
.fb-done {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--da-success);
}
.fb-error {
  flex-basis: 100%;
  color: var(--da-danger);
  font-size: 11px;
}

/* 表格 */
.table-wrap {
  overflow-x: auto;
}
.result-table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.result-table th,
.result-table td {
  border-bottom: 1px solid var(--da-border-light);
  padding: 9px 14px;
  white-space: nowrap;
  text-align: left;
  color: var(--da-text);
}
.result-table th {
  background: var(--da-bg-soft);
  font-weight: 600;
  color: var(--da-text-secondary);
  position: sticky;
  top: 0;
}
.result-table tbody tr:hover {
  background: var(--da-bg-soft);
}

/* ============ 猜你想问 ============ */
.recommend-card {
  border: 1px solid var(--da-border);
  padding: 14px 16px;
}
.recommend-title {
  font-size: 13px;
  color: var(--da-text-muted);
  margin-bottom: 10px;
  font-weight: 500;
}
.recommend-grid {
  display: grid;
  grid-template-columns: repeat(2, calc(50% - 6px));
  gap: 10px;
}
.recommend-chip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid var(--da-border);
  background: var(--da-bg-soft);
  color: var(--da-text);
  font-size: 13px;
  line-height: 20px;
  text-align: left;
  cursor: pointer;
  transition: all 0.15s;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.recommend-chip span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.recommend-chip:hover {
  background: var(--da-accent-soft);
  border-color: var(--da-accent);
  color: var(--da-accent);
  transform: translateY(-1px);
}
.recommend-chip:disabled {
  cursor: wait;
  opacity: 0.55;
  transform: none;
}

/* ============ 错误 ============ */
.error-card {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 12px;
  background: rgba(231, 76, 60, 0.08);
  border: 1px solid rgba(231, 76, 60, 0.25);
}
.error-icon {
  font-size: 16px;
}
.error-text {
  font-size: 13px;
  color: var(--da-danger);
  line-height: 20px;
}
.clarification-help{margin-top:9px;padding-top:9px;border-top:1px solid rgba(184,61,49,.16);font-size:11px;line-height:1.6}.clarification-help strong{display:block;margin-bottom:4px}.clarification-help ul{margin:0;padding-left:17px}.clarification-slots{display:flex;flex-wrap:wrap;gap:4px;margin-top:7px}.clarification-slots code{padding:2px 6px;border-radius:5px;background:rgba(184,61,49,.08);font:9px Consolas,monospace}
.app-toast {
  position: absolute;
  z-index: 80;
  top: 16px;
  left: 50%;
  max-width: min(480px, calc(100% - 28px));
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 13px;
  border: 1px solid #b9e2c8;
  border-radius: 10px;
  background: rgba(240, 251, 244, .98);
  color: #287b4a;
  box-shadow: var(--da-shadow-md);
  font-size: 12px;
  transform: translateX(-50%);
}
.app-toast.error { border-color: #f0c5bf; background: rgba(255, 243, 241, .98); color: #b83d31; }

/* ============ 悬浮输入区 ============ */
.input-wrapper {
  flex: 0 0 auto;
  position: relative;
  padding: 12px 20px 16px;
  background: linear-gradient(to top, var(--da-bg) 60%, rgba(255, 255, 255, 0));
}
.input-box {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 8px 8px 8px 16px;
  border: 1px solid var(--da-border);
  border-radius: 16px;
  background: var(--da-bg-input);
  box-shadow: 0 6px 24px rgba(31, 35, 41, 0.08);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.input-box:focus-within {
  border-color: var(--da-accent);
  box-shadow: 0 6px 24px rgba(64, 158, 255, 0.18);
}
.input-box textarea {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  resize: none;
  font-size: 15px;
  line-height: 24px;
  padding: 8px 0;
  max-height: 160px;
  color: var(--da-text);
  font-family: inherit;
}
.attach-btn {
  flex: 0 0 30px;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: var(--da-text-secondary);
  cursor: pointer;
  font-size: 17px;
}
.attach-btn:hover:not(.disabled) {
  background: var(--da-accent-soft);
  color: var(--da-accent);
}
.attach-btn.disabled {
  opacity: .5;
  cursor: not-allowed;
}
.attach-btn input {
  display: none;
}
.input-box textarea::placeholder {
  color: var(--da-text-muted);
}
.send-btn {
  flex: 0 0 36px;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: var(--da-accent);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}
.send-btn:hover:not(:disabled) {
  background: var(--da-accent-hover);
  transform: scale(1.05);
}
.send-btn:disabled {
  background: var(--da-text-muted);
  cursor: not-allowed;
}
.send-icon {
  font-size: 15px;
  transform: rotate(-90deg);
}
.loader {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.input-hint {
  max-width: 800px;
  margin: 8px auto 0;
  text-align: center;
  font-size: 11px;
  color: var(--da-text-muted);
}
.input-options {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 7px 12px;
  margin-bottom: 5px;
}
.dataset-option { display: inline-flex; align-items: center; flex-wrap: wrap; justify-content: center; }
.data-context-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 30px;
  padding: 0 10px;
  color: #334155;
  border: 1px solid var(--da-border);
  border-radius: 999px;
  background: rgba(255,255,255,.88);
  cursor: pointer;
  font-size: 11px;
  transition: .15s ease;
}
.data-context-pill:hover { border-color: rgba(52,120,246,.42); box-shadow: 0 4px 12px rgba(15,23,42,.06); }
.data-context-pill.upload { color: #0f5f59; border-color: rgba(13,148,136,.28); background: #f0fdfa; }
.data-context-pill em { padding: 2px 6px; color: var(--da-text-muted); border-radius: 999px; background: rgba(148,163,184,.12); font-size: 9px; font-style: normal; }
.data-context-pill.dirty { border-style: dashed; }
.context-dirty-dot { width: 7px; height: 7px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,.14); }
.source-context-audit,
.source-comparison-audit {
  display: grid;
  gap: 4px;
  padding: 10px 12px;
  margin-top: 10px;
  color: #155e75;
  border: 1px solid #bae6fd;
  border-radius: 11px;
  background: #f0f9ff;
  font-size: 11px;
}
.source-context-audit span,
.source-comparison-audit span { color: #334155; }
.source-context-audit small,
.source-comparison-audit small { color: #64748b; font-size: 9px; }
.source-comparison-audit { color: #166534; border-color: #bbf7d0; background: #f0fdf4; }
.source-comparison-details {
  margin-top: 2px;
  color: #166534;
}
.source-comparison-details summary {
  width: fit-content;
  cursor: pointer;
  font-weight: 650;
}
.source-comparison-table-wrap {
  max-height: 230px;
  margin-top: 8px;
  overflow: auto;
  border: 1px solid rgba(22, 101, 52, .14);
  border-radius: 8px;
  background: rgba(255, 255, 255, .75);
}
.source-comparison-table-wrap table {
  width: 100%;
  border-collapse: collapse;
  color: #334155;
  font-size: 10px;
}
.source-comparison-table-wrap th,
.source-comparison-table-wrap td {
  padding: 7px 9px;
  border-bottom: 1px solid rgba(148, 163, 184, .16);
  text-align: right;
  white-space: nowrap;
}
.source-comparison-table-wrap th:first-child,
.source-comparison-table-wrap td:first-child { text-align: left; }
.source-comparison-table-wrap tbody tr:last-child td { border-bottom: 0; }
.source-comparison-table-wrap td small {
  display: inline;
  margin-left: 3px;
}
.clear-adhoc {
  border: none;
  padding: 0;
  background: transparent;
  color: var(--da-accent);
  cursor: pointer;
  font-size: 11px;
}
.compare-toggle {
  margin-left: 8px;
  color: var(--da-accent);
  cursor: pointer;
  font-size: 11px;
}
.compare-toggle input {
  vertical-align: -1px;
  accent-color: var(--da-accent);
}
.analysis-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 8px;
  border: 1px solid var(--da-border);
  border-radius: 999px;
  background: rgba(255, 255, 255, .78);
  color: var(--da-text-secondary);
  cursor: pointer;
  user-select: none;
}
.analysis-toggle:hover { border-color: var(--da-accent); color: var(--da-accent); }
.analysis-toggle input { margin: 0; accent-color: var(--da-accent); }
.analysis-toggle em { color: var(--da-text-muted); font-size: 10px; font-style: normal; }
.analysis-toggle:has(input:checked) { border-color: rgba(52, 120, 246, .42); background: rgba(52, 120, 246, .08); color: var(--da-accent); }
.memory-toggle { min-height: 28px; display: inline-flex; align-items: center; gap: 5px; padding: 4px 9px; border: 1px solid #cbdcf3; border-radius: 999px; background: #f5f9ff; color: #2e67ae; cursor: pointer; font: 10px inherit; }
.user-actions { display: flex; align-items: center; gap: 9px; margin-left: auto; }
.workspace-label { color: var(--da-text-muted); font-size: 11px; white-space: nowrap; }
.logout-btn { border: 1px solid var(--da-border); border-radius: 7px; background: transparent; color: var(--da-text-secondary); padding: 6px 9px; cursor: pointer; font-size: 11px; }
.auth-loading { flex: 1; display: grid; place-items: center; color: var(--da-text-muted); font-size: 13px; }

/* ============ 响应式 ============ */
@media (max-width: 640px) {
  .welcome-suggestions {
    grid-template-columns: 1fr;
  }
  .recommend-grid {
    grid-template-columns: 1fr;
  }
  .messages {
    padding: 16px 12px 180px;
  }
  .topbar-inner,
  .input-box {
    padding-left: 12px;
    padding-right: 12px;
  }
}

/* ============ 2026 visual refresh ============ */
.chat-page {
  background:
    radial-gradient(circle at 50% -20%, rgba(52, 120, 246, .09), transparent 38%),
    var(--da-bg-soft);
}
.topbar {
  position: relative;
  z-index: 15;
  border-bottom-color: rgba(217, 224, 234, .8);
  background: rgba(255, 255, 255, .9);
  box-shadow: 0 1px 0 rgba(24, 39, 75, .02);
  backdrop-filter: blur(18px);
}
.topbar-inner {
  width: 100%;
  max-width: 1540px;
  height: 68px;
  gap: 14px;
  padding: 0 24px;
  justify-content: flex-start;
}
.brand { flex: 0 0 auto; min-width: 164px; gap: 10px; }
.brand-logo {
  width: 35px;
  height: 35px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 3px;
  padding: 8px;
  border-radius: 11px;
  background: linear-gradient(145deg, #3d86ff, #2563df);
  box-shadow: 0 8px 20px rgba(52, 120, 246, .25);
}
.brand-logo i { width: 4px; border-radius: 4px; background: #fff; opacity: .95; }
.brand-logo i:nth-child(1) { height: 8px; opacity: .68; }
.brand-logo i:nth-child(2) { height: 13px; }
.brand-logo i:nth-child(3) { height: 17px; opacity: .82; }
.brand-copy { display: flex; flex-direction: column; line-height: 1.15; }
.brand-name { font-size: 15px; font-weight: 750; letter-spacing: -.02em; }
.brand-copy small { margin-top: 4px; color: var(--da-text-muted); font-size: 9px; font-weight: 650; letter-spacing: .13em; text-transform: uppercase; }
.topbar-tabs {
  min-width: 0;
  flex: 1 1 auto;
  gap: 3px;
  overflow-x: auto;
  padding: 6px;
  border-radius: 13px;
  background: rgba(244, 247, 251, .78);
  scrollbar-width: none;
}
.topbar-tabs::-webkit-scrollbar { display: none; }
.tab {
  flex: 0 0 auto;
  min-height: 35px;
  padding: 7px 11px;
  border: 0;
  border-radius: 9px;
  font-size: 12px;
  font-weight: 520;
  white-space: nowrap;
}
.tab span { margin-right: 3px; filter: saturate(.8); }
.tab:hover { color: var(--da-text); background: rgba(255, 255, 255, .9); box-shadow: var(--da-shadow-sm); }
.tab.active { border: 0; background: #fff; color: var(--da-accent-strong); box-shadow: var(--da-shadow-sm); }
.user-actions { flex: 0 0 auto; margin-left: 0; }
.workspace-label {
  height: 36px;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--da-border-light);
  border-radius: 10px;
  background: #fff;
  color: var(--da-text-secondary);
  box-shadow: var(--da-shadow-sm);
}
.workspace-label i { width: 7px; height: 7px; border-radius: 50%; background: var(--da-success); box-shadow: 0 0 0 3px rgba(31, 157, 104, .11); }
.workspace-label b { max-width: 110px; overflow: hidden; color: var(--da-text); font-size: 11px; font-weight: 650; text-overflow: ellipsis; }
.workspace-label em { padding-left: 6px; border-left: 1px solid var(--da-border-light); color: var(--da-text-muted); font-size: 10px; font-style: normal; }
.logout-btn { height: 34px; border-color: transparent; color: var(--da-text-muted); }
.logout-btn:hover { color: var(--da-danger); background: rgba(214, 69, 69, .07); }
.new-chat-top {
  flex: 0 0 auto;
  height: 36px;
  padding: 0 13px;
  border: 0;
  border-radius: 10px;
  background: var(--da-accent);
  color: #fff;
  font-weight: 650;
  box-shadow: 0 7px 16px rgba(52, 120, 246, .2);
}
.new-chat-top:hover { border-color: transparent; background: var(--da-accent-hover); color: #fff; transform: translateY(-1px); }
.messages { padding-top: 34px; background: transparent; }
.welcome { max-width: 820px; margin-top: clamp(42px, 8vh, 90px); }
.welcome-eyebrow { display: flex; align-items: center; gap: 7px; margin-bottom: 16px; color: var(--da-accent); font-size: 10px; font-weight: 750; letter-spacing: .18em; }
.welcome-eyebrow i { width: 18px; height: 1px; background: var(--da-accent); }
.welcome-logo {
  width: 70px;
  height: 70px;
  margin-bottom: 22px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 5px;
  padding: 17px;
  border: 1px solid rgba(255, 255, 255, .78);
  border-radius: 22px;
  background: linear-gradient(145deg, #4a8eff, #225ad0);
  box-shadow: 0 18px 42px rgba(52, 120, 246, .24), inset 0 1px 1px rgba(255,255,255,.34);
}
.welcome-logo span { width: 7px; border-radius: 6px; background: #fff; }
.welcome-logo span:nth-child(1) { height: 17px; opacity: .65; }
.welcome-logo span:nth-child(2) { height: 27px; }
.welcome-logo span:nth-child(3) { height: 36px; opacity: .82; }
.welcome-title { margin-bottom: 10px; font-size: clamp(27px, 3vw, 36px); font-weight: 650; letter-spacing: -.04em; }
.welcome-title strong { color: var(--da-accent-strong); font-weight: 760; }
.welcome-sub { max-width: 550px; margin-bottom: 14px; color: var(--da-text-secondary); font-size: 14px; line-height: 1.8; }
.welcome-capabilities { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px; margin-bottom: 30px; }
.welcome-capabilities span { display: inline-flex; align-items: center; gap: 5px; padding: 5px 9px; border: 1px solid rgba(52, 120, 246, .12); border-radius: 999px; background: rgba(255, 255, 255, .68); color: var(--da-text-muted); font-size: 10px; }
.welcome-capabilities b { color: var(--da-accent); font-size: 12px; font-weight: 750; line-height: 1; }
.welcome-suggestions { gap: 10px; }
.suggestion-card {
  position: relative;
  min-height: 64px;
  padding: 14px 15px;
  border-color: rgba(217, 224, 234, .9);
  border-radius: 14px;
  box-shadow: var(--da-shadow-sm);
}
.suggestion-card:hover { border-color: rgba(52, 120, 246, .45); background: #fff; box-shadow: var(--da-shadow-md); transform: translateY(-2px); }
.suggestion-icon { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 10px; background: var(--da-accent-soft); }
.suggestion-icon > span { color: var(--da-accent); font-size: 14px; font-weight: 750; letter-spacing: -.03em; }
.suggestion-text { flex: 1; font-size: 13px; font-weight: 550; }
.suggestion-arrow { color: var(--da-text-muted); transition: transform .15s, color .15s; }
.suggestion-card:hover .suggestion-arrow { color: var(--da-accent); transform: translate(2px, -2px); }
.chat-stream { max-width: 900px; }
.row { padding: 18px 0; }
.avatar { width: 36px; height: 36px; flex-basis: 36px; border-radius: 11px; box-shadow: var(--da-shadow-sm); }
.avatar.ai { background: linear-gradient(145deg, #4a8eff, #2869e5); }
.avatar.user { border: 1px solid var(--da-border-light); background: #fff; }
.user-bubble { padding: 11px 15px; border-radius: 16px 16px 5px 16px; background: var(--da-accent); color: #fff; box-shadow: 0 7px 18px rgba(52, 120, 246, .16); }
.steps-card, .chart-card, .table-card, .analysis-card, .recommend-card { border-color: var(--da-border-light); border-radius: var(--da-radius-md); box-shadow: var(--da-shadow-sm); }
.card-head { min-height: 52px; padding: 11px 15px; background: linear-gradient(180deg, #fff, #fbfcfe); }
.card-title { font-size: 14px; }
.card-body { padding: 14px; }
.sql-card { border: 1px solid #28344a; border-radius: var(--da-radius-md); box-shadow: var(--da-shadow-md); }
.result-table tbody tr:nth-child(even) { background: rgba(244, 247, 251, .55); }
.modal-mask { padding: 20px; backdrop-filter: blur(4px); }
.modal { width: min(420px, 100%); border: 1px solid rgba(255,255,255,.7); border-radius: 18px; box-shadow: var(--da-shadow-lg); }
.input-wrapper { padding-bottom: 17px; background: linear-gradient(to top, var(--da-bg-soft) 66%, rgba(244, 247, 251, 0)); }
.input-box { max-width: 900px; min-height: 58px; padding: 9px 9px 9px 13px; border-color: rgba(196, 207, 221, .95); border-radius: 18px; background: rgba(255, 255, 255, .95); box-shadow: 0 12px 34px rgba(31, 55, 96, .13); backdrop-filter: blur(14px); }
.input-box:focus-within { box-shadow: 0 14px 36px rgba(31, 98, 223, .16), 0 0 0 3px rgba(52, 120, 246, .08); }
.input-hint { max-width: 900px; }
.send-btn { width: 40px; height: 40px; flex-basis: 40px; border-radius: 12px; box-shadow: 0 7px 16px rgba(52, 120, 246, .2); }
.attach-btn { width: 36px; height: 36px; flex-basis: 36px; }
.auth-loading { background: var(--da-bg-soft); }

.conversation-context {
  width: min(900px, calc(100% - 40px));
  min-height: 54px;
  flex: 0 0 auto;
  margin: 16px auto -4px;
  padding: 9px 10px 9px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid rgba(52, 120, 246, .16);
  border-radius: 15px;
  background:
    radial-gradient(circle at 0 0, rgba(52, 120, 246, .1), transparent 38%),
    rgba(255, 255, 255, .9);
  box-shadow: 0 8px 24px rgba(31, 55, 96, .07);
  backdrop-filter: blur(12px);
}
.conversation-context.archived {
  border-color: rgba(155, 105, 28, .22);
  background:
    radial-gradient(circle at 0 0, rgba(230, 166, 68, .12), transparent 38%),
    rgba(255, 253, 248, .94);
}
.conversation-context-icon {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: var(--da-accent-soft);
  color: var(--da-accent-strong);
}
.conversation-context.archived .conversation-context-icon {
  background: #fff3dc;
  color: #98651c;
}
.conversation-context-copy {
  min-width: 0;
  flex: 1;
  display: grid;
  gap: 2px;
}
.conversation-context-copy strong,
.conversation-context-copy span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.conversation-context-copy strong {
  color: var(--da-text);
  font-size: 12px;
}
.conversation-context-copy span {
  color: var(--da-text-muted);
  font-size: 10px;
}
.conversation-context-action {
  min-height: 31px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 0 10px;
  border: 1px solid var(--da-border);
  border-radius: 9px;
  background: #fff;
  color: var(--da-text-secondary);
  font-size: 10px;
  font-weight: 650;
}
.conversation-context-action:hover:not(:disabled) {
  border-color: var(--da-accent);
  color: var(--da-accent-strong);
}
.conversation-context-action.primary {
  border-color: var(--da-accent);
  background: var(--da-accent);
  color: #fff;
}
.history-status-card {
  width: 100%;
  min-height: 50px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 11px 13px;
  border: 1px solid var(--da-border-light);
  border-radius: 12px;
  background: linear-gradient(145deg, #fbfcfe, #f6f8fb);
  color: var(--da-text-muted);
  box-shadow: var(--da-shadow-sm);
}
.history-status-card.succeeded { border-color: rgba(43, 169, 119, .16); color: #28825b; }
.history-status-card.failed,
.history-status-card.cancelled { border-color: rgba(212, 92, 88, .18); color: #b14c49; }
.history-status-card.pending { border-color: rgba(228, 155, 48, .2); color: #9b691e; }
.history-status-card > span {
  min-width: 0;
  display: grid;
  gap: 3px;
}
.history-status-card strong { color: var(--da-text); font-size: 11px; }
.history-status-card small { color: var(--da-text-secondary); font-size: 11px; line-height: 1.5; }
.history-status-card em { color: var(--da-text-muted); font-size: 9px; font-style: normal; }
.input-wrapper.readonly .input-box {
  border-color: rgba(155, 105, 28, .22);
  background: rgba(255, 253, 248, .96);
  box-shadow: 0 8px 26px rgba(92, 69, 34, .08);
}
.input-box textarea:disabled { cursor: not-allowed; color: var(--da-text-muted); }
.spinning { animation: history-spin .8s linear infinite; }
@keyframes history-spin { to { transform: rotate(360deg); } }

@media (max-width: 1180px) {
  .brand { min-width: auto; }
  .brand-copy small, .workspace-label em { display: none; }
  .workspace-label b { max-width: 84px; }
  .tab { padding-inline: 9px; }
}
@media (max-width: 900px) {
  .app-shell { flex-direction: column; }
  .topbar-inner { height: auto; min-height: 64px; padding: 9px 12px; flex-wrap: wrap; gap: 7px; }
  .brand { order: 1; }
  .user-actions { order: 2; margin-left: auto; }
  .new-chat-top { order: 3; }
  .topbar-tabs { order: 4; flex-basis: 100%; padding: 4px; }
  .messages { padding-top: 20px; }
  .welcome { margin-top: 28px; }
}
@media (max-width: 640px) {
  .brand-name { font-size: 14px; }
  .brand-logo { width: 32px; height: 32px; }
  .workspace-label b, .logout-btn { display: none; }
  .workspace-label { width: 30px; height: 30px; padding: 0; justify-content: center; border-radius: 9px; }
  .new-chat-top { width: 34px; height: 34px; padding: 0; justify-content: center; }
  .new-chat-label { display: none; }
  .tab { padding: 6px 9px; }
  .welcome { padding-inline: 4px; }
  .welcome-logo { width: 58px; height: 58px; padding: 14px; border-radius: 18px; }
  .welcome-logo span:nth-child(1) { height: 14px; }
  .welcome-logo span:nth-child(2) { height: 21px; }
  .welcome-logo span:nth-child(3) { height: 28px; }
  .welcome-title { font-size: 26px; }
  .welcome-suggestions { grid-template-columns: 1fr; }
  .row { gap: 8px; }
  .avatar { width: 32px; height: 32px; flex-basis: 32px; }
  .card-head { align-items: flex-start; gap: 8px; }
  .card-actions { flex-wrap: wrap; justify-content: flex-end; }
  .input-wrapper { padding: 10px 10px 12px; }
  .input-box { padding-left: 8px; }
  .steps-detail-label { display: none; }
  .steps-header { gap: 6px; padding-inline: 11px; }
  .steps-progress { display: none; }
  .messages { padding-bottom: 22px; }
  .conversation-context {
    width: calc(100% - 20px);
    margin-top: 10px;
    border-radius: 13px;
  }
  .conversation-context-copy span { display: none; }
  .conversation-context-action { padding-inline: 8px; }
}
</style>
