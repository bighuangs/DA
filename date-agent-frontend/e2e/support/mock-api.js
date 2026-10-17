const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

const embeddingRuntime = () => ({
  runtime_mode: "microbatch",
  runtime_profile: "microbatch_inprocess_v1",
  runtime_profile_fingerprint: "sha256:runtime-profile-demo",
  model_instances: 1,
  scheduling: {
    state: "running",
    queue: { capacity: 128, depth: 0, query_depth: 0, document_depth: 0, running_requests: 0, max_depth_seen: 7 },
    policy: { batch_window_ms: 0, max_batch_texts: 32, max_batch_tokens: 8192, request_deadline_ms: 45000 },
    metrics: {
      requests: { submitted: 28, completed: 28, rejected: 0, cancelled: 0, timed_out: 0, failed: 0 },
      latency_ms: { queue_wait: { p95: 2.4 }, inference: { p95: 38.5 }, total: { p95: 41.2 } },
      batching: { last_batch_texts: 6, max_batch_texts_seen: 12, max_requests_per_batch: 7, merged_batch_count: 4 },
    },
  },
});

export const sampleDataset = {
  dataset_id: "dataset-1",
  asset_id: "asset-1",
  version_id: "dataset-1",
  version_no: 1,
  asset_kind: "managed_upload",
  retention_mode: "managed",
  filename: "sales.csv",
  format: "csv",
  row_count: 2,
  column_count: 2,
  status: "ready",
  schema_fingerprint: "sha256:sample-upload-schema",
  content_checksum: "sha256:sample-upload-content",
  byte_size: 128,
  managed: true,
  lifecycle_revision: 1,
  created_at: "2026-07-28T08:00:00",
  updated_at: "2026-07-28T08:00:00",
  columns: [
    { name: "region", display_name: "地区", type: "string" },
    { name: "amount", display_name: "销售额", type: "number" },
  ],
  preview: [{ 地区: "华东", 销售额: 120000 }, { 地区: "华南", 销售额: 98000 }],
  expires_at: "2099-12-31T23:59:59",
};

function session(role = "owner", username = "alice") {
  const workspace = { id: "workspace-1", name: "销售组", role };
  return {
    access_token: `test-${role}-token`,
    token_type: "bearer",
    user: { id: `user-${username}`, username, display_name: username === "alice" ? "Alice" : username, workspace },
    workspace,
  };
}

function sse(events) {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
}

async function bodyOf(request) {
  try { return request.postDataJSON() || {}; } catch { return {}; }
}

function multipartFilename(request, fallback = "uploaded.csv") {
  try {
    const payload = request.postDataBuffer()?.toString("utf8") || "";
    const match = payload.match(/filename="([^"]+)"/i);
    const raw = match?.[1] || fallback;
    return raw.split(/[\\/]/).at(-1) || fallback;
  } catch {
    return fallback;
  }
}

export async function installMockApi(page, options = {}) {
  const role = options.role || "owner";
  const evaluationDifficulties = {
    easy: { label: "简单", suite: "easy_sql_v1", suiteLabel: "简单 SQL 基线集", runId: "20260805_easy_001", total: 50, resultRate: .54, tableRate: .86, executionRate: 1, latencyP95: 13037.2, attempts: 35, tokensMean: 2246.3, cost: null },
    medium: { label: "中等", suite: "medium_sql_v1", suiteLabel: "中等 SQL 基线集", runId: "20260805_medium_001", total: 50, resultRate: .3, tableRate: .98, executionRate: 1, latencyP95: 12374.2, attempts: 37, tokensMean: 2457.8, cost: null },
    hard: { label: "困难", suite: "hard_sql_v1", suiteLabel: "困难 SQL 基线集", runId: "20260805_hard_001", total: 50, resultRate: .16, tableRate: .8, executionRate: .6, latencyP95: 23300.2, attempts: 72, tokensMean: 6144.3, cost: null },
  };
  const freshEvaluationGovernance = {
    easy: {
      experimentId: "todo97-finalflow2-easy-d0-20260809-v1",
      runIds: ["20260809_122528_864451", "20260809_123150_579043", "20260809_123828_451550"],
      results: [48, 49, 47], tables: [46, 47, 45], executions: [50, 50, 48],
      scopeSafe: [50, 50, 48], scopeEvidence: [50, 50, 48], coreFailures: [0, 0, 2],
      status: "incomplete", complete: false, recordable: false, qualified: false,
      checkOverrides: { table_scope_safety: false, table_scope_evidence_complete: false, result_floor_non_regression: false, failed_llm_calls_zero: false },
    },
    medium: {
      experimentId: "todo97-finalflow2-medium-d0-20260809-v1",
      runIds: ["20260809_125256_036055", "20260809_130112_634378", "20260809_130914_348751"],
      results: [46, 46, 46], tables: [50, 50, 49], executions: [50, 50, 50],
      scopeSafe: [50, 50, 50], scopeEvidence: [50, 50, 50], coreFailures: [0, 0, 0],
      status: "qualified", complete: true, recordable: true, qualified: true,
      checkOverrides: {},
    },
    hard: {
      experimentId: "todo97-finalflow2-hard-d0-20260809-v1",
      runIds: ["20260809_132534_667976", "20260809_133958_909883", "20260809_135357_495896"],
      results: [18, 19, 22], tables: [38, 41, 37], executions: [38, 34, 36],
      scopeSafe: [48, 48, 48], scopeEvidence: [48, 48, 48], coreFailures: [0, 0, 0],
      status: "incomplete", complete: false, recordable: false, qualified: false,
      checkOverrides: { table_scope_safety: false, table_scope_evidence_complete: false, semantic_review_first_correct_final_incorrect_zero: false, semantic_review_same_sql_execution_loss_zero: false },
    },
  };
  const freshGovernanceFor = (difficulty) => options.evaluationFreshGovernance
    ? freshEvaluationGovernance[difficulty] || freshEvaluationGovernance.easy
    : null;
  const evaluationFor = (difficulty) => {
    const base = evaluationDifficulties[difficulty] || evaluationDifficulties.easy;
    const fresh = freshGovernanceFor(difficulty);
    if (!fresh) return base;
    const latest = fresh.results.length - 1;
    return {
      ...base,
      runId: fresh.runIds[latest],
      resultRate: fresh.results[latest] / base.total,
      tableRate: fresh.tables[latest] / base.total,
      executionRate: fresh.executions[latest] / base.total,
    };
  };
  const state = {
    session: session(role, options.username || "alice"),
    datasets: options.datasets ? [...options.datasets] : [structuredClone(sampleDataset)],
    metrics: [],
    training: [],
    tables: [{
      id: "table-1", name: "fact_sales", role: "fact", description: "销售事实表",
      columns: [
        { id: "fact_sales.region", name: "region", type: "varchar", role: "dimension", description: "销售地区", sync: true },
        { id: "column-2", name: "amount", type: "decimal", role: "measure", description: "销售额" },
      ],
    }],
    maskingRules: [],
    accessPolicies: [],
    conversations: structuredClone(options.conversations || []),
    conversationTurns: structuredClone(options.conversationTurns || {}),
    baselineStatus: Object.fromEntries(Object.keys(evaluationDifficulties).map((key) => [key, options.evaluationEligible || options.evaluationFreshGovernance ? "candidate" : "unreviewed"])),
    approvedBaselines: Object.fromEntries(Object.entries(evaluationDifficulties).map(([key, item]) => [key, { ...item, runId: `baseline_${key}_approved` }])),
    activeEvaluationDifficulty: "easy",
    feedbackReviewStatus: "pending",
    feedbackCategory: null,
    candidateStatus: "none",
    candidateType: null,
    hybridStatus: options.hybridStatus || null,
    valueIndexStatus: {
      exists: true, ready: true, workspace_id: "workspace-1",
      alias: "data-agent-value-index-workspace-1",
      index_version: "sha256:value-index-v2-demo",
      document_count: 7, responsibility: "es_field_value",
      contract_version: "value-index-v2", mapping_version: "value-mapping-v2",
      query_modes: ["exact", "prefix", "full_text"], sync_column_count: 1,
      policy_precheck_allowed_count: 1, policy_precheck_blocked: [],
      retained_versions: [
        { physical_index: "data-agent-value-index--202607240100-current", index_version: "sha256:value-index-v2-demo", document_count: 7, active: true },
        { physical_index: "data-agent-value-index--202607230100-previous", index_version: "sha256:value-index-v1-demo", document_count: 6, active: false },
      ],
    },
    dashboards: options.emptyDashboards ? [] : structuredClone(options.dashboards || [{ id: "dashboard-1", name: "销售看板", node_type: "dashboard", pid: null }]),
    dashboardBlocks: options.emptyDashboards ? [] : structuredClone(options.dashboardBlocks || [{
      x: 0, y: 0, w: 6, h: 6,
      query: "各地区销售额", sql: "SELECT region, SUM(amount) AS amount FROM fact_sales GROUP BY region",
      chart: { type: "column", title: "地区销售额", axis: { x: { name: "地区", value: "region" }, y: [{ name: "销售额", value: "amount" }] } },
      rows: [{ region: "华东", amount: 120000 }, { region: "华南", amount: 98000 }],
    }]),
    memoryStatus: {
      chat_id: "chat-1",
      schema_version: "memory-context-v2",
      working_memory: { available: true, exists: true, ttl_seconds: 86342 },
      checkpoint: { exists: true, version: 2, updated_at: "2026-07-24T10:00:00" },
      long_term: {
        active_count: 4,
        current_chat_count: 1,
        by_type: { preference: 1, query_episode: 1, profile_fact: 1, long_term_goal: 1 },
        items: [
          {
            memory_id: "memory-pref-1",
            memory_type: "preference",
            content: "以后默认回答用中文",
            source_chat_id: "chat-older",
            status: "active",
            verification_status: "confirmed",
            created_at: "2026-07-24T09:00:00",
          },
          {
            memory_id: "memory-episode-1",
            memory_type: "query_episode",
            content: "各地区的销售总额",
            source_chat_id: "chat-1",
            status: "active",
            verification_status: "executed",
            created_at: "2026-07-24T10:00:00",
          },
          {
            memory_id: "memory-profile-1",
            memory_type: "profile_fact",
            content: "用户负责华东销售分析",
            source_chat_id: "chat-1",
            status: "active",
            verification_status: "confirmed",
            created_at: "2026-07-24T10:01:00",
          },
          {
            memory_id: "memory-goal-1",
            memory_type: "long_term_goal",
            content: "持续追踪区域销售趋势",
            source_chat_id: "chat-1",
            status: "active",
            verification_status: "confirmed",
            created_at: "2026-07-24T10:02:00",
          },
        ],
      },
      summary_memory: {
        enabled: true,
        active_segment_count: 2,
        by_level: { L0: 2 },
        signal_count: 7,
        covered_through_message_id: 18,
        watermark_version: 2,
        source_tokens: 240,
        summary_tokens: 72,
        compression_ratio: 0.3,
        extractor_version: "deterministic-extractive-v1",
        items: [
          { summary_id: "summary-1", level: 0, content: "用户持续查看各地区销售额，并要求后续按品类拆分。", source_start_message_id: 1, source_end_message_id: 8, source_message_count: 8, summary_tokens: 34 },
          { summary_id: "summary-2", level: 0, content: "用户确认默认使用中文，并关注华东区域趋势。", source_start_message_id: 9, source_end_message_id: 18, source_message_count: 10, summary_tokens: 38 },
        ],
      },
      projection: {
        outbox: { backlog: 0, by_target: { es: { completed: 2 }, qdrant: { completed: 2 }, summary_es: { completed: 2 }, summary_qdrant: { completed: 2 } } },
        worker: { running: true, processed: 8, failed: 0 },
        elasticsearch: { engine: "elasticsearch", alias_ready: true, document_count: 2 },
        qdrant: { engine: "qdrant", exists: true, points_count: 2 },
        summary_elasticsearch: { engine: "summary_elasticsearch", alias_ready: true, document_count: 2 },
        summary_qdrant: { engine: "summary_qdrant", exists: true, points_count: 2 },
      },
    },
    dataContexts: {},
    calls: [],
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    state.calls.push({ method, path, query: Object.fromEntries(url.searchParams.entries()), body: await bodyOf(request) });

    const ok = (body, status = 200) => route.fulfill({ status, headers: jsonHeaders, body: JSON.stringify(body) });
    const noContent = () => route.fulfill({ status: 204, body: "" });

    if (path === "/api/auth/me") {
      return options.expiredToken ? ok({ detail: "登录已失效" }, 401) : ok({ user: state.session.user, workspace: state.session.workspace });
    }
    if (path === "/api/auth/login" || path === "/api/auth/register") return ok(state.session, path.endsWith("register") ? 201 : 200);
    if (path === "/api/auth/switch-workspace") return ok(state.session);

    if (path === "/api/chats" && method === "GET") {
      const status = url.searchParams.get("status") || "active";
      const items = state.conversations
        .filter((item) => item.status === status)
        .sort((left, right) => String(right.last_message_at || "").localeCompare(String(left.last_message_at || "")));
      return ok({
        items,
        next_cursor: null,
        has_more: false,
        status,
        retention_days: 30,
      });
    }
    const turnsMatch = path.match(/^\/api\/chats\/([^/]+)\/turns$/);
    if (turnsMatch && method === "GET") {
      const chatId = decodeURIComponent(turnsMatch[1]);
      const conversation = state.conversations.find((item) => item.chat_id === chatId);
      if (!conversation) return ok({ detail: "会话不存在或无权访问" }, 404);
      return ok({
        conversation,
        items: state.conversationTurns[chatId] || [],
        next_cursor: null,
        has_more: false,
        retention_days: 30,
      });
    }
    const dataContextMatch = path.match(/^\/api\/chats\/([^/]+)\/data-context$/);
    if (dataContextMatch && method === "GET") {
      const chatId = decodeURIComponent(dataContextMatch[1]);
      return ok(state.dataContexts[chatId] || {
        chat_id: chatId,
        source_mode: "dw_only",
        dataset_ids: [],
        comparison_intent: null,
        mapping_confirmation_id: null,
        revision: 0,
        invalidation_reason: null,
      });
    }
    if (dataContextMatch && method === "PUT") {
      const chatId = decodeURIComponent(dataContextMatch[1]);
      const payload = await bodyOf(request);
      const current = state.dataContexts[chatId] || { revision: 0 };
      if (Number(payload.expected_revision || 0) !== Number(current.revision || 0)) {
        return ok({ detail: "数据上下文版本冲突，请刷新后重试" }, 409);
      }
      const revision = Number(current.revision || 0) + 1;
      state.dataContexts[chatId] = {
        chat_id: chatId,
        source_mode: payload.source_mode,
        dataset_ids: payload.dataset_ids || [],
        comparison_intent: payload.comparison_intent || null,
        mapping_confirmation_id: payload.mapping_confirmation_id || null,
        revision,
        invalidation_reason: null,
        resolved_context: {
          schema_version: "data-source-context-v1",
          source_mode: payload.source_mode,
          comparison_intent: payload.comparison_intent || null,
          revision,
          datasets: (payload.dataset_ids || []).map((datasetId, index) => ({
            dataset_id: datasetId,
            version_id: datasetId,
            logical_name: `upload_${index + 1}_dataset`,
            state: "ready",
          })),
        },
      };
      return ok(state.dataContexts[chatId]);
    }
    const dataContextResetMatch = path.match(/^\/api\/chats\/([^/]+)\/data-context\/reset$/);
    if (dataContextResetMatch && method === "POST") {
      const chatId = decodeURIComponent(dataContextResetMatch[1]);
      const current = state.dataContexts[chatId] || { revision: 0 };
      const revision = Number(current.revision || 0) + 1;
      state.dataContexts[chatId] = {
        chat_id: chatId,
        source_mode: "dw_only",
        dataset_ids: [],
        comparison_intent: null,
        mapping_confirmation_id: null,
        revision,
        invalidation_reason: null,
      };
      return ok(state.dataContexts[chatId]);
    }
    const conversationMatch = path.match(/^\/api\/chats\/([^/]+)$/);
    if (conversationMatch && method === "PATCH") {
      const chatId = decodeURIComponent(conversationMatch[1]);
      const conversation = state.conversations.find((item) => item.chat_id === chatId);
      if (!conversation) return ok({ detail: "会话不存在或无权访问" }, 404);
      const payload = await bodyOf(request);
      if (payload.title) conversation.title = payload.title;
      if (payload.status) {
        conversation.status = payload.status;
        conversation.archived_at = payload.status === "archived" ? new Date().toISOString() : null;
      }
      conversation.version = Number(conversation.version || 0) + 1;
      return ok(conversation);
    }
    if (conversationMatch && method === "DELETE") {
      const chatId = decodeURIComponent(conversationMatch[1]);
      const index = state.conversations.findIndex((item) => item.chat_id === chatId);
      if (index < 0) return ok({ detail: "会话不存在或无权访问" }, 404);
      state.conversations.splice(index, 1);
      delete state.conversationTurns[chatId];
      return noContent();
    }

    const viewerWrite = role === "viewer" && !["GET", "HEAD", "OPTIONS"].includes(method)
      && (path.startsWith("/api/admin/") || path.startsWith("/api/dashboard"));
    if (viewerWrite) return ok({ detail: "当前角色无权执行此操作" }, 403);

    if (path === "/api/query" && method === "POST") {
      if (options.queryObjectError) {
        return ok({ detail: [{ loc: ["body", "query"], msg: "问题不能为空", type: "value_error" }] }, 422);
      }
      if (options.queryError) return ok({ detail: "受控查询失败" }, 503);
      if (options.queryClarification) {
        return route.fulfill({
          status: 200,
          headers: {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache",
          },
          body: sse([
            { chat_id: "chat-1" },
            { stage: "生成 SQL" },
            {
              clarification: {
                schema_version: "model-clarification-v1",
                action: "clarify_then_retry",
                message: "当前未能在安全时限内完成完整 SQL。",
                suggestions: ["明确要统计的指标及聚合方式", "明确分组维度、时间范围和筛选条件"],
                unresolved_slots: ["operator:set", "filterop:category:negated"],
              },
            },
            { error: "当前未能在安全时限内完成完整 SQL。" },
            { done: true },
          ]),
        });
      }
      const enableAnalysis = Boolean(state.calls.at(-1)?.body?.enable_analysis);
      const queryBody = state.calls.at(-1)?.body || {};
      const activeChatId = queryBody.chat_id || "chat-1";
      if (options.intentResponse) {
        const intentResponse = structuredClone(options.intentResponse);
        const leakedArtifacts = [
          { sql: "SELECT * FROM should_not_render" },
          { result: [{ should_not_render: true }] },
        ];
        const terminalChunks = options.artifactsBeforeIntentResponse
          ? [...leakedArtifacts, { intent_response: intentResponse }]
          : [{ intent_response: intentResponse }, ...leakedArtifacts];
        return route.fulfill({
          status: 200,
          headers: {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache",
          },
          body: sse([
            { chat_id: activeChatId },
            { stage: "识别问题意图" },
            // 同时支持终态前后两种异常顺序，验证浏览器终态隔离和补偿清理。
            ...terminalChunks,
            {
              trace: {
                trace_id: "trace-intent-response",
                status: "handled",
                tt_result_ms: null,
                tt_complete_ms: 18,
                duration_ms: 18,
                llm_calls: 0,
                total_tokens: 0,
                estimated_cost: 0,
                cost_currency: "USD",
              },
            },
            { done: true },
          ]),
        });
      }
      const isFollowUp = queryBody.query === "那销售额呢";
      const isTopProducts = queryBody.query === "销量前十的商品";
      const isPieChartFollowUp = queryBody.query === "改为饼图";
      const isProductChartQuery = isTopProducts || isPieChartFollowUp;
      const sourceMode = queryBody.source_mode || "dw_only";
      const usesUpload = Array.isArray(queryBody.dataset_ids) && queryBody.dataset_ids.length > 0;
      const isIndependentCompare = (
        sourceMode === "upload_compare_dw"
        && queryBody.comparison_intent === "independent_aggregate_compare"
      );
      const sql = isIndependentCompare
        ? "SELECT 'upload' AS source_label, c_001 AS region, SUM(c_002) AS amount FROM upload_1_dataset GROUP BY c_001 UNION ALL SELECT 'dw' AS source_label, region, SUM(amount) AS amount FROM fact_sales GROUP BY region"
        : usesUpload
        ? "SELECT c_001 AS region, SUM(c_002) AS amount FROM upload_1_dataset GROUP BY c_001"
        : isProductChartQuery
          ? "SELECT product_name, SUM(order_quantity) AS total_quantity FROM fact_orders GROUP BY product_name ORDER BY total_quantity DESC LIMIT 10"
        : isFollowUp
          ? "SELECT category, SUM(amount) AS amount FROM fact_sales GROUP BY category"
          : "SELECT region, SUM(amount) AS amount FROM fact_sales GROUP BY region";
      const queryResult = isIndependentCompare
        ? [
          { source_label: "dw", region: "华东", amount: 100000 },
          { source_label: "dw", region: "华南", amount: 80000 },
          { source_label: "upload", region: "华东", amount: 125000 },
          { source_label: "upload", region: "华北", amount: 40000 },
        ]
        : isProductChartQuery
          ? [
            { product_name: "商品 A", total_quantity: 17000 },
            { product_name: "商品 B", total_quantity: 16800 },
            { product_name: "商品 C", total_quantity: 15900 },
            { product_name: "商品 D", total_quantity: 15800 },
            { product_name: "商品 E", total_quantity: 15000 },
            { product_name: "商品 F", total_quantity: 14900 },
            { product_name: "商品 G", total_quantity: 14800 },
            { product_name: "商品 H", total_quantity: 14700 },
            { product_name: "商品 I", total_quantity: 14600 },
            { product_name: "商品 J", total_quantity: 14500 },
          ]
        : [{ region: "华东", amount: 120000 }, { region: "华南", amount: 98000 }];
      const chart = isPieChartFollowUp
        ? {
          type: "pie",
          title: "销量前十的商品，改为饼图",
          axis: {
            series: { name: "商品", value: "product_name" },
            y: { name: "销量", value: "total_quantity" },
          },
        }
        : isTopProducts
          ? {
            type: "column",
            title: "销量前十的商品",
            axis: {
              x: { name: "商品", value: "product_name" },
              y: [{ name: "销量", value: "total_quantity" }],
            },
          }
          : {
            type: "column",
            title: "地区销售额",
            axis: {
              x: { name: "地区", value: "region" },
              y: [{ name: "销售额", value: "amount" }],
            },
          };
      const sourceComparison = isIndependentCompare
        ? {
          contract: "deterministic-source-comparator-v2",
          status: "compared",
          mode: "grouped",
          source_column: "source_label",
          measure: "amount",
          dimensions: ["region"],
          baseline_source: "dw",
          comparison_source: "upload",
          baseline_value: 180000,
          comparison_value: 165000,
          delta: -15000,
          delta_pct: -8.3333,
          compared_key_count: 3,
          matched_key_count: 1,
          baseline_missing_count: 1,
          comparison_missing_count: 1,
          comparisons: [
            { dimensions: { region: "华东" }, baseline_value: 100000, comparison_value: 125000, delta: 25000, delta_pct: 25 },
            { dimensions: { region: "华北" }, baseline_value: null, comparison_value: 40000, delta: null, delta_pct: null, baseline_missing: true },
            { dimensions: { region: "华南" }, baseline_value: 80000, comparison_value: null, delta: null, delta_pct: null, comparison_missing: true },
          ],
          row_level_join_used: false,
        }
        : null;
      const sourceContext = {
        schema_version: "data-source-context-v1",
        source_mode: sourceMode,
        comparison_intent: queryBody.comparison_intent || null,
        mapping_confirmation_id: queryBody.mapping_confirmation_id || null,
        revision: Number(queryBody.data_context_revision || 0),
        context_fingerprint: "sha256:mock-data-context",
        official_table_count: sourceMode === "upload_only" || sourceMode === "multi_upload" ? 0 : 1,
        datasets: (queryBody.dataset_ids || []).map((datasetId, index) => ({
          dataset_id: datasetId,
          version_id: datasetId,
          logical_name: `upload_${index + 1}_dataset`,
          kind: "managed_upload",
          state: "ready",
          schema_fingerprint: "sha256:sample-upload-schema",
        })),
        profiles: (queryBody.dataset_ids || []).map((datasetId, index) => ({
          dataset_id: datasetId,
          logical_name: `upload_${index + 1}_dataset`,
          row_count: 2,
          column_count: 2,
          bounded: true,
          raw_rows_exposed: false,
        })),
      };
      const events = [
            { chat_id: activeChatId },
            { stage: "解析数据源上下文" },
            { source_context_resolved: sourceContext },
            ...(usesUpload ? [
              { stage: "剖析上传 Schema" },
              { source_profile: { dataset_count: queryBody.dataset_ids.length, global_dw_calls: sourceMode === "upload_only" ? 0 : 1, bounded: true } },
            ] : []),
            { memory_context: {
              schema_version: "memory-context-v2",
              working_source: "redis",
              working_version: 1,
              redis_hit: true,
              recent_selected: 2,
              preference_count: 1,
              episode_count: 1,
              profile_fact_count: 1,
              long_term_goal_count: 1,
              summary_count: 1,
              summary_status: "selected",
              summary_trigger: "long_range_cue",
              recent_messages_in_prompt: true,
              es_candidates: 2,
              qdrant_candidates: 2,
              fusion_method: "memory_id_rrf",
              authoritative_recheck: true,
              degradations: [],
            } },
            ...(isFollowUp ? [
              { stage: "闭合多轮指代" },
              { follow_up: { action: "replace", summary: "follow_up_action=replace；变更 指标；上一轮维度 1 项已纳入闭合校验", standalone_question: "各品类的销售额总和", deterministic: true } },
            ] : []),
            { stage: "提取关键字" },
            { stage: "评估规划必要性" },
            { planning_route: {
              contract_version: "planning-route-v1",
              profile: "keyword_length_v1",
              decision: "direct_generate",
              actual_action: "direct_generate",
              score: 0,
              confidence: .84,
              reason_codes: ["legacy_no_trigger"],
              decision_ms: .23,
              raw_text_logged: false,
              shadow_decisions: [
                { profile: "structured_rules_v1", decision: "direct_generate", score: .16 },
                { profile: "weighted_scorecard_v1", decision: "direct_generate", score: .34 },
              ],
            } },
            { stage: "生成 SQL" },
            { stage: "评估 SQL 置信度" },
      { candidate_decision: { version: "candidate-confidence-v2", adaptive_triggered: false, trigger_reason: "high_confidence_fast_path", selected: "A", selection_reason: "primary_fast_path", candidate_count: 1, candidates: [{ label: "A", valid: true, score: .91, band: "high", weaknesses: [] }] } },
            { semantic_repair: { contract_version: "semantic-repair-result-v1", registry_version: "semantic-repair-registry-v1", rule_set_version: "semantic-repair-rules-v1", profile: "registry_same_rules_v1", changed: false, original_sql_hash: "sha256:query-demo", repaired_sql_hash: "sha256:query-demo", applied_rules: [], skipped_rules: [{ rule_code: "join.unique_metadata_edge", rule_version: "1", status: "skipped", reason: "no_unique_invalid_join" }], repair_ms: .42, fallback_used: false, invariants: { single_bounded_pass: true, table_set_unchanged: true, filter_literals_unchanged: true, read_only_preserved: true }, raw_sql_logged: false } },
            { sql },
            { result: queryResult },
            ...(sourceComparison ? [{ source_comparison: sourceComparison }] : []),
            { chart },
            ...(enableAnalysis ? [{ analysis: "华东地区销售额最高，领先华南 22,000 元。" }] : []),
            {
              recommendation_contract: { schema_version: "recommendation-options/v2" },
              recommendations: [
                {
                  id: "rec-monthly-trend",
                  label: "查看月度趋势",
                  query: "用折线图展示各地区每月销售总额趋势",
                  kind: "trend",
                },
                {
                  id: "rec-east-category-breakdown",
                  label: "分析品类贡献",
                  query: "查询华东地区各品类销售额，并按销售额从高到低排序",
                  kind: "breakdown",
                },
                {
                  id: "rec-east-customer-ranking",
                  label: "查看客户排行",
                  query: "查询华东地区销售额最高的10位客户",
                  kind: "ranking",
                },
              ],
              recommend: ["查看月度趋势", "分析品类贡献", "查看客户排行"],
            },
            { memory_write: { mysql_committed: true, checkpoint_version: 2, memory_items_created: 1, outbox_events_created: 2, redis_status: "updated" } },
            { trace: { trace_id: "trace-12345678", status: "success", tt_result_ms: 80, tt_complete_ms: 120, duration_ms: 120, llm_calls: 2, total_tokens: 321, estimated_cost: 0.0012, cost_currency: "USD", candidate_count: 1, candidate_confidence: .91, candidate_band: "high", candidate_decision: { selected: "A" }, data_source_context: sourceContext, planning_route: { profile: "keyword_length_v1", actual_action: "direct_generate", decision_ms: .23, reason_codes: ["legacy_no_trigger"], raw_text_logged: false }, follow_up_action: isFollowUp ? "replace" : null, follow_up_summary: isFollowUp ? "follow_up_action=replace；变更 指标" : null, slowest_nodes: [{ node_id: "decide_plan_route", duration_ms: .23 }, { node_id: "generate_sql", duration_ms: 42 }], retrieval_metrics: usesUpload && sourceMode === "upload_only" ? [{ route: "upload_profile", responsibility: "upload_schema", selected_count: 2, global_dw_calls: 0 }] : [{ route: "column", embedding_mode: "dense_sparse", fusion_method: "entity_rrf", selected_count: 4 }, { route: "value", responsibility: "es_field_value", engine: "elasticsearch", index_version: "sha256:value-index-v2-demo", query_modes: ["exact"], workspace_scope_applied: true, sparse_literal_inference: false, value_search_audits: [{ query_mode: "exact", query_fingerprint: "sha256:test", hit_count: 1, latency_ms: 8, raw_value_logged: false }] }] } },
            { done: true },
          ];
      const now = new Date().toISOString();
      let conversation = state.conversations.find((item) => item.chat_id === activeChatId);
      if (!conversation) {
        conversation = {
          chat_id: activeChatId,
          title: String(queryBody.query || "未命名会话").slice(0, 80),
          preview: String(queryBody.query || "").slice(0, 160),
          status: "active",
          last_turn_status: "succeeded",
          turn_count: 0,
          created_at: now,
          last_message_at: now,
          updated_at: now,
          archived_at: null,
          context_reset_at: null,
          version: 1,
        };
        state.conversations.push(conversation);
      }
      conversation.preview = String(queryBody.query || "").slice(0, 160);
      conversation.last_turn_status = "succeeded";
      conversation.turn_count = Number(conversation.turn_count || 0) + 1;
      conversation.last_message_at = now;
      conversation.updated_at = now;
      conversation.version = Number(conversation.version || 0) + 1;
      state.conversationTurns[activeChatId] ||= [];
      state.conversationTurns[activeChatId].push({
        turn_id: `turn-${activeChatId}-${conversation.turn_count}`,
        chat_id: activeChatId,
        trace_id: "trace-12345678",
        status: "succeeded",
        created_at: now,
        completed_at: now,
        user_text: queryBody.query,
        assistant_summary: "查询成功，返回 2 行。",
        sql,
        result: {
          columns: ["region", "amount"],
          rows: [{ region: "华东", amount: 120000 }, { region: "华南", amount: 98000 }],
          row_count: 2,
          truncated: false,
        },
        replay_fidelity: "result_snapshot",
        unavailable_parts: ["chart", "analysis", "recommendation"],
        data_source_context: sourceContext,
      });
      if (options.queryDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.queryDelayMs));
      }
      return route.fulfill({ status: 200, headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" }, body: sse(events) });
    }
    if (path === "/api/query/exec" && method === "POST") {
      if (options.execError) return ok({ detail: "SQL 只允许只读查询" }, 400);
      return ok({ columns: ["region", "amount"], rows: [{ region: "华北", amount: 86000 }] });
    }
    if (path === "/api/query/export") return route.fulfill({ status: 200, headers: { "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }, body: "xlsx" });
    if (path === "/api/feedback") {
      if (options.feedbackObjectError) {
        return ok({ detail: { message: "反馈服务暂时不可用", reason: "write_rejected" } }, 503);
      }
      return ok({ id: "feedback-1" }, 201);
    }
    if (/^\/api\/chat\/[^/]+\/memory$/.test(path) && method === "GET") {
      return ok(state.memoryStatus);
    }
    if (/^\/api\/chat\/[^/]+\/memory$/.test(path) && method === "DELETE") {
      const includeLongTerm = url.searchParams.get("include_long_term") === "true";
      const revoked = includeLongTerm
        ? state.memoryStatus.long_term.current_chat_count
        : 0;
      state.memoryStatus.working_memory.exists = false;
      state.memoryStatus.working_memory.ttl_seconds = -2;
      state.memoryStatus.checkpoint = { exists: false, version: 0, updated_at: "" };
      if (includeLongTerm) {
        state.memoryStatus.long_term.items = state.memoryStatus.long_term.items.filter(
          (item) => item.source_chat_id !== "chat-1",
        );
        state.memoryStatus.long_term.active_count = state.memoryStatus.long_term.items.length;
        state.memoryStatus.long_term.current_chat_count = 0;
      }
      return ok({
        chat_id: "chat-1",
        redis_deleted: true,
        checkpoint_deleted: 1,
        working_messages_reset: 2,
        archived: { messages: 2, history: 0 },
        long_term_revoked: revoked,
      });
    }

    if (path === "/api/dashboard/tree" && method === "GET") return ok(state.dashboards);
    if (path === "/api/dashboard" && method === "POST") {
      const payload = await bodyOf(request);
      const node = { id: `dashboard-${state.dashboards.length + 1}`, name: payload.name, node_type: payload.node_type, pid: payload.pid ?? null };
      state.dashboards.push(node); return ok(node, 201);
    }
    if (path === "/api/dashboard/dashboard-1" && method === "GET") return ok({ id: "dashboard-1", name: "销售看板", component_data: state.dashboardBlocks });
    if (/^\/api\/dashboard\/[^/]+$/.test(path) && method === "GET") {
      const id = path.split("/").at(-1);
      const node = state.dashboards.find((item) => item.id === id);
      const delayMs = Number(options.dashboardGetDelays?.[id] || 0);
      if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
      const componentData = options.dashboardBlocksById?.[id] || [];
      return node ? ok({ id, name: node.name, component_data: structuredClone(componentData) }) : ok({ detail: "仪表盘不存在" }, 404);
    }
    if (/^\/api\/dashboard\/[^/]+\/block$/.test(path) && method === "POST") return ok({ id: "block-2" }, 201);
    if (path === "/api/dashboard/dashboard-1" && method === "PUT") {
      const payload = await bodyOf(request);
      if (Array.isArray(payload.component_data)) state.dashboardBlocks = structuredClone(payload.component_data);
      if (payload.name) {
        const node = state.dashboards.find((item) => item.id === "dashboard-1");
        if (node) node.name = payload.name;
      }
      return ok({ id: "dashboard-1" });
    }
    if (/^\/api\/dashboard\/[^/]+$/.test(path) && method === "PUT") {
      const id = path.split("/").at(-1);
      const payload = await bodyOf(request);
      const node = state.dashboards.find((item) => item.id === id);
      if (node && payload.name) node.name = payload.name;
      return ok({ id });
    }
    if (/^\/api\/dashboard\/[^/]+$/.test(path) && method === "DELETE") {
      const id = path.split("/").at(-1);
      const removed = new Set([id]);
      let changed = true;
      while (changed) {
        changed = false;
        for (const item of state.dashboards) {
          if (!removed.has(item.id) && removed.has(item.pid)) {
            removed.add(item.id);
            changed = true;
          }
        }
      }
      state.dashboards = state.dashboards.filter((item) => !removed.has(item.id));
      return noContent();
    }

    if (path === "/api/admin/meta/overview") return ok({ metrics: state.metrics.length, terminology: 0, training: state.training.length, tables: state.tables.length });
    if (path === "/api/admin/meta/metrics" && method === "GET") return ok(state.metrics);
    if (path === "/api/admin/meta/metrics" && method === "POST") {
      const payload = await bodyOf(request);
      state.metrics.push({ id: payload.name, kind: "metric", ...payload }); return ok(state.metrics.at(-1), 201);
    }
    if (/^\/api\/admin\/meta\/metrics\//.test(path) && method === "PATCH") return ok({});
    if (/^\/api\/admin\/meta\/metrics\//.test(path) && method === "DELETE") {
      const id = decodeURIComponent(path.split("/").at(-1));
      state.metrics = state.metrics.filter((item) => String(item.id) !== id);
      return noContent();
    }
    if (path === "/api/admin/meta/training" && method === "GET") return ok(state.training);
    if (path === "/api/admin/meta/tables") return ok(state.tables);
    if (path === "/api/admin/meta/value-index/status") return ok(state.valueIndexStatus);
    if (path === "/api/admin/meta/value-index/search") return ok({
      items: [{
        id: "value:hashed", document_id: "value:hashed", value: "华东",
        normalized_value: "华东", table_id: "fact_sales",
        table_name: "fact_sales", column_id: "fact_sales.region",
        column_name: "region", type: "varchar", query_mode: url.searchParams.get("query_mode") || "exact",
        index_version: state.valueIndexStatus.index_version, is_active: true,
        workspace_id: "workspace-1", indexed_at: "2026-07-24T00:00:00Z",
      }],
      total: 1, page: 1, page_size: 50,
      evidence: {
        schema: "value-search-audit-v1", responsibility: "es_field_value",
        engine: "elasticsearch", workspace_id: "workspace-1",
        query_mode: url.searchParams.get("query_mode") || "exact",
        index_version: state.valueIndexStatus.index_version,
        filter_stage: "elasticsearch_query", hit_count: 1,
        returned_count: 1, latency_ms: 8, raw_value_logged: false, status: "ok",
      },
    });
    if (path === "/api/admin/meta/value-index/rollback" && method === "POST") return ok(state.valueIndexStatus);
    if (path === "/api/admin/meta/hybrid/status") return ok(state.hybridStatus || {
      active_mode: "bge_m3_dense_sparse", replacement_strategy: "overwrite_existing_collections",
      v1_fallback: false, ready: true,
      runtime: { model: "BAAI/bge-m3", device: "cpu", dense_size: 1024, embedding_mode: "dense_sparse", ...embeddingRuntime() },
      collections: {
        column: { exists: true, compatible: true, content_compatible: true, matching_points_count: 24, expected_points_count: 24, points_count: 24 },
        metric: { exists: true, compatible: true, content_compatible: true, matching_points_count: 6, expected_points_count: 6, points_count: 6 },
        training: { exists: true, compatible: true, content_compatible: true, matching_points_count: 5, expected_points_count: 5, points_count: 5 },
      },
      point_projection: {
        active_strategy: "single_canonical", identity_version: "point-identity-v2",
        canonical_template_version: "entity-canonical-v1", alias_policy_version: "approved-alias-v1",
        approved_alias_count: 4, supported_strategies: ["single_canonical", "multi_all", "mixed_selective"],
        routes: {
          column: { entity_count: 24, point_count: 24, points_per_entity: 1, point_roles: { canonical: 24 } },
          metric: { entity_count: 6, point_count: 6, points_per_entity: 1, point_roles: { canonical: 6 } },
          training: { entity_count: 5, point_count: 5, points_per_entity: 1, point_roles: { question: 5 } },
        },
      },
    });
    if (path === "/api/admin/meta/rebuild") return ok({
      embedding_mode: "dense_sparse", replacement_strategy: "overwrite_existing_collections",
      v1_fallback: false,
      indexed: { columns: 25, metrics: state.metrics.length, training: state.training.length },
      status: {
        active_mode: "bge_m3_dense_sparse", replacement_strategy: "overwrite_existing_collections",
        v1_fallback: false, ready: true,
        runtime: { model: "BAAI/bge-m3", device: "cpu", dense_size: 1024, ...embeddingRuntime() },
        collections: {
          column: { exists: true, compatible: true, content_compatible: true, matching_points_count: 24, expected_points_count: 24, points_count: 24 },
          metric: { exists: true, compatible: true, content_compatible: true, matching_points_count: 6, expected_points_count: 6, points_count: 6 },
          training: { exists: true, compatible: true, content_compatible: true, matching_points_count: 5, expected_points_count: 5, points_count: 5 },
        },
        point_projection: {
          active_strategy: "single_canonical", identity_version: "point-identity-v2",
          canonical_template_version: "entity-canonical-v1", alias_policy_version: "approved-alias-v1",
          approved_alias_count: 4, supported_strategies: ["single_canonical", "multi_all", "mixed_selective"],
          routes: {
            column: { entity_count: 24, point_count: 24, points_per_entity: 1, point_roles: { canonical: 24 } },
            metric: { entity_count: 6, point_count: 6, points_per_entity: 1, point_roles: { canonical: 6 } },
            training: { entity_count: 5, point_count: 5, points_per_entity: 1, point_roles: { question: 5 } },
          },
        },
      },
    });
    if (path === "/api/admin/meta/schema/preview") return ok({ diff: { counts: { added_tables: 1, removed_tables: 0, changed_tables: 0, added_columns: 2, removed_columns: 0, changed_columns: 0 } } });
    if (path === "/api/admin/meta/schema/sync") {
      const rebuildValue = Boolean(state.calls.at(-1)?.body?.rebuild_value_index);
      return ok({
        source: "mysql", table_count: 1, column_count: 2,
        value_index_rebuilt: rebuildValue,
        value_index: rebuildValue ? {
          ...state.valueIndexStatus, document_count: 7,
          admitted_column_count: 1, blocked_column_count: 0,
        } : null,
        diff: { counts: { added_tables: 1, removed_tables: 0, changed_tables: 0, added_columns: 2, removed_columns: 0, changed_columns: 0 } },
      });
    }

    if (path === "/api/adhoc/datasets" && method === "GET") return ok(state.datasets);
    if (path === "/api/adhoc/upload" && method === "POST") { state.datasets.unshift(structuredClone(sampleDataset)); return ok(sampleDataset, 201); }
    if (/^\/api\/adhoc\/datasets\//.test(path) && method === "DELETE") { state.datasets = []; return noContent(); }
    if (path === "/api/uploaded-datasets/quota" && method === "GET") return ok({
      usage: {
        active_versions: state.datasets.filter(item => item.status === "ready").length,
        bytes: state.datasets.reduce((sum, item) => sum + Number(item.byte_size || 0), 0),
        rows: state.datasets.reduce((sum, item) => sum + Number(item.row_count || 0), 0),
      },
      limits: { active_versions: 20, bytes: 104857600, rows: 200000 },
      remaining: { active_versions: 19, bytes: 104857472, rows: 199998 },
    });
    if (path === "/api/uploaded-datasets" && method === "GET") return ok({
      items: state.datasets.filter(item => item.status !== "deleted"),
      next_cursor: null,
      has_more: false,
    });
    if (path === "/api/uploaded-datasets" && method === "POST") {
      const sequence = state.datasets.length + 1;
      const filename = multipartFilename(request, `uploaded-${sequence}.csv`);
      const dataset = {
        ...structuredClone(sampleDataset),
        dataset_id: `dataset-${sequence}`,
        version_id: `dataset-${sequence}`,
        asset_id: `asset-${sequence}`,
        filename,
      };
      state.datasets.unshift(dataset);
      return ok(dataset, 200);
    }
    const uploadedPreviewMatch = path.match(/^\/api\/uploaded-datasets\/([^/]+)\/preview$/);
    if (uploadedPreviewMatch && method === "GET") {
      const dataset = state.datasets.find(item => item.dataset_id === decodeURIComponent(uploadedPreviewMatch[1]));
      if (!dataset || dataset.status !== "ready") return ok({ detail: "上传数据集不可查询或已过期" }, 404);
      return ok({ dataset_id: dataset.dataset_id, version_id: dataset.version_id, columns: dataset.columns, rows: dataset.preview, bounded: true });
    }
    const uploadedExtendMatch = path.match(/^\/api\/uploaded-datasets\/([^/]+)\/extend$/);
    if (uploadedExtendMatch && method === "POST") {
      const dataset = state.datasets.find(item => item.dataset_id === decodeURIComponent(uploadedExtendMatch[1]));
      if (!dataset) return ok({ detail: "上传数据集不可续期" }, 404);
      dataset.expires_at = "2099-12-31T23:59:59";
      dataset.lifecycle_revision = Number(dataset.lifecycle_revision || 1) + 1;
      return ok(dataset);
    }
    const uploadedArchiveMatch = path.match(/^\/api\/uploaded-datasets\/([^/]+)\/archive$/);
    if (uploadedArchiveMatch && method === "POST") {
      const dataset = state.datasets.find(item => item.dataset_id === decodeURIComponent(uploadedArchiveMatch[1]));
      if (!dataset) return ok({ detail: "上传数据集不可归档" }, 404);
      dataset.status = "archived";
      return ok(dataset);
    }
    const uploadedDatasetMatch = path.match(/^\/api\/uploaded-datasets\/([^/]+)$/);
    if (uploadedDatasetMatch && method === "GET") {
      const dataset = state.datasets.find(item => item.dataset_id === decodeURIComponent(uploadedDatasetMatch[1]));
      return dataset ? ok(dataset) : ok({ detail: "上传数据集不存在或无权访问" }, 404);
    }
    if (uploadedDatasetMatch && method === "DELETE") {
      const dataset = state.datasets.find(item => item.dataset_id === decodeURIComponent(uploadedDatasetMatch[1]));
      if (!dataset) return ok({ detail: "上传数据集不存在" }, 404);
      state.datasets = state.datasets.filter(item => item.dataset_id !== dataset.dataset_id);
      return noContent();
    }
    const sourceMappingMatch = path.match(/^\/api\/source-mappings\/([^/]+)\/confirm$/);
    if (sourceMappingMatch && method === "POST") return ok({
      mapping_id: decodeURIComponent(sourceMappingMatch[1]),
      status: "confirmed",
      evidence: {
        auto_join_allowed: true,
        quality_gates: { type_compatibility: true, coverage: true, fanout: true, grain: true },
      },
    });

    if (path === "/api/admin/meta/masking/rules" && method === "GET") return ok(state.maskingRules);
    if (path === "/api/admin/meta/masking/policy" && method === "GET") return ok({ enabled: true, heuristic_enabled: true, default_strategy: "mask", explicit_override_strategy: "keep", builtins: [{ source: "column_name", data_kind: "name", patterns: ["customer_name"], description: "仅精确匹配高置信姓名字段" }] });
    if (path === "/api/admin/meta/masking/rules" && method === "POST") {
      const payload = await bodyOf(request); state.maskingRules.push({ id: "mask-1", ...payload }); return ok(state.maskingRules.at(-1), 201);
    }
    if (/^\/api\/admin\/meta\/masking\/rules\//.test(path) && method === "PATCH") return ok({});
    if (/^\/api\/admin\/meta\/masking\/rules\//.test(path) && method === "DELETE") {
      const id = path.split("/").at(-1);
      state.maskingRules = state.maskingRules.filter((item) => String(item.id) !== id);
      return noContent();
    }

    if (path === "/api/admin/meta/access-policies" && method === "GET") return ok(state.accessPolicies);
    if (path === "/api/admin/meta/access-policies" && method === "POST") {
      const payload = await bodyOf(request); state.accessPolicies.push({ id: "policy-1", ...payload }); return ok(state.accessPolicies.at(-1), 201);
    }
    if (/^\/api\/admin\/meta\/access-policies\//.test(path) && method === "PATCH") return ok({});
    if (/^\/api\/admin\/meta\/access-policies\//.test(path) && method === "DELETE") {
      const id = path.split("/").at(-1);
      state.accessPolicies = state.accessPolicies.filter((item) => String(item.id) !== id);
      return noContent();
    }

    if (path === "/api/workspaces" && method === "GET") return ok([{ id: "workspace-1", name: "销售组", role }]);
    if (path === "/api/workspaces" && method === "POST") return ok({ id: "workspace-2", name: "分析组", role: "owner" });
    if (path === "/api/workspaces/current/members" && method === "GET") return ok([{ user_id: state.session.user.id, username: state.session.user.username, display_name: state.session.user.display_name, role }]);
    if (path === "/api/workspaces/current/members" && method === "POST") return ok({ user_id: "user-bob" }, 201);
    if (/^\/api\/workspaces\/current\/members\//.test(path) && method === "PATCH") return ok({});

    if (path === "/api/admin/evaluation/nl2sql/baselines") return ok({
      baselines: Object.entries(evaluationDifficulties).map(([difficulty, fallback]) => {
        const item = state.approvedBaselines[difficulty];
        const unresolvable = options.evaluationUnresolvableBaseline === difficulty;
        return {
          difficulty, difficulty_label: fallback.label,
          dataset_suite: fallback.suite, suite_label: fallback.suiteLabel,
          available: Boolean(item) && !unresolvable, status: item ? "approved" : "unreviewed",
          run_id: item?.runId || null, reviewed_at: item ? "2026-08-05T17:00:00+08:00" : null,
          resolution_status: unresolvable ? "artifact_missing" : item ? "resolved" : "not_approved",
          resolution_error: unresolvable ? "已审批，但基线制品不存在或无法访问。" : null,
          metrics: item && !unresolvable ? {
            total: item.total, result_rate: item.resultRate,
            table_exact_rate: item.tableRate, execution_rate: item.executionRate,
            latency_p95: item.latencyP95, llm_attempts: item.attempts,
            tokens_mean: item.tokensMean, estimated_cost_total: item.cost,
            cost_estimate_available: false, cost_currency: "USD",
          } : {},
        };
      }),
    });
    if (path === "/api/admin/evaluation/nl2sql/runs") {
      const evaluationDifficulty = url.searchParams.get("difficulty") || "easy";
      const evaluation = evaluationFor(evaluationDifficulty);
      const fresh = freshGovernanceFor(evaluationDifficulty);
      const hasExperiment = Boolean(options.evaluationEligible || fresh);
      state.activeEvaluationDifficulty = evaluationDifficulty;
      return ok([{
        run_id: evaluation.runId, timestamp: "2026-08-05 16:00:00",
        total: options.evaluationTotal || evaluation.total,
        result_passed: Math.round(evaluation.total * evaluation.resultRate),
        table_exact: Math.round(evaluation.total * evaluation.tableRate),
        latency_p95: evaluation.latencyP95,
        difficulty: evaluationDifficulty, difficulty_label: evaluation.label,
        dataset_suite: options.evaluationDatasetSuite || evaluation.suite,
        suite_label: options.evaluationDatasetSuite ? options.evaluationDatasetSuite : evaluation.suiteLabel,
        experiment_id: hasExperiment ? fresh?.experimentId || `nl2sql-v2-${evaluationDifficulty}-test` : null,
        repeat_index: hasExperiment ? 3 : null,
        cache_state: hasExperiment ? "warm" : null,
        evaluation_profile: "core-default",
        baseline_status: state.baselineStatus[evaluationDifficulty], fingerprint_complete: hasExperiment,
      }]);
    }
    if (path === "/api/admin/evaluation/nl2sql/latest") {
      const evaluationDifficulty = url.searchParams.get("difficulty") || "easy";
      const evaluation = evaluationFor(evaluationDifficulty);
      state.activeEvaluationDifficulty = evaluationDifficulty;
      return ok({
      available: true,
      run: {
        run_id: evaluation.runId, timestamp: "2026-08-05 16:00:00",
        total: options.evaluationTotal || evaluation.total,
        difficulty: evaluationDifficulty, difficulty_label: evaluation.label,
        dataset_suite: options.evaluationDatasetSuite || evaluation.suite,
        suite_label: options.evaluationDatasetSuite ? options.evaluationDatasetSuite : evaluation.suiteLabel,
        evaluation_profile: "core-default",
        candidate_label: "Q1",
        candidate_id: "todo97-q1-qwen3.7-plus-standard-nonthinking-v1",
        model_name: "qwen3.7-plus", provider: "dashscope", model_provider: "openai",
        model_profile: { candidate_id: "todo97-q1-qwen3.7-plus-standard-nonthinking-v1", model_name: "qwen3.7-plus", provider: "dashscope", model_provider: "openai", adapter_version: "openai-compatible-v1", capability_registry_version: "llm-capability-registry-v1" },
        artifact: `D:\\eval\\runs\\run_${evaluation.suite}_20260805.json`,
      },
      metrics: {
        total: evaluation.total, syntax_rate: 1, execution_rate: evaluation.executionRate,
        table_exact_rate: evaluation.tableRate, result_rate: evaluation.resultRate,
        first_result_rate: Math.max(0, evaluation.resultRate - .06), table_f1_mean: .99,
        latency_p95: evaluation.latencyP95,
        first_sql: { evidence_complete: true, non_empty: evaluation.total, syntax_valid: evaluation.total, guard_passed: evaluation.total, exec_success: evaluation.total, tables_correct: Math.round(evaluation.total * evaluation.tableRate), structure_equivalent: Math.max(0, Math.round(evaluation.total * evaluation.resultRate) - 3), result_equivalent: Math.max(0, Math.round(evaluation.total * evaluation.resultRate) - 3), denominator: evaluation.total },
        correction: { triggered: 6, fixed_execution: 1, fixed_result: 3, harmful_modification: 1, unattributable: 1, llm_attempts: 6, additional_tokens: 1920, unknown_token_attempts: 0, additional_latency_ms: 2520, denominator: evaluation.total },
        final_sql: { syntax_valid: evaluation.total, exec_success: Math.round(evaluation.total * evaluation.executionRate), tables_correct: Math.round(evaluation.total * evaluation.tableRate), structure_equivalent: Math.round(evaluation.total * evaluation.resultRate), result_equivalent: Math.round(evaluation.total * evaluation.resultRate), result_evaluated: evaluation.total, denominator: evaluation.total },
      },
      performance: { tt_sql_mean: 3000, tt_sql_p95: 5000, tt_result_mean: 3500, tt_result_p95: 6000, tt_complete_mean: 5200, tt_complete_p95: 12000, timing_protocol_version: "timing-v2", llm_calls_mean: 3, llm_calls_p95: 4, tokens_mean: 1200, tokens_p95: 1800, estimated_cost_total: 0, cost_estimate_available: false, cost_currency: "USD", degraded_cases: 0, llm_attempt_schema: "llm-attempt-v2", llm_attempts: 90, llm_attempts_succeeded: 90, llm_attempts_failed: 0, llm_attempts_timed_out: 0, llm_attempts_cancelled: 0, llm_attempt_unknown_tokens: 0, llm_attempt_accounting_complete: true, llm_recoveries: 1, llm_provider_request_id_unavailable: 2, llm_token_timing_unavailable: 90, llm_queue_wait_p95_ms: 18, llm_provider_p95_ms: 2380, llm_outcome_classes: { success: 90 }, failed_llm_calls: 0, failed_core_llm_calls: 0, failed_optional_llm_calls: 0, core_degradation_count: 0, optional_degradation_count: 0, degradation_evidence_complete: true, structured_output_fallbacks: 0, semantic_coverage_cases: 30, semantic_coverage_mean: 1, semantic_unresolved_cases: 0, semantic_unsafe_compile_cases: 0, request_deadline_contract_complete: true, request_deadline_exhausted_cases: 0, prompt_tokens_before: 3600, prompt_tokens_after: 1800, prompt_compression_rate: .5, adaptive_candidate_rate: .1, candidate_selected_b: 1, candidate_valid_b: 2, candidate_repaired: 1, candidate_failed: 0, candidate_wait_total_ms: 850, candidate_wait_p95_ms: 430, node_performance: [], llm_route_performance: [{ node_id: "generate_sql", route: "sql", models: ["test-strong"], calls: 30, failed_calls: 0, succeeded: 30, errors: 0, timed_out: 0, cancelled: 0, unknown_tokens: 0, tokens_per_call: 800, latency_p95_ms: 2400 }] },
      summary: { failed_count: evaluation.total - Math.round(evaluation.total * evaluation.resultRate), root_causes: [], targeted_actions: [] },
      cases: [{
        id: "eval_001", question: "各地区的销售总额", result_equivalent: true,
        gen_sql: "SELECT r.region_name, SUM(f.order_amount) AS total_amount FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id GROUP BY r.region_name ORDER BY total_amount DESC",
        expected_sql: "SELECT r.region_name, SUM(f.order_amount) AS total_amount FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id GROUP BY r.region_name ORDER BY total_amount DESC",
        attribution: { label: "结果通过", confidence: 1, evidence: [], recommendation: "无需处理" },
        evaluation_profile: "core-default",
        candidate_label: "Q1", candidate_id: "todo97-q1-qwen3.7-plus-standard-nonthinking-v1", model_name: "qwen3.7-plus", provider: "dashscope", model_provider: "openai",
        model_profile: { candidate_id: "todo97-q1-qwen3.7-plus-standard-nonthinking-v1", model_name: "qwen3.7-plus", provider: "dashscope", model_provider: "openai" },
        first_sql_metrics: { non_empty: true, syntax_valid: true, read_only_safe: true, guard_passed: true, execution_attempted: true, exec_success: true, result_evaluated: true, result_equivalent: false },
        correction_metrics: { triggered: true, attribution_valid: true, attribution_status: "direct", trigger_reason_codes: ["sql.semantic_error"], retry_count: 1, correction_output_count: 1, sql_revision_count: 1, llm_attempts: 1, failed_llm_attempts: 0, additional_tokens: 320, additional_tokens_are_lower_bound: false, unknown_token_attempts: 0, additional_latency_ms: 420, fixed_execution: false, fixed_result: true, harmful_modification: false, outcome: "fixed_result" },
        final_sql_metrics: { non_empty: true, syntax_valid: true, read_only_safe: true, guard_passed: true, execution_attempted: true, exec_success: true, result_evaluated: true, result_equivalent: true },
        llm_attempt_accounting: { schema: "llm-attempt-v2", attempts: 1, succeeded: 1, failed: 0, timed_out: 0, cancelled: 0, recoveries: 0, provider_request_id_unavailable: 1, evidence_complete: true, reconciled: true },
        llm_attempts: [{ attempt_id: "attempt-eval-001", node_id: "generate_sql", route: "sql", model: "test-strong", status: "success", reason_code: "llm.success", outcome_class: "success", latency_ms: 2400, timeout_budget_ms: 6000, physical_attempt_budget_ms: 5000, recovery_reserve_ms: 1000, queue_wait_ms: 20, provider_ms: 2380, deadline_before_ms: 25000, deadline_after_ms: 22600, retry_index: 0, recovery: false, provider_request_id: "unavailable", provider_request_id_available: false, provider_request_id_unavailable_reason: "provider_request_id_not_exposed", total_tokens: 800, token_known: true, token_timing_unavailable_reason: "token_timing_not_exposed", affects_core_path: true }],
        degradations: [],
        stage_evidence: {
          semantic_coverage: { schema_version: "semantic-coverage-v1", registry_version: "semantic-composition-v1", question_fingerprint: "sha256:mock", coverage_score: 1, compiler_mode: "deterministic_ast", compiler_supported: true, detected_slots: ["metric:sum", "dimension:region"], represented_slots: ["metric:sum", "dimension:region"], unresolved_slots: [], abstain_reasons: [], raw_question_persisted: false, filter_values_persisted: false },
          request_deadline: { contract_version: "request-deadline-v1", budget_ms: 30000, remaining_ms: 22600, execution_reserve_ms: 3000, phase_budgets_ms: { llm: 24000, execution: 3000 }, absolute_timestamps_persisted: false },
          semantic_ir: { schema_version: "semantic-ir-v1", status: "complete", intent: "aggregate", dimensions: [{ table: "dim_region", column: "region_name" }], measures: [{ table: "fact_order", column: "order_amount", function: "SUM", alias: "total_amount" }], joins: [{ left_table: "dim_region", left_column: "region_id", right_table: "fact_order", right_column: "region_id" }], order_by: [{ target: "total_amount", direction: "DESC" }], limit: null, confidence: .96, compiler: { supported: true, mode: "deterministic_ast", emitted: true } },
          first_sql: "SELECT r.region_name, SUM(f.order_amount) AS total_amount FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id GROUP BY r.region_name",
          correction_sqls: ["SELECT r.region_name, SUM(f.order_amount) AS total_amount FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id GROUP BY r.region_name ORDER BY total_amount DESC"], authorized_sql: "SELECT 1", result_summary: { row_count: 5 }, retrieval: {},
        },
      }],
      run_command: `python -m app.eval.run_eval --workspace-id workspace-1 --suite ${evaluation.suite}`,
    });
    }
    if (path === "/api/admin/evaluation/nl2sql/governance") {
      const evaluationDifficulty = url.searchParams.get("difficulty") || "easy";
      const evaluation = evaluationFor(evaluationDifficulty);
      const fresh = freshGovernanceFor(evaluationDifficulty);
      state.activeEvaluationDifficulty = evaluationDifficulty;
      const eligible = Boolean(options.evaluationEligible);
      const hasExperiment = Boolean(eligible || fresh);
      const runEvidenceComplete = hasExperiment;
      const checklist = { valid_run: true, full_dataset: true, fingerprint_complete: runEvidenceComplete, sample_contract_complete: true, samples_reviewed: true, infrastructure_clean: true, evidence_chain_complete: runEvidenceComplete, no_raw_result_rows: true, evaluation_profile_complete: runEvidenceComplete, llm_attempt_accounting_complete: runEvidenceComplete, degradation_evidence_complete: runEvidenceComplete };
      const experimentChecks = { metadata_complete: true, three_repeats: true, repeat_indexes: true, cache_protocol: true, same_workspace: true, same_candidate: true, same_evaluation_profile: true, full_dataset: true, valid_runs: true, infrastructure_clean: true, fingerprint_v2: true, fingerprints_complete: true, fingerprints_comparable: true, result_denominator_complete: true, run_checklists_eligible: true, approved_baseline_available: true, baseline_non_regression: true, table_exact_non_regression: false, table_scope_safety: true, table_scope_evidence_complete: true, security_guard_integrity: true, result_floor_non_regression: true, ttsql_gate: true, ttresult_gate: true, ttcomplete_gate: true, timing_protocol_complete: true, routing_evidence_complete: true, prompt_compression_evidence: true, llm_attempt_accounting_complete: true, semantic_coverage_evidence_complete: true, request_deadline_evidence_complete: true, llm_outcome_taxonomy_complete: true, llm_recovery_bounded: true, degradation_evidence_complete: true, failed_llm_calls_zero: true, optional_postprocess_clean: true, structured_output_clean: true, candidate_trigger_gate: true, candidate_safety_gate: true, semantic_review_first_correct_final_incorrect_zero: true, semantic_review_same_sql_execution_loss_zero: true, semantic_review_selected_b_no_sql_delta_zero: true, token_non_regression: true, ...(fresh?.checkOverrides || {}) };
      const minimum = evaluationDifficulty === "easy" ? 48 : evaluationDifficulty === "medium" ? 43 : null;
      const results = fresh?.results || [1, 2, 3].map(() => Math.round(evaluation.total * evaluation.resultRate));
      const tables = fresh?.tables || [1, 2, 3].map(() => Math.round(evaluation.total * evaluation.tableRate));
      const scopeSafe = fresh?.scopeSafe || [evaluation.total, evaluation.total, evaluation.total];
      const scopeEvidence = fresh?.scopeEvidence || [evaluation.total, evaluation.total, evaluation.total];
      const experiment = hasExperiment ? {
        available: true, experiment_id: fresh?.experimentId || `nl2sql-v2-${evaluationDifficulty}-test`, candidate_label: fresh ? "D0" : "v2-control", evaluation_profile: "core-default",
        status: fresh?.status || "recordable", complete: fresh?.complete ?? true, recordable: fresh?.recordable ?? true, qualified: fresh?.qualified ?? false, repeat_count: 3, required_repeats: 3,
        checks: experimentChecks,
        blocking_checks: Object.fromEntries(Object.entries(experimentChecks).filter(([key]) => key !== "table_exact_non_regression")),
        observational_checks: { table_exact_non_regression: false },
        failed_checks: Object.entries(experimentChecks).filter(([key, passed]) => key !== "table_exact_non_regression" && !passed).map(([key]) => key), failed_observations: ["table_exact_non_regression"], regressed_case_ids: [],
        promotion_policy: {
          dataset_suite: evaluation.suite,
          case_count: evaluation.total,
          result_rate_exclusive: evaluationDifficulty === "easy" ? .95 : evaluationDifficulty === "medium" ? .85 : null,
          per_run: [1, 2, 3].map((repeat, index) => ({
            run_id: fresh?.runIds[index] || `${evaluation.runId}_r${repeat}`,
            passed: results[index],
            total: evaluation.total,
            minimum,
            met: minimum === null || results[index] >= minimum,
          })),
        },
        worst: { tables_correct: Math.min(...tables), table_scope_safe: Math.min(...scopeSafe), table_scope_evidence_complete: Math.min(...scopeEvidence), result_equivalent: Math.min(...results), tt_sql_p95: 5000, tt_result_p95: 6000, tt_complete_p95: evaluation.latencyP95, prompt_compression_rate: .5, adaptive_candidate_rate: .1, candidate_valid_b: 2, candidate_failed: 0, candidate_wait_total_ms: 850, failed_core_llm_calls: Math.max(...(fresh?.coreFailures || [0, 0, 0])), failed_optional_llm_calls: 0 },
        runs: [1, 2, 3].map((repeat, index) => ({ run_id: fresh?.runIds[index] || `${evaluation.runId}_r${repeat}`, repeat_index: repeat, cache_state: repeat === 1 ? "cold" : "warm", evaluation_profile: "core-default", total: evaluation.total, tables_correct: tables[index], table_scope_safe: scopeSafe[index], table_scope_evidence_complete: scopeEvidence[index], result_equivalent: results[index], tt_sql_p95: 5000, tt_result_p95: 6000, tokens_p95: 1800, adaptive_candidate_rate: .1, candidate_valid_b: 2, candidate_failed: 0, candidate_wait_total_ms: 850 })),
        baseline: { run_id: state.approvedBaselines[evaluationDifficulty]?.runId, suite: evaluation.suite, candidate_suite: evaluation.suite, comparison_applicable: true },
      } : null;
      return ok({
        available: true,
        difficulty: evaluationDifficulty, difficulty_label: evaluation.label,
        dataset_suite: evaluation.suite, runnable_dataset_suite: evaluation.suite,
        sample_contract: { total: evaluation.total, complete: true, complete_count: evaluation.total, reviewed_count: evaluation.total, duplicate_ids: [] },
        selected: {
          run: { run_id: evaluation.runId, total: evaluation.total, difficulty: evaluationDifficulty, difficulty_label: evaluation.label }, status: state.baselineStatus[evaluationDifficulty],
          eligible: runEvidenceComplete, run_eligible: Object.values(checklist).every(Boolean), approval_eligible: eligible || Boolean(fresh?.recordable), checklist,
          experiment: hasExperiment ? { experiment_id: fresh?.experimentId || `nl2sql-v2-${evaluationDifficulty}-test`, repeat_index: 3, cache_state: "warm", evaluation_profile: { name: "core-default" } } : null,
          evaluation_profile: { name: "core-default" },
          fingerprint: runEvidenceComplete ? { complete: true, run_fingerprint: "sha256:1234567890abcdef", dataset_version: "sha256:data", dw_snapshot: "sha256:dw", schema_version: "sha256:schema", knowledge_version: "sha256:knowledge", index_version: "sha256:index", config_hash: "sha256:config", git_commit: "abc123", evaluation_profile: { name: "core-default" }, model: { model_name: "test-model" } } : { complete: false, errors: ["legacy"] },
        },
        approved: { run_id: state.approvedBaselines[evaluationDifficulty]?.runId, difficulty: evaluationDifficulty },
        baseline_resolution: { source: "meta_mysql", status: "approved", schema: "evaluation-baseline-v2", run_id: state.approvedBaselines[evaluationDifficulty]?.runId },
        experiment,
        experiment_setup: { suggested_id: `nl2sql-v2-${evaluationDifficulty}-test`, commands: [1, 2, 3].map((repeat) => `python -m app.eval.run_eval --suite ${evaluation.suite} --experiment-id nl2sql-v2-${evaluationDifficulty}-test --repeat-index ${repeat} --cache-state ${repeat === 1 ? "cold" : "warm"}`) },
        comparison: null,
        baseline_policy: "observational_three_repeat",
      });
    }
    if (path === "/api/admin/evaluation/nl2sql/experiments") return ok([]);
    if (/^\/api\/admin\/evaluation\/nl2sql\/baselines\/[^/]+\/review$/.test(path) && method === "POST") {
      const payload = await bodyOf(request);
      const evaluationDifficulty = state.activeEvaluationDifficulty;
      const fresh = freshGovernanceFor(evaluationDifficulty);
      if (payload.decision === "approve" && !options.evaluationEligible && !fresh?.recordable) return ok({ detail: "该 run 未满足 baseline 准入" }, 409);
      state.baselineStatus[evaluationDifficulty] = payload.decision === "approve" ? "approved" : "rejected";
      if (payload.decision === "approve") {
        const evaluation = evaluationFor(evaluationDifficulty);
        state.approvedBaselines[evaluationDifficulty] = { ...evaluation, runId: evaluation.runId };
      }
      return ok({ status: state.baselineStatus[evaluationDifficulty], difficulty: evaluationDifficulty, reviewed_by: "alice" });
    }
    if (path === "/api/admin/evaluation/resilience/latest") {
      const groups = [["dependency", 6], ["concurrency", 5], ["cancellation", 2], ["resource", 6], ["cache", 3], ["soak", 1]];
      const cases = groups.flatMap(([category, count]) => Array.from({ length: count }, (_, index) => ({
        id: `${category.slice(0, 3).toUpperCase()}-${index + 1}`, category,
        title: `${category} 稳定性探针 ${index + 1}`, risk: index === 0 ? "critical" : "high",
        probe: `${category}_probe`, status: "passed", duration_ms: 3.2,
        metrics: category === "concurrency" ? { concurrency: 5, error_rate: 0, p95_ms: 18.2, peak_active: 5 } : { injected: true, released: true }, error: null,
      })));
      return ok({ available: true, generated_at: "2026-07-16T15:00:00+08:00", source: "deterministic_fault_injection",
        contract: { valid: true, errors: [], total: 23, categories: Object.fromEntries(groups), fingerprint: "sha256:resilience-suite" },
        summary: { total: 23, passed: 23, failed: 0, pass_rate: 1, gate_passed: true },
        certification: { status: "not_run", duration_seconds: 0, required_seconds: 1800, cycles: 0 },
        categories: groups.map(([category, total]) => ({ category, total, passed: total, failed: 0 })),
        resource_limits: { query_chars: 2000, result_rows: 1000, upload_bytes: 10485760, upload_rows: 20000, export_cells: 2000000, global_concurrency: 20, workspace_concurrency: 5 }, cases });
    }
    if (path === "/api/admin/evaluation/adhoc-isolation/latest") {
      const groups = [
        ["lifecycle", 3],
        ["authorization", 3],
        ["schema_index_isolation", 3],
        ["ttl_cleanup", 3],
        ["fault_compensation", 3],
        ["multi_dataset_concurrency", 3],
        ["privacy_artifact", 3],
      ];
      const probes = [
        "csv_xlsx_roundtrip", "ordinary_entry_denied", "schema_snapshot_excludes_upload",
        "expired_context_denied", "drop_failure_reconciled", "parallel_namespace_isolation",
        "report_contains_no_rows",
      ];
      const cases = groups.flatMap(([category, count], groupIndex) => Array.from({ length: count }, (_, index) => ({
        attempt_id: `sha256:attempt-${groupIndex}-${index}`,
        scenario_id: `ADHOC-${String(groupIndex + 1).padStart(2, "0")}-${index + 1}`,
        category,
        fixture_id: `adhoc_fixture_${groupIndex + 1}_v1`,
        fixture_hash: `sha256:fixture-${groupIndex + 1}`,
        probe: probes[groupIndex],
        status: "passed",
        duration_ms: 2.4 + index,
        evidence: { denied: category === "authorization", residual_resources: 0, raw_rows_logged: false },
        trace_count: 1,
      })));
      return ok({
        available: true,
        source: "executable_suite",
        run_id: "adhoc-20260728180000-abcdef12",
        generated_at: "2026-07-28T18:00:00+08:00",
        facet: "adhoc_isolation",
        suite_version: "adhoc-isolation-v1",
        evidence_level: "offline_contract",
        production_equivalent: false,
        contract: { valid: true, total: 21, fixtures: 9, categories: Object.fromEntries(groups) },
        fixture_manifest: {
          fixture_count: 9,
          origin_counts: { synthetic_adhoc_contract: 8, synthetic_negative_payload: 1 },
          provenance_completeness: 1,
          real_user_fixture_count: 0,
        },
        sandbox: {
          manifest: { run_id: "adhoc-20260728180000-abcdef12", namespace_hashes: { dw_database: "sha256:dw" } },
          preflight: { valid: true, errors: [], allowlist_exact: true },
          dependency_evidence: { status: "not_run", evidence_level: "offline_contract" },
        },
        denominator: {
          declared: 21, eligible: 21, selected: 21, attempted: 21,
          passed: 21, failed: 0, skipped: 0, infrastructure_failed: 0,
          cleanup_failed: 0, attempt_records: 21, trace_records: 21,
        },
        metrics: {
          real_upload_content_occurrences: 0,
          shared_resource_delta: 0,
          sandbox_residual_resources: 0,
          unauthorized_db_execution_count: 0,
          schema_index_pollution_count: 0,
          cleanup_success_rate: 1,
          compensation_recovery_rate: 1,
          denominator_consistency: true,
          cross_tenant_leak_count: 0,
          fixture_provenance_completeness: 1,
        },
        summary: { total: 21, passed: 21, failed: 0, pass_rate: 1, gate_passed: true },
        cases,
        privacy: {
          raw_rows_persisted: false,
          filenames_persisted: false,
          physical_table_names_persisted: false,
          real_user_content_collected: false,
        },
        run_command: "python -m app.eval.adhoc_isolation_suite --evidence-level offline_contract --enforce",
      });
    }
    if (path === "/api/admin/evaluation/rag/latest") {
      const routes = {
        column: { cases: 16, recall_at_k: 1, precision_at_k: .5099, mrr: .98, ndcg: .97 },
        metric: { cases: 5, recall_at_k: 1, precision_at_k: .2, mrr: 1, ndcg: 1 },
        value: { cases: 1, recall_at_k: 1, precision_at_k: 1, mrr: 1, ndcg: 1 },
        training: { cases: 4, recall_at_k: 1, precision_at_k: .33, mrr: .88, ndcg: .91 },
      };
      const makeCase = (index) => ({
        id: `RAG-${String(index + 1).padStart(3, "0")}`, rag_case_id: `RAG-${String(index + 1).padStart(3, "0")}`,
        query: index ? "每种商品的平均销量" : "2025年每月销售额趋势",
        expected_tables: index ? ["dim_product", "fact_order"] : ["dim_date", "fact_order"],
        candidate_tables: index ? ["dim_product", "fact_order"] : ["dim_date", "fact_order"],
        filtered_tables: index ? ["dim_product", "fact_order"] : ["dim_date", "fact_order"],
        display_expected: [index ? "dim_product.product_name" : "dim_date.month"],
        display_hit: [index ? "dim_product.product_name" : "dim_date.month"],
        routes: Object.fromEntries(Object.keys(routes).map(name => [name, [{ id: name === "column" ? (index ? "dim_product.product_name" : "dim_date.month") : `${name}-candidate`, rank: 1, score: .92 }]])),
      });
      const tunedSummary = { cases: 16, routes, table_candidate_recall: 1, table_filter_recall: 1, display_hit_rate: 1, tenant_leakage: 0, candidate_count: 212, final_context_count: 196, context_compression_rate: .0755, gate_passed: true };
      const fixedSummary = { ...tunedSummary, routes: { ...routes, column: { ...routes.column, precision_at_k: .2734 } }, candidate_count: 272, final_context_count: 256, gate_passed: false };
      const live = { available: true, run_id: "run-todo43-warm-3", summary: { evaluated_cases: 30, required_cases: 30, column_recall_at_k: .96, column_precision_at_k: .48, effective_context_column_recall: 1, context_tokens_before_mean: 2875, context_tokens_after_mean: 1760, context_compression_rate: .3878, candidate_cache_hit_rate: .83, budget_evidence_available: true, average_dynamic_k: { column: 5.2, metric: 2.1, value: 3, training: 1.8 }, failed_cases: 1, gate_passed: false }, cases: [{ id: "RAG-018", rag_case_id: "RAG-018", query: "各品牌共售出多少件商品", passed: false, diagnostics: { columns: { missing: ["fact_order.order_quantity"] }, effective_context: { missing: [] }, tables: { missing: [] }, display: { missing: [] }, metrics: { missing: [] }, training: { missing: [] }, values: { missing: [] } } }] };
      const hybridSummary = (recall, precision, longTailRecall, longTailMrr, p95) => ({ routes: { column: { recall_at_k: recall, precision_at_k: precision } }, base_column_recall_at_k: recall, base_column_precision_at_k: precision, identifier_subset: { recall_at_k: longTailRecall, mrr: longTailMrr }, latency_ms: { p95 }, degradation_count: 0 });
      const hybrid = {
        available: true, active_mode: "bge_m3_dense_sparse", replacement_strategy: "overwrite_existing_collections", v1_fallback: false,
        contract: { total: 28 }, summary: { gate_passed: true },
        profiles: {
          "R-D1": { summary: hybridSummary(.96, .61, .82, .79, 34) },
          "R-S1": { summary: hybridSummary(.92, .65, .91, .88, 31) },
          "R-H1": { summary: hybridSummary(1, .63, 1, .96, 41) },
          "R-H3": { summary: hybridSummary(1, .68, 1, 1, 44) },
        },
        comparison: { "R-H1_vs_R-D1": { identifier_recall_delta: .18, identifier_mrr_delta: .17 } },
        embedding_latency_ms: { p95: 86 },
        historical_dense_reference: { available: true, run_id: "20260722_163506", column_recall_at_k: 1, column_precision_at_k: .5738 },
      };
      const pointProfile = (entities, points, duplicate, precision, roles) => ({
        eligible: true,
        manifest: {
          entity_count: entities, point_count: points, points_per_entity: points / entities,
          duplicate_occupancy: duplicate, stable_ids_valid: true,
          vector_integrity: { valid: true },
          routes: {
            column: { point_count: roles.column.count, point_roles: roles.column.roles },
            metric: { point_count: roles.metric.count, point_roles: roles.metric.roles },
            training: { point_count: roles.training.count, point_roles: roles.training.roles },
          },
        },
        summary: { routes: { column: { recall_at_k: 1, precision_at_k: precision } }, identifier_subset: { recall_at_k: 1, mrr: 1 }, latency_ms: { p95: 42 }, passed_cases: 28, cases: 28 },
      });
      const pointProjection = {
        available: true, frozen_variable: "entity_to_point_projection_only",
        contract: { total: 28 }, alias_manifest: { approved_alias_count: 4 },
        profiles: {
          single_canonical: pointProfile(35, 35, 0, .5753, { column: { count: 24, roles: { canonical: 24 } }, metric: { count: 6, roles: { canonical: 6 } }, training: { count: 5, roles: { question: 5 } } }),
          multi_all: pointProfile(35, 131, .7328, .5113, { column: { count: 98, roles: { name: 24, description: 24, alias: 50 } }, metric: { count: 28, roles: { name: 6, description: 6, alias: 16 } }, training: { count: 5, roles: { question: 5 } } }),
          mixed_selective: pointProfile(35, 39, .1026, .5753, { column: { count: 26, roles: { canonical: 24, approved_alias: 2 } }, metric: { count: 8, roles: { canonical: 6, approved_alias: 2 } }, training: { count: 5, roles: { question: 5 } } }),
        },
        decision: {
          selected_strategy: "single_canonical", active_strategy_at_evaluation: "single_canonical",
          gate_passed: true, requires_active_rebuild: false,
          reason: "single_canonical 在全部质量门禁上不低于 multi_all，并以最少 Point 满足召回、精度、歧义与隔离要求。",
        },
        comparison: { single_canonical_vs_multi_all: { point_count_delta: -96, column_precision_delta: .064, query_p95_delta_ms: -21 } },
      };
      const embeddingCapacity = {
        available: true, source: "live_bge_m3_current_hardware", live_model_evaluated: true,
        profiles: {
          serial_inprocess_v1: { measurements: { burst_texts_per_second: 66.9, burst_wall_ms: 298.8, idle_query_ms: { p95: 16.0 }, model_invocations_total: 29, max_model_concurrency: 1 }, resource_usage: { available: true, process: { rss_peak_measured_mb: 512, measured_peak_growth_mb: 12 }, cuda_measured: { available: false, peak_reserved_mb: null } } },
          microbatch_inprocess_v1: { measurements: { burst_texts_per_second: 491.6, burst_wall_ms: 40.7, idle_query_ms: { p95: 15.1 }, model_invocations_total: 10, max_model_concurrency: 1 }, resource_usage: { available: true, process: { rss_peak_measured_mb: 548, measured_peak_growth_mb: 18 }, cuda_measured: { available: false, peak_reserved_mb: null } } },
        },
        correctness: { query: { passed: true, minimum_dense_cosine: 1 }, document: { passed: true, minimum_dense_cosine: 1 } },
        comparison: { burst_texts_per_second_ratio: 7.344, idle_query_p95_ms_delta: -0.9, model_invocation_delta: -19, process_rss_peak_mb_delta: 36, process_rss_growth_mb_delta: 6 },
        summary: { gate_passed: true, selected_runtime_profile: "microbatch_inprocess_v1", remote_service_triggered: false, remote_service_decision: "单实例有界微批已满足当前容量门禁，尚无远端服务触发证据。" },
      };
      const signalProfile = (profile, routePolicy, auxiliaryRecall, modifierRecall, columnQueries, valueQueries, latency) => ({
        profile, route_policy: routePolicy, auxiliary_recall: auxiliaryRecall,
        typed_modifier_recall: modifierRecall, standalone_modifier_retention: 1,
        extractor_latency_ms: { warm_p95: latency },
        routing: { column: { remote_queries_mean: columnQueries }, value: { remote_queries_mean: valueQueries } },
      });
      const querySignals = {
        available: true, source: "deterministic_signal_plus_live_rag",
        summary: { gate_passed: true, cases: 20, live_evaluated: true, raw_text_logged: false },
        profiles: {
          K0: signalProfile("full_query_only", "full_query_only_v1", 0, 0, 1, 1, .05),
          K1: signalProfile("jieba_tfidf_v1", "current_multi_query_v1", .5758, 0, 3.55, 3.55, .18),
          K2: signalProfile("domain_signals_v1", "routed_signals_v1", 1, 1, 1, 3.6, .22),
          K3: signalProfile("jieba_textrank_v1", "current_multi_query_v1", .4848, 0, 3.4, 3.4, .31),
        },
        live: {
          evaluated: true,
          profiles: {
            K0: { gate_passed: false, summary: { column_recall_at_k: .9714 }, remote_queries_mean: { column: 1 } },
            K1: { gate_passed: true, summary: { column_recall_at_k: 1 }, remote_queries_mean: { column: 3.562 } },
            K2: { gate_passed: false, summary: { column_recall_at_k: .9714 }, remote_queries_mean: { column: 1 } },
          },
        },
        phase0: { checks: { batch_query_api_available: true } },
        privacy: { gate_passed: true },
        decision: {
          selected_profile: "jieba_tfidf_v1", selected_route_policy: "current_multi_query_v1",
          active_changed: false, status: "retain_active",
          reason: "K1 是本轮唯一通过真实 RAG 门禁的已评测方案；K0/K2 均出现原始字段漏召回，且没有三次端到端 NL2SQL non-regression 证据，因此保留 active。",
          k2_signal_layer_delta_vs_k1: { auxiliary_recall: .4242, typed_modifier_recall: 1, column_remote_queries_mean: -2.55 },
        },
      };
      return ok({ available: true, generated_at: "2026-07-16T18:00:00+08:00", source: "deterministic_offline_index", live_index_evaluated: false,
        contract: { valid: true, errors: [], total: 16, fingerprint: "sha256:rag-contract" }, index: { version: "sha256:rag-index-version", parameters: {} },
        summary: tunedSummary, fixed_k: { summary: fixedSummary }, tuned: { summary: tunedSummary, cases: Array.from({ length: 16 }, (_, index) => makeCase(index)) }, live,
        hybrid, point_projection: pointProjection, embedding_capacity: embeddingCapacity, query_signals: querySignals, comparison: { column_recall_delta: 0, column_precision_delta: .2365, table_filter_recall_delta: 0, display_hit_delta: 0, candidate_count_delta: -60 } });
    }
    if (path === "/api/admin/evaluation/planning/latest") {
      const planningProfile = (profile, precision, recall, overPlan, underPlan, resultRate, ttResult, tokens, decisionP95) => ({
        profile, feature_level: "A4", cases: 12,
        plan_precision: precision, plan_recall: recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0,
        over_plan_rate: overPlan, under_plan_rate: underPlan, harmful_plan_rate: 0,
        selected_path: { result_equivalence_rate: resultRate, safety_rate: 1, tt_result_p50_ms: 6250, tt_result_p95_ms: ttResult, tokens_mean: tokens, estimated_cost_mean: .0013 },
        decision_latency_ms: { p50: decisionP95 * .7, p95: decisionP95, p99: decisionP95 * 1.05, max: decisionP95 * 1.1 },
      });
      const profiles = {
        C0: planningProfile("never_plan", 1, 0, 0, 1, .5, 6250, 920, .18),
        C1: planningProfile("keyword_length_v1", .25, .1667, .5, .8333, .5, 8610, 1097, .21),
        C2: planningProfile("always_plan", .5, 1, 1, 0, .9167, 8610, 1450, .21),
        C3: planningProfile("structured_rules_v1", 1, .8333, 0, .1667, .9167, 8610, 1185, .26),
        C4: planningProfile("weighted_scorecard_v1", .8571, 1, .1667, 0, .9167, 8610, 1229, .24),
        C5: planningProfile("calibrated_linear_v1", .8571, 1, .1667, 0, .9167, 8610, 1229, .31),
      };
      const featureAblation = (recall, f1) => ({ profile: "weighted_scorecard_v1", plan_recall: recall, f1, selected_path: { result_equivalence_rate: .5 + recall * .5 }, decision_latency_ms: { p95: .25 } });
      return ok({
        available: true,
        artifact_schema: "planning-router-ablation-v1",
        generated_at: "2026-07-27T16:26:19+08:00",
        evidence_level: "reviewed_offline_counterfactual",
        contract: { valid: true, version: "planning-reviewed-set-v1", total: 42, categories: 18, raw_questions_in_report: false },
        evaluation_scope: { split: "test", total_cases: 16, comparable_cases: 12, excluded: { both_failed: 1, inconclusive: 1, ir_bypass: 2 } },
        frozen_variables: { semantic_ir_profile: "semantic-ir-v1", semantic_repair_profile: "semantic-repair-inline-v1", cache_ttl_seconds: 30, retrieval_cache_ttl_seconds: 120 },
        profiles,
        feature_ablations: {
          A0: featureAblation(0, 0),
          A1: featureAblation(0, 0),
          A2: featureAblation(.5, .6667),
          A3: featureAblation(1, .9231),
          A4: featureAblation(1, .9231),
        },
        decision: {
          active_profile: "keyword_length_v1", active_unchanged: true,
          offline_recommendation_label: "C4", offline_recommendation: "weighted_scorecard_v1",
          release_qualified: false, fallback_profile: "keyword_length_v1", rollback_without_restart: true,
          release_blockers: ["missing_live_direct_plan_pairs", "missing_cold_warm_warm_full_run", "missing_production_shadow_canary"],
          selection_reason: "离线 reviewed set 上取得 Pareto 候选；真实发布门禁未完成，因此只保留推荐，不切换 active。",
        },
        implementation_gates: { gate_passed: true, active_c1_unchanged: true, privacy_safe_report: true, ttl_frozen_and_disambiguated: true },
        runtime: { configured_profile: "keyword_length_v1", runtime_profile: "keyword_length_v1", runtime_override: null, fallback_profile: "keyword_length_v1", rollback_without_restart: true },
      });
    }
    if (path === "/api/admin/evaluation/planning/rollback" && method === "POST") {
      return ok({
        status: "rolled_back",
        runtime: { configured_profile: "weighted_scorecard_v1", runtime_profile: "keyword_length_v1", runtime_override: "keyword_length_v1", fallback_profile: "keyword_length_v1", rollback_without_restart: true },
        message: "规划路由已在当前进程立即回切到预注册 fallback；无需数据库迁移或索引重建。",
      });
    }
    if (path === "/api/admin/evaluation/semantic-repair/latest") {
      const repairProfile = (profile, rules, rescue, p95) => ({
        profile, rules, cases: 36, positive_cases: 14, negative_cases: 22,
        correct_repairs: Math.round(14 * rescue), changed_cases: Math.round(14 * rescue),
        repair_precision: 1, targeted_rescue_rate: rescue, false_trigger_count: 0,
        near_miss_false_trigger_rate: 0, harmful_repair_count: 0,
        idempotence_rate: 1, unsafe_unchanged_rate: 1,
        repair_latency_ms: { mean: p95 * .62, p95, max: p95 * 1.12 },
      });
      return ok({
        available: true, artifact_schema: "semantic-repair-ablation-v1",
        generated_at: "2026-07-27T18:00:00+08:00", evidence_level: "reviewed_deterministic_offline",
        contract: { valid: true, total: 36, splits: { train: 5, dev: 5, test: 26 }, raw_questions_in_report: false },
        runtime: { runtime_profile: "registry_same_rules_v1", configured_profile: "registry_same_rules_v1", fallback_profile: "inline_v1", rule_set_version: "semantic-repair-rules-v1", rollback_without_restart: true },
        profiles: {
          P0: repairProfile("inline_v1", ["join.unique_metadata_edge", "projection.threshold_measure"], .571, 12.9),
          P1: repairProfile("violation_only_v1", ["join.unique_metadata_edge", "projection.threshold_measure"], .571, 8.3),
          P2: repairProfile("registry_same_rules_v1", ["join.unique_metadata_edge", "projection.threshold_measure"], .571, 9.1),
          P3: repairProfile("registry_extended_v1", ["join.unique_metadata_edge", "projection.threshold_measure", "ir.unique_limit_order"], .857, 7.1),
          P4: repairProfile("targeted_correction_v1", ["join.unique_metadata_edge", "projection.threshold_measure"], .571, 8.1),
          P5: repairProfile("upstream_first_v1", ["join.unique_metadata_edge", "projection.threshold_measure"], .571, 7.5),
          P6: repairProfile("cascade_v1", ["join.unique_metadata_edge", "projection.threshold_measure", "ir.unique_limit_order"], .857, 7.7),
        },
        registry: [
          { rule_code: "join.unique_metadata_edge", rule_version: "1", mode: "active", changed_ast_nodes: ["join.on"] },
          { rule_code: "projection.threshold_measure", rule_version: "1", mode: "active", changed_ast_nodes: ["select.projection"] },
          { rule_code: "ir.unique_limit_order", rule_version: "1", mode: "candidate", changed_ast_nodes: ["select.order", "select.limit"] },
          { rule_code: "ir.unique_group_by", rule_version: "1-shadow", mode: "shadow", changed_ast_nodes: ["select.group"] },
        ],
        taxonomy: [
          { layer: "syntax", prefix: "sql.syntax", default: "llm" },
          { layer: "semantic", prefix: "semantic", default: "deterministic/llm" },
          { layer: "semantic_ir", prefix: "semantic_ir", default: "deterministic/llm" },
          { layer: "metric", prefix: "metric", default: "llm" },
          { layer: "security", prefix: "security", default: "reject" },
          { layer: "cost", prefix: "cost", default: "reject" },
        ],
        governance_layers: [
          { level: "L0", name: "拒绝", scope: "安全、权限、成本；禁止自动扩大边界" },
          { level: "L1", name: "确定性修复", scope: "唯一元数据边、唯一投影、唯一 IR 约束" },
          { level: "L2", name: "定向模型纠错", scope: "稳定 violation code + 有界指令 + 单次重试" },
          { level: "L3", name: "上游再生成", scope: "仅候选；需 Full Run 与影子证据后发布" },
        ],
        decision: {
          active_label: "P2", active_profile: "registry_same_rules_v1", fallback_label: "P0", fallback_profile: "inline_v1",
          active_unchanged_semantics: true, offline_candidates: ["P3", "P4", "P5", "P6"], release_qualified: false,
          release_blockers: ["missing_live_llm_targeted_correction_pairs", "missing_cold_warm_warm_full_run", "missing_shadow_canary_and_negative_feedback_review"],
          selection_reason: "P2 把既有两条修复规则迁入版本化注册表并保持 P0 字符兼容；扩展规则与模型纠错只做离线/影子评测。",
        },
        implementation_gates: { gate_passed: true, active_precision_100: true, active_false_trigger_zero: true, active_idempotence_100: true, active_unsafe_unchanged_100: true },
      });
    }
    if (path === "/api/admin/evaluation/semantic-repair/rollback" && method === "POST") {
      return ok({
        status: "rolled_back",
        runtime: { configured_profile: "registry_same_rules_v1", runtime_profile: "inline_v1", fallback_profile: "inline_v1", rollback_without_restart: true },
        message: "语义修复已在当前后端进程回切 P0 inline_v1；无需数据库迁移、索引重建或重启。",
      });
    }
    if (path === "/api/admin/evaluation/scenarios/matrix") {
      const capabilities = ["conversation", "chart", "knowledge", "adhoc", "masking", "rbac", "access_policy", "dashboard"];
      const cases = capabilities.flatMap((capability, capabilityIndex) => Array.from({ length: 7 }, (_, index) => ({
        id: `SCN-${String(capabilityIndex + 1).padStart(2, "0")}-${String(index + 1).padStart(2, "0")}`,
        capability, title: `${capability} 组合场景 ${index + 1}`,
        risk: index === 0 ? "critical" : index < 3 ? "high" : "medium",
        principal: ["owner", "admin", "analyst", "viewer", "removed"][index % 5],
        entrypoint: "POST /api/query", resource_state: index === 1 ? "foreign_workspace_id" : "active_resource",
        expected: { decision: index === 1 ? "deny" : "allow", http_status: index === 1 ? 404 : 200, side_effect: index === 1 ? "no_read" : "verified", trace: "required", failure_layer: index === 1 ? "resource" : "none" },
        evidence: { type: index % 2 ? "e2e" : "unit", target: `automated evidence ${capability}-${index + 1}` },
        tags: index === 1 ? ["cross_workspace"] : ["sql_entry"], status: "passed", duration_ms: 12.5, error: null,
      })));
      return ok({ available: true, generated_at: "2026-07-16T12:00:00+08:00", source: "standalone_runner", contract: { valid: true, errors: [], total: 56, capabilities: Object.fromEntries(capabilities.map(item => [item, 7])), roles: ["owner", "admin", "analyst", "viewer", "removed"], failure_layers: ["none", "identity", "resource", "policy", "sql", "dependency", "presentation"], matrix_version: "sha256:scenario-matrix" }, summary: { total: 56, executed: 56, passed: 56, failed: 0, not_run: 0, pass_rate: 1, gate_passed: true }, capabilities: capabilities.map(capability => ({ capability, total: 7, passed: 7, failed: 0, not_run: 0 })), cases });
    }
    const feedbackView = () => ({
      id: 7, chat_id: "chat-1", query_summary: "各地区销售额", sql_summary: "SELECT region, SUM(amount) FROM fact_sales GROUP BY region",
      score: -1, comment_summary: "华东金额不正确", created_at: "2026-07-16 10:00:00",
      review_status: state.feedbackReviewStatus, review_category: state.feedbackCategory, review_note: null,
      source_trace_id: "trace-1", candidate_type: state.candidateType, candidate_status: state.candidateStatus,
      candidate_question: state.candidateStatus === "none" ? "" : "各地区销售额",
      candidate_sql_summary: state.candidateStatus === "none" ? null : "SELECT region, SUM(amount) FROM fact_sales GROUP BY region",
      candidate_training_id: state.candidateStatus === "confirmed" && state.candidateType === "training" ? 19 : null,
      trace: { trace_id: "trace-1", status: "success", failure_code: "none", tt_complete_ms: 420, total_tokens: 320, version_run_id: "20260715_160000" },
    });
    if (path === "/api/admin/evaluation/quality/overview") return ok({
      available: true, days: Number(url.searchParams.get("days") || 7), filters: {},
      summary: { traces: 12, success_rate: .917, negative_rate: .125, result_proxy_rate: .833, p50_ms: 850, p95_ms: options.qualityP95 ?? 2300, tokens_mean: 540, estimated_cost: .018, correction_rate: .083, cache_hit_rate: .25, low_confidence_rate: .083, second_candidate_rate: .083, adaptive_repair_rate: 1, feedback_total: 8, negative_feedback: 1 },
      trends: [{ date: "2026-07-15", traces: 5, success_rate: .8, p50_ms: 900, p95_ms: 2500, tokens_mean: 610, estimated_cost: .01 }, { date: "2026-07-16", traces: 7, success_rate: 1, p50_ms: 780, p95_ms: 1900, tokens_mean: 490, estimated_cost: .008 }],
      versions: [{ run_fingerprint: "fp-online", version_run_id: "20260715_160000", version_complete: true, baseline_bound: true, traces: 12, success_rate: .917, p95_ms: 2300, tokens_mean: 540 }],
      categories: [{ category: "aggregation", count: 12 }],
      failure_funnel: [{ failure_code: "sql_execution", count: 1 }],
      slow_nodes: [{ node_id: "generate_sql", calls: 12, failed: 0, mean_ms: 480, p95_ms: 900, total_ms: 5760, tokens: 3200 }],
      feedback_queue: [feedbackView()],
      recent_traces: [{ trace_id: "trace-1", query_summary: "各地区销售额", status: "success", failure_code: "none", query_category: "aggregation", candidate_confidence: .86, candidate_band: "high", adaptive_candidate_triggered: false, tt_complete_ms: 420, total_tokens: 320, estimated_cost: .001, cache_hit: true, version_run_id: "20260715_160000", version_complete: true, started_at: "2026-07-16 10:00:00" }],
      evaluation_runs: [],
      candidate_policy: { available: true, source: "deterministic_candidate_policy_matrix", summary: { baseline_success_rate: .8333, adaptive_success_rate: .9667, second_candidate_rate: .1667, targeted_repair_rate: .8, semantic_contract_cases: 3, semantic_contract_passed: 3, semantic_false_positive_count: 0, unsafe_selected: 0, high_confidence_extra_calls: 0, gate_passed: true } },
      semantic_repair: { observed_traces: 12, changed_traces: 2, change_rate: .1667, fallback_count: 0, fallback_rate: 0, repair_p95_ms: 1.8, profiles: [{ profile: "registry_same_rules_v1", count: 12 }], rules: [{ rule_code: "join.unique_metadata_edge", count: 2 }], violations: [{ violation_code: "semantic.join.missing_predicate", count: 1 }] },
      runtime_governance: { semantic_coverage_observed: 12, semantic_coverage_rate: 1, semantic_coverage_mean: 1, semantic_unresolved_traces: 0, semantic_unsafe_compile_traces: 0, deadline_observed: 12, deadline_coverage_rate: 1, deadline_exhausted_traces: 0, deadline_contract_complete: true, llm_attempt_accounting: { schema: "llm-attempt-v2", attempts: 12, succeeded: 12, failed: 0, recoveries: 1, provider_request_id_unavailable: 2, evidence_complete: true }, llm_queue_wait_p95_ms: 18, llm_provider_p95_ms: 880, llm_outcome_classes: { success: 12 } },
    });
    if (path === "/api/admin/evaluation/quality/traces/trace-1") return ok({
      trace_id: "trace-1", query_summary: "那销售额呢",
      sql_summary: "SELECT category, SUM(amount) FROM fact_sales GROUP BY category",
      status: "success", failure_code: "none", query_category: "aggregation",
      version_run_id: "20260715_160000", run_fingerprint: "fp-online",
      version_complete: true, cache_hit: true, row_count: 2,
      tt_complete_ms: 420, total_tokens: 320, estimated_cost: .001,
      follow_up_action: "replace",
      follow_up_summary: "follow_up_action=replace；变更 指标；继承维度 1 项",
      candidate_decision: {
        version: "candidate-confidence-v2", adaptive_triggered: true,
        candidate_count: 2, selected: "B",
        selection_reason: "alternative_has_material_confidence_gain",
        candidates: [
          { label: "A", valid: true, score: .61, band: "medium", weaknesses: ["display_column_missing"], error_code: null },
          { label: "B", valid: true, score: .88, band: "high", weaknesses: [], error_code: null },
        ],
      },
      planning_route: {
        contract_version: "planning-route-v1", profile: "keyword_length_v1",
        decision: "direct_generate", actual_action: "direct_generate", score: 0,
        confidence: .84, reason_codes: ["legacy_no_trigger"], decision_ms: .23,
        raw_text_logged: false,
        shadow_decisions: [
          { profile: "structured_rules_v1", decision: "direct_generate" },
          { profile: "weighted_scorecard_v1", decision: "direct_generate" },
        ],
      },
      semantic_repair: {
        contract_version: "semantic-repair-result-v1",
        profile: "registry_same_rules_v1", changed: true, repair_ms: .72,
        fallback_used: false, raw_sql_logged: false,
        original_sql_hash: "sha256:before-demo",
        repaired_sql_hash: "sha256:after-demo",
        applied_rules: [{ rule_code: "join.unique_metadata_edge", rule_version: "1", changed_ast_nodes: ["join.on"] }],
        violation: { violation_code: "semantic.join.missing_predicate", repairability: "deterministic" },
      },
      semantic_coverage: {
        schema_version: "semantic-coverage-v1",
        registry_version: "semantic-composition-v1",
        question_fingerprint: "sha256:trace",
        coverage_score: 1, compiler_mode: "deterministic_ast",
        compiler_supported: true,
        detected_slots: ["metric:sum", "dimension:category"],
        represented_slots: ["metric:sum", "dimension:category"],
        unresolved_slots: [], abstain_reasons: [],
        raw_question_persisted: false, filter_values_persisted: false,
      },
      request_deadline: {
        contract_version: "request-deadline-v1", budget_ms: 30000,
        remaining_ms: 25100, execution_reserve_ms: 3000,
        phase_budgets_ms: { llm: 24000, execution: 3000 },
        absolute_timestamps_persisted: false,
      },
      retrieval_evidence: [{
        version: "rule-rerank-v6", route: "column", selected_count: 4,
        dynamic_k: 4, fusion_method: "entity_rrf",
        query_signals: {
          contract_version: "query-signals-v1", extractor_profile: "jieba_tfidf_v1",
          route_policy: "current_multi_query_v1", local_signal_count: 3,
          remote_query_count: 3, rerank_signal_count: 3, query_encode_mode: "query",
          actual_remote_signals: [{ kind: "standalone_query" }, { kind: "lexical_terms" }],
          raw_text_logged: false,
        },
        hybrid_queries: [{
          retrieval_mode: "hybrid", effective_mode: "hybrid",
          fusion_method: "entity_rrf", dense: { raw_points: 8, unique_entities: 8 },
          sparse: { raw_points: 6, unique_entities: 6 }, fused_entities: 8,
          point_strategies: ["single_canonical"], matched_point_roles: ["canonical"],
        }],
      }],
      llm_attempt_accounting: {
        schema: "llm-attempt-v2", attempts: 1, succeeded: 1,
        core_failures: 0, recoveries: 0, provider_request_id_unavailable: 1,
        callback_missing: 0, evidence_complete: true,
      },
      llm_attempts: [{
        attempt_id: "attempt-trace-1", node_id: "generate_sql", route: "sql",
        model: "test-strong", status: "success", reason_code: "llm.success",
        outcome_class: "success", latency_ms: 900, timeout_budget_ms: 6000,
        physical_attempt_budget_ms: 5000, recovery_reserve_ms: 1000,
        queue_wait_ms: 20, provider_ms: 880, deadline_before_ms: 27000,
        deadline_after_ms: 26100, retry_index: 0, recovery: false,
        provider_request_id: "unavailable", provider_request_id_available: false,
        provider_request_id_unavailable_reason: "provider_request_id_not_exposed",
        token_known: true, token_timing_unavailable_reason: "token_timing_not_exposed",
        total_tokens: 320, affects_core_path: true,
      }],
      steps: [
        { step_name: "rewrite_follow_up", status: "success", duration_ms: 1, attempt: 1, error_summary: "" },
        { step_name: "decide_plan_route", status: "success", duration_ms: .23, attempt: 1, error_summary: "" },
        { step_name: "judge_candidate", status: "success", duration_ms: 180, attempt: 1, error_summary: "" },
      ],
    });
    if (path === "/api/admin/evaluation/quality/traces/trace-1") return ok({ trace_id: "trace-1", query_summary: "那销售额呢", sql_summary: "SELECT category, SUM(amount) FROM fact_sales GROUP BY category", status: "success", failure_code: "none", query_category: "aggregation", version_run_id: "20260715_160000", run_fingerprint: "fp-online", version_complete: true, cache_hit: true, row_count: 2, tt_complete_ms: 420, total_tokens: 320, estimated_cost: .001, follow_up_action: "replace", follow_up_summary: "follow_up_action=replace；变更 指标；继承维度 1 项", candidate_decision: { version: "candidate-confidence-v2", adaptive_triggered: true, candidate_count: 2, selected: "B", selection_reason: "alternative_has_material_confidence_gain", candidates: [{ label: "A", valid: true, score: .61, band: "medium", weaknesses: ["display_column_missing"], error_code: null }, { label: "B", valid: true, score: .88, band: "high", weaknesses: [], error_code: null }] }, planning_route: { contract_version: "planning-route-v1", profile: "keyword_length_v1", decision: "direct_generate", actual_action: "direct_generate", score: 0, confidence: .84, reason_codes: ["legacy_no_trigger"], decision_ms: .23, raw_text_logged: false, shadow_decisions: [{ profile: "structured_rules_v1", decision: "direct_generate" }, { profile: "weighted_scorecard_v1", decision: "direct_generate" }] }, semantic_repair: { contract_version: "semantic-repair-result-v1", profile: "registry_same_rules_v1", changed: true, repair_ms: .72, fallback_used: false, raw_sql_logged: false, original_sql_hash: "sha256:before-demo", repaired_sql_hash: "sha256:after-demo", applied_rules: [{ rule_code: "join.unique_metadata_edge", rule_version: "1", changed_ast_nodes: ["join.on"] }], violation: { violation_code: "semantic.join.missing_predicate", repairability: "deterministic" } }, retrieval_evidence: [{ version: "rule-rerank-v6", route: "column", selected_count: 4, dynamic_k: 4, fusion_method: "entity_rrf", query_signals: { contract_version: "query-signals-v1", extractor_profile: "jieba_tfidf_v1", route_policy: "current_multi_query_v1", local_signal_count: 3, remote_query_count: 3, rerank_signal_count: 3, query_encode_mode: "query", actual_remote_signals: [{ kind: "standalone_query" }, { kind: "lexical_terms" }], raw_text_logged: false }, hybrid_queries: [{ retrieval_mode: "hybrid", effective_mode: "hybrid", fusion_method: "entity_rrf", dense: { raw_points: 8, unique_entities: 8 }, sparse: { raw_points: 6, unique_entities: 6 }, fused_entities: 8, point_strategies: ["single_canonical"], matched_point_roles: ["canonical"] }] }], steps: [{ step_name: "rewrite_follow_up", status: "success", duration_ms: 1, attempt: 1, error_summary: "" }, { step_name: "decide_plan_route", status: "success", duration_ms: .23, attempt: 1, error_summary: "" }, { step_name: "judge_candidate", status: "success", duration_ms: 180, attempt: 1, error_summary: "" }] });
    if (path === "/api/admin/evaluation/quality/feedback/7/review" && method === "PATCH") { const payload = await bodyOf(request); state.feedbackReviewStatus = payload.status; state.feedbackCategory = payload.category; return ok(feedbackView()); }
    if (path === "/api/admin/evaluation/quality/feedback/7/candidate" && method === "POST") { const payload = await bodyOf(request); state.candidateType = payload.candidate_type; state.candidateStatus = "draft"; return ok(feedbackView()); }
    if (path === "/api/admin/evaluation/quality/feedback/7/candidate/confirm" && method === "POST") { state.candidateStatus = "confirmed"; return ok(feedbackView()); }
    if (path.startsWith("/api/admin/evaluation/")) return ok({});
    return ok({});
  });
  return state;
}

export async function openAs(page, role = "owner", options = {}) {
  const state = await installMockApi(page, { ...options, role });
  await page.addInitScript((token) => localStorage.setItem("da_access_token", token), state.session.access_token);
  await page.goto("/");
  return state;
}
