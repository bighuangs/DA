# 定义yaml文件的路径
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

# 在加载 yaml 配置前先加载 .env 文件，将敏感信息（数据库密码、LLM api_key）
# 注入环境变量，供 yaml 中的 ${oc.env:VAR_NAME} 插值解析使用。
# .env 里的值不会覆盖系统或当前进程中已经存在的环境变量
_load_dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=_load_dotenv_path, override=False)

# ======日志配置类型=========
# @dataclass 是 Python 的一个装饰器，用来把类变成“主要用于存数据的类”。
# 它会自动帮你生成常用方法，例如：
# __init__：自动初始化字段
# __repr__：方便打印对象
# __eq__：方便比较对象是否相等
# 定义数据的模型
@dataclass
class File:
    enable: bool  # 是否启用文件日志输出
    level: str  # 写入文件的最低日志级别，如 INFO、WARNING
    path: str  # 日志文件或日志目录的保存路径
    rotation: str  # 日志轮转规则，如按文件大小或时间切分
    retention: str  # 轮转日志的保留时长或保留数量


@dataclass
class Console:
    enable: bool  # 是否启用控制台日志输出
    level: str  # 输出到控制台的最低日志级别


@dataclass
class LoggingConfig:
    file: File  # 文件日志配置
    console: Console  # 控制台日志配置

# ==================== database配置模型 ====================

@dataclass
class DBConfig:
    host: str  # 数据库服务器地址或主机名
    port: int  # 数据库服务端口
    user: str  # 数据库连接用户名
    password: str  # 数据库连接密码，通常通过环境变量注入
    database: str  # 连接后默认使用的数据库名称

# ==================== Qdrant 配置模型 ====================

@dataclass
class QdrantConfig:
    host: str  # Qdrant 向量数据库地址
    port: int  # Qdrant 服务端口
    embedding_size: int  # 向量维度，需与嵌入模型输出维度一致
    dense_vector_name: str = "dense"  # BGE-M3 稠密命名向量
    sparse_vector_name: str = "sparse"  # BGE-M3 lexical sparse 命名向量
    trust_env: bool = False  # 服务连接默认绕过桌面/系统 HTTP 代理


# ==================== Embedding 配置模型 ====================


@dataclass
class EmbeddingSchedulerConfig:
    """TODO-60 进程内动态微批调度参数。

    队列容量按“请求”计数，batch 上限按文本数与近似 Token 双重约束。
    offline_corpus_quota 表示有离线任务等待时，最多允许连续执行多少个在线批次，
    从而在在线优先的同时避免索引任务永久饥饿。
    """

    queue_capacity: int = 128
    # 0 表示同一事件循环调度轮次内的机会式合批，不给空闲 query 人为增加等待。
    # 只有目标 Linux/GPU 压测证明有收益时才配置正等待窗口。
    batch_window_ms: float = 0.0
    max_batch_texts: int = 32
    max_batch_tokens: int = 8192
    request_deadline_ms: int = 45000
    online_query_priority: int = 0
    offline_corpus_priority: int = 10
    offline_corpus_quota: int = 4
    shutdown_drain_timeout_ms: int = 30000
    metrics_window_size: int = 512

@dataclass
class EmbeddingConfig:
    model: str  # BGE-M3 模型名称，用于索引合同和可观测证据
    model_path: str  # DA 自己加载的外部模型目录，不依赖其他项目
    revision: str = "local"  # 模型 revision/制品版本，进入索引指纹
    dense_size: int = 1024  # BGE-M3 dense 向量维度
    max_length: int = 512  # 元知识短文本编码上限
    batch_size: int = 8  # 本地编码批大小
    device: str = "auto"  # auto/cpu/cuda:0
    use_fp16: bool = True  # 仅 CUDA 设备启用；CPU 自动关闭
    normalize_dense: bool = True  # Qdrant cosine 前保持单位向量
    warmup_on_startup: bool = False  # false 时只做依赖/模型路径检查，重建时加载模型
    runtime_mode: str = "microbatch"  # serial / microbatch；remote 仅在容量触发后建设
    scheduler: EmbeddingSchedulerConfig = field(
        default_factory=EmbeddingSchedulerConfig
    )


# ==================== Point 投影配置模型 ====================

@dataclass
class PointProjectionConfig:
    strategy: str = "single_canonical"  # single_canonical / multi_all / mixed_selective
    namespace_uuid: str = "6f838686-6c2f-5f66-9015-fc258910fa41"  # UUIDv5 固定命名空间
    identity_version: str = "point-identity-v2"  # v2 仅由业务身份/role/alias 生成 ID
    canonical_template_version: str = "entity-canonical-v1"  # 结构化主 Point 模板版本
    alias_policy_version: str = "approved-alias-v1"  # 选择性别名准入规则版本
    alias_manifest_path: str = "conf/point_projection_aliases.yaml"  # 人工批准别名清单
    max_alias_points_per_entity: int = 1  # mixed_selective 每实体最多附加几个别名 Point


# ==================== ES 配置模型 ====================

@dataclass
class ESConfig:
    host: str  # Elasticsearch 服务地址
    port: int  # Elasticsearch 服务端口
    index_name: str  # 字段值索引的基础名称；运行时按工作空间派生 active alias
    value_contract_version: str = "value-index-v2"  # 字段值文档/发布合同版本
    value_mapping_version: str = "value-mapping-v2"  # mapping/analyzer 合同版本
    value_normalization_version: str = "nfkc-casefold-v1"  # 精确值规范化合同
    value_analyzer: str = "ik_max_word"  # value 全文检索 analyzer
    max_distinct_values_per_column: int = 10000  # 单字段允许发布的 distinct 值上限
    bulk_batch_documents: int = 500  # 单次 ES bulk 的文档数量
    retain_previous_versions: int = 2  # active 之外保留的历史物理索引数量
    allow_sensitive_columns: list[str] = field(default_factory=list)  # 显式批准的敏感字段 glob
    allow_high_cardinality_columns: list[str] = field(default_factory=list)  # 显式批准的高基数字段 glob


# ==================== Redis / 会话记忆配置模型 ====================

@dataclass
class RedisConfig:
    host: str = "127.0.0.1"  # Redis 服务地址
    port: int = 6379  # Redis 服务端口
    database: int = 0  # DA 独占的逻辑数据库编号
    password: str | None = None  # 可选密码；本地开发默认不启用
    socket_timeout_seconds: float = 1.5  # 单次命令超时，超时后降级到 MySQL
    connect_timeout_seconds: float = 1.5  # 建连超时
    max_connections: int = 50  # asyncio 连接池上限
    required_on_startup: bool = False  # false 时 Redis 故障不阻断新话题查询


@dataclass
class MemoryConfig:
    enabled: bool = True  # 是否启用 TODO-59 分层会话记忆
    working_ttl_seconds: int = 86400  # Redis working state 的滑动 TTL
    recent_message_limit: int = 20  # MySQL 重建时最多读取的消息数
    recent_token_budget: int = 900  # 最近消息注入预算
    context_token_budget: int = 700  # 长短期记忆总注入预算
    long_term_enabled: bool = True  # 是否启用 ES/Qdrant 长期记忆双路召回
    long_term_top_k: int = 5  # 融合后最多回表的长期记忆条数
    es_index_name: str = "data-agent-conversation-memory"
    es_projection_version: str = "conversation-memory-lexical-v1"
    qdrant_collection_name: str = "data-agent_conversation_memories_v1"
    qdrant_projection_version: str = "conversation-memory-vector-v2"
    extractor_version: str = "memory-rule-extractor-v1"
    context_schema_version: str = "memory-context-v2"
    fusion_rrf_k: int = 60
    retrieval_timeout_seconds: float = 5.0
    projection_poll_seconds: float = 3.0
    projection_batch_size: int = 20
    projection_max_attempts: int = 8
    projection_lease_seconds: int = 90
    message_content_max_chars: int = 2000
    memory_content_max_chars: int = 1000
    summary_enabled: bool = False
    summary_build_enabled: bool = False
    summary_projection_enabled: bool = False
    summary_turn_high_watermark: int = 12
    summary_turn_low_watermark: int = 4
    summary_token_high_watermark: int = 1800
    summary_token_low_watermark: int = 700
    summary_recent_protected_turns: int = 4
    summary_segment_max_messages: int = 16
    summary_segment_max_tokens: int = 1200
    summary_content_max_chars: int = 2400
    summary_signal_limit: int = 12
    summary_top_k: int = 3
    summary_hierarchy_group_size: int = 4
    summary_max_level: int = 2
    summary_retrieval_always: bool = False
    summary_extractor_version: str = "summary-extractive-v1"
    summary_signal_extractor_version: str = "summary-signals-rule-v1"
    summary_schema_version: str = "summary-segment-v1"
    summary_es_index_name: str = "data-agent-conversation-summaries"
    summary_es_projection_version: str = "conversation-summary-lexical-v2"
    summary_qdrant_collection_name: str = "data-agent_conversation_summaries_v2"
    summary_qdrant_projection_version: str = "conversation-summary-vector-v2"
    summary_points_per_segment: int = 1
    stable_fact_enabled: bool = False
    stable_fact_extractor_version: str = "stable-fact-rule-v1"


# ==================== LLM 配置模型 ====================

@dataclass
class LLMConfig:
    model_name: str  # 默认使用的 LLM 模型名称
    api_key: str = field(repr=False)  # 调用 LLM 服务的 API 密钥，通常通过环境变量注入
    provider: str = "deepseek"  # 部署供应商配置名，例如 deepseek / dashscope
    api_key_env: str = field(default="LLM_API_KEY", repr=False)  # 默认凭证环境变量名
    base_url: str | None = field(default=None, repr=False)  # OpenAI-compatible 地址
    providers: dict[str, dict] = field(default_factory=dict, repr=False)  # provider 合同
    timeout_seconds: float = 30.0  # 单次 LLM 调用的超时时间（秒）
    max_retries: int = 1  # 单次调用失败后的最大重试次数
    max_completion_tokens: int = 1024  # 模型单次生成内容的最大 Token 数
    input_cost_per_million: float = 0.0  # 每百万输入 Token 的费用；0 表示不估算费用
    output_cost_per_million: float = 0.0  # 每百万输出 Token 的费用；0 表示不估算费用
    cost_currency: str = "USD"  # Token 费用统计所使用的币种
    max_concurrency: int = 12  # 单实例跨路由 provider 并发上限
    thinking_mode: str = "auto"  # 通用思考模式：auto/enabled/disabled；具体映射由 provider 适配
    reasoning_effort: str | None = None  # thinking-only 模型的推理强度
    temperature: float = 0.0  # 默认确定性采样；thinking-only 路由必须显式覆盖
    routes: dict[str, dict] = field(default_factory=dict, repr=False)  # 按任务覆盖部署与预算


# ==================== 多轮对话配置模型 ====================

@dataclass
class ChatConfig:
    history_limit: int  # 生成 SQL 时注入的最近对话历史轮数
    directory_page_size: int = 20  # 会话目录首屏/默认分页条数
    transcript_page_size: int = 30  # transcript 默认分页回合数
    history_result_preview_rows: int = 100  # 历史回放单回合最多返回多少行
    history_result_preview_bytes: int = 200_000  # 历史结果预览 JSON 字节上限
    title_max_chars: int = 80  # 首问确定性标题与手工重命名长度上限


# ==================== QuerySignals 配置模型 ====================

@dataclass
class QuerySignalConfig:
    profile: str = "jieba_tfidf_v1"  # full_query_only / jieba_tfidf_v1 / domain_signals_v1
    route_policy: str = "current_multi_query_v1"  # full_query_only_v1/current_multi_query_v1/routed_signals_v1
    lexical_top_k: int = 10
    remote_signal_limit: int = 4
    workspace_dictionary_enabled: bool = True
    custom_idf_enabled: bool = False
    preserve_identifier: bool = True
    preserve_quoted_phrase: bool = True
    preserve_modifier: bool = True
    log_plaintext: bool = False
    cache_namespace_version: str = "query-signals-cache-v1"


# ==================== SQL 规划路由配置模型 ====================

@dataclass
class PlanningRouteConfig:
    active_profile: str = "keyword_length_v1"
    fallback_profile: str = "keyword_length_v1"
    shadow_profiles: list[str] = field(default_factory=lambda: [
        "structured_rules_v1",
        "weighted_scorecard_v1",
        "calibrated_linear_v1",
    ])
    contract_version: str = "planning-route-v1"
    feature_schema: str = "planning-features-v1"
    repair_profile: str = "semantic-repair-inline-v1"
    cache_namespace_version: str = "planning-route-cache-v1"
    log_plaintext: bool = False
    direct_threshold: float = 0.35
    plan_threshold: float = 0.62
    linear_direct_threshold: float = 0.35
    linear_plan_threshold: float = 0.62
    scorecard_weights: dict[str, float] = field(default_factory=dict)
    linear_coefficients: dict[str, float] = field(default_factory=dict)


# ==================== 语义守卫修复配置模型 ====================

@dataclass
class SemanticRepairConfig:
    active_profile: str = "registry_same_rules_v1"
    fallback_profile: str = "inline_v1"
    shadow_profiles: list[str] = field(default_factory=lambda: [
        "registry_extended_v1",
        "targeted_correction_v1",
        "cascade_v1",
    ])
    contract_version: str = "semantic-repair-result-v1"
    registry_version: str = "semantic-repair-registry-v1"
    rule_set_version: str = "semantic-repair-rules-v2"
    max_passes: int = 1
    log_plaintext: bool = False


# ==================== 查询执行配置模型 ====================

@dataclass
class QueryConfig:
    limit_rows: int  # 执行 SQL 时强制附加的行数上限，防止大结果集拖垮服务（无 LIMIT 时包装子查询追加）
    cost_guard_enabled: bool = True  # 是否启用 EXPLAIN 扫描成本守卫
    max_explain_rows: int = 5000000  # 允许中型/大型正式基线聚合，同时阻断笛卡尔级计划
    cache_enabled: bool = True  # 是否启用进程内查询结果缓存
    cache_ttl_seconds: int = 30  # 缓存 TTL，适合短时间重复查询
    cache_max_entries: int = 256  # LRU 最大缓存条目数
    export_max_rows: int = 100000  # 单次导出允许的最大行数
    export_max_columns: int = 200  # 单次导出允许的最大列数
    export_max_cells: int = 2_000_000  # 单次导出允许的最大单元格数量
    export_max_cell_chars: int = 32767  # 导出时单个单元格允许的最大字符数
    chart_llm_enabled: bool = False  # 是否使用 LLM 生成图表建议；关闭时使用确定性规则
    recommend_llm_enabled: bool = False  # 是否使用 LLM 生成后续问题推荐
    value_recall_llm_enabled: bool = False  # 是否使用 LLM 扩展字段值召回关键词
    keyword_recall_llm_enabled: bool = False  # 是否使用 LLM 扩展字段和指标召回关键词
    selection_llm_enabled: bool = False  # 是否使用 LLM 对动态召回结果进行二次筛选
    planning_llm_enabled: bool = False  # 规划默认使用结构化确定性证据；显式开启才调用可选规划模型
    complex_sql_llm_enabled: bool = True  # 启用 SQL 模型路由；正常 DW 问数的默认正确性策略由下方直出开关决定
    deterministic_sql_direct_emission_enabled: bool = False  # 显式消融/兼容开关；默认禁止 compiler 绕过 generate_sql 直出
    deterministic_sql_model_fallback_enabled: bool = False  # 显式消融/兼容开关；模型预算不足或候选被拒时默认不执行自证 coverage 的 compiler 草案
    complex_sql_llm_slot_threshold: int = 6  # aggregate 检出槽位达到该值时不再直接信任模板编译
    postprocess_timeout_seconds: float = 20.0  # 图表、推荐等后处理单阶段的超时时间（秒）
    analysis_enabled_default: bool = False  # AI 数据解读功能是否默认开启
    analysis_timeout_seconds: float = 30.0  # AI 数据解读的超时时间（秒）
    postprocess_budget_ms: int = 60000  # 可选后处理流程的总时间预算（毫秒）
    max_concurrent_queries: int = 20  # 单个服务实例允许的最大并发查询数
    max_concurrent_queries_per_workspace: int = 5  # 单个工作空间允许的最大并发查询数
    concurrency_queue_timeout_seconds: float = 2.0  # 等待并发槽位的超时时间，超时后返回 429（秒）
    optional_dependency_timeout_seconds: float = 5.0  # 可降级依赖服务单阶段的超时时间（秒）
    evaluation_sql_timeout_seconds: float = 6.0  # 评测完整结果复查的单条只读 SQL 上限（秒）
    mysql_cancel_control_timeout_seconds: float = 2.0  # 独立控制连接建连并终止目标会话的总上限（秒）
    mysql_cancel_drain_timeout_seconds: float = 1.0  # 服务端终止后等待业务查询协程收敛的上限（秒）
    request_deadline_ms: int = 60000  # 覆盖必要 SQL 生成、一次语义复核与最终执行的有界总预算
    request_phase_budgets_ms: dict[str, int] = field(default_factory=lambda: {
        "retrieval": 7000,
        "planning": 6000,
        "generation": 22000,
        "correction": 16000,
        "execution": 6000,
    })
    request_execution_reserve_ms: int = 4000  # 模型阶段必须为授权与正常 SQL 执行保留的预算
    llm_recovery_reserve_ms: int = 5000  # 为最多一次紧凑恢复保留尾部预算；可显式设为 0 做消融
    llm_min_attempt_ms: int = 750  # 剩余预算低于该值时不再发起物理模型调用
    llm_recovery_min_budget_ms: int = 4000  # 紧凑恢复至少需要的有效窗口
    planning_llm_min_start_budget_ms: int = 10000  # 可选规划模型启动前需保留的请求预算
    generation_llm_min_start_budget_ms: int = 12000  # 必要 SQL 生成调用的最小可完成预算
    correction_llm_min_start_budget_ms: int = 9000  # 纠错调用的最小可完成预算
    generation_llm_primary_min_budget_ms: int = 12000  # 预留恢复前必须保证的主生成窗口
    correction_llm_primary_min_budget_ms: int = 9000  # 预留恢复前必须保证的主纠错窗口
    llm_route_default_concurrency: int = 6  # 未显式配置时单路由并发上限
    column_recall_score_threshold: float = 0.35  # BGE-M3 dense 字段召回下限（按当前模型分数重标定）
    column_recall_limit: int = 8  # 最终保留的字段召回候选数量上限
    column_recall_min_limit: int = 3  # 字段召回动态收紧时保留的最小候选数
    column_recall_pool_limit: int = 16  # 每个关键词初步召回的字段候选池大小
    metric_recall_score_threshold: float = 0.58  # 指标召回结果保留所需的最低相似度分数
    metric_recall_limit: int = 5  # 最终保留的指标召回候选数量上限
    metric_recall_min_limit: int = 1  # 指标召回动态收紧时保留的最小候选数
    metric_recall_pool_limit: int = 10  # 每个关键词初步召回的指标候选池大小
    training_recall_score_threshold: float = 0.62  # 训练样例召回结果保留所需的最低相似度分数
    training_recall_limit: int = 3  # 最终保留的训练样例候选数量上限
    training_recall_min_limit: int = 1  # 训练样例动态收紧时保留的最小候选数
    training_recall_pool_limit: int = 6  # 每个关键词初步召回的训练样例候选池大小
    column_sparse_score_threshold: float = 0.0  # BGE-M3 sparse 字段召回下限，独立于 cosine
    metric_sparse_score_threshold: float = 0.0  # BGE-M3 sparse 指标召回下限
    training_sparse_score_threshold: float = 0.0  # BGE-M3 sparse样例召回下限
    hybrid_rrf_k: int = 60  # 实体级 reciprocal-rank fusion 常数
    hybrid_oversample_factor: int = 3  # 物理 Point 召回后先按实体去重，再截最终 K
    hybrid_retrieval_timeout_seconds: float = 45.0  # 本地编码 + 双路 Qdrant 的有界时限
    hybrid_allow_sparse_only_fallback: bool = False  # dense 故障时默认只保留 Meta exact/alias
    value_recall_limit: int = 8  # 最终保留的字段值召回候选数量上限
    value_recall_min_limit: int = 2  # 字段值召回动态收紧时保留的最小候选数
    value_recall_pool_limit: int = 12  # 每个关键词初步召回的字段值候选池大小
    retrieval_keyword_limit: int = 4  # 单次召回可使用的关键词数量上限
    retrieval_dynamic_score_gap: float = 0.08  # 候选分数差小于该值时保留相邻候选
    retrieval_cache_enabled: bool = True  # 是否缓存检索结果
    retrieval_cache_ttl_seconds: int = 120  # 检索结果缓存的有效期（秒）
    retrieval_cache_max_entries: int = 512  # 检索结果 LRU 缓存的最大条目数
    context_total_token_budget: int = 2200  # SQL Prompt 上下文的总 Token 预算
    context_schema_token_budget: int = 1300  # SQL Prompt 中 Schema 信息的 Token 预算
    context_metric_token_budget: int = 350  # SQL Prompt 中指标信息的 Token 预算
    context_training_token_budget: int = 450  # SQL Prompt 中训练样例的 Token 预算
    context_max_columns_per_table: int = 12  # 注入 Prompt 时每张表最多携带的字段数
    sql_semantic_review_enabled: bool = True  # 对每条已通过确定性校验的 SQL 再做一次独立语义复核；可关闭用于消融
    adaptive_candidate_enabled: bool = True  # 是否按置信度动态生成备选 SQL
    candidate_confidence_high: float = 0.78  # 视为高置信、无需备选 SQL 的阈值
    candidate_confidence_low: float = 0.58  # 低于此值时判定为低置信的阈值
    candidate_min_confidence_gain: float = 0.03  # 备选 SQL 被采用所需的最小置信度提升
    adaptive_candidate_deadline_ms: int = 52000  # 语义复核/备选 SQL 的最晚启动窗口（毫秒）
    adaptive_candidate_timeout_seconds: float = 18.0  # 单次语义复核/备选生成调用上限（秒）
    prompt_history_limit: int = 2  # 二次压缩后保留的历史对话轮数
    prompt_training_limit: int = 2  # 二次压缩后保留的训练样例数量
    prompt_column_example_limit: int = 0  # 每列保留的示例值数量；0 表示不保留
    prompt_description_max_chars: int = 120  # Prompt 中字段或表描述允许的最大字符数
    recommend_prompt_token_limit: int = 2000  # 推荐 LLM 输入的 Token 估算上限


@dataclass
class AdhocConfig:
    max_file_bytes: int = 10 * 1024 * 1024  # 单个上传文件允许的最大字节数
    max_rows: int = 20000  # 单个上传文件允许解析的最大行数
    max_columns: int = 100  # 单个上传文件允许解析的最大列数
    max_cell_chars: int = 1000  # 单元格内容允许的最大字符数
    ttl_seconds: int = 3600  # 上传文件生成的临时数据有效期（秒）
    managed_ttl_seconds: int = 30 * 24 * 3600  # 受管上传版本默认保留 30 天
    max_extension_seconds: int = 30 * 24 * 3600  # 单次续期最大时长
    max_active_datasets: int = 20  # 单用户/工作空间可查询版本数量
    max_total_bytes: int = 100 * 1024 * 1024  # 活跃上传版本总容量
    max_total_rows: int = 200000  # 活跃上传版本总行数
    max_selected_datasets: int = 2  # 一轮上下文允许选择的上传版本数
    max_compare_tables: int = 50  # 可参与比较的最大临时表数量
    max_compare_columns: int = 50  # 单次表比较允许的最大列数


@dataclass
class MaskingConfig:
    enabled: bool = True  # 是否启用敏感数据脱敏
    heuristic_enabled: bool = True  # 是否启用基于规则的敏感信息识别
    default_strategy: str = "mask"  # 未指定规则时使用的默认脱敏策略
    history_retention_days: int = 30  # 含敏感信息的历史记录保留天数


@dataclass
class AuthConfig:
    enabled: bool = True  # 是否启用用户认证与授权
    secret_key: str = "dev-only-change-me"  # 用于签名和验证访问令牌的密钥
    access_token_ttl_minutes: int = 480  # 访问令牌的有效期（分钟）


# ==================== 应用总配置模型 ====================

@dataclass
class AppConfig:
    logging: LoggingConfig  # 日志相关配置
    db_meta: DBConfig  # 元数据数据库连接配置
    db_dw: DBConfig  # 数据仓库数据库连接配置
    qdrant: QdrantConfig  # Qdrant 向量数据库配置
    embedding: EmbeddingConfig  # 文本嵌入推理服务配置
    point_projection: PointProjectionConfig  # 元知识实体到 Qdrant Point 的投影合同
    es: ESConfig  # Elasticsearch 全文检索配置
    redis: RedisConfig  # 在线短期工作记忆
    memory: MemoryConfig  # 分层会话记忆及长期检索配置
    llm: LLMConfig  # 大语言模型调用配置
    chat: ChatConfig  # 多轮对话配置
    query_signals: QuerySignalConfig  # 版本化查询信号与路由合同
    planning_route: PlanningRouteConfig  # SQL 规划必要性决策、shadow 与回退合同
    semantic_repair: SemanticRepairConfig  # 语义守卫诊断、注册表和分层纠错合同
    query: QueryConfig  # 查询执行、检索与 Prompt 预算配置
    adhoc: AdhocConfig  # 文件上传查询的安全边界配置
    masking: MaskingConfig  # 数据脱敏与合规配置
    auth: AuthConfig  # 用户认证与令牌配置

# yaml配置文件的路径
_yaml_path = Path(__file__).parents[2] / "conf/app_config.yaml"

# 加载yaml文件
_yaml_data = OmegaConf.load(_yaml_path)

# 将_yaml_data转换为指定类型AppConfig的对象
app_config:AppConfig = OmegaConf.to_object(OmegaConf.merge(AppConfig, _yaml_data))
