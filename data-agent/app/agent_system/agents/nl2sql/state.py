import operator
from typing import Annotated, NotRequired, TypedDict

from app.agent_system.agents.nl2sql.constants import MAX_RETRY as MAX_RETRY
from app.agent_system.agents.nl2sql.contracts.retrieval import (
    ColumnCandidate,
    MetricCandidate,
    ValueCandidate,
)

# 当前日期时间的状态信息
class DateInfoState(TypedDict):
    date: str # 年月日
    weekday: str # 星期
    quarter: str # 季度  Q1-Q4

# 数据库信息
class DBInfoState(TypedDict):
    version: str # 版本号
    dialect: str # 数据库名称

class MetricInfoState(TypedDict):
    name:str
    description:str
    relevant_columns:list
    alias:list
    formula_rule: dict | None

class ColumnInfoState(TypedDict):
    name: str
    type:str
    role: str
    examples:list
    description:str
    alias: list
    is_display: bool
    value_match_hints: NotRequired[list[dict]]

class TableInfoState(TypedDict):
    name:str
    role:str
    description:str
    columns: list[ColumnInfoState]

# 自定义State模型
class DataAgentState(TypedDict):
    query: str # 提问
    original_query: str # 用户本轮原始输入；query 可由 TODO-44 闭合为 standalone question
    follow_up: dict # TODO-44 可审计 action/operations/preserve/standalone
    query_intent: dict # M3 版本化意图决策；仅含枚举、规则和问题哈希
    intent_response: dict # 引导、拒绝或澄清的正常终态；业务查询为空
    query_signals: dict # TODO-61 有类型、版本和来源的查询信号合同
    sql: str # 生成的sql
    authorized_sql: str # 经过行列权限编译、实际执行的 SQL
    error: str # 存储校验SQL产生的错误信息
    recall_columns: list[ColumnCandidate] # 列信息对象列表
    recall_metrics: list[MetricCandidate] # 指标信息对象列表
    recall_values: list[ValueCandidate] # 字段值信息对象列表
    recall_training: list[dict] # 训练样例召回结果（问题-SQL 对），作为 few-shot 示例注入 generate_sql
    retrieval_metrics: Annotated[list[dict], operator.add] # 四路动态 K、缓存、精排与裁剪证据
    relation_closure: dict # 当前工作空间元数据推导的桥接表、连接键与歧义证据
    table_infos: list[TableInfoState] # 合并3路召回产生的相关表及其字段信息状态列表
    metric_infos: list[MetricInfoState] # 指标信息状态对象列表
    date_info: DateInfoState # 当前日期时间信息状态对象
    db_info: DBInfoState # 数据库信息状态对象
    retry_count: int # SQL 校正重试计数，达到 MAX_RETRY 后终止重试
    chat_history: list[dict] # 多轮对话历史（旧→新），每项含 query 与 sql，供 generate_sql 注入上下文
    result_references: dict # 紧邻成功轮投影出的单值、白名单结果指代，不包含原始结果行
    memory_context: dict # TODO-59 有类型/来源/预算的短期 working state 与长期记忆上下文
    result: list[dict] # execute_sql 执行成功后的查询结果行，供 generate_chart / generate_analysis 等后处理节点使用
    chart: dict # generate_chart 产出的图表配置（type/title/axis 或 columns），供前端渲染
    analysis: str # generate_analysis 产出的自然语言解读，供前端展示
    analysis_enabled: bool # 当前请求是否启用可选 AI 数据解读
    plan: str # plan_sql 产出的查询规划（三步：表连接/口径/排序），供 generate_sql 落实为 SQL；简单问题为空
    clarification: dict # 必要模型路径不可用时的结构化补充建议；不包含问题或 SQL 明文
    model_phase_budget: dict # 物理模型调用前的可完成预算决策证据
    planning_route: dict # TODO-63 版本化规划决策、实际动作、理由与 shadow 证据
    planning_evidence: dict # 可选模型规划或确定性规划的来源、预算和物理调用证据
    semantic_ir: dict # TODO-42 结构化语义意图、编译判定与审计信息
    context_budget: dict # SQL Prompt 上下文装包前后 Token、裁剪原因与强制保留项
    validation_evidence: dict # 当前 SQL 独立安全/语义/成本校验的结构化结果
    semantic_repair_ledger: Annotated[list[dict], operator.add] # 每次候选修复调用的有界脱敏证据
    repair_attempts: Annotated[list[dict], operator.add] # TODO-64 sql_hash+violation 防循环纠错记录
    candidate_decision: dict # TODO-45 置信度、触发原因、候选淘汰与选择证据
    recommend: list[str] # v1 兼容标签数组；滚动发布期间供旧客户端继续渲染
    recommendations: list[dict] # v2 结构化推荐项（id/label/query/kind），点击提交完整 query
    recommendation_contract: dict # 推荐项版本合同，当前为 recommendation-options/v2
    performance_started_at: float # perf_counter 起点，仅用于本次请求相对耗时
    request_deadline: dict # TODO-93 单调时钟总截止时间与分阶段预算（绝对值不持久化）
    node_metrics: Annotated[list[dict], operator.add] # 稳定 node_id 的并行安全耗时记录
    degradations: Annotated[list[dict], operator.add] # 增强节点超时/跳过，不影响 SQL 与结果
    llm_attempts: Annotated[list[dict], operator.add] # 节点自主管理 timeout/cancel 时的统一模型尝试证据
    data_source_context: dict # TODO-87 每轮冻结且可公开的数据源上下文快照
    source_table_infos: list[TableInfoState] # 上传数据集逻辑表 Schema overlay
    upload_profiles: list[dict] # 有界、脱敏的 UploadSchemaProfileV1
    source_mapping: dict # TODO-88 关联映射及质量证据
    source_sqls: dict # 独立聚合对比时按来源分离的 SQL（不做行级 JOIN）
