"""Agent 工作流节点集合。

本目录下的每个模块导出一个或多个 LangGraph 节点函数，承担智能问数流程中的
单一处理步骤。各节点按“输入 state → 处理 → 输出 state 增量”的契约工作，
由同一 Agent 包中的 ``graph.py`` 按状态字段与条件边组合编排执行。

节点职责概览（按流水线大致顺序）：
    - rewrite_follow_up：多轮指代闭合（TODO-44），把省略语改写为独立问题
    - extract_keywords：jieba 非语义分词，产出基础关键词
    - recall_column / recall_metric / recall_value / recall_training：四路召回
    - merge_retrieved_info：合并四路召回，组装表/指标结构
    - build_semantic_ir：构建 TODO-42 语义 IR（无 LLM 调用）
    - filter_table / filter_metric：LLM/确定性筛选表与指标
    - pack_context / add_extra_context：装包上下文与补充日期/库信息
    - plan_sql：复杂问题查询规划（条件边接入）
    - generate_sql：生成 SQL（可被语义 IR 确定性编译短路）
    - validate_sql / correct_sql / execute_sql：校验-校正-执行循环（受 MAX_RETRY 限制）
    - judge_candidate：置信度门控的备选候选生成与判优（TODO-45）
    - generate_chart / generate_analysis / generate_recommend：后处理三件套
"""
