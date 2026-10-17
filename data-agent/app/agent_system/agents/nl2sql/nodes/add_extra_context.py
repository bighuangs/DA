from datetime import datetime

from langgraph.runtime import Runtime

from app.agent_system.agents.nl2sql.context import DataAgentContext
from app.agent_system.agents.nl2sql.state import DateInfoState, DBInfoState
import logging

logger = logging.getLogger(__name__)


async def add_extra_context(state, runtime: Runtime[DataAgentContext]):
    """
    为 SQL 生成补充额外的上下文信息（当前日期信息 + 数据库元信息）。

    核心功能：在生成 SQL 前，向 state 注入两类与提问无关但生成 SQL 必需的
             上下文——当前日期/星期/季度（用于处理"今天""本季度"等相对时间
             表达）以及目标数据库的基本信息（用于让大模型对齐库的方言与
             约束），从而提高生成 SQL 的准确率。

    业务逻辑：
        1. 向前端推送当前流水线阶段"添加额外信息"
        2. 取运行时上下文中的 dw_mysql_repo（数仓库访问对象）
        3. 收集当前日期时间，计算 date / weekday / quarter 组装为 DateInfoState
        4. 查询数仓库的数据库基本信息，转换为 DBInfoState
        5. 返回 {"date_info": ..., "db_info": ...} 更新到 state

    Args:
        state: 流水线状态（本节点不读取其字段，仅做补充写入）
        runtime: LangGraph 运行时，携带上下文与流式写入器

    Returns:
        dict: {"date_info": DateInfoState, "db_info": DBInfoState}
    """
    runtime.stream_writer({"stage": "添加额外信息"})
    try:
        dw_mysql_repo = runtime.context["dw_mysql_repo"]

        # 1.收集当前日期时间
        # 将"今天""本周""本季度"等相对时间表达落地为具体日期信息，供生成 SQL 时使用
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")
        weekday = today.strftime("%A")
        quarter = f"Q{(today.month+2)//3}"  # Q1-Q4  由月份推算季度：1-3月→Q1，依此类推
        date_info = DateInfoState(
            date=date,
            weekday=weekday,
            quarter=quarter,
        )

        # 2.收集数据库信息
        # 让大模型感知目标数据库的方言、版本等约束，避免生成不兼容的 SQL
        db_info_mysql = await dw_mysql_repo.get_db_infos()
        db_info = DBInfoState(**db_info_mysql)

        logger.info(f"添加额外信息完成：date_info={date_info}, db_info={db_info}")

        return {"date_info": date_info, "db_info": db_info}
    except Exception as e:
        logger.error(f"添加额外信息信息失败：{str(e)}")
        raise
