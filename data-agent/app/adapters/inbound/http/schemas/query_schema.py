from typing import Optional

from pydantic import BaseModel, Field


class QuerySchema(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    chat_id: Optional[str] = Field(default=None, max_length=64)
    dataset_id: Optional[str] = Field(default=None, max_length=36)
    dataset_ids: list[str] = Field(default_factory=list, max_length=2)
    compare_dw: bool = False  # TODO-32 是否允许与已配置 DW 表联合查询
    source_mode: Optional[str] = Field(
        default=None,
        pattern=r"^(dw_only|upload_only|upload_compare_dw|upload_join_dw|multi_upload)$",
    )
    comparison_intent: Optional[str] = Field(default=None, max_length=48)
    mapping_confirmation_id: Optional[str] = Field(default=None, max_length=64)
    data_context_revision: Optional[int] = Field(default=None, ge=0)
    enable_analysis: Optional[bool] = None  # 不传时使用服务端默认值（当前默认关闭）


class ExecSqlSchema(BaseModel):
    """手动编辑 SQL 重跑请求体（TODO-22）。"""
    sql: str = Field(min_length=1, max_length=20000)
    dataset_id: Optional[str] = Field(default=None, max_length=36)
    dataset_ids: list[str] = Field(default_factory=list, max_length=2)
    compare_dw: bool = False  # TODO-32 编辑 SQL 时是否保留 DW 对比白名单
    source_mode: Optional[str] = Field(default=None, max_length=32)
    comparison_intent: Optional[str] = Field(default=None, max_length=48)
    mapping_confirmation_id: Optional[str] = Field(default=None, max_length=64)


class ExportSchema(BaseModel):
    """导出当前结果表格；结果由前端按当前查询快照传回，避免再次执行 SQL。"""
    columns: list[str] = Field(default_factory=list, max_length=200)
    rows: list[dict] = Field(default_factory=list, max_length=100000)
    filename: str = Field(default="data-agent-result", max_length=80)
