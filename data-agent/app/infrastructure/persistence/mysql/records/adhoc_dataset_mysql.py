from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class AdhocDatasetMySQL(Base):
    """上传数据资产的不可变版本记录。

    ``asset_id`` 表示逻辑数据集，``id``/``version_id`` 表示一次不可变版本。
    物理表名只在服务端保存，任何面向浏览器的 DTO 都不得暴露它。
    """

    __tablename__ = "adhoc_dataset"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    version_no: Mapped[int] = mapped_column(Integer, default=1)
    asset_kind: Mapped[str] = mapped_column(String(32), default="managed_upload", index=True)
    retention_mode: Mapped[str] = mapped_column(String(32), default="temporary", index=True)
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    workspace_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    table_name: Mapped[str] = mapped_column(String(128), unique=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_format: Mapped[str] = mapped_column(String(16))
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    columns: Mapped[list] = mapped_column(JSON)
    preview: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    content_checksum: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    schema_fingerprint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    parser_profile: Mapped[str] = mapped_column(String(64), default="upload-parser-v2")
    byte_size: Mapped[int] = mapped_column(BigInteger, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    policy_version: Mapped[str] = mapped_column(String(64), default="upload-policy-v2")
    lifecycle_revision: Mapped[int] = mapped_column(Integer, default=1)
    managed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleanup_attempts: Mapped[int] = mapped_column(Integer, default=0)
    cleanup_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
