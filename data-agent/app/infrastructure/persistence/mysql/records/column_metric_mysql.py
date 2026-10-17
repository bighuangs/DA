from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.mysql.records.base import Base


class ColumnMetricMySQL(Base):
    __tablename__ = "column_metric"

    column_id: Mapped[str] = mapped_column(
        ForeignKey("column_info.id", ondelete="CASCADE"),
        primary_key=True,
        comment="列编号"
    )
    metric_id: Mapped[str] = mapped_column(
        ForeignKey("metric_info.id", ondelete="CASCADE"),
        primary_key=True,
        comment="指标编号"
    )
