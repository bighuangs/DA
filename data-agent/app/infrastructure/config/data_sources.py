"""把外层配置投影为数据来源 Application 的不可变设置。"""

from app.application.ports.data_sources import (
    AdhocDatasetSettings,
    DataSourceContextSettings,
)
from app.config.app_config import app_config


def adhoc_dataset_settings() -> AdhocDatasetSettings:
    """从进程配置生成上传用例设置快照。"""

    value = app_config.adhoc
    return AdhocDatasetSettings(
        max_file_bytes=value.max_file_bytes,
        max_rows=value.max_rows,
        max_columns=value.max_columns,
        max_cell_chars=value.max_cell_chars,
        ttl_seconds=value.ttl_seconds,
        managed_ttl_seconds=value.managed_ttl_seconds,
        max_extension_seconds=value.max_extension_seconds,
        max_active_datasets=value.max_active_datasets,
        max_total_bytes=value.max_total_bytes,
        max_total_rows=value.max_total_rows,
        max_selected_datasets=value.max_selected_datasets,
    )


def data_source_context_settings() -> DataSourceContextSettings:
    """从进程配置生成来源上下文设置快照。"""

    return DataSourceContextSettings(
        max_selected_datasets=app_config.adhoc.max_selected_datasets,
    )


__all__ = ["adhoc_dataset_settings", "data_source_context_settings"]
