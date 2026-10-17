from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf

from app.domain.query.semantic_aliases import merge_business_aliases

# ==================== 字段信息配置模型 ====================
@dataclass
class ColumnConfig:
    name: str
    role: str
    description: str
    alias: list[str]
    sync: bool
    is_display: bool = False

# ==================== 表信息配置模型 ====================
@dataclass
class TableConfig:
    name: str
    role: str
    description: str
    columns: list[ColumnConfig]

# ==================== 指标信息配置模型 ====================
@dataclass
class MetricConfig:
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]
    formula_rule: dict | None = None

# ==================== SQL 样例库配置模型 ====================
@dataclass
class TrainingConfig:
    question: str
    sql: str

# ==================== 元数据的总配置模型 ====================
@dataclass
class MetaConfig:
    tables: list[TableConfig]
    metrics: list[MetricConfig]
    trainings: list[TrainingConfig]

# 得到yaml配置文件的绝对路径
_yaml_path = Path(__file__).parents[2] / "conf/meta_config.yaml"

# 根据路径加载配置文件数据 =》DictConfig对象
_yaml_data = OmegaConf.load(_yaml_path)

# 将数据转换为MetaConfig类型的对象
meta_config: MetaConfig = OmegaConf.to_object(OmegaConf.merge(MetaConfig, _yaml_data))

# Keep the YAML aliases as human-curated authority while exposing the same
# governed vocabulary to fresh installs, evaluation coverage checks and schema
# refreshes.  The helper is stable and idempotent, so repeated imports/syncs do
# not grow the list.
for _table in meta_config.tables:
    for _column in _table.columns:
        _column.alias = merge_business_aliases(
            _table.name,
            _column.name,
            _column.alias,
        )
