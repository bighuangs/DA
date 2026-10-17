from pydantic import BaseModel, Field


class MetricSchema(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=5000)
    relevant_columns: list[str] = Field(default_factory=list, max_length=200)
    alias: list[str] = Field(default_factory=list, max_length=100)
    enabled: bool = True
    formula_rule: dict | None = None


class TrainingSchema(BaseModel):
    question: str = Field(min_length=1, max_length=255)
    sql: str = Field(min_length=1, max_length=20000)
    enabled: bool = True


class RebuildSchema(BaseModel):
    target: str = Field(default="all", pattern="^(column|metric|training|all)$")


class SchemaSyncSchema(BaseModel):
    rebuild_value_index: bool = False


class ValueIndexRollbackSchema(BaseModel):
    physical_index: str = Field(min_length=1, max_length=255)


class MetricPatchSchema(BaseModel):
    description: str = Field(default="", max_length=5000)
    relevant_columns: list[str] = Field(default_factory=list, max_length=200)
    alias: list[str] = Field(default_factory=list, max_length=100)
    enabled: bool = True
    formula_rule: dict | None = None


class TrainingPatchSchema(BaseModel):
    question: str = Field(min_length=1, max_length=255)
    sql: str = Field(min_length=1, max_length=20000)
    enabled: bool = True


class MaskingRuleSchema(BaseModel):
    column_pattern: str = Field(min_length=1, max_length=255)
    data_kind: str = Field(default="generic", pattern="^(generic|phone|email|id_card|bank_card|name)$")
    strategy: str = Field(default="mask", pattern="^(mask|hash|keep_last4|null|keep)$")
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=10000)
    description: str = Field(default="", max_length=5000)


class MaskingRulePatchSchema(MaskingRuleSchema):
    pass


class AccessPolicySchema(BaseModel):
    role: str = Field(default="all", pattern="^(all|owner|admin|analyst|viewer)$")
    user_id: str | None = Field(default=None, max_length=36)
    table_pattern: str = Field(default="*", min_length=1, max_length=128)
    allowed_columns: list[str] | None = Field(default=None, max_length=200)
    row_filter: str | None = Field(default=None, max_length=5000)
    access: bool = True
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=10000)
    description: str = Field(default="", max_length=5000)


class AccessPolicyPatchSchema(AccessPolicySchema):
    pass
