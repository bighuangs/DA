from pydantic import BaseModel, Field


class DatasetExtendSchema(BaseModel):
    seconds: int = Field(default=86400, ge=60, le=30 * 24 * 3600)


class MappingDecisionSchema(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class ConversationDataContextSchema(BaseModel):
    source_mode: str = Field(
        pattern=r"^(dw_only|upload_only|upload_compare_dw|upload_join_dw|multi_upload)$"
    )
    dataset_ids: list[str] = Field(default_factory=list, max_length=2)
    comparison_intent: str | None = Field(default=None, max_length=48)
    mapping_confirmation_id: str | None = Field(default=None, max_length=64)
    expected_revision: int | None = Field(default=None, ge=0)


class DataContextResetSchema(BaseModel):
    expected_revision: int | None = Field(default=None, ge=0)
