from pydantic import BaseModel, Field


class ConversationUpdateSchema(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    status: str | None = Field(
        default=None, pattern=r"^(active|archived)$"
    )
