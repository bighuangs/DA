from pydantic import BaseModel, Field


class RegisterSchema(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=128)
    workspace_name: str = Field(default="我的工作空间", min_length=1, max_length=128)


class LoginSchema(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class WorkspaceSchema(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class SwitchWorkspaceSchema(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=36)


class MemberSchema(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    role: str = Field(default="viewer", pattern="^(admin|analyst|viewer)$")


class MemberRoleSchema(BaseModel):
    role: str = Field(pattern="^(admin|analyst|viewer)$")
