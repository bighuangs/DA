"""API 认证服务：scrypt 密码哈希、HMAC 访问令牌与主体模型。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    display_name: str
    workspace_id: str
    workspace_name: str
    role: str


@dataclass(frozen=True, slots=True)
class AuthTokenPolicy:
    """访问令牌签名与有效期策略。"""

    secret_key: str
    access_token_ttl_minutes: int

    def __post_init__(self) -> None:
        if not self.secret_key:
            raise ValueError("secret_key 不能为空")
        if self.access_token_ttl_minutes < 1:
            raise ValueError("access_token_ttl_minutes 必须大于 0")


class InvalidAccessTokenError(ValueError):
    """访问令牌无效或已经过期。"""


class PasswordPolicyError(ValueError):
    """密码不满足注册策略。"""


_COMPATIBILITY_TOKEN_POLICY = AuthTokenPolicy(
    secret_key="dev-only-change-me",
    access_token_ttl_minutes=480,
)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1${}${}".format(
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        prefix, salt_text, digest_text = encoded.rsplit("$", 2)
        _, n, r, p = prefix.split("$")
        salt = base64.urlsafe_b64decode(salt_text.encode())
        expected = base64.urlsafe_b64decode(digest_text.encode())
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_access_token(
    *,
    user_id: str,
    workspace_id: str,
    policy: AuthTokenPolicy = _COMPATIBILITY_TOKEN_POLICY,
) -> str:
    payload = {
        "sub": user_id,
        "workspace_id": workspace_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + policy.access_token_ttl_minutes * 60,
        "jti": str(uuid.uuid4()),
    }
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64(
        hmac.new(policy.secret_key.encode(), body.encode(), hashlib.sha256).digest()
    )
    return f"da1.{body}.{signature}"


def decode_access_token(
    token: str,
    *,
    policy: AuthTokenPolicy = _COMPATIBILITY_TOKEN_POLICY,
) -> dict:
    try:
        version, body, signature = token.split(".", 2)
        if version != "da1":
            raise ValueError("unsupported token")
        expected = _b64(
            hmac.new(policy.secret_key.encode(), body.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError("bad signature")
        payload = json.loads(_unb64(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception as exc:
        raise InvalidAccessTokenError("登录已失效，请重新登录") from exc


def validate_password(password: str) -> None:
    if len(password or "") < 8:
        raise PasswordPolicyError("密码至少需要 8 个字符")


class AuthTokenService:
    """使用 Bootstrap 注入策略签发和校验访问令牌。"""

    def __init__(self, policy: AuthTokenPolicy) -> None:
        self._policy = policy

    def create(self, *, user_id: str, workspace_id: str) -> str:
        return create_access_token(
            user_id=user_id,
            workspace_id=workspace_id,
            policy=self._policy,
        )

    def decode(self, token: str) -> dict:
        return decode_access_token(token, policy=self._policy)


__all__ = [
    "AuthTokenPolicy",
    "AuthTokenService",
    "InvalidAccessTokenError",
    "PasswordPolicyError",
    "Principal",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "validate_password",
    "verify_password",
]
