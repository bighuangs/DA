"""Safe parsing helpers for provider structured output.

Provider JSON mode is not assumed: responses may be fenced, prefixed with a short
sentence, or already decoded by a test/model adapter. Empty and wrong-shaped output
raise a typed error without copying raw model content into logs or API responses.
"""
from __future__ import annotations

import json
from typing import Any


class StructuredOutputError(ValueError):
    pass


def _balanced_fragment(text: str, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def parse_json_response(raw: Any, expected_type: type | tuple[type, ...] | None = None) -> Any:
    if isinstance(raw, (dict, list)):
        value = raw
    else:
        text = str(raw or "").strip()
        if not text:
            raise StructuredOutputError("empty_structured_output")
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        candidates = [text]
        for opening, closing in (("{", "}"), ("[", "]")):
            fragment = _balanced_fragment(text, opening, closing)
            if fragment and fragment not in candidates:
                candidates.append(fragment)
        value = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except (TypeError, ValueError):
                continue
        if value is None:
            raise StructuredOutputError("invalid_structured_output")
    if expected_type is not None and not isinstance(value, expected_type):
        raise StructuredOutputError("unexpected_structured_output_shape")
    return value
