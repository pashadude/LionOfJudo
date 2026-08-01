"""Strict loading and validated atomic persistence for canonical review JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pipeline.video_review_contract import validate_review_payload


def load_review_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON konstanta nije dozvoljena: {value}")

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("review.json nije validan strogi JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("review.json mora sadržati JSON objekat")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_review(path: Path, payload: Mapping[str, Any]) -> None:
    validate_review_payload(payload)
    atomic_write_json(path, payload)


__all__ = ["atomic_write_json", "atomic_write_review", "load_review_json"]
