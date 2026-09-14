from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_ENCODER_CONFIG = Path(__file__).resolve().parent.parent / "config" / "encoder_config.yaml"


@lru_cache(maxsize=8)
def _load_common_dn_values(config_path: str) -> frozenset[int]:
    try:
        with Path(config_path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        values: Iterable[object] = (config.get("size_processing") or {}).get("common_dn_values") or []
        return frozenset(int(value) for value in values)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return frozenset()


def get_common_dn_values(config_path: str | Path | None = None) -> frozenset[int]:
    """Return the configured nominal-size whitelist shared by rule extractors."""

    resolved = Path(config_path or DEFAULT_ENCODER_CONFIG).expanduser().resolve()
    return _load_common_dn_values(str(resolved))
