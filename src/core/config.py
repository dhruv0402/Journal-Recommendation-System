from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


# =====================
# Pydantic Models
# =====================

class ThresholdConfig(BaseModel):
    exact_match: float = Field(1.0, ge=0.0, le=1.0)
    fuzzy_match_min: int = Field(85, ge=0, le=100)


class StrategyConfig(BaseModel):
    use_exact: bool = True
    use_fuzzy: bool = True
    use_abbreviation: bool = False


class PreprocessingConfig(BaseModel):
    lowercase: bool = True
    strip_punctuation: bool = True
    normalize_whitespace: bool = True


class OutputConfig(BaseModel):
    return_top_k: int = Field(3, ge=1, le=20)
    include_score: bool = True
    include_strategy: bool = True


class DetectionConfig(BaseModel):
    enabled: bool = True
    thresholds: ThresholdConfig = ThresholdConfig()
    strategies: StrategyConfig = StrategyConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    output: OutputConfig = OutputConfig()


class AppConfig(BaseModel):
    detection: DetectionConfig = DetectionConfig()


# =====================
# Loader
# =====================

def load_config(config_path: str | Path) -> AppConfig:
    """
    Load and validate application configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated AppConfig object

    Raises:
        RuntimeError: If config is invalid or unreadable
    """
    path = Path(config_path)

    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_config: Dict[str, Any] = yaml.safe_load(f) or {}

        return AppConfig(**raw_config)

    except ValidationError as e:
        raise RuntimeError(
            f"Configuration validation failed:\n{e}"
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Failed to load configuration from {path}: {e}"
        ) from e
