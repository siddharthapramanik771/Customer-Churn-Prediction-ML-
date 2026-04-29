from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable model bundle used by both training and prediction."""

    pipeline: Any
    feature_columns: list[str] = field(default_factory=list)
    numeric_defaults: dict[str, float] = field(default_factory=dict)
    categorical_defaults: dict[str, str] = field(default_factory=dict)
    target_column: str = ""
    positive_target_label: str = "Yes"
    negative_target_label: str = "No"
    prediction_threshold: float = 0.5
    artifact_version: int = ARTIFACT_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelArtifact":
        if "pipeline" not in payload:
            raise ValueError(
                "Invalid model artifact: expected a unified bundle with a "
                "'pipeline' key."
            )

        return cls(
            pipeline=payload["pipeline"],
            feature_columns=payload.get("feature_columns", []),
            numeric_defaults=payload.get("numeric_defaults", {}),
            categorical_defaults=payload.get("categorical_defaults", {}),
            target_column=payload.get("target_column", ""),
            positive_target_label=payload.get("positive_target_label", "Yes"),
            negative_target_label=payload.get("negative_target_label", "No"),
            prediction_threshold=payload.get("prediction_threshold", 0.5),
            artifact_version=payload.get("artifact_version", 1),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "numeric_defaults": self.numeric_defaults,
            "categorical_defaults": self.categorical_defaults,
            "target_column": self.target_column,
            "positive_target_label": self.positive_target_label,
            "negative_target_label": self.negative_target_label,
            "prediction_threshold": self.prediction_threshold,
        }


class ModelArtifactRepository:
    """Persists model artifacts without leaking joblib details to services."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def load(self) -> ModelArtifact:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Train the model first with `python -m src.train`."
            )

        payload = joblib.load(self.model_path)
        if isinstance(payload, ModelArtifact):
            return payload
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported model artifact format at {self.model_path}: "
                f"{type(payload).__name__}"
            )
        return ModelArtifact.from_payload(payload)

    def save(self, artifact: ModelArtifact) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact.to_payload(), self.model_path)
