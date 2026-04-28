from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelArtifact:
    pipeline: Any | None = None
    feature_columns: list[str] = field(default_factory=list)
    numeric_defaults: dict[str, float] = field(default_factory=dict)
    categorical_defaults: dict[str, str] = field(default_factory=dict)
    legacy_preprocessor: Any | None = None
    legacy_model: Any | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelArtifact":
        return cls(
            pipeline=payload.get("pipeline"),
            feature_columns=payload.get("feature_columns", []),
            numeric_defaults=payload.get("numeric_defaults", {}),
            categorical_defaults=payload.get("categorical_defaults", {}),
            legacy_preprocessor=payload.get("preprocessor"),
            legacy_model=payload.get("model"),
        )

    @property
    def is_modern_bundle(self) -> bool:
        return self.pipeline is not None

    def to_payload(self) -> dict[str, Any]:
        if self.is_modern_bundle:
            return {
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "numeric_defaults": self.numeric_defaults,
                "categorical_defaults": self.categorical_defaults,
            }

        return {
            "preprocessor": self.legacy_preprocessor,
            "model": self.legacy_model,
        }
