import joblib
import pandas as pd

from src.config import RUNTIME_CONFIG
from src.model_bundle import ModelArtifact


class FeaturePayloadBuilder:
    def __init__(self, bundle: ModelArtifact) -> None:
        self.bundle = bundle

    def prepare(self, payload: dict) -> pd.DataFrame:
        if not self.bundle.feature_columns:
            return pd.DataFrame([payload])

        row = {}
        for column in self.bundle.feature_columns:
            value = payload.get(column)

            if column in self.bundle.numeric_defaults:
                try:
                    row[column] = float(value)
                except (TypeError, ValueError):
                    row[column] = self.bundle.numeric_defaults[column]
            else:
                fallback = self.bundle.categorical_defaults.get(column, "missing")
                row[column] = str(value).strip() if value not in (None, "") else fallback

        return pd.DataFrame([row], columns=self.bundle.feature_columns)


class LegacyPredictorAdapter:
    def __init__(self, bundle: ModelArtifact) -> None:
        self.preprocessor = bundle.legacy_preprocessor
        self.model = bundle.legacy_model

    def predict_probability(self, payload: dict) -> float:
        import numpy as np

        frame = pd.DataFrame([payload])
        required_columns = []
        column_kind = {}

        for name, _, columns in getattr(self.preprocessor, "transformers_", []):
            if columns in ("remainder", None):
                continue
            if isinstance(columns, (list, tuple)):
                for column in columns:
                    if isinstance(column, str):
                        required_columns.append(column)
                        column_kind[column] = name
            elif isinstance(columns, slice):
                if hasattr(self.preprocessor, "feature_names_in_"):
                    for column in list(self.preprocessor.feature_names_in_[columns]):
                        required_columns.append(column)
                        column_kind[column] = name
            else:
                try:
                    if isinstance(columns, np.ndarray) and hasattr(
                        self.preprocessor, "feature_names_in_"
                    ):
                        for index in columns.tolist():
                            column = self.preprocessor.feature_names_in_[int(index)]
                            required_columns.append(column)
                            column_kind[column] = name
                except Exception:
                    pass

        deduped_columns = list(dict.fromkeys(required_columns))
        for column in deduped_columns:
            if column not in frame.columns:
                frame[column] = 0.0 if column_kind.get(column) == "num" else "missing"

        if deduped_columns:
            frame = frame.reindex(columns=deduped_columns)

        encoded = self.preprocessor.transform(frame)
        probability = self.model.predict_proba(encoded)[0, 1]
        return float(probability)


class ChurnPredictor:
    def __init__(self, model_path=RUNTIME_CONFIG.model_path) -> None:
        self.model_path = model_path
        self._artifact: ModelArtifact | None = None

    def load_artifact(self) -> ModelArtifact:
        if self._artifact is None:
            payload = joblib.load(self.model_path)
            self._artifact = ModelArtifact.from_payload(payload)
        return self._artifact

    def predict_probability(self, payload: dict) -> float:
        artifact = self.load_artifact()
        if artifact.is_modern_bundle:
            features = FeaturePayloadBuilder(artifact).prepare(payload)
            probability = artifact.pipeline.predict_proba(features)[0, 1]
            return float(probability)
        return LegacyPredictorAdapter(artifact).predict_probability(payload)


_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


def predict_proba_single(payload: dict) -> float:
    return get_predictor().predict_probability(payload)
