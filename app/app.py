from dataclasses import dataclass
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from app.data_analysis import DataAnalysisRenderer
from app.training_methodology import TrainingMethodologyRenderer
from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.predict import ChurnPrediction, ChurnPredictor
from src.preprocessing import DataPreprocessor


@dataclass(frozen=True)
class ReferenceDataset:
    frame: pd.DataFrame
    source_path: Path


class ReferenceDataService:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor.from_config(config)

    def load(self) -> ReferenceDataset | None:
        data_path = self.resolve_reference_data_path()
        if not data_path.exists():
            return None
        raw_df = self.config.load_dataset(data_path)
        return ReferenceDataset(self.preprocessor.clean(raw_df), data_path)

    def resolve_reference_data_path(self) -> Path:
        metrics_data_path = self.read_metrics_data_path()
        if metrics_data_path and metrics_data_path.exists():
            return metrics_data_path
        return self.config.data_path

    def read_metrics_data_path(self) -> Path | None:
        if not self.config.metrics_path.exists():
            return None

        try:
            payload = json.loads(self.config.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        training_data_path = payload.get("training_data_path")
        if not training_data_path:
            return None

        path = Path(training_data_path)
        if path.is_absolute():
            return path
        return self.config.project_root / path


class LocalPredictionService:
    def __init__(
        self,
        predictor: ChurnPredictor | None = None,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> None:
        self.predictor = predictor or ChurnPredictor(config)
        self.config = config

    def predict(self, payload: dict) -> ChurnPrediction:
        return self.predictor.predict(payload)


class DashboardRenderer:
    def __init__(
        self,
        reference_data_service: ReferenceDataService | None = None,
        prediction_service: LocalPredictionService | None = None,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> None:
        self.reference_data_service = reference_data_service or ReferenceDataService(
            config
        )
        self.prediction_service = prediction_service or LocalPredictionService(
            config=config
        )
        self.data_analysis_renderer = DataAnalysisRenderer(config)
        self.training_methodology_renderer = TrainingMethodologyRenderer(config)
        self.config = config

    @st.cache_data
    def load_reference_data(_self) -> ReferenceDataset | None:
        return _self.reference_data_service.load()

    def render_feature_inputs(self, feature_df: pd.DataFrame) -> dict:
        payload = {}
        columns = st.columns(2)

        for index, column_name in enumerate(feature_df.columns):
            series = feature_df[column_name]
            panel = columns[index % 2]

            with panel:
                if is_numeric_dtype(series):
                    payload[column_name] = self.render_numeric_input(column_name, series)
                else:
                    payload[column_name] = self.render_categorical_input(
                        column_name, series
                    )

        return payload

    @staticmethod
    def render_numeric_input(column_name: str, series: pd.Series) -> int | float:
        median_value = series.median()
        min_value = series.min()
        max_value = series.max()

        if is_integer_dtype(series):
            return int(
                st.number_input(
                    column_name,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(round(median_value)),
                    step=1,
                )
            )

        step = max((float(max_value) - float(min_value)) / 100, 0.1)
        return float(
            st.number_input(
                column_name,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(round(median_value, 2)),
                step=float(step),
                format="%.4f",
            )
        )

    @staticmethod
    def render_categorical_input(column_name: str, series: pd.Series) -> str:
        options = sorted(series.dropna().astype(str).unique().tolist())
        modes = series.mode(dropna=True)
        default_value = str(modes.iloc[0]) if not modes.empty else options[0]
        default_index = options.index(default_value) if default_value in options else 0
        return st.selectbox(column_name, options, index=default_index)

    @staticmethod
    def render_prediction_result(result: ChurnPrediction, elapsed_seconds: float) -> None:
        st.success("Prediction complete")
        left, right, extra = st.columns(3)
        left.metric("Prediction", result.prediction)
        right.metric("Probability", f"{result.churn_probability:.2%}")
        extra.metric("Latency", f"{elapsed_seconds:.2f}s")

        st.progress(float(result.churn_probability))
        st.json(result.to_dict())

    def render_prediction_tab(self, feature_df: pd.DataFrame) -> None:
        with st.form("prediction_form"):
            payload = self.render_feature_inputs(feature_df)
            submitted = st.form_submit_button("Predict churn risk")

        if submitted:
            try:
                start = time.time()
                result = self.prediction_service.predict(payload)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                self.render_prediction_result(result, time.time() - start)

    def render(self) -> None:
        st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
        st.title("Customer Churn Dashboard")
        st.caption(
            "Submit a customer record for local churn scoring and inspect the reference dataset."
        )

        reference_dataset = self.load_reference_data()
        if reference_dataset is None:
            st.error(
                f"Dataset not found at {self.config.data_path}. "
                "Add the training data to enable the dashboard."
            )
            st.stop()

        df = reference_dataset.frame
        feature_df = df.drop(columns=[self.config.target_column], errors="ignore")
        st.write(
            f"Loaded {len(df)} cleaned rows and {feature_df.shape[1]} input "
            f"features from `{reference_dataset.source_path.name}`."
        )

        prediction_tab, analysis_tab, methodology_tab = st.tabs(
            ["Predict", "Data Analysis", "Training Methodology"]
        )

        with prediction_tab:
            self.render_prediction_tab(feature_df)
        with analysis_tab:
            self.data_analysis_renderer.render(df, feature_df)
        with methodology_tab:
            self.training_methodology_renderer.render()


def main() -> None:
    DashboardRenderer().render()
