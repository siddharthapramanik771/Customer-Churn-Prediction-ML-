from dataclasses import dataclass
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from app.data_analysis import DataAnalysisRenderer
from app.styles import GITHUB_REPOSITORY_URL, apply_page_styles
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

    def render_sidebar(self, reference_dataset: ReferenceDataset | None = None) -> None:
        with st.sidebar:
            st.markdown("### Customer Churn ML")
            st.markdown(
                "Interactive scoring, dataset analysis, and training notes for the "
                "customer churn model."
            )
            st.link_button("View GitHub repository", GITHUB_REPOSITORY_URL)
            st.divider()
            st.markdown("### Reference Data")

            if reference_dataset is None:
                st.info("Reference data will appear here after the dataset loads.")
                return

            csv_data = reference_dataset.frame.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download cleaned CSV",
                data=csv_data,
                file_name="customer_churn_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander("Preview data", expanded=True):
                st.caption(
                    f"{len(reference_dataset.frame):,} rows from "
                    f"{reference_dataset.source_path.name}"
                )
                st.dataframe(
                    reference_dataset.frame.head(50),
                    use_container_width=True,
                    height=320,
                )

    @staticmethod
    def render_hero() -> None:
        st.markdown(
            f"""
<section class="hero">
    <div class="hero__eyebrow">Machine learning dashboard</div>
    <h1>Customer Churn Dashboard</h1>
    <p>
        Score customer churn risk, inspect the reference data, and review the
        training workflow behind the deployed XGBoost model.
    </p>
    <div class="hero__actions">
        <a class="hero__link" href="{GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener">
            GitHub repository
        </a>
        <span class="hero__note">Local prediction from the saved model artifact</span>
    </div>
</section>
""",
            unsafe_allow_html=True,
        )

    def render_status_strip(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, source_path: Path
    ) -> None:
        churn_rate = "n/a"
        if self.config.target_column in df.columns:
            churn_rate = (
                f"{df[self.config.target_column].astype(str).eq(self.config.positive_target_label).mean():.1%}"
            )

        st.markdown(
            f"""
<div class="status-strip">
    <div class="status-tile">
        <span>Clean rows</span>
        <strong>{len(df):,}</strong>
        <small>Ready for exploration</small>
    </div>
    <div class="status-tile">
        <span>Input features</span>
        <strong>{feature_df.shape[1]:,}</strong>
        <small>Used by prediction form</small>
    </div>
    <div class="status-tile">
        <span>Churn rate</span>
        <strong>{churn_rate}</strong>
        <small>From reference dataset</small>
    </div>
    <div class="status-tile">
        <span>Data source</span>
        <strong>{source_path.name}</strong>
        <small>Loaded at runtime</small>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

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
        st.subheader("Predict Churn Risk")
        st.caption("Enter a customer profile and run the saved pipeline locally.")
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
        st.set_page_config(
            page_title="Customer Churn Dashboard",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        apply_page_styles()
        self.render_hero()

        reference_dataset = self.load_reference_data()
        if reference_dataset is None:
            self.render_sidebar()
            st.error(
                f"Dataset not found at {self.config.data_path}. "
                "Add the training data to enable the dashboard."
            )
            st.stop()

        df = reference_dataset.frame
        feature_df = df.drop(columns=[self.config.target_column], errors="ignore")
        self.render_sidebar(reference_dataset)
        self.render_status_strip(df, feature_df, reference_dataset.source_path)

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
