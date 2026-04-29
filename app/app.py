import time

import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.predict import ChurnPrediction, ChurnPredictor
from src.preprocessing import DataPreprocessor


class ReferenceDataService:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor.from_config(config)

    def load(self) -> pd.DataFrame | None:
        if not self.config.data_path.exists():
            return None
        return self.preprocessor.clean(self.config.load_dataset())


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
        self.config = config

    @st.cache_data
    def load_reference_data(_self) -> pd.DataFrame | None:
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

    def render_data_explorer(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        st.header("Data Explorer")
        left, right = st.columns(2)
        left.metric("Rows", len(df))
        right.metric("Features", feature_df.shape[1])

        if self.config.target_column in df.columns:
            st.subheader("Target distribution")
            target_counts = (
                df[self.config.target_column]
                .value_counts()
                .rename_axis("target")
                .reset_index(name="count")
            )
            st.bar_chart(target_counts, x="target", y="count")

        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Summary statistics")
        st.write(df.describe(include="all"))

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        st.subheader("Numeric feature distributions")
        numeric_choice = st.selectbox("Choose numeric feature", [""] + numeric_cols)
        if numeric_choice:
            series = df[numeric_choice].dropna()
            counts = (
                series.groupby(pd.cut(series, bins=20))
                .size()
                .rename_axis("bucket")
                .reset_index(name="count")
            )
            counts["bucket"] = counts["bucket"].astype(str)
            st.bar_chart(counts, x="bucket", y="count")

        st.subheader("Categorical feature counts")
        categorical_choice = st.selectbox(
            "Choose categorical feature", [""] + categorical_cols
        )
        if categorical_choice:
            category_counts = (
                df[categorical_choice]
                .fillna("(missing)")
                .value_counts()
                .nlargest(50)
                .rename_axis("category")
                .reset_index(name="count")
            )
            st.bar_chart(category_counts, x="category", y="count")

        if len(numeric_cols) >= 2:
            st.subheader("Correlation matrix")
            st.dataframe(df[numeric_cols].corr())

    def render(self) -> None:
        st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
        st.title("Customer Churn Dashboard")
        st.caption(
            "Submit a customer record for local churn scoring and inspect the reference dataset."
        )

        df = self.load_reference_data()
        if df is None:
            st.error(
                f"Dataset not found at {self.config.data_path}. "
                "Add the training data to enable the dashboard."
            )
            st.stop()

        feature_df = df.drop(columns=[self.config.target_column], errors="ignore")
        st.write(
            f"Loaded {len(df)} cleaned rows and {feature_df.shape[1]} input "
            f"features from `{self.config.data_path.name}`."
        )

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

        self.render_data_explorer(df, feature_df)


def main() -> None:
    DashboardRenderer().render()
