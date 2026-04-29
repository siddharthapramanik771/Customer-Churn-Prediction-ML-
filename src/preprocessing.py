from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RUNTIME_CONFIG, RuntimeConfig


NUMERIC_CONVERSION_THRESHOLD = 0.95
MISSING_CATEGORY = "missing"


@dataclass(frozen=True)
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns


@dataclass(frozen=True)
class FeatureDefaults:
    numeric_defaults: dict[str, float]
    categorical_defaults: dict[str, str]


@dataclass(frozen=True)
class DataPreprocessor:
    """Owns data cleaning and feature preparation rules."""

    target_column: str
    id_column: str
    positive_target_label: str
    negative_target_label: str
    numeric_conversion_threshold: float = NUMERIC_CONVERSION_THRESHOLD

    @classmethod
    def from_config(cls, config: RuntimeConfig = RUNTIME_CONFIG) -> "DataPreprocessor":
        return cls(
            target_column=config.target_column,
            id_column=config.id_column,
            positive_target_label=config.positive_target_label,
            negative_target_label=config.negative_target_label,
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = cleaned.columns.str.strip()
        cleaned = self._normalize_object_columns(cleaned)
        cleaned = cleaned.dropna()

        if self.id_column in cleaned.columns:
            cleaned = cleaned.drop(columns=[self.id_column])

        return cleaned

    def build_transformer(
        self, df: pd.DataFrame
    ) -> tuple[ColumnTransformer, FeatureSchema]:
        feature_df = df.drop(columns=[self.target_column], errors="ignore")
        schema = self.infer_schema(feature_df)

        transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), schema.numeric_columns),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    schema.categorical_columns,
                ),
            ]
        )
        return transformer, schema

    def infer_schema(self, feature_df: pd.DataFrame) -> FeatureSchema:
        numeric_columns = [
            column
            for column in feature_df.columns
            if is_numeric_dtype(feature_df[column])
        ]
        categorical_columns = [
            column
            for column in feature_df.columns
            if not is_numeric_dtype(feature_df[column])
        ]
        return FeatureSchema(numeric_columns, categorical_columns)

    def encode_target(self, target: pd.Series) -> pd.Series:
        normalized = target.astype(str).str.strip()
        mapping = {
            self.negative_target_label: 0,
            self.positive_target_label: 1,
        }
        encoded = normalized.map(mapping)

        if encoded.isna().any():
            invalid_values = sorted(normalized[encoded.isna()].unique().tolist())
            raise ValueError(
                f"Unexpected values in target column '{self.target_column}': "
                f"{invalid_values}. Expected only {self.negative_target_label!r} "
                f"and {self.positive_target_label!r}."
            )

        return encoded.astype(int)

    def derive_feature_defaults(self, feature_df: pd.DataFrame) -> FeatureDefaults:
        numeric_defaults: dict[str, float] = {}
        categorical_defaults: dict[str, str] = {}

        for column in feature_df.columns:
            series = feature_df[column]
            if is_numeric_dtype(series):
                numeric_defaults[column] = float(series.median())
            else:
                modes = series.mode(dropna=True)
                categorical_defaults[column] = (
                    str(modes.iloc[0]) if not modes.empty else MISSING_CATEGORY
                )

        return FeatureDefaults(numeric_defaults, categorical_defaults)

    def _normalize_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=["object", "string"]).columns:
            normalized = df[column].astype("string").str.strip().replace({"": pd.NA})
            converted = pd.to_numeric(normalized, errors="coerce")
            non_null_mask = normalized.notna()

            if not non_null_mask.any():
                df[column] = normalized
                continue

            numeric_ratio = converted[non_null_mask].notna().mean()
            df[column] = (
                converted
                if numeric_ratio >= self.numeric_conversion_threshold
                else normalized
            )

        return df
