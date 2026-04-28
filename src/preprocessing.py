from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ID_COLUMN,
    NEGATIVE_TARGET_LABEL,
    POSITIVE_TARGET_LABEL,
    TARGET_COLUMN,
)


NUMERIC_CONVERSION_THRESHOLD = 0.95


@dataclass(frozen=True)
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True)
class FeatureDefaults:
    numeric_defaults: dict[str, float]
    categorical_defaults: dict[str, str]


class DataPreprocessor:
    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        id_column: str = ID_COLUMN,
        positive_target_label: str = POSITIVE_TARGET_LABEL,
        negative_target_label: str = NEGATIVE_TARGET_LABEL,
        numeric_conversion_threshold: float = NUMERIC_CONVERSION_THRESHOLD,
    ) -> None:
        self.target_column = target_column
        self.id_column = id_column
        self.positive_target_label = positive_target_label
        self.negative_target_label = negative_target_label
        self.numeric_conversion_threshold = numeric_conversion_threshold

    def _normalize_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=["object", "string"]).columns:
            normalized = df[column].astype("string").str.strip().replace({"": pd.NA})
            converted = pd.to_numeric(normalized, errors="coerce")
            non_null_mask = normalized.notna()

            if non_null_mask.any():
                numeric_ratio = converted[non_null_mask].notna().mean()
                df[column] = (
                    converted
                    if numeric_ratio >= self.numeric_conversion_threshold
                    else normalized
                )
            else:
                df[column] = normalized

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        df = self._normalize_object_columns(df)
        df = df.dropna()
        if self.id_column in df.columns:
            df = df.drop(columns=[self.id_column])
        return df

    def build_preprocessor(
        self, df: pd.DataFrame
    ) -> tuple[ColumnTransformer, FeatureSchema]:
        feature_df = df.drop(columns=[self.target_column], errors="ignore")
        numeric_columns = [c for c in feature_df.columns if is_numeric_dtype(feature_df[c])]
        categorical_columns = [
            c for c in feature_df.columns if not is_numeric_dtype(feature_df[c])
        ]

        transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ]
        )
        return transformer, FeatureSchema(numeric_columns, categorical_columns)

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
                f"Unexpected values in target column '{self.target_column}': {invalid_values}. "
                f"Expected only {self.negative_target_label!r} and {self.positive_target_label!r}."
            )

        return encoded.astype(int)

    def derive_feature_defaults(self, feature_df: pd.DataFrame) -> FeatureDefaults:
        numeric_defaults = {}
        categorical_defaults = {}

        for column in feature_df.columns:
            series = feature_df[column]
            if is_numeric_dtype(series):
                numeric_defaults[column] = float(series.median())
            else:
                modes = series.mode(dropna=True)
                categorical_defaults[column] = (
                    str(modes.iloc[0]) if not modes.empty else "missing"
                )

        return FeatureDefaults(numeric_defaults, categorical_defaults)


DEFAULT_PREPROCESSOR = DataPreprocessor()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return DEFAULT_PREPROCESSOR.clean(df)


def build_preprocessor(df: pd.DataFrame):
    transformer, schema = DEFAULT_PREPROCESSOR.build_preprocessor(df)
    return transformer, schema.numeric_columns, schema.categorical_columns


def encode_target(target: pd.Series) -> pd.Series:
    return DEFAULT_PREPROCESSOR.encode_target(target)


def derive_feature_defaults(feature_df: pd.DataFrame):
    defaults = DEFAULT_PREPROCESSOR.derive_feature_defaults(feature_df)
    return defaults.numeric_defaults, defaults.categorical_defaults
