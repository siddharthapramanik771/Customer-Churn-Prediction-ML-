import altair as alt
import pandas as pd
import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig


class DataAnalysisRenderer:
    """Renders interactive analysis for the reference churn dataset."""

    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def render(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        st.header("Data Analysis")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        target_col = self.config.target_column
        has_target = target_col in df.columns

        self.render_overview(df, feature_df, has_target)

        tabs = st.tabs(
            [
                "Churn Story",
                "Segments",
                "Numeric Trends",
                "Relationships",
                "Data Table",
            ]
        )

        with tabs[0]:
            self.render_churn_story(df, has_target)
        with tabs[1]:
            self.render_segments(df, categorical_cols, has_target)
        with tabs[2]:
            self.render_numeric_trends(df, numeric_cols, has_target)
        with tabs[3]:
            self.render_relationships(df, numeric_cols, has_target)
        with tabs[4]:
            self.render_table(df)

    def render_overview(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, has_target: bool
    ) -> None:
        columns = st.columns(4)
        columns[0].metric("Rows", f"{len(df):,}")
        columns[1].metric("Features", f"{feature_df.shape[1]:,}")

        if has_target:
            churn_rate = self.churn_mask(df).mean()
            columns[2].metric("Churn rate", f"{churn_rate:.1%}")
        else:
            columns[2].metric("Churn rate", "n/a")

        revenue_col = self.find_column(df, ["TotalRevenue", "MonthlyCharges"])
        if revenue_col:
            columns[3].metric(f"Avg {revenue_col}", f"{df[revenue_col].mean():,.2f}")
        else:
            numeric_field_count = len(df.select_dtypes(include=["number"]).columns)
            columns[3].metric("Numeric fields", f"{numeric_field_count:,}")

    def render_churn_story(self, df: pd.DataFrame, has_target: bool) -> None:
        if not has_target:
            st.info("Target column is not available in this dataset.")
            return

        left, right = st.columns(2)

        target_counts = (
            df[self.config.target_column]
            .value_counts()
            .rename_axis("churn")
            .reset_index(name="customers")
        )
        target_chart = (
            alt.Chart(target_counts)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("customers:Q"),
                color=alt.Color("churn:N", legend=alt.Legend(title="Churn")),
                tooltip=["churn:N", "customers:Q"],
            )
            .properties(height=300)
        )
        left.altair_chart(target_chart, use_container_width=True)

        revenue_col = self.find_column(df, ["TotalRevenue"])
        if revenue_col:
            revenue_by_churn = (
                df.groupby(self.config.target_column, as_index=False)[revenue_col]
                .mean()
                .rename(
                    columns={
                        self.config.target_column: "churn",
                        revenue_col: "average_revenue",
                    }
                )
            )
            revenue_chart = (
                alt.Chart(revenue_by_churn)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("churn:N", title="Churn"),
                    y=alt.Y("average_revenue:Q", title=f"Average {revenue_col}"),
                    color=alt.Color("churn:N", legend=None),
                    tooltip=[
                        "churn:N",
                        alt.Tooltip("average_revenue:Q", format=",.2f"),
                    ],
                )
                .properties(height=300)
            )
            right.altair_chart(revenue_chart, use_container_width=True)
        else:
            right.info("Revenue column is not available.")

    def render_segments(
        self, df: pd.DataFrame, categorical_cols: list[str], has_target: bool
    ) -> None:
        if not categorical_cols:
            st.info("No categorical features are available.")
            return

        preferred = [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
            "InternationalPlan",
            "VoiceMailPlan",
        ]
        options = [col for col in preferred if col in categorical_cols]
        options.extend([col for col in categorical_cols if col not in options])
        segment_col = st.selectbox("Segment by", options)

        left, right = st.columns(2)

        category_counts = (
            df[segment_col]
            .fillna("(missing)")
            .value_counts()
            .nlargest(12)
            .rename_axis(segment_col)
            .reset_index(name="customers")
        )
        count_chart = (
            alt.Chart(category_counts)
            .mark_bar()
            .encode(
                x=alt.X("customers:Q", title="Customers"),
                y=alt.Y(f"{segment_col}:N", sort="-x", title=segment_col),
                tooltip=[f"{segment_col}:N", "customers:Q"],
            )
            .properties(height=360)
        )
        left.altair_chart(count_chart, use_container_width=True)

        if has_target:
            churn_rates = (
                df.assign(churned=self.churn_mask(df))
                .groupby(segment_col, dropna=False, as_index=False)
                .agg(churn_rate=("churned", "mean"), customers=("churned", "size"))
                .sort_values("churn_rate", ascending=False)
                .head(12)
            )
            churn_chart = (
                alt.Chart(churn_rates)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "churn_rate:Q",
                        title="Churn rate",
                        axis=alt.Axis(format="%"),
                    ),
                    y=alt.Y(f"{segment_col}:N", sort="-x", title=segment_col),
                    color=alt.Color("churn_rate:Q", legend=None),
                    tooltip=[
                        f"{segment_col}:N",
                        alt.Tooltip("churn_rate:Q", format=".1%"),
                        "customers:Q",
                    ],
                )
                .properties(height=360)
            )
            right.altair_chart(churn_chart, use_container_width=True)
        else:
            right.info("Target column is not available.")

    def render_numeric_trends(
        self, df: pd.DataFrame, numeric_cols: list[str], has_target: bool
    ) -> None:
        if not numeric_cols:
            st.info("No numeric features are available.")
            return

        preferred = [
            "tenure",
            "TotalRevenue",
            "CustomerServiceCalls",
            "TotalDayMinutes",
            "TotalCall",
        ]
        options = [col for col in preferred if col in numeric_cols]
        options.extend([col for col in numeric_cols if col not in options])
        numeric_col = st.selectbox("Analyze numeric feature", options)

        left, right = st.columns(2)

        histogram = (
            alt.Chart(df)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X(f"{numeric_col}:Q", bin=alt.Bin(maxbins=30), title=numeric_col),
                y=alt.Y("count():Q", title="Customers"),
                color=(
                    alt.Color(f"{self.config.target_column}:N", title="Churn")
                    if has_target
                    else alt.value("#4c78a8")
                ),
                tooltip=[alt.Tooltip("count():Q", title="Customers")],
            )
            .properties(height=330)
        )
        left.altair_chart(histogram, use_container_width=True)

        if has_target:
            boxplot = (
                alt.Chart(df)
                .mark_boxplot(size=45)
                .encode(
                    x=alt.X(f"{self.config.target_column}:N", title="Churn"),
                    y=alt.Y(f"{numeric_col}:Q", title=numeric_col),
                    color=alt.Color(f"{self.config.target_column}:N", legend=None),
                )
                .properties(height=330)
            )
            right.altair_chart(boxplot, use_container_width=True)
        else:
            summary = df[numeric_col].describe().rename("value").reset_index()
            right.dataframe(summary, use_container_width=True)

    def render_relationships(
        self, df: pd.DataFrame, numeric_cols: list[str], has_target: bool
    ) -> None:
        if len(numeric_cols) < 2:
            st.info("At least two numeric features are needed for relationship charts.")
            return

        left, right = st.columns(2)
        x_col = left.selectbox("X axis", numeric_cols, index=0)
        y_default = min(1, len(numeric_cols) - 1)
        y_col = right.selectbox("Y axis", numeric_cols, index=y_default)

        scatter = (
            alt.Chart(df)
            .mark_circle(size=48, opacity=0.55)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_col),
                y=alt.Y(f"{y_col}:Q", title=y_col),
                color=(
                    alt.Color(f"{self.config.target_column}:N", title="Churn")
                    if has_target
                    else alt.value("#4c78a8")
                ),
                tooltip=[
                    alt.Tooltip(f"{x_col}:Q", format=",.2f"),
                    alt.Tooltip(f"{y_col}:Q", format=",.2f"),
                ],
            )
            .interactive()
            .properties(height=420)
        )
        st.altair_chart(scatter, use_container_width=True)

        corr = df[numeric_cols].corr(numeric_only=True).reset_index().melt(
            id_vars="index", var_name="feature", value_name="correlation"
        )
        heatmap = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x=alt.X("feature:N", title=None),
                y=alt.Y("index:N", title=None),
                color=alt.Color(
                    "correlation:Q",
                    scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                    title="Correlation",
                ),
                tooltip=[
                    alt.Tooltip("index:N", title="Feature"),
                    alt.Tooltip("feature:N", title="Compared with"),
                    alt.Tooltip("correlation:Q", format=".2f"),
                ],
            )
            .properties(height=520)
        )
        st.altair_chart(heatmap, use_container_width=True)

    @staticmethod
    def render_table(df: pd.DataFrame) -> None:
        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    def churn_mask(self, df: pd.DataFrame) -> pd.Series:
        return df[self.config.target_column].astype(str).eq(
            self.config.positive_target_label
        )

    @staticmethod
    def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None
