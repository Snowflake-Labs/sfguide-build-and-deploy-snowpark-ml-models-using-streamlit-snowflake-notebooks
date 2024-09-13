import altair as alt
import pandas as pd
import streamlit as st
from common import get_col_types
from snowflake.snowpark import functions as F


@st.cache_data(show_spinner=False)
def prepare_eda_data(_df, name) -> pd.DataFrame:
    categoricals = get_col_types(_df, "string")
    stats = _df.describe()
    unique_vals = _df.select(
        [F.count_distinct(F.col(i)).alias(f"{i}") for i in _df.columns],
    )
    missing_vals = _df.select(
        [
            F.call_builtin("COUNT_IF", F.col(i).is_null()).alias(f"{i}")
            for i in _df.columns
        ],
    )
    missing_vals = missing_vals.with_column("SUMMARY", F.lit("missing"))
    unique_vals = unique_vals.with_column("SUMMARY", F.lit("unique"))

    combined = unique_vals.union(missing_vals).select(
        [i for i in ["SUMMARY"] + list(_df.columns)]
    )
    combined = combined.to_pandas().set_index("SUMMARY").T.rename_axis("feature")
    stats = stats.to_pandas().set_index("SUMMARY").T.rename_axis("feature")
    joined = combined.join(stats, how="left").reset_index()
    joined["type"] = joined["feature"].apply(
        lambda c: "Categorical" if c in categoricals else "Numeric"
    )
    joined.loc[joined["type"] == "Categorical", "min"] = None
    joined.loc[joined["type"] == "Categorical", "max"] = None
    joined.reset_index(drop=False, inplace=True)
    selector = {
        "feature": "Feature Name",
        "index": "Index",
        "type": "Var Type",
        "unique": "Unique",
        "missing": "Missing",
        "count": "Count",
        "mean": "Mean",
        "stddev": "Std Dev",
        "min": "Min",
        "max": "Max",
    }
    return joined.rename(columns=selector)[[*selector.values()]]


class AutoHistogram:
    def __init__(self, df, name) -> None:
        self.df = df
        self.name = name
        self.eval_df = prepare_eda_data(self.df, self.name)

    def render_histograms(self, eval_df) -> None:
        x_axis = eval_df["Feature Name"]
        x_axis_type = eval_df["Var Type"]

        max_bins = st.radio(
            "Maximum Bins",
            options=[5, 10, 15, 20],
            horizontal=True,
            disabled=x_axis_type == "Categorical",
        )
        hist = (
            alt.Chart(self.df.to_pandas())
            .mark_bar(color="#11567F")
            .encode(
                x=alt.X(
                    f"{x_axis}",
                    type="nominal",
                    title=x_axis,
                    bin=(
                        alt.Bin(maxbins=max_bins)
                        if x_axis_type == "Numeric"
                        else False
                    ),
                ),
                y=alt.Y(f"{x_axis}", type="quantitative", aggregate="count"),
            ).properties(title=x_axis)
        )
        st.altair_chart(hist, use_container_width=True)

    @st.experimental_dialog("Exploratory Data Analysis")
    def render_grid(self):
        st.info("Click the cell to the left of a feature name to display a histogram.")
        ordered_df = self.eval_df.sort_values(by="Index").fillna('')
        selected = st.dataframe(ordered_df,use_container_width=True, on_select="rerun", hide_index=True, key='histogram_select', selection_mode="single-row")
        if selected:
            if selected.get("selection").get("rows", False):
                selection_index = selected.get("selection").get("rows")[0]
                self.render_histograms(ordered_df.iloc[selection_index])
