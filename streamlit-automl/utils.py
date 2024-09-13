import base64
import sys

import pandas as pd
import snowflake.ml.modeling.pipeline
import streamlit as st
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.snowpark import Session, exceptions
from snowflake.snowpark import functions as F
from snowflake.snowpark.context import get_active_session


def connect() -> Session:
    if sys._xoptions.get("snowflake_import_directory"):
        try:
            session = get_active_session()
        except exceptions.SnowparkSessionException:
            session = Session.builder.create()
    else:
        try:
            session = Session.builder.create()
        except exceptions.SnowparkSessionException as e:
            st.toast(e)
    return session


def get_model(model_type: str, estimator: str, label_cols: list):
    """
    Get the specified model based on the model type, estimator, and label columns.

    Args:
        model_type (str): The type of model (regression, classification, or clustering).
        estimator (str): The name of the estimator to retrieve.
        label_cols (list): The list of label columns.

    Returns:
        object: The specified model object.

    Raises:
        IndexError: If the specified estimator is not found in the model list.
    """
    model_types = dict(
        regression=[
            XGBRegressor(
                random_state=42, n_jobs=-1, scoring="accuracy", label_cols=label_cols
            ),
            "LightGBMRegressor",
            "RandomForestRegressor",
            "LinearRegression",
            "Lasso",
            "Ridge",
            "ElasticNet",
        ],
        classification=[
            XGBClassifier(
                random_state=42, n_jobs=-1, scoring="accuracy", label_cols=label_cols
            ),
            "LightGBMClassifier",
            "RandomForestClassifier",
            "LogisticRegression",
            "SVM",
        ],
        clustering=["KMeans", "DBSCAN", "SpectralClustering"],
    )
    model_list = model_types.get(model_type, [])
    selected_models = [_ for _ in model_list if _.__class__.__name__ == estimator]
    if len(selected_models) == 0:
        raise IndexError(f"No model found with the name '{estimator}'")
    return selected_models[0]


@st.cache_data
def get_databases(_session: Session) -> list:
    databases = _session.sql("SHOW TERSE DATABASES").select('"name"').collect()
    return [database.name for database in databases]


@st.cache_data
def get_schemas(_session: Session, database_name: str) -> list:
    schemas = (
        _session.sql(f"SHOW TERSE SCHEMAS IN DATABASE {database_name}")
        .select('"name"')
        .collect()
    )
    return [schemas.name for schemas in schemas if schemas.name != "INFORMATION_SCHEMA"]


@st.cache_data
def get_tables(_session: Session, database_name: str, schema_name: str) -> list:
    tables = (
        _session.sql(f"SHOW TABLES IN SCHEMA {database_name}.{schema_name}")
        .filter(F.col('"kind"') != F.lit("TEMPORARY"))
        .select('"name"')
        .collect()
    )
    return [table.name for table in tables]


@st.cache_data
def get_views(_session: Session, database_name: str, schema_name: str) -> list:
    views = (
        _session.sql(f"SHOW VIEWS IN SCHEMA {database_name}.{schema_name}")
        .select('"name"')
        .collect()
    )
    return [view.name for view in views]


def stratify_sample_fn(df, column_name, fraction, positive_class):
    # some how get the negative class (need Tyler help)
    neg_class = (
        df.select(F.col(column_name))
        .filter(F.col(column_name) != positive_class)
        .distinct()
        .collect()[0][0]
    )
    test = df.sample_by(column_name, {positive_class: fraction, neg_class: fraction})
    train = df.minus(test)
    return train, test


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    return f"""
    <style>
    div[data-testid="stApp"] {{
        background-color: #1d799a;
        background-image: url("data:image/png;base64,{bin_str}");
        background-position:  center bottom;
        background-size: contain;
        background-repeat: no-repeat;
        }}
    </style>
    """


def initialize_session_state():
    if "workflow" not in st.session_state:
        st.session_state["workflow"] = 0

    if "app_state" not in st.session_state:
        st.session_state["app_state"] = 0

    if "recorded_steps" not in st.session_state:
        st.session_state["recorded_steps"] = [0]
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "session" not in st.session_state:
        st.session_state["session"] = None
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None
    if "preprocessing_steps" not in st.session_state:
        st.session_state["preprocessing_steps"] = []
    if "context" not in st.session_state:
        st.session_state["context"] = {}
    if "pipeline_run" not in st.session_state:
        st.session_state["pipeline_run"] = False
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "persisted_fields" not in st.session_state:
        st.session_state["persisted_fields"] = {}
    if "eda_mode" not in st.session_state:
        st.session_state["eda_mode"] = False
    if "histogram_select" not in st.session_state:
        st.session_state["histogram_select"] = None
    if "model_ran" not in st.session_state:
        st.session_state.model_ran = False
    if "ts_col" not in st.session_state:
        st.session_state["ts_col"] = None
    if st.session_state["logged_in"] is False:
        try:
            with st.spinner("Connecting to Snowflake"):
                st.session_state["session"] = connect()
                st.session_state["logged_in"] = True
            st.toast("Connected!")
        except Exception as e:
            st.toast(e)
    if "timeseries_deploy_sequence" not in st.session_state:
        st.session_state["timeseries_deploy_sequence"] = 0
    if "model_card_kwargs" not in st.session_state:
        st.session_state["model_card_kwargs"] = {}
    if "ml_model_predictions" not in st.session_state:
        st.session_state["ml_model_predictions"] = None
    if "pipeline_object" not in st.session_state:
        st.session_state["pipeline_object"] = None
    if "pipeline_metrics" not in st.session_state:
        st.session_state["pipeline_metrics"] = None
    if "environment" not in st.session_state:
        st.session_state["environment"] = (
            "sis" if sys._xoptions.get("snowflake_import_directory") else "oss"
        )


def get_feature_importance_df(
    pipeline: snowflake.ml.modeling.pipeline.Pipeline,
) -> pd.DataFrame:
    estimator_step = pipeline.to_sklearn()[-1]
    model_type = estimator_step.__class__.__name__
    feature_names = estimator_step.feature_names_in_
    if model_type in ("XGBClassifier", "XGBRegressor"):
        feature_importances = estimator_step.feature_importances_
    elif model_type in (
        "ElasticNet",
        "LinearRegression",
    ):
        feature_importances = estimator_step.coef_
    elif model_type == "LogisticRegression":
        feature_importances = estimator_step.coef_[0]
    feat_imp_df = pd.DataFrame(
        zip(feature_names, feature_importances),
        columns=("FEATURE", "IMPORTANCE"),
    ).sort_values("IMPORTANCE", key=abs, ascending=False)

    return feat_imp_df
