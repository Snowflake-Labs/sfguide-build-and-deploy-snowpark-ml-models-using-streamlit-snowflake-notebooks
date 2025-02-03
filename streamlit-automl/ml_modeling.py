from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from callbacks import Callbacks
from code_exporter import create_notebook
from common import get_col_types
from histograms import AutoHistogram
from model_metrics import ModelMetrics
from preprocessing import AutoPreProcessor
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
)
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session
from streamlit.components.v1 import html
from utils import get_databases, get_feature_importance_df, get_schemas, get_tables

AVATAR_PATH = str(Path(__file__).parent / "resources" / "Snowflake_ICON_Chat.png")


def set_state(state: int):
    st.session_state["app_state"] = state
    if state not in st.session_state["recorded_steps"]:
        st.session_state["recorded_steps"].append(state)


def create_metric_card(label, value):
    return f"""
             <span class="property_container">
                <span class="property_title">{label}</span>
                <span class="property_pill_current">{value}</span>
            </span>
                """


class TopMenu:
    def __init__(self) -> None:
        header_menu_c = st.container(border=False, height=60)
        header_menu = header_menu_c.columns(3)
        header_menu[0].button(
            "Select Dataset",
            key="btn_select",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 0 else "secondary",
            disabled=0 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[0],
        )
        header_menu[1].button(
            "Pre-Processing",
            key="btn_preprocess",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 1 else "secondary",
            disabled=1 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[1],
        )
        header_menu[2].button(
            "Modeling",
            key="btn_modeling",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 2 else "secondary",
            disabled=2 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[2],
        )


class AutoMLModeling:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.step_data = {}

    def render_ml_builder(self):
        with st.container(height=101, border=False):
            if st.button("🔍"):
                if st.session_state["dataset"]:
                    AutoHistogram(
                        df=st.session_state["dataset"],
                        name=f"{st.session_state.get('aml_mpa.sel_db','-')}.{st.session_state.get('aml_mpa.sel_schema','-')}.{st.session_state.get('aml_mpa.sel_table','-')}",
                    ).render_grid()
                else:
                    st.toast("You must select a dataset before.")

        TopMenu()
        if st.session_state["app_state"] > -1:
            dataset_chat = st.chat_message(
                name="assistant",
                avatar=AVATAR_PATH,
            )
            with dataset_chat:
                st.write("Let's begin by selecting a source dataset.")
                with st.popover(
                    "Dataset Selection",
                    disabled=not (st.session_state["app_state"] == 0),
                    use_container_width=True,
                ):
                    context_menu_cols = st.columns((1, 2))
                    databases = get_databases(self.session)
                    db = context_menu_cols[0].selectbox(
                        "Source Database",
                        index=None,
                        options=databases,
                        placeholder="Select a database",
                        key="aml_mpa.sel_db",
                    )
                    if db:
                        st.session_state["context"]["database"] = db
                        st.session_state["context"]["schemas"] = get_schemas(
                            self.session, db
                        )
                    else:
                        st.session_state["context"]["schemas"] = []

                    schema = context_menu_cols[0].selectbox(
                        "Source Schema",
                        st.session_state["context"].get("schemas", []),
                        index=None,
                        placeholder="Select a schema",
                        key="aml_mpa.sel_schema",
                    )

                    if schema:
                        st.session_state["context"]["tables"] = get_tables(
                            self.session, db, schema
                        )
                    else:
                        st.session_state["context"]["tables"] = []

                    table = context_menu_cols[0].selectbox(
                        "Source Table",
                        st.session_state["context"].get("tables", []),
                        index=None,
                        placeholder="Select a table",
                        key="aml_mpa.sel_table",
                        on_change=Callbacks.set_dataset,
                        args=[self.session, db, schema, "aml_mpa.sel_table"],
                    )
                    if all([db, schema, table]):
                        context_menu_cols[1].dataframe(
                            st.session_state["dataset"].limit(5),
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.container(border=False, height=10)
                        dataset_chat_cols = dataset_chat.columns(3)
                        dataset_chat_cols[2].button(
                            "Next",
                            use_container_width=True,
                            type="primary",
                            on_click=set_state,
                            args=[1],
                            disabled=not (st.session_state["app_state"] == 0),
                        )

        if 1 in st.session_state["recorded_steps"] and st.session_state["dataset"]:
            preproc_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with preproc_chat:
                st.write("Now, let's pre-process the source dataset.")
                st.info(
                    "Click the :mag: icon above to dig deeper into your data. If you have null values, add a **SimpleImputer** step. If you have string features, add a **OneHotEncoder** step. "
                )
                with st.expander(
                    "Pre-Processing Options",
                    expanded=st.session_state["app_state"] == 1,
                ):
                    st.header("Preprocessing Options")
                    st.caption(":red[*] required fields")
                    feature_cols = st.multiselect(
                        "Select the feature columns.:red[*]",
                        options=st.session_state["dataset"].columns,
                    )
                    target_col = st.selectbox(
                        "Select the target column.:red[*]",
                        st.session_state["dataset"].columns,
                        index=None,
                    )

                    if feature_cols and target_col:
                        t_sub = st.session_state["dataset"].select(
                            feature_cols + [target_col]
                        )
                        cat_cols = get_col_types(t_sub, "string")
                        num_cols = get_col_types(t_sub, "numeric")

                        if target_col in cat_cols:
                            cat_cols.remove(target_col)
                        if target_col in num_cols:
                            num_cols.remove(target_col)

                        preprocessor_options = [
                            "SimpleImputer (numeric)",
                            "SimpleImputer (categorical)",
                            "OneHotEncoder",
                            "StandardScaler",
                            "MinMaxScaler",
                            "MaxAbsScaler",
                        ]
                        steps_container = st.container(border=False)
                        st.divider()
                        st.button(
                            "Add Step",
                            on_click=Callbacks.add_step,
                            use_container_width=True,
                            type="primary",
                        )

                        st.container(border=False, height=25)
                        processors_map = {
                            "SI": SimpleImputer,
                            "OHE": OneHotEncoder,
                            "SS": StandardScaler,
                            "MMS": MinMaxScaler,
                            "MAS": MaxAbsScaler,
                        }
                        if bool(st.session_state["preprocessing_steps"]):
                            steps_container.divider()
                        for steps in st.session_state["preprocessing_steps"]:
                            with steps_container:
                                definition = AutoPreProcessor(
                                    id=steps,
                                    preprocessor_options=preprocessor_options,
                                    num_cols=num_cols,
                                    cat_cols=cat_cols,
                                )
                                if definition.step_return:
                                    self.step_data[steps] = definition.step_return

                        pprocessing_steps = []

                        for seq, data in enumerate(self.step_data.values()):
                            c_step = processors_map.get(data.get("preprocess_type"))
                            if data.get("is_valid"):
                                pprocessing_steps.append(
                                    (data.get("title"), c_step(**data.get("kw")))
                                )

                        progress_cont = st.empty()
                        if (
                            len(pprocessing_steps)
                            == len(st.session_state["preprocessing_steps"])
                            and len(st.session_state["preprocessing_steps"]) > 0
                        ):
                            pproc_btn = st.button(
                                "Generate Preview",
                                use_container_width=True,
                            )

                            if pproc_btn:
                                prproc_prg = progress_cont.progress(
                                    value=0, text="Pre-Processing Dataset"
                                )
                                with prproc_prg:
                                    transform_pipeline = Pipeline(
                                        steps=pprocessing_steps
                                    )
                                    prproc_prg.progress(
                                        33, "Pipeline Transform Updated"
                                    )
                                    transform_pipeline.fit(st.session_state["dataset"])
                                    prproc_prg.progress(66, "Pipeline Fit Completed")
                                    st.session_state["processed_df"] = (
                                        transform_pipeline.transform(
                                            st.session_state["dataset"]
                                        )
                                    )
                                    prproc_prg.progress(
                                        100,
                                        "Pipeline Transform Completed - Preview Available",
                                    )
                                    progress_cont.empty()
                                    st.session_state["pipeline_run"] = True

                            preproc_preview = st.popover(
                                "Preview", use_container_width=True
                            )
                            if (
                                st.session_state["pipeline_run"]
                                and st.session_state["processed_df"]
                            ):
                                preproc_preview.container(height=20, border=False)
                                preproc_preview.dataframe(
                                    st.session_state["processed_df"].limit(10),
                                    hide_index=True,
                                )
                            st.button(
                                "Next",
                                use_container_width=True,
                                type="primary",
                                on_click=set_state,
                                args=[2],
                                key="pproc_nxt",
                            )
        if 2 in st.session_state["recorded_steps"] and st.session_state["dataset"]:
            modeling_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with modeling_chat:
                st.write("Review your model options")
                with st.expander(
                    "Modeling", expanded=st.session_state["app_state"] == 2
                ):
                    st.header("Modeling Options")
                    model_types = [
                        {
                            "type": "Regression",
                            "models": [
                                "XGBRegressor",
                                "LinearRegression",
                                "ElasticNet",
                            ],
                        },
                        {
                            "type": "Classification",
                            "models": [
                                "XGBClassifier",
                                "LogisticRegression",
                            ],
                        },
                    ]

                    model_type = st.radio(
                        "Model Type",
                        options=[i.get("type") for i in model_types],
                        horizontal=True,
                    )
                    available_models = [
                        i for i in model_types if i.get("type") == model_type
                    ][0].get("models")
                    model_selections = st.selectbox("Model", options=available_models)
                    if bool(model_selections):
                        fit_menu = st.columns(4, gap="large")
                        show_metrics = fit_menu[0].toggle(
                            "Retrieve Model Metrics", value=True
                        )
                        fit_btn = fit_menu[1].button(
                            "Fit Model & Run Prediction(s)", use_container_width=True
                        )

                        with fit_menu[2].popover(
                            "Download Notebook",
                            use_container_width=True,
                            disabled=not st.session_state["model_ran"],
                        ):
                            st.markdown("Please input Project Name, then hit Save")
                            proj_name = st.text_input("Project Name")
                            registry_db = st.selectbox(
                                "Database",
                                options=get_databases(),
                                index=None,
                                placeholder="Choose a Database",
                                label_visibility="collapsed",
                                key="registry_nb_db",
                            )
                            if registry_db:
                                registry_schema = st.selectbox(
                                    "Schema",
                                    options=get_schemas(
                                        _session=self.session,
                                        database_name=(
                                            registry_db if registry_db else []
                                        ),
                                    ),
                                    index=None,
                                    placeholder="Choose a Schema",
                                    label_visibility="collapsed",
                                    key="registry_nb_schema",
                                )
                                # TODO: Make this not if-if-if nested.
                                if all([proj_name, registry_db, registry_schema]):
                                    download_column1, download_column2 = st.columns(2)
                                    st.session_state.notebook_btn = (
                                        download_column1.download_button(
                                            label="Download",
                                            data=create_notebook(
                                                st.session_state[
                                                    "full_qualified_table_nm"
                                                ],
                                                st.session_state.complete_pipeline,
                                                project_name=proj_name,
                                                registry_database=registry_db,
                                                registry_schema=registry_schema,
                                                context="local",
                                            ),
                                            file_name=proj_name + ".ipynb",
                                            mime="application/x-ipynb+json",
                                        )
                                    )
                                    upload_button = download_column2.button(
                                        label="Create Snowflake Notebook"
                                    )
                                    if upload_button:
                                        # TODO: Context is being set in app based on source data. Notebook creation should take user's context.
                                        stage_name = str(
                                            st.session_state["session"]
                                            .get_session_stage()
                                            .replace('"', "")
                                            .replace("@", "")
                                        )
                                        target_path = (
                                            f"@{stage_name}/ntbk/{proj_name}.ipynb"
                                        )
                                        self.session.file.put_stream(
                                            create_notebook(
                                                st.session_state[
                                                    "full_qualified_table_nm"
                                                ],
                                                st.session_state.complete_pipeline,
                                                project_name=proj_name,
                                                registry_database=registry_db,
                                                registry_schema=registry_schema,
                                                context="snowflake",
                                            ),
                                            target_path,
                                            auto_compress=False,
                                        )
                                        self.session.sql(
                                            f"""CREATE NOTEBOOK {registry_db}.{registry_schema}.{proj_name}
 FROM '@{stage_name}/ntbk'
 MAIN_FILE = '{proj_name}.ipynb'
 QUERY_WAREHOUSE = {self.session.get_current_warehouse()}"""
                                        ).collect()

                        if fit_btn:
                            fit_prg_cont = st.empty()
                            fit_prg = fit_prg_cont.progress(
                                value=0, text="Fitting Model"
                            )

                            model_classes = {
                                "XGBClassifier": (XGBClassifier, {}),
                                "LinearRegression": (LinearRegression, {}),
                                "ElasticNet": (ElasticNet, {}),
                                "XGBRegressor": (XGBRegressor, {}),
                                "LogisticRegression": (LogisticRegression, {}),
                            }

                            shared_params = {
                                "random_state": 42,
                                "input_cols": [],
                                "n_jobs": -1,
                                "label_cols": target_col,
                            }

                            if model_selections in model_classes:
                                model_class, specific_params = model_classes[
                                    model_selections
                                ]
                                if model_selections == "LinearRegression":
                                    del shared_params["random_state"]
                                elif model_selections == "ElasticNet":
                                    del shared_params["n_jobs"]
                                model = model_class(**shared_params, **specific_params)
                                pprocessing_steps.append((model_selections, model))

                            complete_pipeline = Pipeline(steps=pprocessing_steps)
                            complete_pipeline.fit(
                                st.session_state["dataset"].select(
                                    feature_cols + [target_col]
                                )
                            )
                            fit_prg.progress(
                                value=50, text="Model Fitted, running predictions"
                            )
                            st.session_state["complete_pipeline"] = complete_pipeline
                            predictions = complete_pipeline.predict(
                                st.session_state["dataset"]
                            )
                            st.session_state["ml_model_predictions"] = predictions
                            fit_prg.progress(value=100, text="Predictions Complete")
                            st.session_state["model_ran"] = True
                            if show_metrics:
                                metric_results = {}
                                model_metrics = ModelMetrics(targel_col=target_col)
                                metrics = model_metrics.metrics_map.get(
                                    str(model_type).title()
                                )
                                for idx, metric in enumerate(metrics):
                                    metric_fn = metrics.get(metric).get("fn")
                                    metric_kw = metrics.get(metric).get("kw")
                                    metric_results[metric] = metric_fn(
                                        st.session_state["ml_model_predictions"],
                                        **metric_kw,
                                    )

                            if st.session_state["model_ran"]:
                                st.session_state["pipeline_object"] = complete_pipeline
                                if show_metrics:
                                    st.session_state["pipeline_metrics"] = (
                                        metric_results
                                    )
                                st.rerun()

                        if st.session_state["ml_model_predictions"]:
                            st.dataframe(
                                st.session_state["ml_model_predictions"].limit(15),
                                hide_index=True,
                                use_container_width=True,
                            )
                            with fit_menu[3].popover(
                                "Save to Registry",
                                use_container_width=True,
                                disabled=not st.session_state["model_ran"],
                            ):
                                st.markdown("Please input Project Name, then hit Save")
                                if st.session_state["environment"] == "sis":
                                    tgt_database = st.session_state[
                                        "session"
                                    ].get_current_database()
                                    tgt_schema = st.session_state[
                                        "session"
                                    ].get_current_schema()
                                    st.caption(
                                        f"{tgt_database[1:-1]}.{tgt_schema[1:-1]}"
                                    )

                                else:
                                    tgt_database = st.selectbox(
                                        "Database",
                                        options=get_databases(),
                                        index=None,
                                        placeholder="Choose a Database",
                                        label_visibility="collapsed",
                                        key="tgt_db",
                                    )
                                    tgt_schema = st.selectbox(
                                        "Schema",
                                        options=(
                                            get_schemas(
                                                database_name=(
                                                    tgt_database if tgt_database else []
                                                ),
                                                _session=self.session,
                                            )
                                            if tgt_database
                                            else []
                                        ),
                                        index=None,
                                        placeholder="Choose a Schema",
                                        label_visibility="collapsed",
                                        key="tgt_schema",
                                    )
                                tgt_model_name = st.text_input(
                                    "",
                                    label_visibility="collapsed",
                                    placeholder="Model Name",
                                    key="tgt_forecast_name",
                                )
                                target_location = (
                                    ".".join([tgt_database, tgt_schema, tgt_model_name])
                                    if all([tgt_database, tgt_schema, tgt_model_name])
                                    else None
                                )
                                if target_location:
                                    reg = Registry(
                                        self.session,
                                        database_name=tgt_database,
                                        schema_name=tgt_schema,
                                    )
                                    try:
                                        reg.get_model(tgt_model_name)
                                        st.write(
                                            f"Model {tgt_model_name} already exists in {tgt_database}.{tgt_schema}. Would you like to save as a new version?"
                                        )
                                        button_text = "Save New Version"
                                    except ValueError:
                                        button_text = "Register"

                                    register_columns = st.columns(2)

                                    if register_columns[0].button(button_text):
                                        try:
                                            with register_columns[1]:
                                                with st.spinner("Saving..."):
                                                    query_tag_comment = '{"origin": "sf_sit", "name": "ml_sidekick", "version": {"major":1, "minor":0}, "attributes":{"component":"model"}}'
                                                    reg.log_model(
                                                        model=st.session_state[
                                                            "pipeline_object"
                                                        ],
                                                        model_name=tgt_model_name,
                                                        metrics=st.session_state[
                                                            "pipeline_metrics"
                                                        ],
                                                        comment = query_tag_comment
                                                    )
                                                    reg.get_model(tgt_model_name).comment = query_tag_comment
                                                    st.toast("Model Registered")

                                        except Exception as e:
                                            st.toast(
                                                f"Failed to register model \n\n {e}"
                                            )

                        if st.session_state["pipeline_metrics"]:
                            st.header("Model Metrics", anchor=False)
                            metric_columns = st.columns(2, gap="small")
                            metric_pills = []
                            for key, metric in st.session_state[
                                "pipeline_metrics"
                            ].items():
                                if key == "Confusion Matrix":
                                    tp = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            0
                                        ][0]
                                    )
                                    fp = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            0
                                        ][1]
                                    )
                                    fn = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            1
                                        ][0]
                                    )
                                    tn = int(
                                        st.session_state["pipeline_metrics"].get(key)[
                                            1
                                        ][1]
                                    )

                                    data = [
                                        [
                                            tp,
                                            "TP",
                                            "Positive",
                                            "Positive",
                                            "pos",
                                        ],
                                        [fp, "FP", "Positive", "Negative", "neg"],
                                        [fn, "FN", "Negative", "Positive", "neg"],
                                        [tn, "TN", "Negative", "Negative", "pos"],
                                    ]

                                    df = pd.DataFrame(
                                        data,
                                        columns=[
                                            "value",
                                            "label",
                                            "predicted",
                                            "actual",
                                            "color",
                                        ],
                                    )
                                    df["calculated_text"] = df.apply(
                                        lambda x: x["label"] + ":" + str(x["value"]),
                                        axis=1,
                                    )
                                    colors = ["#29b5e8", "grey"]
                                    domains = ["pos", "neg"]
                                    base = (
                                        alt.Chart(df)
                                        .mark_rect(height=90, width=90, cornerRadius=5)
                                        .encode(
                                            x=alt.X(
                                                "actual",
                                                type="nominal",
                                                sort="y",
                                                title="Actual Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="top",
                                                    labelAngle=0,
                                                    labelFontSize=20,
                                                    labelColor="black",
                                                    titleColor="black",
                                                ),
                                            ),
                                            y=alt.Y(
                                                "predicted",
                                                type="nominal",
                                                sort="x",
                                                title="Predicted Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="left",
                                                    labelAngle=-90,
                                                    labelFontSize=20,
                                                    labelColor="black",
                                                    titleColor="black",
                                                ),
                                            ),
                                            color=alt.Color(
                                                field="color",
                                                type="nominal",
                                                scale=alt.Scale(
                                                    domain=domains, range=colors
                                                ),
                                                title="Region",
                                                legend=None,
                                            ),
                                            tooltip=alt.value(None),
                                        )
                                    )
                                    text = (
                                        alt.Chart(df)
                                        .mark_text(
                                            fontSize=16,
                                        )
                                        .encode(
                                            x=alt.X(
                                                "actual",
                                                type="nominal",
                                                sort="y",
                                                title="Actual Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="top",
                                                    labelAngle=0,
                                                ),
                                            ),
                                            y=alt.Y(
                                                "predicted",
                                                type="nominal",
                                                sort="x",
                                                title="Predicted Values",
                                                axis=alt.Axis(
                                                    labelAlign="center",
                                                    orient="left",
                                                    labelAngle=-90,
                                                ),
                                            ),
                                            color=alt.value("white"),
                                            text=alt.Text("calculated_text"),
                                            tooltip=alt.value(None),
                                        )
                                    )
                                    layered = base + text
                                    metric_columns[1].subheader(
                                        "Confusion Matrix", divider=True
                                    )
                                    metric_columns[1].altair_chart(
                                        layered.properties(
                                            width=360, height=330, padding={"left": 100}
                                        )
                                    )
                                else:
                                    metric_pills.append(create_metric_card(key, metric))
                            with metric_columns[0]:
                                st.subheader("Metrics", divider=True)
                                pills = "\n".join(metric_pills)
                                html(
                                    f"""{st.session_state["css_styles"]}
                                    <div class="contact_card">
                                                {pills}
                                            </div>
                                                """,
                                )
                            feat_imp_df = get_feature_importance_df(
                                st.session_state.get("pipeline_object")
                            )
                            feat_imp_df["ABS_VALUE"] = (
                                feat_imp_df["IMPORTANCE"].abs().astype(float)
                            )
                            feat_imp_df["IMPORTANCE"] = (
                                feat_imp_df["IMPORTANCE"].round(3).astype(float)
                            )
                            feat_imp_df = feat_imp_df[
                                ["FEATURE", "ABS_VALUE", "IMPORTANCE"]
                            ]

                            metric_columns[0].subheader(
                                "Feature Importance", divider=True
                            )

                            metric_columns[0].dataframe(
                                feat_imp_df,
                                column_config={
                                    "ABS_VALUE": st.column_config.ProgressColumn(
                                        "Relative Importance",
                                        format=" ",
                                        min_value=feat_imp_df["ABS_VALUE"].min(),
                                        max_value=feat_imp_df["ABS_VALUE"].max(),
                                        width="large",
                                    ),
                                    "FEATURE": st.column_config.TextColumn(
                                        "Feature", width="small"
                                    ),
                                    "IMPORTANCE": st.column_config.TextColumn(
                                        "Importance", width="small"
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )
