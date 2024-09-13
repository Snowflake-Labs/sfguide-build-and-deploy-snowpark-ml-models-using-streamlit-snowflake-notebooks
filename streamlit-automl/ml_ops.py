import json
from textwrap import dedent
from typing import Union

import pandas as pd
import streamlit as st
from model_metrics import ModelMetrics

# Snowpark ML
from snowflake.ml.registry import Registry
from snowflake.snowpark import DataFrame, Session
from streamlit.components.v1 import html
from utils import get_databases, get_schemas, get_tables, get_views

class ModelReg:
    def __init__(self, session: Session) -> None:
        self.session = session

    def set_version_cb(version_key: str):
        pass

    def selected_model_card(self, loc):
        if kwargs := st.session_state["model_card_kwargs"].get(loc):
            if "current_type" in kwargs:
                type_pill = f"""
                            <span class="property_container">
                                <span class="property_title">VERSION TYPE</span>
                                <span class="property_pill_current">{kwargs.get("current_type")}</span>
                            </span>
                        """
            else:
                type_pill = ""
            if "selected_version" in kwargs:
                sel_ver = f"""
                            <span class="property_container">
                                <span class="property_title">SELECTED VERSION</span>
                                <span class="property_pill_current">{kwargs.get("selected_version")}</span>
                            </span>
                        """
            else:
                sel_ver = ""
        else:
            sel_ver = ""
            type_pill = ""
        html(
            f"""{st.session_state["css_styles"]}
            <div class="contact_card">
                        {type_pill}
                        {sel_ver}
                    </div>
                        """,
        )

    def model_contact_card(self, model, loc):
        model_location = ".".join(
            model[["database_name", "schema_name", "name"]].iloc[loc]
        )

        html(
            dedent(
                f"""{st.session_state["css_styles"]}
                <div class="contact_card">

                    <span class="property_container">
                        <span class="property_title">MODEL</span>
                        <span class="property_pill">{model["name"].values[loc]}</span>
                    </span>

                    <span class="property_container">
                        <span class="property_title">CREATED</span>
                        <span class="property_pill">{str(model["created_on"].values[loc]).replace("T", " ")}</span>
                    </span>

                    <span class="property_container">
                        <span class="property_title">LOCATION</span>
                        <span class="property_pill">{model_location}</span>
                    </span>

                    <span class="property_container">
                        <span class="property_title">DEFAULT VERSION</span>
                        <span class="property_pill">{model["default_version_name"].values[loc]}</span>
                    </span>

                    <span class="property_container">
                        <span class="property_title">MODEL TYPE</span>
                        <span class="property_pill">{'CLASSIFICATION' if model["is_classification"].values[loc] else 'REGRESSION'}</span>
                    </span>
                </div>
                        """
            ),
        )

        return model[["database_name", "schema_name", "name"]].iloc[loc]

    def calculate_metrics(self, df, true_col_name, model_type, db, schema):
        metric_results = {}
        model_metrics = ModelMetrics(targel_col=true_col_name)
        metrics = model_metrics.metrics_map.get(model_type)
        for idx, metric in enumerate(metrics):
            metric_fn = metrics.get(metric).get("fn")
            metric_kw = metrics.get(metric).get("kw")
            metric_results[metric] = metric_fn(
                df,
                **metric_kw,
            )

        return metric_results

    @st.experimental_dialog("Model Test")
    def call_test_models(self, df: Union[pd.DataFrame, DataFrame], tbl_name: str):
        test_columns = st.columns((1, 2, 2))
        test_status = test_columns[0].status("Model Test", expanded=True)
        warning = test_columns[0].empty()
        warning.warning(
            "Model Testing in progress, please do not close this window.", icon="⚠️"
        )

        for k, v in st.session_state["model_card_kwargs"].items():
            self.test_model_version(
                loc=k,
                df=df,
                source_table_name=tbl_name,
                test_status=test_status,
                test_columns=test_columns,
            )
        test_status.update(state="complete")
        warning.empty()
        warning.info(
            "Model Testing Complete, You may close this window now.", icon="✅"
        )

    def test_model_version(
        self,
        loc: int,
        df: Union[pd.DataFrame, DataFrame],
        source_table_name: str,
        test_status: st.status,
        test_columns: st.columns,
    ) -> None:
        try:
            model_type = st.session_state["model_card_kwargs"][loc].get("current_type")
            database = st.session_state["model_card_kwargs"][loc].get("db")
            schema = st.session_state["model_card_kwargs"][loc].get("schema")
            model_name = st.session_state["model_card_kwargs"][loc].get("model_name")
            version = st.session_state["model_card_kwargs"][loc].get("selected_version")

            model_reg = Registry(
                session=self.session, database_name=database, schema_name=schema
            )
            test_columns[loc + 1].subheader(f"{model_name}-{version}", anchor=False)
            selected_version = model_reg.get_model(model_name).version(version)
            predict_status = test_status.empty()
            predict_status.info(f"Predicting:\n\r({model_name}-{version})", icon="⏳")
            test_run = selected_version.run(df, function_name="PREDICT")
            true_col_name = [
                i.replace("OUTPUT_", "")
                for i in test_run.columns
                if i.startswith("OUTPUT_")
            ][0]

            test_columns[loc + 1].write("**Test Results**")
            test_columns[loc + 1].dataframe(
                test_run, use_container_width=True, hide_index=True
            )

            predict_status.empty()
            predict_status.success(
                f"Predicting:\n\r({model_name}-{version})", icon="✅"
            )
            test_status_cont = test_status.empty()
            test_status_cont.info(
                f"Fetching Metrics:\n\r({model_name}-{version})", icon="⏳"
            )
            metrics = self.calculate_metrics(
                df=test_run,
                true_col_name=true_col_name,
                model_type=str(model_type).title(),
                db=database,
                schema=schema,
            )
            test_columns[loc + 1].write("**Test Metrics**")
            test_columns[loc + 1].dataframe(
                metrics, use_container_width=True, hide_index=True
            )
            test_status_cont.empty()
            test_status_cont.success(
                f"Fetching Metrics:\n\r({model_name}-{version})", icon="✅"
            )

        except Exception as e:
            st.error(f"Error testing model: \n\n {e}")

    def create_version_card(self, version_data):
        with st.container(border=False):
            st.write("**Functions**")
            st.write("$~~$⇢$~~$".join(version_data.get("current_model_functions")))
            st.write("**Metrics**")
            st.dataframe(version_data.get("current_metrics"), use_container_width=True)

    def compare_versions(self, version_data) -> None:
        with st.spinner("Fetching Metadata"):
            compare_versions_columns = st.columns(2, gap="medium")
            with compare_versions_columns[0]:
                self.selected_model_card(loc=0)
                self.create_version_card(version_data=version_data[0])
            with compare_versions_columns[1]:
                self.selected_model_card(loc=1)
                self.create_version_card(version_data=version_data[1])

    # @st.experimental_dialog("Comparison")
    def compare_models(self, versions, models, mode) -> None:
        compare_version_columns = st.columns(2, gap="medium")
        with st.spinner("Fetching Metadata"):
            with compare_version_columns[0]:
                self.create_version_card(version_data=versions[0])
            with compare_version_columns[1]:
                self.create_version_card(version_data=versions[1])

    @st.cache_data(show_spinner=False)
    def get_model_version_details(
        _self, model_name: str, model_version, database: str, schema: str
    ) -> tuple:
        model_reg = Registry(
            session=_self.session, database_name=database, schema_name=schema
        )
        selected_version = model_reg.get_model(model_name).show_versions()
        curr_model_fns = [
            i.get("name")
            for i in model_reg.get_model(model_name)
            .version(model_version)
            ._get_functions()
        ]
        curr_model_metrics = (
            model_reg.get_model(model_name).version(model_version).show_metrics()
        )
        curr_model_type = (
            "CLASSIFICATION" if "PREDICT_PROBA" in curr_model_fns else "REGRESSION"
        )
        return selected_version, dict(
            current_type=curr_model_type,
            selected_version=model_version,
            current_model_functions=curr_model_fns,
            current_metrics=curr_model_metrics,
            db=database,
            schema=schema,
            model_name=model_name,
        )

    def create_models_grid(self, models, pos):
        if models.shape[0] == 1:
            st.subheader("Compare versions", anchor=False)
            model_info = self.model_contact_card(model=models, loc=0)
            compare_versions_columns = st.columns(2, gap="medium")
            model_reg = Registry(
                session=self.session,
                database_name=model_info[0],
                schema_name=model_info[1],
            )
            versions = model_reg.get_model(model_info[2]).show_versions()


            if len(versions) == 1:
                st.info("Model contains only 1 version, compare mode disabled")
            with compare_versions_columns[0]:
                left_ver = st.selectbox(
                    "Version",
                    options=versions["name"],
                    key="model_version_left",
                    index=None,
                    placeholder="Select a Version",
                    disabled=len(versions) == 1,
                )
                if left_ver:
                    left_version_data = self.get_model_version_details(
                        database=model_info[0],
                        schema=model_info[1],
                        model_name=model_info[2],
                        model_version=left_ver,
                    )
            with compare_versions_columns[1]:
                right_ver = st.selectbox(
                    "Version",
                    options=versions["name"],
                    key="model_version_right",
                    index=None,
                    placeholder="Select a Version",
                    disabled=len(versions) == 1,
                )
                if right_ver:
                    right_ver_version_data = self.get_model_version_details(
                        database=model_info[0],
                        schema=model_info[1],
                        model_name=model_info[2],
                        model_version=right_ver,
                    )

            metrics, test = st.tabs(["Metrics", "Test"])
            if all([right_ver, left_ver]):
                st.session_state["model_card_kwargs"][0] = left_version_data[1]
                st.session_state["model_card_kwargs"][1] = right_ver_version_data[1]
                with metrics:
                    with st.spinner("Fetching Metrics"):
                        self.compare_versions(
                            version_data=st.session_state["model_card_kwargs"]
                        )
                with test:
                    src_pop = st.popover("Source Data", use_container_width=True)
                    database = src_pop.selectbox(
                        "Database",
                        options=get_databases(_session=self.session),
                        index=None,
                        placeholder="Choose a Database",
                        label_visibility="collapsed",
                    )
                    schema = src_pop.selectbox(
                        "Schema",
                        options=(
                            get_schemas(database_name=database, _session=self.session)
                            if database
                            else []
                        ),
                        index=None,
                        placeholder="Choose a Schema",
                        label_visibility="collapsed",
                    )
                    object_type = src_pop.radio(
                        "",
                        label_visibility="collapsed",
                        options=["Table", "View"],
                        horizontal=True,
                    )
                    if object_type == "Table":
                        table = src_pop.selectbox(
                            "Table",
                            options=(
                                get_tables(
                                    database_name=database,
                                    schema_name=schema,
                                    _session=self.session,
                                )
                                if schema
                                else []
                            ),
                            index=None,
                            placeholder="Choose a Table",
                            label_visibility="collapsed",
                        )
                    else:
                        table = src_pop.selectbox(
                            "View",
                            options=(
                                get_views(
                                    database_name=database,
                                    schema_name=schema,
                                    _session=self.session,
                                )
                                if schema
                                else []
                            ),
                            index=None,
                            placeholder="Choose a View",
                            label_visibility="collapsed",
                        )

                    source_location = (
                        ".".join([database, schema, table])
                        if all([database, schema, table])
                        else None
                    )
                    if source_location:
                        test_df = self.session.table(source_location)
                        st.caption(source_location)
                        if st.button(
                            "Start Test", use_container_width=True, type="primary"
                        ):
                            self.call_test_models(df=test_df, tbl_name=source_location)
                    else:
                        st.caption(":red[Select a Source Dataset]")

        else:
            st.subheader("Compare & Test Models", anchor=False)

            model_colums = st.columns(2, gap="medium")
            reg_versions = {}
            for k, v in models.iterrows():
                key = "model_version_left" if k == 0 else "model_version_right"
                with model_colums[k]:
                    model_info = self.model_contact_card(model=models, loc=k)
                    curr_ver = st.selectbox(
                        "Version",
                        options=v["versions"],
                        key=key,
                        index=None,
                        placeholder="Select a Version",
                    )
                    if curr_ver:
                        curr_version_data = self.get_model_version_details(
                            database=model_info[0],
                            schema=model_info[1],
                            model_name=model_info[2],
                            model_version=curr_ver,
                        )
                        reg_versions[k] = curr_version_data[0]
                        st.session_state["model_card_kwargs"][k] = curr_version_data[1]
                        self.selected_model_card(k)
                    else:
                        st.session_state["model_card_kwargs"][k] = {}
            metrics, test = st.tabs(["Metrics", "Test"])

            if all(
                [
                    st.session_state["model_version_right"],
                    st.session_state["model_version_left"],
                ]
            ):
                with metrics:
                    with st.spinner("Fetching Metrics"):
                        self.compare_models(
                            versions=st.session_state["model_card_kwargs"],
                            models=models,
                            mode=1,
                        )
                with test:
                    st.info(
                        "You can only test data against two versions of the same model at this time."
                    )
                   

    @st.cache_data(show_spinner=False)
    def get_models(_self):
        return _self.session.sql("show models in account").collect()

    @st.cache_data(show_spinner=False)
    def get_current_version(_self, model, database, schema):
        model_reg = Registry(
            session=_self.session, database_name=database, schema_name=schema
        )
        versions = model_reg.get_model(model).show_versions()
        versions = versions.loc[versions["is_default_version"] == "true"]
        return "PREDICT_PROBA" in versions["functions"].values[0]
        # return  'REGRESSION' if 'PREDICT_PROBA' not in list(versions["functions"].values[0]) else 'CLASSIFICATION'

    def render_registry(self):
        st.header("Model Registry", anchor=False)
        st.caption("Select One or Two models to compare.")
        models = self.get_models()
        models_df = pd.DataFrame(models)
        if models_df.empty is False:
            models_df["versions"] = models_df["versions"].apply(
                lambda x: x.replace("[", "")
                .replace("]", "")
                .replace('"', "")
                .split(",")
            )
            st.dataframe(
                models_df,
                hide_index=True,
                use_container_width=True,
                selection_mode=["multi-row"],
                on_select="rerun",
                key="selected_models",
            )
            if st.session_state["selected_models"].get("selection").get("rows"):
                selected = (
                    st.session_state["selected_models"].get("selection").get("rows")
                )
                if len(selected) > 2:
                    st.warning("Select a maximum of two(2) models.")
                else:
                    selected_models = models_df.iloc[selected].reset_index(drop=True).reset_index()
                    selected_models["side"] = selected_models["index"].apply(
                        lambda x: "left" if x == 0 else "right"
                    )
                    selected_models.drop(columns=["index"], inplace=True)
                    selected_models["is_classification"] = selected_models.apply(
                        lambda x: self.get_current_version(
                            model=x["name"],
                            database=x["database_name"],
                            schema=x["schema_name"],
                        ),
                        axis=1,
                    )
                    self.create_models_grid(models=selected_models, pos=selected)
