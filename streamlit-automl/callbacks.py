import uuid

import streamlit as st
from snowflake.snowpark import Session


class Callbacks:
    @staticmethod
    def add_step():
        st.session_state["preprocessing_steps"].append(str(uuid.uuid4()))

    @staticmethod
    def remove_step(step_id):
        if step_id in st.session_state["preprocessing_steps"]:
            st.session_state["preprocessing_steps"].remove(step_id)

    @staticmethod
    def eda_mode_switch(mode: bool):
        st.session_state["eda_mode"] = st.session_state[mode]

    @staticmethod
    def set_dataset(session: Session, db: str, schema: str, table: str) -> None:
        db = f'"{db}"'
        schema = f'"{schema}"'
        table = f'"{st.session_state[table]}"'
        fully_qualified_name = f"{db}.{schema}.{table}"
        st.session_state["full_qualified_table_nm"] = fully_qualified_name
        st.session_state["dataset"] = session.table(fully_qualified_name)

    @staticmethod
    def set_workflow(workflow_id: int):
        if workflow_id == 0:
            st.cache_data.clear()
        st.session_state["workflow"] = workflow_id

    @staticmethod
    def set_timeseries_seq(seq_id: int):
        # Manages workflow state for timeseries page
        st.session_state["timeseries_deploy_sequence"] = seq_id
