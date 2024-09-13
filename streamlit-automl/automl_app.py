from pathlib import Path

import streamlit as st
from callbacks import Callbacks
from ml_modeling import AutoMLModeling
from ml_ops import ModelReg
from utils import initialize_session_state, set_png_as_page_bg

if "sidebar_state" not in st.session_state:
    st.session_state["sidebar_state"] = "collapsed"

st.set_page_config(
    layout="wide",
    page_title="Snowflake Auto ML",
    page_icon="❄️",
    initial_sidebar_state=st.session_state["sidebar_state"],
)

initialize_session_state()

with open(Path(__file__).parent / "styles" / "css_bootstrap.html", "r") as r:
    styles = r.read()
    st.markdown(styles, unsafe_allow_html=True)
    if "css_styles" not in st.session_state:
        st.session_state["css_styles"] = styles

st.markdown(
    set_png_as_page_bg(Path(__file__).parent / "resources" / "background.png"),
    unsafe_allow_html=True,
)

with st.container(height=135, border=False):
    st.title("ML Sidekick")
    st.caption("A no-code application for leveraging the snowflake-ml-python package")
    if st.session_state["workflow"] == 0:
        with st.container(border=False, height=51):
            with st.popover("Create Project", use_container_width=True):
                st.button(
                    "ML Model",
                    use_container_width=True,
                    on_click=Callbacks.set_workflow,
                    args=[1],
                )
    else:
        with st.container(border=False, height=49):
            st.button("←", on_click=Callbacks.set_workflow, args=[0])


if st.session_state["logged_in"]:
    session = st.session_state["session"]
    if st.session_state["workflow"] == 0:
        ModelReg(session=session).render_registry()
        pass

    # Workflow 1 = ML Model
    if st.session_state["workflow"] == 1:
        AutoMLModeling(session=session).render_ml_builder()
