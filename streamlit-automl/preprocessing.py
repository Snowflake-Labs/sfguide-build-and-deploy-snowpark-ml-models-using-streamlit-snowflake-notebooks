import streamlit as st
from callbacks import Callbacks
from common import Images


class AutoPreProcessor:
    def __init__(
        self,
        id,
        preprocessor_options,
        cat_cols,
        num_cols,
        # passthrough_preds,
        # passthrough_cols,
    ) -> None:
        self.id = id
        self.preprocessor_options = preprocessor_options
        preproc_cols = st.columns((0.15, 1, 1, 0.15))
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        # self.passthrough_preds = passthrough_preds
        # self.passthrough_cols = passthrough_cols

        preproc = preproc_cols[1].selectbox(
            "Step Type",
            placeholder="Step Type",
            label_visibility="collapsed",
            options=self.preprocessor_options,
            key=f"step_sel_{self.id}",
        )

        with preproc_cols[2]:
            if preproc == "SimpleImputer (numeric)":
                input_cols = st.multiselect(
                    "Select the numeric columns to impute.",
                    options=self.num_cols,
                    label_visibility="collapsed",
                    placeholder="Select the numeric columns to impute.",
                    key=f"si_n_{self.id}",
                )
                processor = "SI"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    strategy="mean",
                    drop_input_cols=True,
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )

            if preproc == "SimpleImputer (categorical)":
                input_cols = st.multiselect(
                    "Select the categorical columns to impute.",
                    label_visibility="collapsed",
                    placeholder="Select the categorical columns to impute.",
                    options=self.cat_cols,
                    key=f"si_c_{self.id}",
                )
                processor = "SI"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    strategy="most_frequent",
                    drop_input_cols=True,
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )
            if preproc == "OneHotEncoder":
                input_cols = st.multiselect(
                    "Select the categorical columns to one-hot encode.",
                    options=self.cat_cols,
                    label_visibility="collapsed",
                    placeholder="Select the categorical columns to one-hot encode.",
                    key=f"ohe_{self.id}",
                )
                processor = "OHE"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    drop_input_cols=True,
                    drop="first",
                    handle_unknown="ignore",
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )
            if preproc == "StandardScaler":
                input_cols = st.multiselect(
                    "Select the numeric columns to Standard Scale.",
                    options=self.num_cols,
                    label_visibility="collapsed",
                    placeholder="Select the numeric columns to Standard Scale.",
                    key=f"ss_{self.id}",
                )
                processor = "SS"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )
            if preproc == "MinMaxScaler":
                input_cols = st.multiselect(
                    "Select the numeric columns to MinMax Scale.",
                    options=self.num_cols,
                    label_visibility="collapsed",
                    key=f"mms_{self.id}",
                )
                processor = "MMS"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )
            if preproc == "MaxAbsScaler":
                input_cols = st.multiselect(
                    "Select the numeric columns to MaxAbs Scale.",
                    options=self.num_cols,
                    label_visibility="collapsed",
                    placeholder="Select the numeric columns to MaxAbs Scale.",
                    key=f"mas_{self.id}",
                )
                processor = "MAS"
                kw = dict(
                    input_cols=input_cols,
                    output_cols=input_cols,
                    # passthrough_cols=self.passthrough_preds + self.passthrough_cols,
                )

        preproc_cols[3].button(
            "🗑️",
            on_click=Callbacks.remove_step,
            args=[self.id],
            key=f"btn_del_step{self.id}",
        )
        if input_cols:
            self.step_return = dict(
                title=preproc, preprocess_type=processor, kw=kw, is_valid=True
            )
            preproc_cols[0].image(Images.check_icon, width=40)
        else:
            self.step_return = dict(
                title=preproc, preprocess_type=processor, kw=kw, is_valid=False
            )
            preproc_cols[0].image(Images.alert_icon, width=40)
