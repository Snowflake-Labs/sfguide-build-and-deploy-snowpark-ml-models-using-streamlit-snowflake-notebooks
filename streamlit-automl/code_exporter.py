from io import BytesIO
from typing import Literal

import black
from black import format_str
from nbconvert.exporters import NotebookExporter
from nbformat import v4
from snowflake.ml.modeling.pipeline import Pipeline

# List of estimators that don't have hyperparameters to tune, so we don't need to give GridSearchCV code for them
estimators_without_hp_tuning = ["LinearRegression", "KMeans"]

# Some potential hp grids to include as examples of how to use GridSearchCV
example_hyperparam_grids = {
    "XGBRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.001],
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.001],
    },
    "LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "max_iter": [500, 1000, 5000],
    },
}


def get_session_code(context: Literal["local", "snowflake"]) -> tuple:
    if context == "local":
        return (
            "from snowflake.snowpark import Session",
            "session = Session.builder.create()",
        )
    elif context == "snowflake":
        return (
            "from snowflake.snowpark.context import get_active_session",
            "session = get_active_session()",
        )


def create_notebook(
    fully_qual_tbl_nm: str,
    model_pipeline: Pipeline,
    project_name: str,
    registry_database: str,
    registry_schema: str,
    context: Literal["local", "snowflake"],
) -> BytesIO:
    notebook = v4.new_notebook()

    session_context_code = get_session_code(context)

    imports_dict = {}
    for step in model_pipeline.steps:
        step_class = step[1].__class__
        imports_dict[f"{step_class.__name__}"] = (
            f"from {step_class.__module__} import {step_class.__name__}"
        )

    imports_string = f"""{session_context_code[0]}
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.registry import Registry
from numpy import nan, array"""

    for item in imports_dict:
        imports_string = imports_string + "\n" + imports_dict[item]

    cells = []
    cells.append(v4.new_markdown_cell(f"# Snowpark ML - {project_name}"))
    cells.append(v4.new_markdown_cell("### Imports"))
    cells.append(v4.new_code_cell(source=imports_string))
    cells.append(v4.new_markdown_cell(source="Establish a Snowpark session."))
    cells.append(v4.new_code_cell(source=session_context_code[1]))
    cells.append(
        v4.new_markdown_cell(
            source="""Establish training DataFrame.

Consider using [`random_split`](https://docs.snowflake.com/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.DataFrame.random_split)
if your data is not already split or [`sample_by`](https://docs.snowflake.com/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.DataFrame.sample_by)
for stratified sampling."""
        )
    )
    cells.append(
        v4.new_code_cell(source=f"""train_df = session.table('{fully_qual_tbl_nm}')""")
    )
    cells.append(v4.new_markdown_cell(source="Preview the DataFrame."))
    cells.append(v4.new_code_cell(source="train_df.show()"))
    cells.append(v4.new_markdown_cell(source="Build model pipeline."))

    big_string = "pipeline = Pipeline(["
    for step in model_pipeline.steps:
        params = step[1].get_params()
        big_string += f'("{step[0]}", '
        big_string += step[1].__class__.__name__
        big_string += "("
        for k, v in params.items():
            if v:
                if isinstance(v, str):
                    big_string += f"{k}='{v}',"
                else:
                    big_string += f"{k}={v},"
        big_string += ")),"

    big_string += "])"

    formatted_string = format_str(big_string, mode=black.FileMode())

    cells.append(v4.new_code_cell(source=formatted_string))

    # Create a formatted string to display possible GridSearchCV code in a Markdown cell (if not LinearRegression or Clustering)
    estimator_step = model_pipeline.steps[-1]
    estimator_class_nm = estimator_step[1].__class__.__name__

    if (
        estimator_class_nm not in estimators_without_hp_tuning
        and estimator_class_nm in example_hyperparam_grids
    ):
        gscv_big_string = f'("GridSearchCV", GridSearchCV(estimator={estimator_class_nm}(), param_grid={example_hyperparam_grids[estimator_class_nm]},'
        for k, v in estimator_step[1].get_params().items():
            gscv_big_string += f"{k}={v},"
        gscv_big_string += "))"

        gscv_formatted_string = format_str(gscv_big_string, mode=black.FileMode())
        gscv_markdown_string = f"""To perform hyperparameter tuning in the pipeline, replace the final estimator step with the following code.
This is runnable example code containing a possible grid of hyperparameter combinations.
To learn more about GridSearchCV take a look at the [docs](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/modeling/snowflake.ml.modeling.model_selection.GridSearchCV).

```python
{gscv_formatted_string}"""

        cells.append(v4.new_markdown_cell(source=gscv_markdown_string))

    cells.append(v4.new_markdown_cell(source="Fit the pipeline on training data"))
    cells.append(v4.new_code_cell(source="""pipeline.fit(train_df)"""))

    cells.append(v4.new_markdown_cell(source="Predict on Test set."))
    cells.append(v4.new_code_cell(source="""result = pipeline.predict(train_df)"""))
    cells.append(v4.new_markdown_cell(source="Review the results."))
    cells.append(v4.new_code_cell(source="result.show()"))
    cells.append(v4.new_markdown_cell(source="Log the model to the registry."))
    reg_instance_str = f"""reg = Registry(session, database_name="{registry_database}", schema_name="{registry_schema}")"""
    cells.append(
        v4.new_code_cell(
            source=f"""{reg_instance_str}
reg.log_model(model_name="{project_name}", model=pipeline)"""
        )
    )

    cells.append(
        v4.new_code_cell(source=f"""reg.get_model("{project_name}").show_versions()""")
    )

    notebook["cells"] = cells

    exporter = NotebookExporter()
    body, resources = exporter.from_notebook_node(notebook)

    data = BytesIO()

    data.write(body.encode())

    return data
