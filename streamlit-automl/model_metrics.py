import math

from pandas import DataFrame as PandasDataFrame
from snowflake.ml.modeling.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from snowflake.snowpark import DataFrame as SnowparkDataFrame
from snowflake.snowpark import functions as F


class RegressionMetrics:
    @staticmethod
    def get_mae(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        mae = mean_absolute_error(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )

        return round(mae, 8)

    @staticmethod
    def get_mape(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        mape = mean_absolute_percentage_error(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )

        return round(mape, 8)

    @staticmethod
    def get_rmse(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        mse = mean_squared_error(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )

        rmse = math.sqrt(mse)

        return round(rmse, 8)


class ClassificationMetrics:
    @staticmethod
    def get_accuracy(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        accuracy = accuracy_score(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )
        return round(accuracy, 8)

    @staticmethod
    def get_precision(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        precision = precision_score(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )
        return round(precision, 8)

    @staticmethod
    def get_recall(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        recall = recall_score(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )
        return round(recall, 8)

    @staticmethod
    def get_f1(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> float:
        f1 = f1_score(
            df=prediction_df, y_true_col_names=y_true_name, y_pred_col_names=y_pred_name
        )
        return round(f1, 8)

    @staticmethod
    def get_auc(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_proba_name: str
    ) -> float:
        roc_auc = roc_auc_score(
            df=prediction_df,
            y_true_col_names=y_true_name,
            y_score_col_names=y_pred_proba_name,
        )
        return round(roc_auc, 8)

    @staticmethod
    def get_confusion_matrix(
        prediction_df: SnowparkDataFrame, y_true_name: str, y_pred_name: str
    ) -> list:
        conf_mx = confusion_matrix(
            df=prediction_df,
            y_true_col_name=y_true_name,
            y_pred_col_name=y_pred_name,
            labels=[0, 1],
        )
        return conf_mx.tolist()


class ClusteringMetrics:
    @staticmethod
    def get_cluster_summary(
        numeric_columns: list,
        cluster_results_df: SnowparkDataFrame,
        cluster_col_name: str,
    ) -> PandasDataFrame:
        agg_list = [F.count(cluster_col_name).alias("RECORD_COUNT")] + [
            F.mean(F.col(colnm)).alias(colnm) for colnm in numeric_columns
        ]

        cluster_summary = (
            cluster_results_df.groupBy(cluster_col_name)
            .agg(agg_list)
            .sort(cluster_col_name)
            .to_pandas()
        )
        return cluster_summary


class ModelMetrics:
    def __init__(self, targel_col) -> None:
        self.target_col = targel_col
        self.metrics_map = {
            "Regression": {
                "MSE": {
                    "fn": RegressionMetrics.get_rmse,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                "MAE": {
                    "fn": RegressionMetrics.get_mae,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                "MAPE": {
                    "fn": RegressionMetrics.get_mape,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
            },
            "Classification": {
                "accuracy_score": {
                    "fn": ClassificationMetrics.get_accuracy,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                "f1-score": {
                    "fn": ClassificationMetrics.get_f1,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                "Recall": {
                    "fn": ClassificationMetrics.get_recall,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                "Precision": {
                    "fn": ClassificationMetrics.get_precision,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
                # TODO: In order to get AUC, w need to use .predict_proba to get a column of probabilities (instead of using .predict)
                # "AUC":{"fn": ClassificationMetrics.get_auc,
                #              "kw":{"y_true_name": target_col,
                #                    "y_pred_name": f"OUTPUT_{target_col}",}},
                "Confusion Matrix": {
                    "fn": ClassificationMetrics.get_confusion_matrix,
                    "kw": {
                        "y_true_name": self.target_col,
                        "y_pred_name": f"OUTPUT_{ self.target_col}",
                    },
                },
            },
            "Clustering": {"cluster_summary": ClusteringMetrics.get_cluster_summary},
        }
