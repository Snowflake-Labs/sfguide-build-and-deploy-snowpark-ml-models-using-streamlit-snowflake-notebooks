import unittest

from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from utils import get_model


class TestUtils(unittest.TestCase):
    def test_get_model_regression(self):
        model = get_model("regression", "XGBRegressor", ["target"])
        self.assertIsInstance(model, XGBRegressor)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, -1)
        self.assertEqual(model.scoring, "accuracy")
        self.assertEqual(model.label_cols, ["target"])

    def test_get_model_classification(self):
        model = get_model("classification", "XGBClassifier", ["target"])
        self.assertIsInstance(model, XGBClassifier)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, -1)
        self.assertEqual(model.scoring, "accuracy")
        self.assertEqual(model.label_cols, ["target"])

    def test_get_model_clustering(self):
        model = get_model("clustering", "KMeans", [])
        self.assertEqual(model, "KMeans")

    def test_get_model_invalid_estimator(self):
        with self.assertRaises(IndexError):
            get_model("regression", "InvalidEstimator", ["target"])


if __name__ == "__main__":
    unittest.main()
