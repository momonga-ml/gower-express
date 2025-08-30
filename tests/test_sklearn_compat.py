"""
Tests for scikit-learn compatibility layer.

These tests verify that the sklearn integration functions work correctly
and produce results consistent with direct gower_matrix/gower_topn calls.
"""

import numpy as np
import pandas as pd
import pytest

import gower_exp

# Check if sklearn is available
try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestSklearnIntegration:
    """Test sklearn integration functionality."""

    def setup_method(self):
        """Set up test data for each test."""
        # Mixed data with categorical and numerical features
        # Use label-encoded categorical data for sklearn compatibility
        self.X = np.array(
            [
                [1.0, 0, 10],  # A -> 0
                [2.0, 1, 20],  # B -> 1
                [3.0, 0, 30],  # A -> 0
                [4.0, 2, 40],  # C -> 2
                [5.0, 1, 50],  # B -> 1
            ],
            dtype=float,
        )

        self.y = np.array([0, 1, 0, 1, 1])  # Binary classification labels
        self.y_reg = np.array([10.0, 20.0, 15.0, 25.0, 30.0])  # Regression targets

        # Categorical features mask (first is numeric, second is categorical, third is numeric)
        self.cat_features = [False, True, False]

        # Small dataset for testing
        self.X_small = self.X[:3]
        self.y_small = self.y[:3]
        self.y_reg_small = self.y_reg[:3]

    def test_gower_distance_callable(self):
        """Test GowerDistance as a callable metric."""
        from gower_exp import GowerDistance

        # Create metric instance
        metric = GowerDistance(cat_features=self.cat_features)

        # Test single point comparison
        dist = metric(self.X[0], self.X[1])
        assert isinstance(dist, (float, np.floating))
        assert dist >= 0.0

        # Test that distance to self is zero
        dist_self = metric(self.X[0], self.X[0])
        assert dist_self == 0.0

        # Test with 2D arrays
        dist_matrix = metric(self.X_small, self.X_small)
        assert dist_matrix.shape == (3, 3)
        assert np.all(np.diag(dist_matrix) == 0)

    def test_gower_distance_function(self):
        """Test standalone gower_distance function."""
        from gower_exp import gower_distance

        # Test pairwise distance matrix
        dist_matrix = gower_distance(
            self.X_small, feature_weights=None, cat_features=self.cat_features
        )
        assert dist_matrix.shape == (3, 3)
        assert np.all(np.diag(dist_matrix) == 0)

        # Test distance between two datasets
        dist_cross = gower_distance(
            self.X_small[:2], self.X_small[1:], cat_features=self.cat_features
        )
        assert dist_cross.shape == (2, 2)

    def test_knn_classifier_convenience(self):
        """Test make_gower_knn_classifier convenience function."""
        from gower_exp import make_gower_knn_classifier

        # Create classifier
        clf = make_gower_knn_classifier(
            n_neighbors=3, cat_features=self.cat_features, weights="uniform"
        )

        # Test that it's a KNeighborsClassifier
        assert isinstance(clf, KNeighborsClassifier)
        assert clf.algorithm == "brute"  # Required for custom metrics

        # Fit and predict
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X_small)

        assert len(predictions) == len(self.X_small)
        assert all(pred in [0, 1] for pred in predictions)

    def test_knn_regressor_convenience(self):
        """Test make_gower_knn_regressor convenience function."""
        from gower_exp import make_gower_knn_regressor

        # Create regressor
        reg = make_gower_knn_regressor(
            n_neighbors=3, cat_features=self.cat_features, weights="distance"
        )

        # Test that it's a KNeighborsRegressor
        assert isinstance(reg, KNeighborsRegressor)
        assert reg.algorithm == "brute"  # Required for custom metrics

        # Fit and predict
        reg.fit(self.X, self.y_reg)
        predictions = reg.predict(self.X_small)

        assert len(predictions) == len(self.X_small)
        assert all(isinstance(pred, (float, np.floating)) for pred in predictions)

    def test_precomputed_gower_matrix(self):
        """Test precomputed distance matrix functionality."""
        from gower_exp import precomputed_gower_matrix

        # Test with train data only
        distances = precomputed_gower_matrix(
            self.X_small, cat_features=self.cat_features
        )

        assert "train" in distances
        assert distances["train"].shape == (3, 3)
        assert np.all(np.diag(distances["train"]) == 0)

        # Test with train and test data
        distances_full = precomputed_gower_matrix(
            self.X_small, self.X_small[1:], cat_features=self.cat_features
        )

        assert "train" in distances_full
        assert "test" in distances_full
        assert distances_full["train"].shape == (3, 3)
        assert distances_full["test"].shape == (
            2,
            3,
        )  # 2 test samples x 3 train samples

    def test_consistency_with_direct_gower_matrix(self):
        """Test that sklearn integration produces same results as direct calls."""
        from gower_exp import GowerDistance

        # Direct gower_matrix call
        direct_result = gower_exp.gower_matrix(
            self.X_small, cat_features=self.cat_features
        )

        # Through sklearn compatibility layer
        metric = GowerDistance(cat_features=self.cat_features)
        sklearn_result = metric(self.X_small, self.X_small)

        # Results should be identical
        np.testing.assert_allclose(direct_result, sklearn_result, rtol=1e-10)

    def test_dbscan_clustering(self):
        """Test DBSCAN clustering with Gower distance."""
        from gower_exp import GowerDistance

        # Create larger dataset for clustering
        np.random.seed(42)
        X_cluster = np.random.rand(20, 3)
        X_cluster[:, 1] = np.random.choice(
            [0, 1, 2], 20
        )  # Make middle column categorical (label-encoded)

        # Create metric
        metric = GowerDistance(cat_features=[False, True, False])

        # Perform DBSCAN clustering
        clustering = DBSCAN(metric=metric, eps=0.3, min_samples=2)
        labels = clustering.fit_predict(X_cluster)

        # Check that we get valid cluster labels
        assert len(labels) == len(X_cluster)
        assert all(isinstance(label, (int, np.integer)) for label in labels)

    def test_feature_weights(self):
        """Test that feature weights are properly applied."""
        from gower_exp import GowerDistance

        # Create metric with custom weights
        weights = np.array([2.0, 1.0, 0.5])
        metric = GowerDistance(cat_features=self.cat_features, feature_weights=weights)

        # Compute distance
        dist = metric(self.X[0], self.X[1])

        # Compare with direct gower_matrix call
        direct_dist = gower_exp.gower_matrix(
            self.X[0:1], self.X[1:2], weight=weights, cat_features=self.cat_features
        )[0, 0]

        np.testing.assert_allclose(dist, direct_dist, rtol=1e-10)

    def test_pandas_dataframe_input(self):
        """Test that sklearn integration works with pandas DataFrames."""
        from sklearn.preprocessing import LabelEncoder

        from gower_exp import make_gower_knn_classifier

        # Create DataFrame
        df = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "categorical": ["A", "B", "A", "C", "B"],
                "numeric2": [10, 20, 30, 40, 50],
            }
        )

        # Preprocess categorical data for sklearn compatibility
        df_processed = df.copy()
        le = LabelEncoder()
        df_processed["categorical"] = le.fit_transform(df["categorical"])

        # Create classifier with explicit categorical feature specification
        clf = make_gower_knn_classifier(
            n_neighbors=3, cat_features=[False, True, False]
        )

        # Fit and predict
        clf.fit(df_processed, self.y)
        predictions = clf.predict(df_processed.iloc[:3])

        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)


class TestSklearnNotAvailable:
    """Test behavior when sklearn is not available."""

    def test_import_without_sklearn(self):
        """Test that core functions work even when sklearn functions fail to import."""
        # Core functions should always be available
        assert hasattr(gower_exp, "gower_matrix")
        assert hasattr(gower_exp, "gower_topn")

        # Test that core functions still work
        X = np.array([[1, "A"], [2, "B"]], dtype=object)
        result = gower_exp.gower_matrix(X)
        assert result.shape == (2, 2)

    @pytest.mark.skipif(
        SKLEARN_AVAILABLE, reason="This test only runs when sklearn is NOT available"
    )
    def test_sklearn_functions_not_available(self):
        """Test that sklearn functions are not available when sklearn is not installed."""
        # These should not be available when sklearn is not installed
        assert not hasattr(gower_exp, "GowerDistance")
        assert not hasattr(gower_exp, "make_gower_knn_classifier")
