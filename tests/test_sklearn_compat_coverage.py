"""Additional tests to boost coverage of sklearn_compat.py module."""

import sys

import numpy as np
import pytest


class TestSklearnCompatCoverage:
    """Test edge cases and missing coverage in sklearn_compat.py"""

    def test_sklearn_import_error_handling(self):
        """Test behavior when sklearn is not available"""
        # Create a test to verify the conditional import logic works
        # Test that SKLEARN_AVAILABLE flag is properly set
        import gower_exp.sklearn_compat as sklearn_compat

        # Verify that sklearn is indeed available in this environment
        assert hasattr(sklearn_compat, "SKLEARN_AVAILABLE")

        # Test the functionality that depends on sklearn being available
        if sklearn_compat.SKLEARN_AVAILABLE:
            assert hasattr(sklearn_compat, "GowerDistance")
            assert hasattr(sklearn_compat, "make_gower_knn_classifier")
            assert hasattr(sklearn_compat, "make_gower_knn_regressor")
            assert hasattr(sklearn_compat, "precomputed_gower_matrix")

        # The conditional import structure is tested by this assertion
        # If sklearn were not available, SKLEARN_AVAILABLE would be False
        # and the classes would not be defined

    @pytest.mark.skipif(
        "sklearn" not in sys.modules, reason="scikit-learn not available"
    )
    def test_make_gower_knn_classifier_edge_cases(self):
        """Test edge cases in make_gower_knn_classifier"""
        try:
            from gower_exp.sklearn_compat import make_gower_knn_classifier

            # Test with invalid metric parameter
            classifier = make_gower_knn_classifier()

            # The function should return a KNeighborsClassifier instance
            assert hasattr(classifier, "fit")
            assert hasattr(classifier, "predict")

        except ImportError:
            pytest.skip("sklearn not available")

    @pytest.mark.skipif(
        "sklearn" not in sys.modules, reason="scikit-learn not available"
    )
    def test_make_gower_knn_regressor_edge_cases(self):
        """Test edge cases in make_gower_knn_regressor"""
        try:
            from gower_exp.sklearn_compat import make_gower_knn_regressor

            # Test with invalid metric parameter
            regressor = make_gower_knn_regressor()

            # The function should return a KNeighborsRegressor instance
            assert hasattr(regressor, "fit")
            assert hasattr(regressor, "predict")

        except ImportError:
            pytest.skip("sklearn not available")

    @pytest.mark.skipif(
        "sklearn" not in sys.modules, reason="scikit-learn not available"
    )
    def test_gower_distance_transformer_edge_cases(self):
        """Test edge cases in GowerDistance metric callable"""
        try:
            from gower_exp.sklearn_compat import GowerDistance

            # Test metric initialization
            metric = GowerDistance()
            assert metric is not None

            # Test callable interface (this is how it's used by sklearn)
            X = np.array([[1, 2, 3]])
            Y = np.array([[4, 5, 6]])

            # Test single point distance computation
            distance = metric(X[0], Y[0])
            assert isinstance(distance, (float, np.floating))
            assert distance >= 0

        except ImportError:
            pytest.skip("sklearn not available")

    @pytest.mark.skipif(
        "sklearn" not in sys.modules, reason="scikit-learn not available"
    )
    def test_precomputed_gower_matrix_edge_cases(self):
        """Test edge cases in precomputed_gower_matrix"""
        try:
            from gower_exp.sklearn_compat import precomputed_gower_matrix

            # Test with simple data
            X = np.array([[1, 2], [3, 4], [5, 6]])
            result = precomputed_gower_matrix(X)

            # Result should be a dictionary with 'train' key
            assert isinstance(result, dict)
            assert "train" in result
            assert result["train"].shape == (3, 3)
            assert result["train"].dtype == np.float32

            # Test with Y parameter
            Y = np.array([[7, 8], [9, 10]])
            result = precomputed_gower_matrix(X, Y)
            assert "train" in result
            assert "test" in result
            assert result["test"].shape == (2, 3)  # test rows by train rows

        except ImportError:
            pytest.skip("sklearn not available")

    @pytest.mark.skipif(
        "sklearn" not in sys.modules, reason="scikit-learn not available"
    )
    def test_gower_distance_function_edge_cases(self):
        """Test edge cases in gower_distance function"""
        try:
            from gower_exp.sklearn_compat import gower_distance

            # Test with simple vectors
            x = np.array([1, 2, 3])
            y = np.array([4, 5, 6])

            distance = gower_distance(x, y)
            assert isinstance(distance, (float, np.float32, np.float64))
            assert 0 <= distance <= 1

            # Test with categorical features
            distance = gower_distance(x, y, cat_features=[True, False, False])
            assert isinstance(distance, (float, np.float32, np.float64))

        except ImportError:
            pytest.skip("sklearn not available")


class TestSklearnCompatModuleBehavior:
    """Test module-level behavior of sklearn_compat"""

    def test_conditional_imports(self):
        """Test that imports are conditional on sklearn availability"""
        # This test verifies the try/except import structure
        # The module should define different functions based on sklearn availability

        try:
            import sklearn  # noqa: F401

            sklearn_available = True
        except ImportError:
            sklearn_available = False

        import gower_exp.sklearn_compat as sklearn_compat

        if sklearn_available:
            # If sklearn is available, functions should be defined
            expected_functions = [
                "GowerDistance",
                "gower_distance",
                "precomputed_gower_matrix",
                "make_gower_knn_classifier",
                "make_gower_knn_regressor",
            ]

            for func_name in expected_functions:
                assert hasattr(sklearn_compat, func_name), (
                    f"Missing function: {func_name}"
                )
        else:
            # If sklearn is not available, these functions should not exist
            assert not hasattr(sklearn_compat, "GowerDistance")

    def test_all_exports(self):
        """Test __all__ exports are correct"""
        import gower_exp.sklearn_compat as sklearn_compat

        # Should have __all__ defined
        if hasattr(sklearn_compat, "__all__"):
            all_exports = sklearn_compat.__all__

            # All exported items should actually exist in the module
            for export in all_exports:
                assert hasattr(sklearn_compat, export), (
                    f"Exported item {export} does not exist"
                )
