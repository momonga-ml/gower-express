import numpy as np
import pandas as pd
import pytest

import gower_exp


class TestGowerTopN:
    def test_basic_gower_topn(self):
        """Test basic top-N functionality"""
        X = np.array([[1.0, "A", 10]], dtype=object)  # Query
        Y = np.array(
            [
                [1.0, "A", 15],
                [2.0, "B", 20],
                [3.0, "A", 25],
                [1.5, "A", 12],
                [4.0, "C", 30],
            ],
            dtype=object,
        )  # Dataset

        result = gower_exp.gower_topn(X, Y, n=3)
        assert "index" in result
        assert "values" in result
        assert len(result["index"]) == 3
        assert len(result["values"]) == 3

    def test_gower_topn_default_n(self):
        """Test with default n=5"""
        X = np.array([[1.0, 2.0]], dtype=object)
        Y = np.random.rand(10, 2)

        result = gower_exp.gower_topn(X, Y)
        assert len(result["index"]) == 5  # Default n=5

    def test_gower_topn_n_greater_than_dataset(self):
        """Test when n is greater than dataset size"""
        X = np.array([[1.0, 2.0]], dtype=object)
        Y = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=object)

        result = gower_exp.gower_topn(X, Y, n=10)
        assert len(result["index"]) == 2  # Only 2 rows available

    def test_gower_topn_with_weights(self):
        """Test top-N with custom weights"""
        X = np.array([[1.0, "A"]], dtype=object)
        Y = np.array([[1.0, "B"], [2.0, "A"], [3.0, "A"]], dtype=object)

        weights = np.array([0.1, 0.9])  # Heavily weight categorical
        result = gower_exp.gower_topn(X, Y, weight=weights, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_with_cat_features(self):
        """Test with explicit categorical features"""
        X = np.array([[1, 2, 3]])
        Y = np.array([[1, 2, 4], [1, 5, 3], [2, 2, 3]])

        cat_features = [True, True, False]
        result = gower_exp.gower_topn(X, Y, cat_features=cat_features, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_multiple_rows_error(self):
        """Test error when X has multiple rows"""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6]])

        with pytest.raises(TypeError, match="Only support.*1 row"):
            gower_exp.gower_topn(X, Y)

    def test_gower_topn_pandas_dataframe(self):
        """Test with pandas DataFrame input"""
        X_df = pd.DataFrame({"a": [1.0], "b": ["A"]})
        Y_df = pd.DataFrame({"a": [1.5, 2.0, 1.2], "b": ["A", "B", "A"]})

        result = gower_exp.gower_topn(X_df, Y_df, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_optimized_path(self):
        """Test optimized path for large datasets"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(6000, 2)  # Large dataset triggers optimization

        result = gower_exp.gower_topn(X, Y, n=10, use_optimized=True)
        assert len(result["index"]) == 10

    def test_gower_topn_non_optimized_path(self):
        """Test non-optimized path explicitly"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(100, 2)

        result = gower_exp.gower_topn(X, Y, n=5, use_optimized=False)
        assert len(result["index"]) == 5

    def test_gower_topn_small_dataset_no_optimization(self):
        """Test that small datasets don't trigger optimization"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(100, 2)  # Small dataset

        result = gower_exp.gower_topn(X, Y, n=5, use_optimized=True)
        assert len(result["index"]) == 5

    def test_gower_topn_with_missing_values(self):
        """Test handling of missing values"""
        X = np.array([[1.0, "A"]], dtype=object)
        Y = np.array([[np.nan, "A"], [2.0, None], [3.0, "B"]], dtype=object)

        result = gower_exp.gower_topn(X, Y, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_all_categorical(self):
        """Test with all categorical features"""
        X = np.array([["A", "X", "P"]], dtype=object)
        Y = np.array([["A", "X", "Q"], ["B", "Y", "P"], ["A", "Y", "P"]], dtype=object)

        result = gower_exp.gower_topn(X, Y, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_all_numerical(self):
        """Test with all numerical features"""
        X = np.array([[1.0, 2.0, 3.0]])
        Y = np.array([[1.1, 2.1, 3.1], [2.0, 3.0, 4.0], [0.9, 1.9, 2.9]])

        result = gower_exp.gower_topn(X, Y, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_negative_values(self):
        """Test with negative values"""
        X = np.array([[-1.0, 2.0]])
        Y = np.array([[-1.5, 2.5], [0.0, 1.0], [-2.0, 3.0]])

        result = gower_exp.gower_topn(X, Y, n=2)
        assert len(result["index"]) == 2

    def test_gower_topn_ordering(self):
        """Test that results are ordered by distance"""
        X = np.array([[0.0, 0.0]])
        Y = np.array(
            [
                [1.0, 1.0],  # Farther
                [0.1, 0.1],  # Closest
                [0.5, 0.5],  # Middle
            ]
        )

        result = gower_exp.gower_topn(X, Y, n=3)
        # Check that distances are in ascending order
        assert result["values"][0] <= result["values"][1]
        assert result["values"][1] <= result["values"][2]

    def test_gower_topn_single_result(self):
        """Test with n=1"""
        X = np.array([[1.0, "A"]], dtype=object)
        Y = np.array([[2.0, "B"], [1.1, "A"], [3.0, "C"]], dtype=object)

        result = gower_exp.gower_topn(X, Y, n=1)
        assert len(result["index"]) == 1
        assert len(result["values"]) == 1

    def test_gower_topn_optimized_heap_algorithm(self):
        """Test the heap-based algorithm specifically"""
        X = np.array([[1.0, 2.0]])
        Y = np.random.rand(5500, 2)  # Large enough for optimization

        # Force optimized path
        result = gower_exp.gower_topn(X, Y, n=20, use_optimized=True)
        assert len(result["index"]) == 20

        # Verify results are sorted
        for i in range(len(result["values"]) - 1):
            assert result["values"][i] <= result["values"][i + 1]

    def test_gower_topn_column_mismatch(self):
        """Test error with column mismatch in DataFrames"""
        X = pd.DataFrame({"a": [1]})
        Y = pd.DataFrame({"b": [2]})

        with pytest.raises(TypeError, match="must have same columns"):
            gower_exp.gower_topn(X, Y)
