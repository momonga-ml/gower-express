"""
Scikit-learn compatibility layer for Gower distance.

This module provides lightweight integration with scikit-learn without
modifying the core gower_dist.py implementation.
"""

import numpy as np


def gower_distance(X, Y=None, feature_weights=None, cat_features=None):
    """
    Sklearn-compatible Gower distance function.

    This function provides a simple interface for using Gower distance
    with scikit-learn algorithms that accept custom distance functions.

    Parameters:
    -----------
    X : array-like, shape (n_samples_X, n_features)
        First dataset
    Y : array-like, shape (n_samples_Y, n_features), optional
        Second dataset. If None, computes pairwise distances within X
    feature_weights : array-like, shape (n_features,), optional
        Feature weights. If None, uses equal weights
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features

    Returns:
    --------
    ndarray
        Distance matrix or scalar distance
    """
    from .gower_dist import gower_matrix

    # Handle sklearn's pairwise distance API
    if Y is None:
        return gower_matrix(X, weight=feature_weights, cat_features=cat_features)
    else:
        # For single point comparisons (used by KNN)
        X_input = np.atleast_2d(X)
        Y_input = np.atleast_2d(Y)

        result = gower_matrix(
            X_input, Y_input, weight=feature_weights, cat_features=cat_features
        )

        # Return scalar for single point comparison
        if result.shape == (1, 1):
            return result[0, 0]
        return result


class GowerDistance:
    """
    Stateful Gower distance for sklearn with persistent feature configuration.

    This class creates a callable object that can be used as a metric
    with scikit-learn algorithms, while maintaining consistent configuration
    for categorical features and feature weights.

    Parameters:
    -----------
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features. If None, will attempt
        to auto-detect based on data types
    feature_weights : array-like, shape (n_features,), optional
        Feature weights. If None, uses equal weights
    n_jobs : int, default=1
        Number of parallel jobs for distance computation

    Examples:
    ---------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from gower_exp.sklearn_compat import GowerDistance
    >>>
    >>> # Create metric with configuration
    >>> gower_metric = GowerDistance(cat_features=[True, False, True])
    >>>
    >>> # Use with sklearn classifier
    >>> knn = KNeighborsClassifier(metric=gower_metric, algorithm='brute')
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, cat_features=None, feature_weights=None, n_jobs=1):
        self.cat_features = cat_features
        self.feature_weights = feature_weights
        self.n_jobs = n_jobs

    def __call__(self, X, Y):
        """
        Callable interface for sklearn.

        Parameters:
        -----------
        X : array-like, shape (n_samples_X, n_features) or (n_features,)
            First dataset or single sample
        Y : array-like, shape (n_samples_Y, n_features) or (n_features,)
            Second dataset or single sample

        Returns:
        --------
        float or ndarray
            Distance(s) between X and Y
        """
        from .gower_dist import gower_matrix

        # Handle single vectors (sklearn sometimes passes 1D arrays)
        X_input = np.atleast_2d(X)
        Y_input = np.atleast_2d(Y)

        # Compute distance matrix
        result = gower_matrix(
            X_input,
            Y_input,
            weight=self.feature_weights,
            cat_features=self.cat_features,
            n_jobs=self.n_jobs,
        )

        # Return scalar for single point comparison
        if result.shape == (1, 1):
            return result[0, 0]
        return result


def make_gower_knn_classifier(
    n_neighbors=5, cat_features=None, feature_weights=None, **kwargs
):
    """
    Convenience function to create KNN classifier with Gower distance.

    This function creates a KNeighborsClassifier configured to use Gower distance
    with the specified parameters.

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features
    feature_weights : array-like, optional
        Feature weights
    **kwargs : dict
        Additional keyword arguments passed to KNeighborsClassifier

    Returns:
    --------
    KNeighborsClassifier
        Configured classifier instance

    Examples:
    ---------
    >>> from gower_exp.sklearn_compat import make_gower_knn_classifier
    >>>
    >>> classifier = make_gower_knn_classifier(
    ...     n_neighbors=3,
    ...     cat_features=[True, False, True],
    ...     weights='distance'
    ... )
    >>> classifier.fit(X_train, y_train)
    >>> predictions = classifier.predict(X_test)
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for sklearn integration. "
            "Install it with: pip install scikit-learn"
        ) from err

    # Create Gower distance metric
    metric = GowerDistance(cat_features=cat_features, feature_weights=feature_weights)

    # Create and return classifier
    # Note: algorithm='brute' is required for custom metrics
    return KNeighborsClassifier(
        n_neighbors=n_neighbors, metric=metric, algorithm="brute", **kwargs
    )


def make_gower_knn_regressor(
    n_neighbors=5, cat_features=None, feature_weights=None, **kwargs
):
    """
    Convenience function to create KNN regressor with Gower distance.

    This function creates a KNeighborsRegressor configured to use Gower distance
    with the specified parameters.

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features
    feature_weights : array-like, optional
        Feature weights
    **kwargs : dict
        Additional keyword arguments passed to KNeighborsRegressor

    Returns:
    --------
    KNeighborsRegressor
        Configured regressor instance

    Examples:
    ---------
    >>> from gower_exp.sklearn_compat import make_gower_knn_regressor
    >>>
    >>> regressor = make_gower_knn_regressor(
    ...     n_neighbors=5,
    ...     cat_features=[True, False, False],
    ...     weights='distance'
    ... )
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_test)
    """
    try:
        from sklearn.neighbors import KNeighborsRegressor
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for sklearn integration. "
            "Install it with: pip install scikit-learn"
        ) from err

    # Create Gower distance metric
    metric = GowerDistance(cat_features=cat_features, feature_weights=feature_weights)

    # Create and return regressor
    # Note: algorithm='brute' is required for custom metrics
    return KNeighborsRegressor(
        n_neighbors=n_neighbors, metric=metric, algorithm="brute", **kwargs
    )


def precomputed_gower_matrix(
    X_train, X_test=None, feature_weights=None, cat_features=None, n_jobs=1
):
    """
    Compute precomputed distance matrices for efficient sklearn usage.

    This function computes Gower distance matrices that can be used with
    sklearn algorithms that support precomputed distances, providing better
    performance for repeated operations on the same datasets.

    Parameters:
    -----------
    X_train : array-like, shape (n_train_samples, n_features)
        Training dataset
    X_test : array-like, shape (n_test_samples, n_features), optional
        Test dataset. If None, returns only the training distance matrix
    feature_weights : array-like, optional
        Feature weights
    cat_features : array-like, bool, optional
        Boolean mask indicating categorical features
    n_jobs : int, default=1
        Number of parallel jobs

    Returns:
    --------
    dict
        Dictionary containing:
        - 'train': Training set distance matrix (n_train x n_train)
        - 'test': Test to train distance matrix (n_test x n_train), if X_test provided

    Examples:
    ---------
    >>> from gower_exp.sklearn_compat import precomputed_gower_matrix
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>>
    >>> # Precompute distance matrices
    >>> distances = precomputed_gower_matrix(X_train, X_test)
    >>>
    >>> # Train with precomputed distances
    >>> knn = KNeighborsClassifier(metric='precomputed')
    >>> knn.fit(distances['train'], y_train)
    >>>
    >>> # Predict using test distances
    >>> predictions = knn.predict(distances['test'])
    """
    from .gower_dist import gower_matrix

    result = {}

    # Compute training set distance matrix
    result["train"] = gower_matrix(
        X_train, weight=feature_weights, cat_features=cat_features, n_jobs=n_jobs
    )

    # Compute test to train distances if test set provided
    if X_test is not None:
        result["test"] = gower_matrix(
            X_test,
            X_train,
            weight=feature_weights,
            cat_features=cat_features,
            n_jobs=n_jobs,
        )

    return result
