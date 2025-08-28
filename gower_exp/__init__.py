from .gower_dist import gower_matrix as gower_matrix
from .gower_dist import gower_topn as gower_topn

# Core functions are always available
__all__ = ["gower_matrix", "gower_topn"]

# Optional sklearn compatibility functions (requires scikit-learn)
try:
    from .sklearn_compat import (
        GowerDistance,
        gower_distance,
        make_gower_knn_classifier,
        make_gower_knn_regressor,
        precomputed_gower_matrix
    )
    
    # Add sklearn functions to __all__ if successfully imported
    __all__.extend([
        "GowerDistance",
        "gower_distance", 
        "make_gower_knn_classifier",
        "make_gower_knn_regressor",
        "precomputed_gower_matrix"
    ])
    
except ImportError:
    # sklearn not available, sklearn compatibility functions not exported
    pass
