import logging

from .gower_dist import gower_matrix as gower_matrix
from .gower_dist import gower_topn as gower_topn

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Core functions are always available
__all__ = ["gower_matrix", "gower_topn"]

# Optional sklearn compatibility functions (requires scikit-learn)
try:
    from .sklearn_compat import (
        GowerDistance,  # noqa: F401
        gower_distance,  # noqa: F401
        make_gower_knn_classifier,  # noqa: F401
        make_gower_knn_regressor,  # noqa: F401
        precomputed_gower_matrix,  # noqa: F401
    )

    # Add sklearn functions to __all__ if successfully imported
    __all__.extend(
        [
            "GowerDistance",
            "gower_distance",
            "make_gower_knn_classifier",
            "make_gower_knn_regressor",
            "precomputed_gower_matrix",
        ]
    )

except ImportError:
    # sklearn not available, sklearn compatibility functions not exported
    pass
