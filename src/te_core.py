"""
Transfer Entropy Core Module

This module provides the main interface to TE algorithms.
All heavy lifting is delegated to algorithms.py (pure implementations).

For backward compatibility, we expose the same API as before.
"""

# Import pure algorithm implementations
try:
    # Try relative import (when imported as package)
    from .algorithms import (
        compute_linear_te_matrix,
        compute_nio,
        compute_precision_recall_f1
    )
except ImportError:
    # Fall back to direct import (when run as script)
    from algorithms import (
        compute_linear_te_matrix,
        compute_nio,
        compute_precision_recall_f1
    )

__all__ = [
    'compute_linear_te_matrix',
    'compute_nio',
    'compute_precision_recall_f1'
]
