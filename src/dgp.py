"""
Unified Data Generating Process Module

All DGP functions called from here.

Functions:
- generate_sparse_var: Base DGP (Gaussian/GARCH/GARCH+Factor)
- generate_var_with_nio_premium: DGP with planted NIO signal
- All imports redirect to extended_dgp modules
"""

# Import from extended_dgp modules
try:
    from .extended_dgp import generate_sparse_var_extended
    from .extended_dgp_planted_signal import generate_sparse_var_with_nio_premium
except ImportError:
    from extended_dgp import generate_sparse_var_extended
    from extended_dgp_planted_signal import generate_sparse_var_with_nio_premium

# Expose with cleaner names
generate_sparse_var = generate_sparse_var_extended

__all__ = [
    'generate_sparse_var',
    'generate_sparse_var_extended',
    'generate_sparse_var_with_nio_premium'
]
