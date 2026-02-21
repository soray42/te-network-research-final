"""
Diagnostic: Check variance after factor-neutral preprocessing
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from extended_dgp import generate_sparse_var_extended
from factor_neutral_preprocessing import preprocess_returns

# Generate test data
R, A_coef, A_true, F = generate_sparse_var_extended(
    N=100, T=100, density=0.05, seed=42, dgp='garch_factor', return_factors=True
)

print("="*60)
print("Variance Diagnostic")
print("="*60)

# Raw returns
print(f"\nRaw returns:")
print(f"  Mean variance: {R.var(axis=0).mean():.6f}")
print(f"  Min variance: {R.var(axis=0).min():.6f}")
print(f"  Max variance: {R.var(axis=0).max():.6f}")

# Estimated factor-neutral
R_fn, diag = preprocess_returns(R, mode='estimated_fn', n_factors=3)
print(f"\nEstimated factor-neutral:")
print(f"  Mean variance: {R_fn.var(axis=0).mean():.6f}")
print(f"  Min variance: {R_fn.var(axis=0).min():.6f}")
print(f"  Max variance: {R_fn.var(axis=0).max():.6f}")
print(f"  R² mean: {diag['r2_mean']:.4f}")
print(f"  Variance reduction: {(1 - R_fn.var(axis=0).mean() / R.var(axis=0).mean())*100:.1f}%")

# Check if any variance is zero or near-zero
n_zero = np.sum(R_fn.var(axis=0) < 1e-10)
print(f"\n  Stocks with near-zero variance: {n_zero}/{R.shape[1]}")

if n_zero > 0:
    print(f"  ⚠️ WARNING: Some stocks have collapsed to zero variance!")
    print(f"  This will cause LASSO to fail.")
