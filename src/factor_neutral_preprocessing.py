"""
Factor-Neutral Preprocessing for Simulation
Implements FIX 1: Add factor-neutral preprocessing to match empirical pipeline

This module provides three preprocessing modes:
  1. raw: No preprocessing (current baseline)
  2. oracle_fn: Factor-neutral using TRUE factors from DGP
  3. estimated_fn: Factor-neutral using PCA-estimated factors (realistic)

Usage:
  from factor_neutral_preprocessing import preprocess_returns
  
  # Generate data
  R, A_true_coef, A_true = generate_sparse_var_extended(dgp='garch_factor', ...)
  
  # Preprocess
  R_processed, diagnostics = preprocess_returns(
      R, 
      mode='estimated_fn',
      true_factors=F  # optional, only needed for oracle_fn mode
  )
  
  # Then estimate TE on R_processed instead of R
"""

import numpy as np
from sklearn.decomposition import PCA


def preprocess_returns(R, mode='raw', true_factors=None, n_factors=3):
    """
    Apply factor-neutral preprocessing to simulated returns.
    
    Parameters
    ----------
    R : ndarray, shape (T, N)
        Raw return matrix
    mode : str
        'raw'          : No preprocessing (return R as-is)
        'oracle_fn'    : Regress on true factors F (requires true_factors)
        'estimated_fn' : Regress on PCA-estimated factors (realistic)
    true_factors : ndarray, shape (T, K), optional
        True factor matrix from DGP (only used for oracle_fn)
    n_factors : int
        Number of factors to estimate (for estimated_fn mode)
    
    Returns
    -------
    R_processed : ndarray, shape (T, N)
        Preprocessed returns
    diagnostics : dict
        Contains R² values, factor loadings, etc. for verification
    """
    T, N = R.shape
    diagnostics = {}
    
    if mode == 'raw':
        # No preprocessing
        return R.copy(), {'mode': 'raw', 'r2_mean': np.nan}
    
    elif mode == 'oracle_fn':
        # Factor-neutral using TRUE factors from DGP
        if true_factors is None:
            raise ValueError("oracle_fn mode requires true_factors argument")
        
        F = true_factors  # shape (T, K)
        K = F.shape[1]
        
        # Regress each stock on true factors
        # r_i,t = α_i + β_i' F_t + ε_i,t
        residuals = np.zeros((T, N))
        r2_values = np.zeros(N)
        loadings = np.zeros((N, K))
        
        for i in range(N):
            # OLS: β = (F'F)^{-1} F'r
            F_with_const = np.column_stack([np.ones(T), F])
            coef = np.linalg.lstsq(F_with_const, R[:, i], rcond=None)[0]
            fitted = F_with_const @ coef
            residuals[:, i] = R[:, i] - fitted
            
            # R²
            ss_res = np.sum(residuals[:, i]**2)
            ss_tot = np.sum((R[:, i] - R[:, i].mean())**2)
            r2_values[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            loadings[i] = coef[1:]  # exclude intercept
        
        diagnostics = {
            'mode': 'oracle_fn',
            'r2_mean': r2_values.mean(),
            'r2_median': np.median(r2_values),
            'loadings': loadings,
            'n_factors': K
        }
        
        return residuals, diagnostics
    
    elif mode == 'estimated_fn':
        # Factor-neutral using PCA-estimated factors (realistic)
        # This is what we do in empirical Section 5
        
        # Step 1: PCA on return covariance
        pca = PCA(n_components=n_factors)
        F_est = pca.fit_transform(R)  # shape (T, n_factors)
        
        # Step 2: Regress each stock on estimated factors
        residuals = np.zeros((T, N))
        r2_values = np.zeros(N)
        loadings = np.zeros((N, n_factors))
        
        for i in range(N):
            F_with_const = np.column_stack([np.ones(T), F_est])
            coef = np.linalg.lstsq(F_with_const, R[:, i], rcond=None)[0]
            fitted = F_with_const @ coef
            residuals[:, i] = R[:, i] - fitted
            
            ss_res = np.sum(residuals[:, i]**2)
            ss_tot = np.sum((R[:, i] - R[:, i].mean())**2)
            r2_values[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            loadings[i] = coef[1:]
        
        diagnostics = {
            'mode': 'estimated_fn',
            'r2_mean': r2_values.mean(),
            'r2_median': np.median(r2_values),
            'loadings': loadings,
            'n_factors': n_factors,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
        
        return residuals, diagnostics
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'raw', 'oracle_fn', or 'estimated_fn'.")


def extract_factors_from_dgp(N=50, T=500, density=0.05, seed=42):
    """
    Wrapper to generate data AND extract true factors from garch_factor DGP.
    
    Returns
    -------
    R : ndarray (T, N)
        Raw returns
    F : ndarray (T, K)
        True factor matrix
    A_true : ndarray (N, N)
        True adjacency matrix
    """
    # We need to modify extended_dgp.py to ALSO return F
    # For now, re-implement the factor generation here
    
    rng = np.random.RandomState(seed)
    
    # Generate factors (same as in extended_dgp.py)
    K = 3
    sigma_base = 0.01
    factor_vol = sigma_base * 1.5
    
    F = np.zeros((T, K))
    for t in range(1, T):
        F[t] = 0.1 * F[t-1] + rng.normal(0, factor_vol, K)
    
    # Generate returns (simplified from extended_dgp.py)
    # This is just for testing — in production, we'll modify extended_dgp.py
    B = rng.normal(0, 1, (N, K)) * 0.4
    
    # Idiosyncratic + VAR
    omega = (sigma_base * 0.7)**2 * 0.05
    alpha = 0.08
    beta = 0.90
    
    u = np.zeros((T, N))
    h = np.full(N, (sigma_base * 0.7)**2)
    
    for t in range(T):
        z = rng.standard_t(df=5, size=N)
        u[t] = np.sqrt(h) * z
        if t < T - 1:
            h = omega + alpha * u[t]**2 + beta * h
            h = np.maximum(h, 1e-10)
    
    innovations = F @ B.T + u
    
    # VAR(1) coefficient matrix (simplified)
    A = np.zeros((N, N))
    n_edges = int(N * N * density)
    off_diag = [(i, j) for i in range(N) for j in range(N) if i != j]
    edge_idx = rng.choice(len(off_diag), size=n_edges, replace=False)
    
    for idx in edge_idx:
        i, j = off_diag[idx]
        A[i, j] = rng.uniform(0.05, 0.15) * rng.choice([-1, 1])
    
    eigvals = np.abs(np.linalg.eigvals(A))
    if eigvals.max() > 0.9:
        A = A * (0.9 / eigvals.max())
    
    A_true = (A != 0).astype(int)
    
    # Generate returns
    R = np.zeros((T, N))
    R[0] = rng.normal(0, sigma_base, N)
    for t in range(1, T):
        R[t] = A @ R[t-1] + innovations[t]
    
    return R, F, A_true


# ============================================================================
# TESTING / VERIFICATION
# ============================================================================

def test_preprocessing():
    """
    Verify that preprocessing works correctly.
    """
    print("Testing Factor-Neutral Preprocessing")
    print("=" * 60)
    
    # Generate test data
    R, F_true, A_true = extract_factors_from_dgp(N=50, T=500, seed=42)
    
    print(f"Generated data: R shape {R.shape}, F_true shape {F_true.shape}")
    print(f"Raw return std: {R.std():.6f}")
    print()
    
    # Test all three modes
    for mode in ['raw', 'oracle_fn', 'estimated_fn']:
        print(f"\nMode: {mode}")
        print("-" * 60)
        
        if mode == 'oracle_fn':
            R_proc, diag = preprocess_returns(R, mode=mode, true_factors=F_true)
        else:
            R_proc, diag = preprocess_returns(R, mode=mode)
        
        print(f"Processed return std: {R_proc.std():.6f}")
        print(f"Diagnostics: {diag}")
        
        # Check orthogonality to factors (for factor-neutral modes)
        if mode in ['oracle_fn', 'estimated_fn']:
            # Residuals should have low correlation with factors
            if mode == 'oracle_fn':
                F_test = F_true
            else:
                from sklearn.decomposition import PCA
                F_test = PCA(n_components=3).fit_transform(R)
            
            # Compute correlation between residuals and factors
            corr = np.abs(np.corrcoef(R_proc.T, F_test.T)[:R_proc.shape[1], R_proc.shape[1]:])
            mean_corr = corr.mean()
            print(f"Mean |correlation| between residuals and factors: {mean_corr:.6f}")
            print(f"  (Should be close to 0 for factor-neutral modes)")


if __name__ == '__main__':
    test_preprocessing()
