"""
Unified Evaluation Metrics Module

All evaluation logic called from here. No duplicates.

Functions:
- compute_precision_recall_f1: Basic metrics
- eval_metrics: Extended metrics with hub recovery
- cross_sectional_tstat: t-stat for NIO-return regression
"""

import numpy as np


def compute_precision_recall_f1(A_true, A_pred):
    """
    Compute precision, recall, and F1 score.
    
    Parameters
    ----------
    A_true : ndarray (N, N)
        True adjacency matrix
    A_pred : ndarray (N, N)
        Predicted adjacency matrix
    
    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    y_true = A_true[mask].flatten()
    y_pred = A_pred[mask].flatten()
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def eval_metrics(A_true, A_pred, top_k=5):
    """
    Extended evaluation with hub recovery and density.
    
    Parameters
    ----------
    A_true : ndarray (N, N)
        True adjacency matrix
    A_pred : ndarray (N, N)
        Predicted adjacency matrix
    top_k : int
        Number of top hubs to consider
    
    Returns
    -------
    metrics : dict
        Contains: precision, recall, f1, hub_recovery, density
    """
    N = A_true.shape[0]
    
    # Basic metrics
    precision, recall, f1 = compute_precision_recall_f1(A_true, A_pred)
    
    # Hub recovery
    out_deg_true = A_true.sum(axis=1)
    out_deg_pred = A_pred.sum(axis=1)
    
    if top_k < N:
        top_hubs_true = np.argsort(out_deg_true)[-top_k:]
        top_hubs_pred = np.argsort(out_deg_pred)[-top_k:]
        hub_recovery = len(set(top_hubs_true) & set(top_hubs_pred)) / top_k
    else:
        hub_recovery = np.nan
    
    # Density
    density = A_pred.sum() / (N * (N - 1))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hub_recovery': hub_recovery,
        'density': density
    }


def cross_sectional_tstat(returns, nio):
    """
    Compute t-statistic from cross-sectional regression: returns ~ NIO.
    
    Used in oracle NIO power analysis.
    
    Parameters
    ----------
    returns : ndarray (T, N) or (N,)
        If 2D: use mean returns across time
        If 1D: use as-is
    nio : ndarray (N,)
        Net information outflow
    
    Returns
    -------
    t_stat : float
        t-statistic for NIO coefficient
    """
    if returns.ndim == 2:
        # Average returns across time
        avg_returns = returns.mean(axis=0)
    else:
        avg_returns = returns
    
    N = len(avg_returns)
    
    # Cross-sectional regression: avg_returns ~ intercept + nio
    X = np.column_stack([np.ones(N), nio])
    
    try:
        beta = np.linalg.lstsq(X, avg_returns, rcond=None)[0]
        resid = avg_returns - X @ beta
        
        # t-stat for NIO coefficient
        dof = N - 2
        s2 = (resid ** 2).sum() / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(cov[1, 1])
        
        t_stat = beta[1] / se if se > 0 else 0.0
    except:
        t_stat = 0.0
    
    return t_stat
