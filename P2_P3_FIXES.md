# P2-P3 Remaining Fixes Summary

## P2-9: cwd issue - DOCUMENTED (no code change needed)
The `run_all_experiments.py` uses `cwd=SRC_DIR` which can cause relative path issues.
However, all current scripts use `Path(__file__).parent` for relative paths, so they're resilient.
**Action**: Document this as a known constraint in README.

## P2-10: compare_runs dead code - WILL FIX
The column check for 'Avg Precision' never triggers because actual columns are 'Precision/Recall/F1'.
**Action**: Fix the column name check or remove the dead code.

## P2-11: Density definition - DOCUMENT
DGP uses `n_edges = int(N*N*density)` instead of `N*(N-1)*density`.
**Action**: Add explicit documentation in extended_dgp.py docstring.

## P3-12: Nonparametric stability indicators - SKIP (low priority)
Would require modifying nonparametric_te.py to track NaN rates, epsilon corrections.
**Action**: Defer to future enhancement (not blocking publication).

## P3-13: CLI parameter centralization - SKIP (low priority)
Would require creating centralized SimulationConfig class.
**Action**: Defer to future refactor (current params are traceable in run_metadata).

## Summary
- P2-10: QUICK FIX (5 min)
- P2-9, P2-11: DOCUMENTATION (already handled in docs)
- P3-12, P3-13: DEFERRED (non-blocking, low ROI for immediate publication)
