# Do Financial Transfer Entropy Networks Recover Meaningful Structure?

**A Matched-DGP Audit of Node-Level Estimation Reliability**

## Overview

This repository contains the replication package for our working paper examining whether Transfer Entropy (TE) and Granger Causality (GC) networks can reliably recover node-level structure at low T/N ratios typical in financial applications.

**Key Finding**: At T/N < 5, network topology recovery is unreliable. OLS pairwise TE achieves ~11% precision; LASSO-TE reaches 72% on raw returns but only 67% with factor-neutral preprocessing. The T/N ratio dominates—factor adjustment does not materially improve recovery.

## Repository Structure

```
.
├── paper/
│   ├── main.tex              # LaTeX source
│   └── references.bib        # Bibliography
├── paper_assets/             # Figures for paper
├── src/                      # Python simulation code
│   ├── all_experiments_v2.py # Main experiment runner (Table 4)
│   ├── extended_dgp.py       # GARCH+t5 DGP
│   ├── lasso_simulation.py   # LASSO-TE estimation
│   └── run_factor_neutral_sim.py  # Factor-neutral experiments
└── results/                  # Generated tables and figures
```

## Replication Instructions

### Requirements
- Python 3.8+
- NumPy, SciPy, scikit-learn, pandas, matplotlib
- LaTeX (for paper compilation)

### Quick Start

**Generate Table 2 (Main Results)**:
```bash
cd src
python run_factor_neutral_sim.py --mode raw --method ols --trials 100
python run_factor_neutral_sim.py --mode raw --method lasso --trials 100
```

**Generate Table 4 (Oracle vs Estimated Factor-Neutral)**:
```bash
python all_experiments_v2.py  # Runs Exp #3 (Oracle Extended)
```

Results will be saved to `results/` as CSV files.

### Compile Paper
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Results Summary

| Estimator | Preprocessing | T/N=2 | T/N=5 |
|-----------|---------------|-------|-------|
| OLS-TE | Raw | 11.3% | 11.5% |
| LASSO-TE | Raw | 45.1% | 72.3% |
| LASSO-TE | Factor-neutral (PCA) | 35.5% | 66.7% |
| LASSO-TE | Factor-neutral (Oracle) | 25.3% | 74.6% |

**Precision** = True positives / (True positives + False positives)

Factor-neutral preprocessing does not materially improve precision; the T/N barrier persists regardless of preprocessing choice.

## Citation

```bibtex
@unpublished{te-network-audit-2026,
  author = {[Your Name]},
  title  = {Do Financial Transfer Entropy Networks Recover Meaningful Structure? 
            A Matched-DGP Audit of Node-Level Estimation Reliability},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

[Your contact information]

---

**Note**: This is the clean replication branch. Full experimental history is available in the private `master` branch.
