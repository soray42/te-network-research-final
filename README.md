# Do Financial Transfer Entropy Networks Recover Meaningful Structure?
## A Matched-DGP Audit of Node-Level Estimation Reliability

**Haotian Yang**  
Bocconi University  
📧 sorayoung.42@outlook.com

**Final Replication Package**

🔗 **GitHub**: https://github.com/sora42y/te-network-research-final

This repository contains the complete replication materials for the paper.

## Structure

```
te-network-research-final/
├── src/                           # Core simulation and analysis code
│   ├── extended_dgp.py            # DGP: Gaussian / GARCH / GARCH+Factor
│   ├── lasso_simulation.py        # OLS-TE and LASSO-TE estimators
│   ├── nonparametric_te.py        # KNN-based nonparametric TE comparison
│   ├── run_main_sim_100.py        # Main simulation (Table 2)
│   ├── run_aggregate_recovery.py  # Aggregate connectedness recovery
│   ├── run_sector_hub.py          # Hub sector identification test
│   └── generate_nonparametric_figure.py  # Nonparametric TE visualization
├── paper/
│   ├── main.tex                   # LaTeX manuscript (final version)
│   └── references.bib             # Bibliography
├── paper_assets/                  # Generated figures for paper
├── results/                       # Simulation outputs (CSV + PNG)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

Python >= 3.10 required.

## Key Results

### Main Simulation (Table 2)
```bash
cd src
python run_main_sim_100.py
# Output: ../results/main_sim_100_summary.csv (~2-3 hours, 100 trials)
```

### Nonparametric TE Comparison (Section 3.7)
```bash
python nonparametric_te.py
# Output: ../results/nonparametric_te_comparison.csv (~30-60 min)
```

### Paper Figures
```bash
python generate_nonparametric_figure.py
# Output: ../paper_assets/figure_nonparametric_te.png
```

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yang2026te,
  title={Do Financial Transfer Entropy Networks Recover Meaningful Structure? A Matched-DGP Audit of Node-Level Estimation Reliability},
  author={Yang, Haotian},
  institution={Bocconi University},
  year={2026},
  note={Working Paper}
}
```

## License

MIT License

## Contact

**Haotian Yang**  
Bocconi University  
📧 sorayoung.42@outlook.com  
🔗 https://github.com/sora42y/te-network-research-final
