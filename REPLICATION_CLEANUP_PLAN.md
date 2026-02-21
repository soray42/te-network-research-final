# TE Network Research - Replication Package Structure (Cleaned)

## Core Scripts (Keep These)

### Data Generation & Preprocessing
```
src/
├── extended_dgp.py                    # DGP with 3 modes: gaussian/garch/garch_factor
├── factor_neutral_preprocessing.py   # Factor-neutral preprocessing (3 modes: raw/oracle/estimated)
└── run_factor_neutral_sim.py         # MAIN simulation script (replaces ALL old scripts)
```

### Empirical Analysis
```
src/
├── factor_neutral_te.py              # Empirical TE estimation on S&P 500 data
└── oracle_nio_power_analysis.py      # Power analysis with planted signal
```

### Figures & Tables
```
src/
├── generate_paper_assets_fixed.py    # Generate all figures for paper
└── generate_nonparametric_figure.py  # Nonparametric TE comparison
```

---

## Scripts to DELETE (Redundant)

### Old simulation scripts (replaced by run_factor_neutral_sim.py)
```
src/
├── lasso_simulation.py               # ❌ DELETE - old version
├── all_experiments_v2.py             # ❌ DELETE - superseded
├── run_main_sim_100.py               # ❌ DELETE - superseded
├── run_aggregate_recovery.py         # ❌ DELETE - can be integrated
└── run_sector_hub.py                 # ❌ DELETE - can be integrated
```

### Why delete these?
- `run_factor_neutral_sim.py` now does EVERYTHING with `--mode` and `--method` flags
- Old scripts don't support factor-neutral preprocessing
- Keeping both will confuse readers

---

## New Unified Replication Commands

### Table 2: Main Simulation (Factor-Neutral)
```bash
# PRIMARY RESULT (Paper Table 2)
python src/run_factor_neutral_sim.py --mode estimated_fn --method ols --trials 100

# LASSO version
python src/run_factor_neutral_sim.py --mode estimated_fn --method lasso --trials 100

# Appendix: Raw returns (no factor adjustment)
python src/run_factor_neutral_sim.py --mode raw --method ols --trials 100
```

### Table 5: Mechanism Decomposition
```bash
# Gaussian baseline
python src/run_factor_neutral_sim.py --mode estimated_fn --method ols --trials 100 --dgp gaussian

# GARCH (intermediate)
python src/run_factor_neutral_sim.py --mode estimated_fn --method ols --trials 100 --dgp garch

# GARCH+Factor (realistic, already run above)
```

### Table 3: Density Sensitivity (TODO:阳菜还没写这个)
```bash
# Need to add --density flag to run_factor_neutral_sim.py
python src/run_factor_neutral_sim.py --mode estimated_fn --density 0.02 --trials 50
python src/run_factor_neutral_sim.py --mode estimated_fn --density 0.10 --trials 50
python src/run_factor_neutral_sim.py --mode estimated_fn --density 0.20 --trials 50
```

### Empirical Results (Section 5)
```bash
# Table 6, 8: NIO portfolio sorts
python src/factor_neutral_te.py  # (existing, already works)

# Table 7: Power analysis
python src/oracle_nio_power_analysis.py --lambda_range 0.001,0.005,0.010,0.020
```

### Figures
```bash
# Generate all paper figures
python src/generate_paper_assets_fixed.py
python src/generate_nonparametric_figure.py
```

---

## Benefits of Cleanup

### Before (messy)
- 12 different simulation scripts
- Unclear which one produces which table
- No factor-neutral preprocessing
- Hard to verify/extend

### After (clean)
- 1 main simulation script with flags
- Clear command → table mapping
- Factor-neutral built-in
- Easy to extend (just add flags)

---

## README.md for Replication

```markdown
# Replication Instructions

## Setup
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Reproduce All Results

### Simulation (Table 2, 3, 4, 5)
```bash
# Main result (estimated factor-neutral, ~8 hours)
python src/run_factor_neutral_sim.py --mode estimated_fn --method ols --trials 100
python src/run_factor_neutral_sim.py --mode estimated_fn --method lasso --trials 100

# Baseline comparison (appendix)
python src/run_factor_neutral_sim.py --mode raw --method ols --trials 100

# Gaussian DGP (Table 5)
python src/run_factor_neutral_sim.py --mode estimated_fn --dgp gaussian --trials 100
```

### Empirical Analysis (Table 6, 7, 8)
```bash
python src/factor_neutral_te.py
python src/oracle_nio_power_analysis.py
```

### Figures
```bash
python src/generate_paper_assets_fixed.py
python src/generate_nonparametric_figure.py
```

Results saved to `results/` directory.
```

---

## Action Plan

阳菜建议 Sora：
1. **等第一个任务跑完** (tonight)
2. **看结果**，如果方向对，阳菜明天做：
   - 删除冗余脚本
   - 写新的 README.md
   - 整理 replication package
3. **打包 v22_replication_clean.zip** 给 referee

这样复现包会**干净10倍**，而且所有命令都统一了！

Sora 觉得这个方向对吗？💤
