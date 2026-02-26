# DATA MANIFEST

## Overview
This document catalogs all empirical data files used in the replication package.

**CRITICAL**: These files MUST match the checksums below for exact replication.

---

## Data Files

### 1. te_features_weekly.csv

**Location**: `data/empirical/te_features_weekly.csv`  
**Size**: 33,157,876 bytes (31.6 MB)  
**SHA256**: `87544851c75673c0cc99823953ce90d917210a5312d7342dab83f8795d380056`

**Description**: Weekly Transfer Entropy features for S&P 500 stocks (2005-2025)

**Columns**:
- `ticker` (str): Stock ticker symbol
- `formation_date` (datetime): Week-ending date for feature formation
- `nio` (float): Network In-Out measure (standardized)
- `next_week_ret` (float): Forward 1-week return (used for portfolio sorts)
- Additional TE-derived features...

**Source**: Computed from CRSP daily returns via `weekly_te_pipeline_500.py`  
**Coverage**: 1,096 weeks from 2005-01-07 to 2025-12-26  
**Missing Data**: Stocks with <52 weeks history excluded per week

**Generation Script**: `weekly_te_pipeline_500.py` (see paper appendix or contact authors)

---

### 2. universe_500.csv

**Location**: `data/empirical/universe_500.csv`  
**Size**: 4,837,126 bytes (4.6 MB)  
**SHA256**: `8cee923a3099f501b488b616d0baf4cce4db6c38bb5143fbfb695fba121f3835`

**Description**: S&P 500 stock universe metadata

**Columns**:
- `ticker` (str): Stock ticker symbol
- `first_date` (datetime): First observed date in CRSP
- `last_date` (datetime): Last observed date in CRSP
- `n_obs` (int): Total number of daily observations
- Additional metadata...

**Source**: Built from CRSP constituents via `build_universe_500.py`

---

## Data Provenance

### Original Source
- **Database**: CRSP (Center for Research in Security Prices)
- **Access**: WRDS (Wharton Research Data Services)
- **License**: Institutional subscription required
- **Query Date**: 2025-12-31

### Processing Pipeline
1. `build_universe_500.py` → filters S&P 500 stocks from CRSP
2. `weekly_te_pipeline_500.py` → computes weekly TE features
3. Output: `te_features_weekly.csv` (this file)

**IMPORTANT**: We provide the **processed** data files to enable replication without WRDS access. Original CRSP data cannot be redistributed per license terms.

---

## Verification

To verify data integrity:

```bash
# Compute SHA256 (PowerShell)
Get-FileHash data/empirical/te_features_weekly.csv -Algorithm SHA256
Get-FileHash data/empirical/universe_500.csv -Algorithm SHA256

# Or use Python
python -c "import hashlib; print(hashlib.sha256(open('data/empirical/te_features_weekly.csv','rb').read()).hexdigest())"
```

Expected output:
```
te_features_weekly.csv: 87544851c75673c0cc99823953ce90d917210a5312d7342dab83f8795d380056
universe_500.csv:       8cee923a3099f501b488b616d0baf4cce4db6c38bb5143fbfb695fba121f3835
```

---

## Runtime Verification

**NEW**: All experiment runs now automatically compute and record input data SHA256 in `run_metadata.json`:

```json
{
  "data_sources": {
    "empirical": {
      "file": "data/empirical/te_features_weekly.csv",
      "sha256": "87544851c75673c0cc99823953ce90d917210a5312d7342dab83f8795d380056",
      "verified": true
    }
  }
}
```

If SHA256 mismatch detected, experiment will **FAIL with warning**.

---

## License & Citation

**Data License**: This processed data is derived from CRSP, which requires proper attribution:

> "Data provided by the Center for Research in Security Prices (CRSP), University of Chicago Booth School of Business."

**Paper Citation**: If you use this data, please cite our paper:
> [Your Paper Citation]

---

## Contact

Questions about data processing or discrepancies:
- Open an issue: https://github.com/sora42y/te-network-research-final/issues
- Email: [Your Email]

---

**Last Updated**: 2026-02-26  
**Manifest Version**: 2.0
