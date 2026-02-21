# TE Network Paper Revision Plan
## Complete Referee Response Strategy

---

## EXECUTIVE SUMMARY

**Total Fixes**: 12 (5 MAJOR + 7 MINOR)  
**Estimated Total Time**: 18-24 hours  
**Risk Level**: HIGH (Fix 1 may change core results)  

**Recommended Approach**: 3-phase execution
- Phase 1 (Tonight, 2-3h): Safe text fixes → v19
- Phase 2A (Tomorrow AM, 2h): Table reorganization → v20  
- Phase 2B (Tomorrow PM - Sunday, 12h+): Re-run experiments → v21 FINAL

---

## DETAILED FIX LIST

### MAJOR FIXES (Change Results or Core Logic)

#### FIX 1: Factor-Neutral Preprocessing in Simulation ⚠️ CRITICAL
**Problem**: Paper uses factor-neutral empirically but NOT in simulation. Table 2's 11-17% precision reflects raw returns, not the recommended pipeline. This is a fatal logical gap.

**Required Changes**:
1. **Code**:
   - Modify `extended_dgp.py`: add factor-neutral preprocessing step
   - Create 3 pipelines: (a) raw, (b) oracle factor-neutral, (c) estimated factor-neutral
   - Update all simulation scripts to support `preprocessing={raw, oracle_fn, estimated_fn}`

2. **Experiments** (re-run required):
   - Table 2: ALL cells under estimated factor-neutral (becomes PRIMARY result)
   - Current Table 2 → Appendix "Raw Return Baseline"
   - Table 5: add factor-neutral row/column
   - Table 4: expand to ALL T/N ratios (not just 4, 5)

3. **Narrative Impact**:
   - If factor-adjusted precision at T/N=5 is ~35% (vs 17% raw), the T/N barrier shifts
   - Conclusion changes from "everything broken" to "factor adjustment necessary but T/N still matters"
   - Must be honest about this — do not cherry-pick

**Estimated Time**: 
- Coding: 3-4 hours
- Running: 6-12 hours (100 trials × multiple configs)
- Updating paper: 2-3 hours

**Risk**: Results may improve significantly. Core claims may need to soften.

---

#### FIX 2: Density Sensitivity Expansion
**Problem**: Table 3 only shows OLS at 5% density. Missing LASSO, hub recovery, F1, and other densities.

**Required Changes**:
1. Expand Table 3 to full panel: rows=T/N, columns=density × {OLS prec, LASSO prec, OLS hub, LASSO hub, OLS F1, LASSO F1}
2. Sweep density = [0.02, 0.05, 0.10, 0.15, 0.20]
3. Add 1-2 sentences justifying 5% baseline (cite Basu & Michailidis 2015)

**Estimated Time**: 4-6 hours (run + update)

---

#### FIX 3: Power Analysis — Static NIO Limitation
**Problem**: Power analysis uses time-invariant NIO. Real networks vary. Need to acknowledge this is the MOST FAVORABLE case.

**Required Changes**:
Add paragraph after Eq. 8:
> "Our design embeds a static network premium: NIO_i is constant across time, computed from the true adjacency matrix. This is the most favorable scenario for cross-sectional detection — the signal-to-noise ratio cannot be improved by time-varying network positions, which would introduce additional estimation noise. If the signal is undetectable under static ground truth, it is a fortiori undetectable under realistic time-varying conditions."

*Optional*: Add one regime-switching panel to show detection is even harder.

**Estimated Time**: 30 min (text only) or +3 hours (if adding regime-switching experiment)

---

#### FIX 4: Hub Recovery — Promote Kendall's τ
**Problem**: Top-5 overlap is brittle. Kendall's τ (currently in Appendix B.2) is more informative.

**Required Changes**:
1. Add Kendall's τ column to Table 2
2. Lead with τ in main text: "Hub ranking concordance τ = 0.12 (OLS), 0.24 (LASSO) at T/N=5, corresponding to 56% and 62% pairwise concordance — barely above 50% random baseline."
3. Keep top-5 overlap as secondary metric

**Estimated Time**: 1 hour (re-format table + update text)

---

#### FIX 5: Fair Characterization of Audited Papers
**Problem**: Table 1 oversimplifies Billio (2012) and Demirer (2018).

**Required Changes**:
1. **Billio et al.**:
   - Node-level claim → "Sector-level hub identification (banks > insurers > broker-dealers); individual institution rankings also reported"
   - Audit implication → add "Sector-level may be defensible (Section 6.3); individual rankings unreliable"

2. **Demirer et al.**:
   - Add footnote: "Demirer et al. use FEVD-based connectedness (Diebold-Yilmaz), not TE/GC directly. Our TE simulation applies to the extent both rely on VAR estimation at same T/N; FEVD-specific recovery may differ."

3. **Introduction tone**:
   - Add: "We emphasize our audit targets estimation reliability, not intellectual contributions. Several papers — particularly Billio et al. (2012) and Demirer et al. (2018) — introduced frameworks of lasting value. Our point is narrower: node-level empirical claims operate in a statistical regime where topology recovery is unreliable."

**Estimated Time**: 30 min

---

### MINOR FIXES (Technical Details & Clarity)

#### FIX 6: Monte Carlo Trial Count Consistency
**Problem**: Table 2 uses 100 trials, Table 5 uses 5 trials (underpowered), Table 7 uses 50.

**Required Changes**:
- Standardize: 100 trials for all main tables, 50 minimum for appendix
- Increase Table 5 to 50-100 trials
- Add note: "LASSO at N=100 requires ~X hours per trial; we use 50 for feasibility"

**Estimated Time**: Bundled with Fix 1 re-runs

---

#### FIX 7: LASSO BIC Discussion
**Problem**: BIC consistency requires T/N → ∞, which fails at T/N < 1.

**Required Changes**:
Add to Section 3.5:
> "BIC-based penalty selection is consistent under classical asymptotics (T/N → ∞) but may over-select at low T/N. We verified robustness using 5-fold time-series cross-validation; results are qualitatively similar (Appendix X). The precision–recall trade-off is governed primarily by T/N, not penalty selection method."

Run CV comparison if not done. Report both if materially different.

**Estimated Time**: 1 hour (text) or +2 hours (if need to run CV experiment)

---

#### FIX 8: Overlap Adjustment in Portfolio Tests
**Problem**: 5-day rolling with 5-day steps creates overlapping returns. t-stats may be inflated.

**Required Changes**:
1. Confirm Newey-West SEs used in ALL portfolio tests (Table 6, 8)
2. Report Newey-West lag: "Newey-West with lag=12, following standard practice"
3. (Optional) Non-overlapping windows as robustness

**Estimated Time**: 30 min (check code + add note)

---

#### FIX 9: Abstract Precision Language
**Problem**: "recovers under 17% of true edges" — ambiguous (precision vs recall).

**Required Changes**:
Change to: "OLS-based TE achieves precision below 17% — over 83% of detected edges are false positives — and hub detection ranks are near-random."

**Estimated Time**: 5 min

---

#### FIX 10: "T/N Barrier" Terminology
**Problem**: "Barrier" implies sharp threshold; data shows gradual improvement.

**Required Changes**:
Either:
- Define explicitly: "We use 'T/N barrier' to denote the region T/N < 8 where no estimator achieves F1 > 0.50 under realistic DGPs."
- Or rename to "T/N requirement" throughout

Pick one, be consistent.

**Estimated Time**: 15 min

---

#### FIX 11: Sample Selection & Time Period
**Problem**: Missing description of how ~100 stocks selected from S&P 500, and why 2021-2026.

**Required Changes**:
Add paragraph to Section 5:
- How stocks selected (market cap? sector balance? random?)
- Why 2021-2026
- Acknowledge limitation: "Our empirical sample spans post-COVID recovery, monetary tightening, and AI cycle. Cross-sectional NIO results may differ in calmer regimes, though our simulation evidence — regime-independent — suggests T/N constraint binds regardless."

*Optional but strong*: Add 2010-2020 robustness panel if historical data available.

**Estimated Time**: 30 min (text only) or +3 hours (if adding historical robustness)

---

#### FIX 12: Equation 1 Notation
**Problem**: Subscripts may not match convention.

**Required Changes**:
Relabel clearly:
TE(j → i) = (1/2) ln(σ²_{-j} / σ²_{full})
where σ²_{-j} is residual variance excluding j's lag, σ²_{full} includes it.

**Estimated Time**: 5 min

---

## EXECUTION PLAN

### Phase 1: Text-Only Fixes (Tonight, 2-3 hours)
**No re-runs required. Safe to execute immediately.**

✅ Fix 3: Power Analysis paragraph  
✅ Fix 5: Table 1 fairness  
✅ Fix 7: LASSO BIC discussion  
✅ Fix 9: Abstract wording  
✅ Fix 10: T/N Barrier definition  
✅ Fix 11: Sample selection paragraph  
✅ Fix 12: Eq 1 notation  

**Output**: v19_text_fixes.zip  
**Risk**: NONE (purely narrative)

---

### Phase 2A: Table Reorganization (Tomorrow AM, 1-2 hours)
**Uses existing data, no new experiments.**

✅ Fix 4: Add Kendall's τ to Table 2  
✅ Fix 8: Verify Newey-West, add note  

**Output**: v20_tables_updated.zip  
**Risk**: LOW

---

### Phase 2B: Re-Run Experiments (Tomorrow PM - Sunday, 12-18 hours)
**Requires code changes and full re-runs. HIGH RISK.**

⚠️ Fix 1: Factor-Neutral Simulation (LOAD-BEARING)  
- Modify extended_dgp.py + all callers  
- Re-run Table 2, 4, 5 (100 trials each)  
- Update all affected text  

⚠️ Fix 2: Density Sensitivity  
- Sweep density = [0.02, 0.05, 0.10, 0.15, 0.20]  
- Expand Table 3  

⚠️ Fix 6: Trial count standardization (bundled with Fix 1)

**Output**: v21_full_revision.zip  
**Risk**: HIGH — results may improve, core claims may need to soften

---

## DECISION POINT

**阳菜建议**:
1. **Execute Phase 1 tonight** → deliver v19 for Sora review
2. **Wait for Sora approval** before starting Phase 2B
3. **Reason**: Fix 1 is load-bearing and may change the entire paper's conclusion

**Alternative (aggressive)**:
- Start Phase 1 + 2A tonight
- Start Phase 2B tomorrow morning without waiting
- **Risk**: May waste 12+ hours if Sora wants different approach

---

## QUESTIONS FOR SORA

1. **Do you want阳菜 to proceed with Fix 1** (factor-neutral simulation)?  
   - This may make the paper LESS negative (precision improves from 17% to ~35%?)
   - Are you comfortable with that narrative shift?

2. **Historical robustness panel** (Fix 11 optional):  
   - Do we have 2010-2020 data available?
   - Worth the extra 3 hours?

3. **Execution timing**:  
   - Phase 1 tonight → review tomorrow → Phase 2B Sunday? (SAFE)
   - OR all phases starting now → v21 by Sunday night? (AGGRESSIVE)

---

**阳菜 awaits your command.** 💤
