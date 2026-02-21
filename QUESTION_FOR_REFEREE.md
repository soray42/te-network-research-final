# Question for Referee - Factor-Neutral Preprocessing Results

## One-Sentence Summary for Referee

"We implemented factor-neutral preprocessing in simulation (regressing simulated returns on PCA-estimated factors before TE estimation), but precision **worsened** from 17% to 11% at T/N=5 — should we interpret this as evidence that TE networks inherently rely on cross-sectional correlation, making factor-neutral preprocessing inappropriate for network recovery tasks, or did we misunderstand your suggested approach?"

---

## Extended Version (if needed)

Following your suggestion to match the empirical factor-neutral pipeline in simulation, we:

1. **Implemented**: After generating returns from the GARCH+Factor DGP, we regress each series on 3 PCA-estimated factors and use residuals as TE input (matching Section 5's empirical preprocessing exactly).

2. **Result**: Precision at T/N=5 dropped from 17% (raw returns) to **11%** (factor-neutral). Hub recovery and Kendall's τ also worsened.

3. **Question**: Does this suggest:
   - (a) Factor-neutral preprocessing is **inappropriate** for TE network estimation (networks require cross-sectional correlation to function), OR
   - (b) PCA factor estimation at low T/N introduces additional noise (we should test with oracle factors), OR  
   - (c) We misunderstood your recommendation?

We can report this honestly ("factor adjustment worsens recovery, suggesting TE networks rely on cross-sectional structure"), but wanted to confirm we're interpreting your Fix 1 correctly before proceeding.

**Attached**: CSV showing precision decline across all T/N ratios under estimated factor-neutral vs. raw returns.

---

## Chinese Version (for Sora)

"我们按你建议在模拟里加了 factor-neutral preprocessing（用 PCA 估计 factors 然后回归残差），但 precision 从 17% **降到了 11%**（T/N=5）——这是不是说明 TE 网络本质上就需要 cross-sectional correlation，所以 factor-neutral 不适合网络恢复任务？还是我们理解错了你的意思？"
