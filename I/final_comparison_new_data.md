# FINAL BENCHMARK COMPARISON: NEW DATA (July-September 2025)

## Question: Is Grouped MinVar better than LassoCV or RidgeCV on NEW test data?

---

## Results on NEW Test Data (2025-07-01 to 2025-09-30)

### Sharpe Ratios

| Algorithm | Mode 1 (Standard) | Mode 3 (Grouped) | Difference |
|-----------|-------------------|------------------|------------|
| **LASSO** | **0.1467** | 0.0905 | -0.0562 |
| **Ridge** | **0.1524** | 0.0821 | -0.0703 |
| **MinVar** | 0.0273 | **0.0919** | +0.0646 |

---

## Answer: **NO - Standard LASSO/Ridge are BETTER!**

### Standard Methods Win on NEW Data

**Standard LASSO vs Grouped MinVar:**
- Standard LASSO: **0.1467**
- Grouped MinVar: **0.0919**
- Standard LASSO is **59.7% BETTER**

**Standard Ridge vs Grouped MinVar:**
- Standard Ridge: **0.1524**  
- Grouped MinVar: **0.0919**
- Standard Ridge is **65.8% BETTER**

---

## Key Findings

1. **Standard LASSO and Ridge significantly outperform Grouped MinVar** on the NEW test data
2. **Grouped MinVar (0.0919) beats Grouped LASSO/Ridge** within Mode 3
3. **Standard MinVar fails completely** (0.0273) - grouping helps MinVar
4. **But standard regularization methods are the clear winners** on this validation period

---

## Detailed Comparison

### Mode 1 (Standard - 100 Portfolios)
```
LASSO:   Sharpe 0.1467  (14.18% return, 96.65% vol)
Ridge:   Sharpe 0.1524  (14.15% return, 92.81% vol)
MinVar:  Sharpe 0.0273  (4.81% return, 176.31% vol)
EW:      Sharpe 0.0493  (5.36% return, 108.66% vol)
```

### Mode 3 (Grouped - 9 Groups)
```
MinVar:  Sharpe 0.0919  (8.09% return, 88.03% vol)
LASSO:   Sharpe 0.0905  (7.92% return, 87.54% vol)
Ridge:   Sharpe 0.0821  (7.29% return, 88.79% vol)
EW:      Sharpe 0.0493  (5.36% return, 108.66% vol)
```

---

## Conclusion

**On the NEW validation period (July-September 2025), standard LassoCV and RidgeCV significantly outperform the grouped MinVar approach.**

While grouping helped improve MinVar performance compared to standard MinVar, the standard regularization methods (LASSO/Ridge on 100 portfolios) achieved the best risk-adjusted returns.

**Winner: Standard Ridge (Sharpe: 0.1524)**

---

*Analysis Date: November 27, 2025*
*Validation Period: 2025-07-01 to 2025-09-30 (24 trading days)*
