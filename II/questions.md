# Questions for Professor Consultation - Credit Default Prediction Project

## 1. Model Architecture: LinearSVC vs SVC
- **Question:** Is LinearSVC appropriate for credit default prediction, or should we explore non-linear kernels despite computational cost?
- **Context:** Standard SVC with RBF kernel took >15 minutes on our dataset (30,000 samples). LinearSVC converges in <1 minute.
- **Trade-off:** What predictive power (AUC improvement) justifies the 15x computational cost in practice?

## 2. Cost-Sensitive Learning Strategy: When to Apply Business Cost?
**Three possible approaches:**

**Option A (Post-hoc - Current):**
1. GridSearchCV optimizes each model using standard metric (AUC for LR/SVM, MSE for Ridge)
2. Select best model by validation AUC
3. **Then** optimize decision threshold using business cost function

**Option B (During Model Selection):**
1. GridSearchCV optimizes each model using standard metric (AUC/MSE)
2. **Select best model by minimum business cost** (instead of AUC)
3. Then optimize threshold using business cost

**Option C (During Training):**
1. Use **custom scorer in GridSearchCV** that directly optimizes business cost
2. Each model learns to minimize business cost during cross-validation
3. Select model and threshold by business cost

**Questions:**
- Which approach is preferred in industry practice?
- Does **Option A** (separate optimization) provide enough flexibility?
- Does **Option C** risk overfitting to validation set cost structure?
- How should we balance statistical performance (AUC) vs business objective (cost)?

---

## 3. Business Cost Ratio: Empirical Justification
**Current assumption:** FN cost = 10 × FP cost

**Questions:**
- Should we derive this ratio empirically from data?
  ```python
  avg_principal = data['LIMIT_BAL'].mean()  # ~$167k
  avg_interest = principal × monthly_rate   # ~$3.3k (assuming 2%)
  cost_ratio = principal / interest ≈ 50:1
  ```
- Is 10:1 too conservative? Should we use 50:1 based on actual credit limits?
- How sensitive are results to this ratio (sensitivity analysis needed)?

## 5. Feature Engineering: Validation & Selection
**8 engineered features created:**
- Temporal slopes (bill_slope, pay_slope, utilization_trend)
- Delinquency metrics (max_delinquency, delinquency_persistence)
- Financial ratios (repayment_ratio, utilization_rate, repayment_stability)

**Questions:**
- Are these features sufficient, or should we engineer more? 34 features for linear regression vs 30000 data points.
- Should we perform feature selection (e.g., remove correlated features)? maybe not since it is not overfitting.
- How to validate feature importance: coefficient analysis, permutation importance, or SHAP? this probably needed.
- Should we check residual patterns by class (defaulter vs non-defaulter)?

---

## 6. Class Imbalance: Explicit vs Implicit Handling
**Current approach:** `class_weight='balanced'` in Logistic Regression and SVM

**Alternative approaches:**
- SMOTE oversampling
- Random undersampling
- Threshold moving (already doing this)

**Questions:**
- Is `class_weight='balanced'` sufficient, or should we compare resampling methods?
- Which metric should we use to compare: AUC, business cost, or F1?
- Could resampling introduce data leakage or overfitting?

---

## 7. Three-Zone Decision System: Evaluation Framework
**Proposed strategy:**
- High-risk (P ≥ τ_high): **Auto-reject**
- Medium-risk (τ_low < P < τ_high): **Manual underwriting**
- Low-risk (P ≤ τ_low): **Auto-approve**

**Evaluation challenges:**
- **Model performance:** Per-zone calibration (predicted vs actual default rate)
- **Business efficiency:** Total cost = FN cost + FP cost + manual review cost
- **Operational metrics:** Automation rate vs manual workload (feasibility constraint)

**Questions:**
- How to optimize (τ_high, τ_low) jointly? Grid search with cost function + manual capacity constraint?
- Should we report per-zone precision/recall, or total business cost?
- How to compare three-zone system to single-threshold baseline?

**Proposed KPIs:**
| Metric | Definition | Target |
|--------|------------|--------|
| Total Cost | FN×10 + FP×1 + Manual×0.5 | Minimize |
| Automation Rate | (Auto-decisions) / Total | >70% |
| Manual Workload | Manual reviews / Total | <30% |
| Low-zone Default Rate | Actual defaults in auto-approve | <10% |
| High-zone Precision | True positives in auto-reject | >80% |

---

## 8. Bonus Question: Project 1 Revisitation
**Requirement:** Compare your portfolio selection to LassoCV/RidgeCV on new test data

**Questions:**
- Can we update our original Project 1 submission with new test results, or must we redo the entire analysis? Or just re-explain what to do with this.
---

## Summary: Key Decisions Needed
1. ✅ **LinearSVC vs SVC:** Confirm LinearSVC is acceptable
2. ✅ **Cost optimization:** Post-hoc threshold tuning (Option A) preferred?
3. ⚠️ **Cost ratio:** Need empirical justification (50:1 vs 10:1)
4. ✅ **Model selection:** AUC → threshold tuning approach confirmed?
5. ⚠️ **Feature validation:** Need feature selection analysis
6. ⚠️ **Imbalance handling:** Should we compare resampling methods?
7. ⚠️ **Three-zone evaluation:** Need optimization framework clarification
8. ❓ **Bonus logistics:** Resubmit vs redo Project 1

---