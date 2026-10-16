# ML-Based Employee Attrition Prediction and Risk Scoring — Palo Alto Networks

## 1. Data
1,470 employees, 31 fields, no missing values. 237 left (16.1%), 1,233 stayed, so the classes are imbalanced (about 5:1). The file has no employee ID; IDs 1–1470 are assigned by row order.

## 2. Key EDA findings (observed attrition rates)
| Segment | Attrition |
|---|---|
| Works overtime | 30.5% (vs 10.4% without) |
| Sales Representative | 39.8% (highest role); Research Director 2.5% (lowest) |
| Laboratory Technician | 23.9% |
| Single | 25.5% (vs ~10–12.5% divorced/married) |
| Stock option level 0 | 24.4% (vs 7.6–9.4% at levels 1–2) |
| Income ≤ $3,000/month | 28.6% (vs ~10% above $5,000) |
| Department | Sales 20.6%, HR 19.0%, R&D 13.8% |

Attrition concentrates among junior, lower-paid, overtime-working, single employees with no equity.

## 3. Method
- **Preprocessing:** one-hot encoding for 7 categorical fields, standard scaling for numerics, stratified 80/20 split (seed 42), class weights for imbalance (`balanced`, `scale_pos_weight` for XGBoost; SMOTE was not needed).
- **Engineered features:** income-to-experience ratio, promotion-delay ratio and a 4+-years-without-promotion flag, engagement composite (mean of five satisfaction/involvement scores), workload-stress flag (overtime AND frequent travel or WLB ≤ 2), companies-per-year.
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost. Decision thresholds were tuned for F1 on training-fold predictions only, so the test set was untouched.
- **Selection:** by 5-fold CV AUC on the training set.

## 4. Results (20% held-out test set, n = 294)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.850 | 0.529 | 0.574 | 0.551 | **0.816** |
| Random Forest | 0.827 | 0.460 | 0.489 | 0.474 | 0.781 |
| Gradient Boosting | 0.813 | 0.439 | 0.617 | 0.513 | 0.794 |
| XGBoost | 0.844 | 0.515 | 0.362 | 0.425 | 0.766 |

The interpretable baseline matched or beat the tree ensembles. This is plausible for a small (1,470-row), mostly linear tabular problem, and it means HR gets the most transparent model without a performance cost. With only ~47 positives in the test set, differences of a few AUC points between models are within noise, so the ranking should not be over-read.

## 5. Risk scoring
Final scores are calibrated (sigmoid, 5-fold) probabilities from the selected model, computed **out-of-fold** so no employee is scored by a model that saw their own outcome. Out-of-fold AUC 0.826, Brier 0.097.

| Category | Employees | Observed attrition |
|---|---|---|
| Low (<30%) | 1,213 | 8.1% |
| Medium (30–60%) | 190 | 43.7% |
| High (>60%) | 67 | 83.6% |

The tiers are well separated and the probabilities are reasonably calibrated.

## 6. Explainability
Global drivers (mean |SHAP|, XGBoost explainer): engagement composite, overtime, stock option level, companies-per-year, monthly income, age, distance from home, job level, frequent travel, years since promotion. Each employee receives top-3 SHAP reason codes in the scored file and dashboard. These describe association, not proven cause.

## 7. Recommendations
1. Review the 67 high-risk employees first, then the 190 medium-risk, prioritising critical or high-performing people.
2. Reduce chronic overtime, especially where combined with frequent travel (workload-stress flag).
3. Review pay and equity for low-income, entry-level and Sales Representative / Lab Technician populations; stock option level 0 is a strong marker.
4. Audit promotion pipelines for people stalled 4+ years.
5. Use scores to prioritise conversations, never as the sole basis for employment decisions.

## 8. Limitations
- The dataset appears to be the widely used IBM HR sample (synthetic); real Palo Alto Networks results may differ and the model should be re-validated on internal data.
- Attrition is a one-time label with no leave date, so "near future" cannot be modelled; scores indicate relative propensity, not a time-bound forecast.
- Small sample and modest precision (~53%) mean roughly half of flagged employees at the F1 threshold would not leave. Use scores as a triage tool.
- Gender, age and marital status are model inputs; review fairness before any operational use, and consider excluding protected attributes.
- What-if outputs are model associations, not causal effects.

---

# Executive Summary

**Problem.** Resignations are discovered too late for effective retention action.

**Solution.** A machine learning model scores every employee's probability of leaving (0–100%) and assigns Low / Medium / High risk, with plain-language reasons.

**Performance.** ROC-AUC ≈ 0.82. Among employees rated High risk, 84% actually left, versus 8% for Low risk, a tenfold difference that lets HR focus on about 5% of staff (67 of 1,470) who account for a disproportionate share of exits.

**Main drivers.** Overtime, low engagement/satisfaction, no stock options, low pay, junior level, long stretches without promotion, frequent travel, and Sales Representative / Lab Technician roles.

**Actions.** Target retention conversations at High and Medium risk employees; cut chronic overtime; revisit pay, equity and promotion cadence in the highest-risk groups.

**Caveats.** Built on a sample dataset; validate on internal data, review fairness, and use scores to inform, not replace, managerial judgement.
