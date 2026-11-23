# Telecom Churn Prediction with SHAP

Predicts which mobile subscribers will churn, explains **why** for each customer in plain English, and works out how many customers a retention campaign should contact to make the most money.

## Data

50,000 synthetic subscribers with a 19% churn rate. Churn comes from a hidden model, so the real drivers are known: contract type, tenure, support calls, dropped-call rate, falling data usage, device age, bill size, autopay, number of lines, and **asking about porting out**. Pure-noise columns (region, a survey answer) are included to check that the explanations ignore them.

## Models (test set: 10,000 subscribers)

| Model | ROC-AUC | PR-AUC | Top-10% lift | Churners in top 20% |
|---|---|---|---|---|
| **Logistic regression** (baseline) | **0.808** | **0.543** | **3.24** | **52.5%** |
| XGBoost (early stopped at 246 trees) | 0.804 | 0.534 | 3.16 | 51.5% |

**The baseline wins.** The churn process here is linear in log-odds, so the boosted trees have nothing extra to find. That's why a simple baseline is always worth training: model complexity has to earn its place. XGBoost is used below for its fast, exact SHAP values.

Calling the top 10% riskiest customers reaches **3.2× the average churn rate**. Calling the top 20% reaches more than half of all churners.

## What drives churn (mean |SHAP|, log-odds)

```
contract              0.669   ████████████████
usage_trend           0.446   ███████████
tenure_months         0.297   ███████
support_calls_90d     0.258   ██████
port_out_inquiry      0.188   ████
device_age_months     0.187   ████
dropped_call_rate     0.171   ████
autopay               0.155   ████
...
region                0.007   (noise, correctly ignored)
favorite_color_survey 0.006   (noise, correctly ignored)
```

Correlated features share credit: `lines`, `monthly_bill` and `bill_per_line` move together, so SHAP spreads the effect of the number of lines across them.

## Why each customer is at risk

The top three positive SHAP contributions for each customer become **reason codes** a retention agent can act on:

```
risk 90%: asked about porting out; high dropped-call rate; short time as a customer
risk 90%: data usage is falling; high dropped-call rate; several support calls recently
risk 89%: asked about porting out; data usage is falling; on a month-to-month contract
```

## How many customers to contact

Assumptions: a $40 credit per contacted customer, 25% of contacted churners are saved, and a saved customer is worth 12 months of margin.

| Contact top | Customers | Net value |
|---|---|---|
| 10% | 1,000 | $66,938 |
| 20% | 2,000 | $87,081 |
| **30%** | 3,000 | **$90,541** |
| 50% | 5,000 | $63,207 |
| everyone | 10,000 | **−$97,155** |

Contacting everyone loses money. The model's value is in **who not to call**.

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=src python report.py     # prints everything above, saves shap_summary.png and lift_by_decile.png
pytest
```

The tests check that real drivers outrank noise features by at least 5×, that SHAP values add up to the model's output (up to a constant base-score difference between some shap and xgboost versions), and that the model gives at least 2.5× lift in the top decile.

## Next steps
- Uplift modeling: target customers the offer actually *changes*, not just those at risk
- Calibrate the scores before using them as probabilities in campaign math
