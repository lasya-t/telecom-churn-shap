"""End-to-end run: train, compare with the baseline, explain, and size a retention campaign."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from churn.evaluate import decile_table, summary
from churn.explain import global_importance, reason_codes, shap_values
from churn.train import make_split, train_baseline, train_xgb

s = make_split()
print(f"train {len(s.y_train):,} / valid {len(s.y_valid):,} / test {len(s.y_test):,} subscribers, "
      f"churn rate {s.y_test.mean():.1%}\n")

base = train_baseline(s)
model = train_xgb(s)
p_base = base.predict_proba(s.X_test)[:, 1]
p = model.predict_proba(s.X_test)[:, 1]
print(f"{'model':22s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'top-10% lift':>13s} {'churners in top 20%':>20s}")
for name, pp in (("logistic baseline", p_base), (f"xgboost ({model.best_iteration} trees)", p)):
    m = summary(s.y_test, pp)
    print(f"{name:22s} {m['ROC-AUC']:8.3f} {m['PR-AUC']:7.3f} {m['top-10% lift']:13.2f} {m['churners caught in top 20%']:20.1%}")

print("\nDecile table (xgboost):")
print(decile_table(s.y_test, p).to_string())

sample = s.X_test.iloc[:4000]
sv = shap_values(model, sample)
print("\nGlobal drivers (mean |SHAP|, log-odds):")
print(global_importance(sv, sample).round(3).to_string())

print("\nTop 5 highest-risk customers and why:")
ps = model.predict_proba(sample)[:, 1]
for i in np.argsort(ps)[::-1][:5]:
    print(f"  risk {ps[i]:.0%}: " + "; ".join(reason_codes(sv[i:i + 1], sample.iloc[i:i + 1])[0]))

# retention campaign sizing: $40 credit per contacted customer, 25% of contacted churners are saved,
# a saved customer is worth 12 months of margin (~35% of their bill)
credit, save_rate = 40.0, 0.25
value = 12 * 0.35 * s.X_test["monthly_bill"].to_numpy()
order = np.argsort(p)[::-1]
print("\nCampaign size vs net value (test set):")
best = None
for share in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    k = int(share * len(order))
    idx = order[:k]
    net = (s.y_test[idx] * value[idx]).sum() * save_rate - credit * k
    best = max(best or (net, share), (net, share))
    print(f"  contact top {share:>4.0%}: {k:6,d} customers  net ${net:>10,.0f}")
print(f"  -> best: contact the top {best[1]:.0%}")

shap.summary_plot(sv, sample, max_display=12, show=False)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=110); plt.close()
t = decile_table(s.y_test, p)
plt.figure(figsize=(6, 3.5)); plt.bar(t.index, t["lift"]); plt.axhline(1, ls="--", c="gray")
plt.xlabel("risk decile (1 = highest)"); plt.ylabel("lift"); plt.title("Churn lift by decile"); plt.tight_layout()
plt.savefig("lift_by_decile.png", dpi=110)
print("\nsaved shap_summary.png, lift_by_decile.png")
