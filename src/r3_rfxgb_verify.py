# -*- coding: utf-8 -*-
"""Independent re-verification of the Table 2 Random Forest / XGBoost rows
(AUC with bootstrap CI, Brier, and sens/spec/PPV/NPV with bootstrap CIs at each
model's locked development threshold: RF 0.423, XGBoost 0.401).
The XGBoost threshold-based cells in Table 2 were updated to these values
(previous cells traced to a stale pre-final run; AUC/Brier were unchanged).
Output: results/r3_rfxgb_verify.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier
import data_prep as dp
from nested_cv import bootstrap_ci

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "results"

def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return tp/(tp+fn), tn/(tn+fp), tp/(tp+fp), tn/(tn+fn)

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]).fit(dev[common], dev[T])
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=SEED,
                        eval_metric="logloss").fit(dev[common], dev[T])
    rng = np.random.default_rng(SEED)
    rows = []
    for mname, mdl, thr in [("RF", rf, 0.423), ("XGB", xgb, 0.401)]:
        for cname, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
            y = d[T].values; p = mdl.predict_proba(d[common])[:, 1]
            auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
            pt = metrics(y, p, thr)
            B = [[], [], [], []]
            for _ in range(1000):
                idx = rng.integers(0, len(y), len(y))
                try:
                    mm = metrics(y[idx], p[idx], thr)
                    for k in range(4): B[k].append(mm[k])
                except Exception:
                    pass
            ci = [f"{pt[k]:.3f} ({np.percentile(B[k],2.5):.3f}-{np.percentile(B[k],97.5):.3f})" for k in range(4)]
            rows.append(dict(model=mname, cohort=cname, auc=f"{auc:.3f} ({lo:.3f}-{hi:.3f})",
                             brier=round(brier_score_loss(y, p), 3),
                             sens_ci=ci[0], spec_ci=ci[1], ppv_ci=ci[2], npv_ci=ci[3]))
            print(mname, cname, rows[-1]["auc"], "Brier", rows[-1]["brier"], ci)
    pd.DataFrame(rows).to_csv(OUT/"r3_rfxgb_verify.csv", index=False)
    print("saved", OUT/"r3_rfxgb_verify.csv")

if __name__ == "__main__":
    main()
