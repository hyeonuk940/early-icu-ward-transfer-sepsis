# -*- coding: utf-8 -*-
"""Sensitivity / specificity / PPV / NPV with bootstrap 95% CIs at each model's
locked development-set threshold.

Thresholds are the Youden-index cut-points obtained from 5-fold out-of-fold
predictions on the DEVELOPMENT set only (same procedure as nested_cv.py), rounded
to three decimals so that the value reported in the manuscript and the value used
for the table are identical and a reader can reproduce them.

Bootstrap: 1,000 patient-level resamples, seed 42, percentile interval -- the same
scheme nested_cv.bootstrap_ci uses for the AUC.

Cohort labels:
    primary      = eICU APACHE sepsis admission-diagnosis cohort (n=13,384, 200 hospitals)
    sensitivity  = eICU Sepsis-3 definition-concordant cohort   (n= 2,409,  84 hospitals)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import data_prep as dp

SEED = 42; N_BOOT = 1000
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"


def counts(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def metrics(y, p, thr):
    tn, fp, fn, tp = counts(y, p, thr)
    return dict(sens=tp / (tp + fn) if tp + fn else np.nan,
                spec=tn / (tn + fp) if tn + fp else np.nan,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                npv=tn / (tn + fn) if tn + fn else np.nan)


def boot_ci(y, p, thr, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% CI for sens/spec/PPV/NPV by patient-level resampling."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    acc = {k: [] for k in ("sens", "spec", "ppv", "npv")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx], thr)
        for k, v in m.items():
            if not np.isnan(v):
                acc[k].append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
            for k, v in acc.items()}


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def dev_threshold(est, X, y):
    """cv=5 (plain int, unshuffled) mirrors nested_cv.py exactly."""
    oof = cross_val_predict(est, X, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    return youden(y, oof)


def main():
    mimic, eicu, fm, fe, common = dp.load(); feats = common
    mimic = dp.apply_physiologic_ranges(mimic, feats)
    eicu = dp.apply_physiologic_ranges(eicu, feats)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    T = dp.TARGET_PRIMARY; ydev = dev[T]; med = dev["apsiii"].median()

    cohorts = [
        ("Internal test (MIMIC-IV)", mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)),
        ("External: primary (eICU)", eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)),
        ("External: Sepsis-3 sensitivity (eICU)", eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)),
    ]

    X_full = lambda d: d[feats]
    X_aps = lambda d: d[["apsiii"]].fillna(med)
    specs = [("LightGBM (primary)",
              lambda: LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED), X_full),
             ("Random Forest",
              lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]), X_full),
             ("XGBoost",
              lambda: XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                    random_state=SEED, eval_metric="logloss"), X_full),
             ("APS III alone", lambda: LogisticRegression(), X_aps)]

    print("Locked thresholds (Youden on 5-fold OOF, development set only):")
    fitted = {}
    for nm, factory, X in specs:
        raw = dev_threshold(factory(), X(dev), ydev)
        thr = round(raw, 3)                      # report and compute at the same value
        fitted[nm] = (factory().fit(X(dev), ydev), X, thr, raw)
        print(f"  {nm:22s} {raw:.6f}  ->  reported/used {thr:.3f}")

    rows = []
    for coh, d in cohorts:
        y = d[T].values
        for nm, (mdl, X, thr, raw) in fitted.items():
            p = mdl.predict_proba(X(d))[:, 1]
            m = metrics(y, p, thr); ci = boot_ci(y, p, thr)
            tn, fp, fn, tp = counts(y, p, thr)
            r = dict(cohort=coh, model=nm, n=len(y), events=int(y.sum()), threshold=thr,
                     TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))
            for k in ("sens", "spec", "ppv", "npv"):
                r[k] = round(m[k], 3)
                r[k + "_ci"] = f"{ci[k][0]:.3f}-{ci[k][1]:.3f}"
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_metrics_ci.csv", index=False)

    print()
    for coh, _ in cohorts:
        g = df[df.cohort == coh]
        print(f"=== {coh}   n={g.n.iloc[0]:,}  events={g.events.iloc[0]:,} ===")
        print(f"  {'model':22s}{'thr':>7s}{'Sensitivity':>22s}{'Specificity':>22s}"
              f"{'PPV':>22s}{'NPV':>22s}")
        for _, r in g.iterrows():
            print(f"  {r.model:22s}{r.threshold:7.3f}"
                  + "".join(f"{r[k]:.3f} ({r[k+'_ci']})".rjust(22) for k in ("sens", "spec", "ppv", "npv")))
        print()
    print("[SAVE]", OUT / "threshold_metrics_ci.csv")
    print(f"CIs: {N_BOOT} patient-level bootstrap resamples, seed {SEED}, percentile method.")


if __name__ == "__main__":
    main()
