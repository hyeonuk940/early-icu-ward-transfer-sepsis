"""
Nested-CV pipeline (core headline result).
- outer5 x inner5 nested CV on MIMIC development set: HP + threshold chosen
  in the inner loop; outer folds give honest performance + selection variance.
- temporal internal test via anchor_year_group (dev 2008-2016, test 2017-2022).
- native missing handling (LightGBM) -> primary; NO listwise deletion.
- physiologic-range screening (fixed bounds, value-level) via data_prep.
- threshold locked from development inner-CV, applied unchanged to test/external.
A single locked model is evaluated ONCE on internal test and ONCE on each external cohort.
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42; N_BOOT = 1000
OUT = Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(exist_ok=True)
THRESHOLD_FILE = OUT / "locked_threshold.json"

def locked_threshold(model="lightgbm"):
    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"{THRESHOLD_FILE} not found -- run nested_cv.py to lock the threshold first.")
    return json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))[model]

def bootstrap_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); n = len(y); v = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        v.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

def cal_metrics(y, p):
    eps = 1e-6; p = np.clip(p, eps, 1 - eps); logit = np.log(p/(1-p)).reshape(-1,1)
    lr = LogisticRegression(solver="lbfgs").fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def metrics_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else np.nan; spec = tn/(tn+fp) if (tn+fp) else np.nan
    ppv = tp/(tp+fp) if (tp+fp) else np.nan; npv = tn/(tn+fn) if (tn+fn) else np.nan
    return dict(sens=sens, spec=spec, ppv=ppv, npv=npv)

LGB_GRID = {
    "n_estimators": [300, 500], "learning_rate": [0.01, 0.05],
    "num_leaves": [15, 31], "min_child_samples": [20, 50],
}

def lgb(seed=SEED):
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1)

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p); return thr[np.argmax(tpr - fpr)]

def nested_cv_lgb(X, y):
    """outer5 x inner5. Returns outer-fold AUCs and inner-selected Youden thresholds."""
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    aucs, thrs = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        gs = GridSearchCV(lgb(), LGB_GRID, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        p_te = best.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p_te))
        # inner-CV threshold from out-of-fold preds on the outer-train
        oof = cross_val_predict(gs.best_estimator_, Xtr, ytr, cv=5,
                                method="predict_proba", n_jobs=-1)[:, 1]
        thrs.append(youden_threshold(ytr, oof))
        print(f"    outer fold {k+1}: AUC={aucs[-1]:.3f}  thr={thrs[-1]:.3f}  {gs.best_params_}")
    return np.array(aucs), np.array(thrs)

def lock_lgb(Xdev, ydev):
    gs = GridSearchCV(lgb(), LGB_GRID, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                      scoring="roc_auc", n_jobs=-1).fit(Xdev, ydev)
    model = gs.best_estimator_
    oof = cross_val_predict(model, Xdev, ydev, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    thr = youden_threshold(ydev, oof)
    model.fit(Xdev, ydev)
    return model, thr, gs.best_params_

def evaluate(name, model, thr, X, y):
    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
    cs, ci = cal_metrics(y, p); br = brier_score_loss(y, p)
    m = metrics_at_threshold(y, p, thr)
    row = dict(cohort=name, n=len(y), events=int(y.sum()), auc=auc, auc_lo=lo, auc_hi=hi,
               brier=br, cal_slope=cs, cal_intercept=ci, **m)
    print(f"  {name:22s} n={len(y):6d} AUC={auc:.3f} ({lo:.3f}-{hi:.3f}) "
          f"Brier={br:.3f} slope={cs:.3f} int={ci:.3f} sens={m['sens']:.3f} spec={m['spec']:.3f}")
    return row, p

def main():
    TARGET = dp.TARGET_PRIMARY
    mimic, eicu, fm, fe, common = dp.load()
    feats = common
    mimic = dp.apply_physiologic_ranges(mimic, feats)
    eicu = dp.apply_physiologic_ranges(eicu, feats)

    # temporal split
    test_groups = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test_groups)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test_groups)].reset_index(drop=True)
    print(f"MIMIC dev n={len(dev)} (event {dev[TARGET].mean():.3f}) | "
          f"internal test n={len(itest)} (event {itest[TARGET].mean():.3f})")

    Xdev, ydev = dev[feats], dev[TARGET]
    print("\n[Nested CV] LightGBM (native missing) on development set:")
    aucs, thrs = nested_cv_lgb(Xdev, ydev)
    print(f"  >> nested-CV AUC = {aucs.mean():.3f} (outer folds {aucs.min():.3f}-{aucs.max():.3f}, "
          f"SD {aucs.std():.3f})   [captures selection uncertainty]")

    print("\n[Lock] refit on full development set + fixed threshold:")
    model, thr, bp = lock_lgb(Xdev, ydev)
    print(f"  locked HP={bp}  locked threshold={thr:.3f}")

    # external cohorts
    e_prim = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    e_alt = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)

    print("\n[Evaluate locked LightGBM once per cohort]:")
    rows = []
    for nm, d in [("MIMIC internal test", itest), ("eICU Sepsis-3 sensitivity", e_prim),
                  ("eICU primary", e_alt)]:
        r, _ = evaluate(nm, model, thr, d[feats], d[TARGET]); rows.append(r)

    # APS III baseline (LR on apsiii; median-impute from dev)
    print("\n[APS III alone baseline] LR on apsiii:")
    med = dev["apsiii"].median()
    def aps_X(d): return d[["apsiii"]].fillna(med)
    aps = LogisticRegression().fit(aps_X(dev), ydev)
    aps_oof = cross_val_predict(LogisticRegression(), aps_X(dev), ydev, cv=5,
                                method="predict_proba")[:, 1]
    aps_thr = youden_threshold(ydev, aps_oof)
    for nm, d in [("MIMIC internal test", itest), ("eICU Sepsis-3 sensitivity", e_prim),
                  ("eICU primary", e_alt)]:
        p = aps.predict_proba(aps_X(d))[:, 1]
        auc = roc_auc_score(d[TARGET], p); lo, hi = bootstrap_ci(d[TARGET], p)
        rows.append(dict(cohort="APS III: " + nm, n=len(d), events=int(d[TARGET].sum()),
                         auc=auc, auc_lo=lo, auc_hi=hi, brier=brier_score_loss(d[TARGET], p),
                         cal_slope=np.nan, cal_intercept=np.nan,
                         **metrics_at_threshold(d[TARGET], p, aps_thr)))
        print(f"  APS III {nm:22s} AUC={auc:.3f} ({lo:.3f}-{hi:.3f})")

    THRESHOLD_FILE.write_text(json.dumps(
        {"lightgbm": round(float(thr), 3), "apsiii": round(float(aps_thr), 3),
         "hyperparams": bp}, indent=2), encoding="utf-8")
    print(f"\n[SAVE] {THRESHOLD_FILE}")

    res = pd.DataFrame(rows)
    res.insert(0, "nested_cv_dev_auc", round(aucs.mean(), 3))
    res.to_csv(OUT / "headline_nested_cv.csv", index=False)
    print(f"[SAVE] {OUT/'headline_nested_cv.csv'}")

if __name__ == "__main__":
    main()
