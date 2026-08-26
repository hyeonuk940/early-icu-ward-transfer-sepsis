# -*- coding: utf-8 -*-
"""Overall (variable-level) p-values for the multi-category Table 1 variables, plus
calibration intercepts for every model and cohort.

Table 1 currently reports one p-value per category, which is a set of separate 2x2
tests. For a multi-level variable the conventional summary is a single test on the
full contingency table (2 x k), so those are computed here. Fisher's exact test is
substituted automatically whenever any expected cell count falls below 5.

Cohort labels: primary = APACHE sepsis admission
diagnosis, n=13,384; sensitivity = Sepsis-3 concordant, n=2,409.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.stats as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import data_prep as dp

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"

GROUPS = {
    "Sex": ["male"],                       # binary, handled as 2x2
    "Race": ["race_white", "race_black", "race_hispanic", "race_asian", "race_others"],
    "Admission type": ["admission_emergency", "admission_elective", "admission_other"],
}


def overall_p(ct):
    """Chi-squared on the full table; Fisher's exact if any expected count < 5."""
    ct = np.asarray(ct)
    ct = ct[:, ct.sum(0) > 0]
    if ct.shape[1] < 2:
        return np.nan, "—"
    chi2, p, dof, exp = st.chi2_contingency(ct)
    if exp.min() < 5:
        if ct.shape == (2, 2):
            return st.fisher_exact(ct)[1], "Fisher exact"
        return p, f"chi-squared (min expected {exp.min():.1f})"
    return p, "chi-squared"


def pfmt(p):
    return "—" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cal_intercept(y, p):
    eps = 1e-6
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs").fit(lg, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common)
    eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)

    # ---------- 1. overall p-values ----------
    print("=" * 78)
    print("1. Overall (variable-level) p-values for Table 1 categorical variables")
    print("=" * 78)
    rows = []
    for cname, df in [("MIMIC-IV", mimic), ("eICU primary", ea)]:
        ev, ne = df[df[T] == 1], df[df[T] == 0]
        for gname, cols in GROUPS.items():
            excl = df[cols].sum(axis=1)
            note = "" if (excl == 1).all() else f"  [WARN] not mutually exclusive: {excl.value_counts().to_dict()}"
            ct = [[int(ev[c].sum()) for c in cols], [int(ne[c].sum()) for c in cols]]
            p, how = overall_p(ct)
            rows.append(dict(cohort=cname, variable=gname, k=len(cols), p=pfmt(p), test=how))
            print(f"  {cname:14s} {gname:16s} k={len(cols)}  p={pfmt(p):8s} ({how}){note}")
        # CCI: 4 categories, and a version that keeps missing as its own level
        cats = [(ev.cci == 0), (ev.cci == 1), (ev.cci == 2), (ev.cci >= 3)]
        catn = [(ne.cci == 0), (ne.cci == 1), (ne.cci == 2), (ne.cci >= 3)]
        ct4 = [[int(c.sum()) for c in cats], [int(c.sum()) for c in catn]]
        p4, how4 = overall_p(ct4)
        rows.append(dict(cohort=cname, variable="CCI (0/1/2/>=3, complete)", k=4, p=pfmt(p4), test=how4))
        print(f"  {cname:14s} {'CCI (complete)':16s} k=4  p={pfmt(p4):8s} ({how4})")
        if df.cci.isna().any():
            ct5 = [ct4[0] + [int(ev.cci.isna().sum())], ct4[1] + [int(ne.cci.isna().sum())]]
            p5, how5 = overall_p(ct5)
            rows.append(dict(cohort=cname, variable="CCI (incl. missing level)", k=5, p=pfmt(p5), test=how5))
            print(f"  {cname:14s} {'CCI (+missing)':16s} k=5  p={pfmt(p5):8s} ({how5})")
            pm, howm = overall_p([[int(ev.cci.isna().sum()), int(ev.cci.notna().sum())],
                                  [int(ne.cci.isna().sum()), int(ne.cci.notna().sum())]])
            rows.append(dict(cohort=cname, variable="CCI missing vs observed", k=2, p=pfmt(pm), test=howm))
            print(f"  {cname:14s} {'CCI missingness':16s} k=2  p={pfmt(pm):8s} ({howm})")
        print()
    pd.DataFrame(rows).to_csv(OUT / "table1_overall_pvalues.csv", index=False)

    # ---------- 2. calibration intercepts ----------
    print("=" * 78)
    print("2. Calibration intercept (and slope) for every model and cohort")
    print("=" * 78)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ydev = dev[T]; med = dev["apsiii"].median(); feats = common
    X_full = lambda d: d[feats]; X_aps = lambda d: d[["apsiii"]].fillna(med)
    models = {
        "LightGBM (primary)": (LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED), X_full),
        "Random Forest": (Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                    ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]), X_full),
        "XGBoost": (XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                  random_state=SEED, eval_metric="logloss"), X_full),
        "Severity score alone": (LogisticRegression(), X_aps),
    }
    for nm, (est, X) in models.items():
        est.fit(X(dev), ydev)
    crows = []
    print(f"  {'cohort':38s}{'model':22s}{'intercept':>11s}{'slope':>9s}")
    for cn, d in [("Temporal internal test (MIMIC-IV)", itest),
                  ("Primary external (eICU)", ea),
                  ("Sepsis-3 sensitivity (eICU)", ep)]:
        for nm, (est, X) in models.items():
            p = est.predict_proba(X(d))[:, 1]
            ic, sl = cal_intercept(d[T].values, p)
            crows.append(dict(cohort=cn, model=nm, cal_intercept=round(ic, 3), cal_slope=round(sl, 3)))
            print(f"  {cn:38s}{nm:22s}{ic:11.3f}{sl:9.3f}")
        print()
    pd.DataFrame(crows).to_csv(OUT / "calibration_intercepts.csv", index=False)
    print("[SAVE]", OUT / "table1_overall_pvalues.csv")
    print("[SAVE]", OUT / "calibration_intercepts.csv")


if __name__ == "__main__":
    main()
