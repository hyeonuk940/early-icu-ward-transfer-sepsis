# -*- coding: utf-8 -*-
"""R3 — threshold-performance table + recalibration (editor Major 4 & 5).

1) Threshold sweep (full + transportable models; internal, eICU primary, and the
   mechanically-ventilated subgroup of eICU primary): sens/spec/PPV/NPV, % flagged,
   FP and FN counts per 100 patients, at thresholds 0.20-0.60 (step 0.05) plus the
   model's locked threshold.
2) Recalibration in the primary eICU external cohort (exploratory):
   calibration-in-the-large, intercept-only recalibration, intercept+slope
   recalibration -> slope/intercept/Brier after each.
Outputs: results/r3_threshold_table.csv, results/r3_recalibration.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import cal_metrics

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
THR_FULL = 0.396
THR_T = None  # filled from r3_transportable_performance.csv

def lgb(): return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED)

def sweep_row(model_name, cohort, y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    n = len(y)
    return dict(model=model_name, cohort=cohort, threshold=round(thr, 3),
                flagged_pct=round(yh.mean()*100, 1),
                sens=round(tp/(tp+fn), 3) if tp+fn else np.nan,
                spec=round(tn/(tn+fp), 3) if tn+fp else np.nan,
                ppv=round(tp/(tp+fp), 3) if tp+fp else np.nan,
                npv=round(tn/(tn+fn), 3) if tn+fn else np.nan,
                fp_per100=round(fp/n*100, 1), fn_per100=round(fn/n*100, 1))

def main():
    global THR_T
    tperf = pd.read_csv(OUT/"r3_transportable_performance.csv")
    THR_T = float(tperf.threshold.iloc[0])
    print(f"transportable locked threshold = {THR_T}")

    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    T = dp.TARGET_PRIMARY
    feats_full = common; feats_t = [c for c in common if c != "apsiii"]
    mfull = lgb().fit(dev[feats_full], dev[T])
    mt = lgb().fit(dev[feats_t], dev[T])

    grids = sorted(set(list(np.arange(0.20, 0.601, 0.05).round(2)) + [THR_FULL, THR_T]))
    rows = []
    ea_mv = ea[ea.invasive_mechanical_ventilation == 1]
    for mname, model, feats, locked in [("full", mfull, feats_full, THR_FULL),
                                        ("transportable", mt, feats_t, THR_T)]:
        for cname, d in [("internal", itest), ("eICU_primary", ea), ("eICU_primary_MV", ea_mv)]:
            p = model.predict_proba(d[feats])[:, 1]; y = d[T].values
            for thr in grids:
                r = sweep_row(mname, cname, y, p, thr)
                r["locked"] = (abs(thr-locked) < 1e-9)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"r3_threshold_table.csv", index=False)
    print(df[(df.model == "full") & (df.cohort == "eICU_primary")].to_string(index=False))
    print(f"[SAVE] {OUT/'r3_threshold_table.csv'} ({len(df)} rows)")

    # ---- recalibration (eICU primary, exploratory) ----
    rec = []
    eps = 1e-6
    for mname, model, feats in [("full", mfull, feats_full), ("transportable", mt, feats_t)]:
        p = model.predict_proba(ea[feats])[:, 1]; y = ea[T].values
        lp = np.log(np.clip(p, eps, 1-eps)/(1-np.clip(p, eps, 1-eps)))
        s0, i0 = cal_metrics(y, p)
        citl = float(np.log(y.mean()/(1-y.mean())) - np.log(p.mean()/(1-p.mean())))  # crude CITL on means
        # calibration-in-the-large: intercept from logistic model with slope fixed at 1 (offset)
        lr_off = LogisticRegression(solver="lbfgs")
        # fit intercept with offset: y ~ 1 + offset(lp) -> use statsmodels-like trick
        import statsmodels.api as sm
        fam = sm.families.Binomial()
        citl_model = sm.GLM(y, np.ones((len(y), 1)), family=fam, offset=lp).fit()
        a_citl = float(citl_model.params[0])
        # intercept-only recalibration
        p1 = 1/(1+np.exp(-(lp + a_citl)))
        s1, i1 = cal_metrics(y, p1)
        # slope+intercept recalibration
        ab = sm.GLM(y, sm.add_constant(lp), family=fam).fit()
        p2 = 1/(1+np.exp(-(ab.params[0] + ab.params[1]*lp)))
        s2, i2 = cal_metrics(y, p2)
        rec += [dict(model=mname, method="none (locked)", slope=round(s0, 3), intercept=round(i0, 3),
                     brier=round(brier_score_loss(y, p), 3), citl=round(a_citl, 3)),
                dict(model=mname, method="intercept-only", slope=round(s1, 3), intercept=round(i1, 3),
                     brier=round(brier_score_loss(y, p1), 3), citl=0.0),
                dict(model=mname, method="intercept+slope", slope=round(s2, 3), intercept=round(i2, 3),
                     brier=round(brier_score_loss(y, p2), 3), citl=0.0)]
        print(f"[recal {mname}] CITL={a_citl:+.3f} | locked slope {s0:.3f}/int {i0:.3f}/Brier {brier_score_loss(y,p):.3f}"
              f" -> int-only {s1:.3f}/{i1:.3f}/{brier_score_loss(y,p1):.3f}"
              f" -> int+slope {s2:.3f}/{i2:.3f}/{brier_score_loss(y,p2):.3f}")
    pd.DataFrame(rec).to_csv(OUT/"r3_recalibration.csv", index=False)
    print(f"[SAVE] {OUT/'r3_recalibration.csv'}")

if __name__ == "__main__":
    main()
