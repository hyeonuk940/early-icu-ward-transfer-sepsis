# -*- coding: utf-8 -*-
"""Supplementary figures:
  (a) development / temporal internal test set sizes and event counts
  (b) 19-feature parsimonious model AUC with bootstrap 95% CI in all three cohorts
  (c) 48 h and 96 h horizon AUCs with bootstrap 95% CI

Bootstrap: 1,000 patient-level resamples, seed 42, percentile interval -- identical
to nested_cv.bootstrap_ci, which produced the AUC intervals already in the paper.

Cohort labels:
    primary      = eICU APACHE sepsis admission-diagnosis cohort (n=13,384, 200 hospitals)
    sensitivity  = eICU Sepsis-3 definition-concordant cohort   (n= 2,409,  84 hospitals)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"


def horizon_target(df, H):
    return ((df.icu_duration_hours <= H) & (df.left_icu_alive == 1) & (df.went_to_ward == 1)
            & (df.icu_readmit_7d == 0) & (df.death_7d_post == 0)).astype(int)


def main():
    mimic, eicu, fm, fe, common = dp.load(); feats = list(common)
    mimic = dp.apply_physiologic_ranges(mimic, feats)
    eicu = dp.apply_physiologic_ranges(eicu, feats)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)     # primary
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)    # sensitivity
    T = dp.TARGET_PRIMARY

    # ---------- (a) split sizes ----------
    print("=" * 74)
    print("(a) Development and temporal internal test set")
    print("=" * 74)
    rows = []
    for lab, d in [("Development set (2008-2010, 2011-2013, 2014-2016)", dev),
                   ("Temporal internal test set (2017-2019, 2020-2022)", itest),
                   ("MIMIC-IV landmark cohort (total)", mimic)]:
        n, ev = len(d), int(d[T].sum())
        rows.append(dict(set=lab, n=n, events=ev, event_rate=round(ev / n, 3), non_events=n - ev))
        print(f"  {lab:52s} n={n:6,}  events={ev:6,} ({ev/n*100:4.1f}%)")
    print()
    print("  anchor-year group breakdown")
    g = mimic.groupby("anchor_year_group").agg(n=(T, "size"), events=(T, "sum"))
    for k, r in g.iterrows():
        part = "test" if k in test else "development"
        print(f"    {k:14s} n={int(r.n):6,}  events={int(r.events):5,} ({r.events/r.n*100:4.1f}%)   {part}")
    pd.DataFrame(rows).to_csv(OUT / "split_sizes.csv", index=False)

    cohorts = [("Temporal internal test (MIMIC-IV)", itest),
               ("Primary external (eICU)", ea),
               ("Sepsis-3 sensitivity (eICU)", ep)]

    # ---------- (b) 19-feature parsimonious model ----------
    print()
    print("=" * 74)
    print("(b) 19-feature parsimonious model, AUC with 95% CI")
    print("=" * 74)
    sub = pd.read_csv(OUT / "parsimonious_features.csv")["parsimonious_features"].tolist()
    print(f"  features (n={len(sub)}): {', '.join(sub)}")
    pm = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[sub], dev[T])
    full = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[feats], dev[T])
    prows = []
    print()
    print(f"  {'cohort':36s}{'19-feature AUC (95% CI)':>28s}{'full 56 (95% CI)':>28s}")
    for cn, d in cohorts:
        y = d[T].values
        p19 = pm.predict_proba(d[sub])[:, 1]; a19 = roc_auc_score(y, p19); l19, h19 = bootstrap_ci(y, p19)
        p56 = full.predict_proba(d[feats])[:, 1]; a56 = roc_auc_score(y, p56); l56, h56 = bootstrap_ci(y, p56)
        prows.append(dict(cohort=cn, n=len(y), events=int(y.sum()), n_features=19,
                          auc_19=round(a19, 3), ci_19=f"{l19:.3f}-{h19:.3f}",
                          auc_full=round(a56, 3), ci_full=f"{l56:.3f}-{h56:.3f}"))
        print(f"  {cn:36s}{f'{a19:.3f} ({l19:.3f}-{h19:.3f})':>28s}{f'{a56:.3f} ({l56:.3f}-{h56:.3f})':>28s}")
    pd.DataFrame(prows).to_csv(OUT / "parsimonious_auc_ci.csv", index=False)

    # ---------- (c) horizon sensitivity ----------
    print()
    print("=" * 74)
    print("(c) Prediction-horizon sensitivity, AUC with 95% CI")
    print("=" * 74)
    hrows = []
    for H in (48, 72, 96):
        yd = horizon_target(dev, H)
        mdl = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[feats], yd)
        r = {"horizon_h": H, "dev_event_rate": round(float(yd.mean()), 3)}
        line = f"  {H:3d} h  dev rate {yd.mean():.3f} |"
        for cn, d in cohorts:
            yy = horizon_target(d, H); p = mdl.predict_proba(d[feats])[:, 1]
            a = roc_auc_score(yy, p); lo, hi = bootstrap_ci(yy, p)
            key = cn.split(" (")[0].replace(" ", "_")
            r[key + "_auc"] = round(a, 3); r[key + "_ci"] = f"{lo:.3f}-{hi:.3f}"
            r[key + "_rate"] = round(float(yy.mean()), 3)
            line += f"  {cn.split(' (')[0][:22]}: {a:.3f} ({lo:.3f}-{hi:.3f})"
        hrows.append(r); print(line)
    pd.DataFrame(hrows).to_csv(OUT / "horizon_auc_ci.csv", index=False)

    print()
    print("[SAVE]", OUT / "split_sizes.csv")
    print("[SAVE]", OUT / "parsimonious_auc_ci.csv")
    print("[SAVE]", OUT / "horizon_auc_ci.csv")
    print("CIs: 1,000 patient-level bootstrap resamples, seed 42, percentile method.")


if __name__ == "__main__":
    main()
