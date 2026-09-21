# -*- coding: utf-8 -*-
"""R3 — bootstrap 95% CIs for the transportable model's threshold metrics
(sens/spec/PPV/NPV at locked threshold 0.414), used in manuscript Table 2.
Output: results/r3_transportable_thr_ci.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
THR = 0.414
OUT = Path(__file__).resolve().parents[1] / "results"

def mets(y, p):
    yh = (p >= THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return (tp/(tp+fn), tn/(tn+fp),
            tp/(tp+fp) if tp+fp else np.nan, tn/(tn+fn) if tn+fn else np.nan)

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common)
    eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)]
    T = dp.TARGET_PRIMARY
    feats = [c for c in common if c != "apsiii"]
    mt = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[feats], dev[T])
    rng = np.random.default_rng(SEED)
    rows = []
    for nm, d in [("internal", mimic[mimic.anchor_year_group.isin(test)]),
                  ("eICU_primary", eicu[eicu.sepsis_admitdx == 1]),
                  ("sepsis3", eicu[eicu.sepsis3_primary == 1])]:
        y = d[T].values; p = mt.predict_proba(d[feats])[:, 1]; n = len(y)
        pt = mets(y, p); B = [[], [], [], []]
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            try:
                m = mets(y[idx], p[idx])
                for k in range(4): B[k].append(m[k])
            except Exception:
                pass
        out = [nm] + [f"{pt[k]:.3f} ({np.percentile(B[k],2.5):.3f}-{np.percentile(B[k],97.5):.3f})"
                      for k in range(4)]
        rows.append(out); print(nm, out[1:])
    pd.DataFrame(rows, columns=["cohort", "sens_ci", "spec_ci", "ppv_ci", "npv_ci"]) \
      .to_csv(OUT/"r3_transportable_thr_ci.csv", index=False)
    print("saved", OUT/"r3_transportable_thr_ci.csv")

if __name__ == "__main__":
    main()
