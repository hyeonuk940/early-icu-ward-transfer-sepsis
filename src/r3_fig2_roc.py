# -*- coding: utf-8 -*-
"""Regenerate manuscript Figure 2 for R3:
(a) temporal internal test — full LightGBM, transportable, APS III-only LR;
(b) primary eICU external — full, transportable (n=13,384) + database-native
    APACHE-IVa benchmark among patients with a documented score (n=11,768).
Output: figures/figR3_roc_main.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400, "savefig.bbox": "tight"})

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    T = dp.TARGET_PRIMARY
    featsF = common; featsT = [c for c in common if c != "apsiii"]
    mf = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[featsF], dev[T])
    mt = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[featsT], dev[T])
    med = dev["apsiii"].median()
    aps = LogisticRegression().fit(dev[["apsiii"]].fillna(med), dev[T])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    # (a) internal
    y = itest[T].values
    curves_a = [("LightGBM, full model", mf.predict_proba(itest[featsF])[:, 1], "#2166ac", "-"),
                ("LightGBM, transportable model", mt.predict_proba(itest[featsT])[:, 1], "#1b7837", "-"),
                ("APS III alone", aps.predict_proba(itest[["apsiii"]].fillna(med))[:, 1], "#b2182b", "-")]
    ax = axes[0]
    for nm, p, col, ls in curves_a:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2.0, label=f"{nm} (AUC {roc_auc_score(y,p):.3f})")
    ax.set_title(f"(a) Temporal internal test (MIMIC-IV)\nn = {len(itest):,}", fontsize=12)
    # (b) external
    ax = axes[1]
    y = ea[T].values
    for nm, p, col in [("LightGBM, full model", mf.predict_proba(ea[featsF])[:, 1], "#2166ac"),
                       ("LightGBM, transportable model", mt.predict_proba(ea[featsT])[:, 1], "#1b7837")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=2.0, label=f"{nm} (AUC {roc_auc_score(y,p):.3f})")
    mask = ea["apsiii"].notna().values
    ym = y[mask]; pn = -ea.loc[mask, "apsiii"].values
    fpr, tpr, _ = roc_curve(ym, pn)
    ax.plot(fpr, tpr, color="#b2182b", lw=2.0,
            label=f"APACHE-IVa, native (AUC {roc_auc_score(ym,pn):.3f})")
    ax.set_title(f"(b) Primary eICU external cohort\nmodels n = {len(ea):,}; native benchmark n = {mask.sum():,}",
                 fontsize=12)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", fontsize=9.5, frameon=True)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG/"figR3_roc_main.png"); plt.close(fig)
    print(f"[FIG] {FIG/'figR3_roc_main.png'}")

if __name__ == "__main__":
    main()
