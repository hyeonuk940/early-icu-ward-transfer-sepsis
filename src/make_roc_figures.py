# -*- coding: utf-8 -*-
"""Two-panel ROC (internal + external) at 600 dpi in Arial, for both eICU cohorts.

Panel labels:
  - the severity comparator is named for what each database actually contains:
    MIMIC-IV holds a true APS III, eICU-CRD holds an APACHE-IVa score in the
    same column, so the eICU panel is labelled APACHE-IVa. A single-predictor
    logistic model is a monotone transform of its input, so the curve and AUC
    are identical to those of the raw score.
  - primary      = eICU APACHE sepsis admission-diagnosis cohort (n=13,384, 200 hospitals)
    sensitivity  = eICU Sepsis-3 definition-concordant cohort   (n= 2,409,  84 hospitals)

Model and preprocessing are identical to make_figures.py.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(exist_ok=True)
BLUE, RED = "#2166ac", "#b2182b"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    mimic, eicu, fm, fe, common = dp.load(); feats = list(common)
    mimic = dp.apply_physiologic_ranges(mimic, feats)
    eicu = dp.apply_physiologic_ranges(eicu, feats)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    T = dp.TARGET_PRIMARY; med = dev["apsiii"].median()

    lg = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[feats], dev[T])
    sev = LogisticRegression().fit(dev[["apsiii"]].fillna(med), dev[T])

    def curves(d):
        return (d[T].values,
                lg.predict_proba(d[feats])[:, 1],
                sev.predict_proba(d[["apsiii"]].fillna(med))[:, 1])

    def panel(ax, title, d, sev_label):
        y, pm, ps = curves(d)
        for lab, p, c in [(f"LightGBM (AUC {roc_auc_score(y, pm):.3f})", pm, BLUE),
                          (f"{sev_label} (AUC {roc_auc_score(y, ps):.3f})", ps, RED)]:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=c, lw=2, label=lab)
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(title); ax.legend(loc="lower right", frameon=True)
        ax.grid(alpha=.3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    variants = [
        ("roc_primary.png",
         eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True), "Primary cohort"),
        ("roc_sepsis3_sensitivity.png",
         eicu[eicu.sepsis3_primary == 1].reset_index(drop=True), "Sepsis-3 sensitivity cohort"),
    ]
    for fname, ext, ext_name in variants:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.3))
        panel(ax[0], f"(a) Internal validation (MIMIC-IV)\nTemporal test set (n = {len(itest):,})",
              itest, "APS III alone")
        panel(ax[1], f"(b) External validation (eICU-CRD)\n{ext_name} (n = {len(ext):,})",
              ext, "APACHE-IVa alone")
        plt.tight_layout()
        plt.savefig(OUT / fname); plt.close()
        y, pm, ps = curves(ext)
        print(f"[FIG] {fname:46s} LightGBM {roc_auc_score(y, pm):.3f}  "
              f"APACHE-IVa {roc_auc_score(y, ps):.3f}  n={len(ext):,}")

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
