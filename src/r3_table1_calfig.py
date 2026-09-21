# -*- coding: utf-8 -*-
"""R3 — Table 1 rework data + full-model calibration figure with CI bands.

Table 1 (editor minor comments): continuous -> median (IQR); p-values -> SMD;
laboratory variables split to a supplementary table.
Outputs: results/r3_table1_main.csv, results/r3_table1_labs.csv
Figure: figures/figR3_cal_full.png (locked full model, decile CI bands)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import brier_score_loss
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import cal_metrics

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400, "savefig.bbox": "tight"})

def smd_cont(a, b):
    a, b = a.dropna(), b.dropna()
    s = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / s if s > 0 else 0.0

def smd_bin(p1, p2):
    v = (p1*(1-p1) + p2*(1-p2)) / 2
    return abs(p1 - p2) / np.sqrt(v) if v > 0 else 0.0

def fmt_med(s):
    v = s.dropna()
    return f"{v.median():.1f} ({v.quantile(.25):.1f}–{v.quantile(.75):.1f})"

def fmt_n(s):
    n = int(s.sum()); return f"{n} ({n/len(s)*100:.1f})"

CONT = [("Age (years)", "age"), ("BMI (kg/m²)", "bmi")]
VITALS = [("Heart rate, min (beats/min)", "heart_rate_min"), ("Heart rate, max (beats/min)", "heart_rate_max"),
          ("SBP, min (mmHg)", "sbp_min"), ("SBP, max (mmHg)", "sbp_max"),
          ("DBP, min (mmHg)", "dbp_min"), ("DBP, max (mmHg)", "dbp_max"),
          ("MBP, min (mmHg)", "mbp_min"), ("MBP, max (mmHg)", "mbp_max"),
          ("Respiratory rate, min (breaths/min)", "respiratory_rate_min"),
          ("Respiratory rate, max (breaths/min)", "respiratory_rate_max"),
          ("Temperature, min (°C)", "temperature_min"), ("Temperature, max (°C)", "temperature_max"),
          ("SpO₂, min (%)", "spo2_min"), ("SpO₂, max (%)", "spo2_max")]
LABS = [("BUN, min (mg/dL)", "bun_min"), ("BUN, max (mg/dL)", "bun_max"),
        ("Creatinine, min (mg/dL)", "creatinine_min"), ("Creatinine, max (mg/dL)", "creatinine_max"),
        ("WBC, min (10³/µL)", "wbc_min"), ("WBC, max (10³/µL)", "wbc_max"),
        ("Hemoglobin, min (g/dL)", "hemoglobin_min"), ("Hemoglobin, max (g/dL)", "hemoglobin_max"),
        ("Platelets, min (10³/µL)", "platelet_min"), ("Platelets, max (10³/µL)", "platelet_max"),
        ("Sodium, min (mEq/L)", "sodium_min"), ("Sodium, max (mEq/L)", "sodium_max"),
        ("Potassium, min (mEq/L)", "potassium_min"), ("Potassium, max (mEq/L)", "potassium_max"),
        ("Chloride, min (mEq/L)", "chloride_min"), ("Chloride, max (mEq/L)", "chloride_max"),
        ("Bicarbonate, min (mEq/L)", "bicarbonate_min"), ("Bicarbonate, max (mEq/L)", "bicarbonate_max"),
        ("Calcium, min (mg/dL)", "calcium_min"), ("Calcium, max (mg/dL)", "calcium_max"),
        ("Glucose, min (mg/dL)", "glucose_min"), ("Glucose, max (mg/dL)", "glucose_max")]
BINS = [("CRRT", "crrt"), ("Invasive mechanical ventilation", "invasive_mechanical_ventilation"),
        ("Norepinephrine use", "norepinephrine"), ("Vasopressin use", "vasopressin"),
        ("Epinephrine use", "epinephrine")]

def build_rows(df, T, sev_label):
    e = df[df[T] == 1]; ne = df[df[T] == 0]
    rows = []
    def hdr(name): rows.append((name, "", "", ""))
    def cont_row(label, col, dfset=None):
        a, b = e[col], ne[col]
        rows.append((label, fmt_med(a), fmt_med(b), f"{smd_cont(a,b):.3f}"))
    def bin_row(label, col, indent="  "):
        p1, p2 = e[col].mean(), ne[col].mean()
        rows.append((label, fmt_n(e[col]), fmt_n(ne[col]), f"{smd_bin(p1,p2):.3f}"))
    hdr("Demographics")
    cont_row("Age (years)", "age")
    rows.append(("Sex (n, %)", "", "", ""))
    bin_row("  Female", "female") if "female" in df.columns else None
    if "female" not in df.columns:
        df["female"] = 1 - df["male"]; e = df[df[T] == 1]; ne = df[df[T] == 0]
        bin_row("  Female", "female"); bin_row("  Male", "male")
    else:
        bin_row("  Male", "male")
    rows.append(("Race (n, %)", "", "", ""))
    for lab, col in [("  White", "race_white"), ("  Black", "race_black"), ("  Hispanic", "race_hispanic"),
                     ("  Asian", "race_asian"), ("  Other", "race_others")]:
        bin_row(lab, col)
    cont_row("BMI (kg/m²)", "bmi")
    rows.append(("Admission type (n, %)", "", "", ""))
    for lab, col in [("  Emergency", "admission_emergency"), ("  Elective", "admission_elective"),
                     ("  Other", "admission_other")]:
        bin_row(lab, col)
    hdr("Treatments (n, %)")
    for lab, col in BINS: bin_row("  " + lab, col)
    hdr("Scores")
    rows.append(("CCI (n, %)", "", "", ""))
    cci = df.cci
    for lab, cond in [("  0", cci == 0), ("  1", cci == 1), ("  2", cci == 2), ("  ≥ 3", cci >= 3)]:
        p1 = cond[df[T] == 1].mean(); p2 = cond[df[T] == 0].mean()
        n1 = int(cond[df[T] == 1].sum()); n2 = int(cond[df[T] == 0].sum())
        rows.append((lab, f"{n1} ({p1*100:.1f})", f"{n2} ({p2*100:.1f})", f"{smd_bin(p1,p2):.3f}"))
    miss = cci.isna()
    p1 = miss[df[T] == 1].mean(); p2 = miss[df[T] == 0].mean()
    rows.append(("  Missing", f"{int(miss[df[T]==1].sum())} ({p1*100:.1f})",
                 f"{int(miss[df[T]==0].sum())} ({p2*100:.1f})",
                 f"{smd_bin(p1,p2):.3f}" if miss.any() else "—"))
    cont_row(sev_label, "apsiii")
    cont_row("GCS score", "gcs_score")
    cont_row("24-hour urine output (mL)", "urine_output")
    hdr("Vital signs")
    for lab, col in VITALS: cont_row(lab, col)
    lab_rows = []
    for lab, col in LABS:
        a, b = e[col], ne[col]
        lab_rows.append((lab, fmt_med(a), fmt_med(b), f"{smd_cont(a,b):.3f}"))
    return rows, lab_rows

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    mrows, mlabs = build_rows(mimic.copy(), T, "APS III")
    erows, elabs = build_rows(ea.copy(), T, "APACHE-IVa score")
    norm = lambda rows: ["SEV" if r[0] in ("APS III", "APACHE-IVa score") else r[0] for r in rows]
    assert norm(mrows) == norm(erows), "row label mismatch"
    main_df = pd.DataFrame({
        "Characteristic": ["APS III / APACHE-IVa score*" if r[0] == "APS III" else r[0] for r in mrows],
        "MIMIC_met": [r[1] for r in mrows], "MIMIC_notmet": [r[2] for r in mrows],
        "MIMIC_SMD": [r[3] for r in mrows],
        "eICU_met": [r[1] for r in erows], "eICU_notmet": [r[2] for r in erows],
        "eICU_SMD": [r[3] for r in erows]})
    labs_df = pd.DataFrame({
        "Characteristic": [r[0] for r in mlabs],
        "MIMIC_met": [r[1] for r in mlabs], "MIMIC_notmet": [r[2] for r in mlabs],
        "MIMIC_SMD": [r[3] for r in mlabs],
        "eICU_met": [r[1] for r in elabs], "eICU_notmet": [r[2] for r in elabs],
        "eICU_SMD": [r[3] for r in elabs]})
    main_df.to_csv(OUT/"r3_table1_main.csv", index=False)
    labs_df.to_csv(OUT/"r3_table1_labs.csv", index=False)
    print(f"[SAVE] r3_table1_main.csv ({len(main_df)} rows), r3_table1_labs.csv ({len(labs_df)} rows)")
    print(main_df.head(12).to_string(index=False))

    # ---- full-model calibration figure with decile CI bands ----
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    mfull = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[common], dev[T])
    def cal_bins(y, p, nb=10):
        df = pd.DataFrame({"y": y, "p": p}); df["bin"] = pd.qcut(df.p, nb, labels=False, duplicates="drop")
        g = df.groupby("bin").agg(mp=("p", "mean"), ob=("y", "mean"), n=("y", "size"))
        se = np.sqrt(g.ob*(1-g.ob)/g.n)
        return g.assign(lo=(g.ob-1.96*se).clip(0), hi=(g.ob+1.96*se).clip(upper=1))
    panels = [("(a) Temporal internal test", itest), ("(b) Primary eICU external cohort", ea),
              ("(c) Sepsis-3 sensitivity cohort", ep)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (nm, d) in zip(axes, panels):
        p = mfull.predict_proba(d[common])[:, 1]; y = d[T].values
        g = cal_bins(y, p); s, i = cal_metrics(y, p)
        ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, label="Perfect calibration")
        ax.errorbar(g.mp, g.ob, yerr=[g.ob-g.lo, g.hi-g.ob], fmt="-o", color="#2166ac",
                    ms=5, lw=1.5, capsize=2.5, label="LightGBM (full model)")
        ax.text(0.03, 0.965, f"Slope {s:.3f}\nIntercept {i:.3f}\nBrier {brier_score_loss(y,p):.3f}",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted probability")
        ax.set_title(nm, fontsize=11); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed event frequency"); axes[1].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG/"figR3_cal_full.png"); plt.close(fig)
    print(f"[FIG] {FIG/'figR3_cal_full.png'}")

if __name__ == "__main__":
    main()
