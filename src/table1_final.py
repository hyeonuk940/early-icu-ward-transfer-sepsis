# -*- coding: utf-8 -*-
"""Final Table 1 for the locked analysis: MIMIC-IV landmark cohort and the eICU
primary cohort side by side, each split by the primary composite outcome.

Conventions follow Methods 2.8 and reproduce fill_table1.py exactly:
  continuous  mean +/- SD, Welch's t-test
  categorical n (%),       chi-squared test
Percentages are out of the whole outcome group, so category blocks with missing
values (eICU CCI) do not sum to the group total; explicit missing rows are added.

Cohort labels: primary = APACHE sepsis admission
diagnosis, n=13,384, 200 hospitals.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.stats as st
from pathlib import Path
import data_prep as dp

OUT = Path(__file__).resolve().parents[1] / "results"
CONT, BINV = "cont", "bin"

ROWS = [
    ("Age, years", "age", CONT, None),
    ("Female", "male", BINV, 0),
    ("Male", "male", BINV, 1),
    ("Race: White", "race_white", BINV, 1),
    ("Race: Black", "race_black", BINV, 1),
    ("Race: Hispanic", "race_hispanic", BINV, 1),
    ("Race: Asian", "race_asian", BINV, 1),
    ("Race: Other", "race_others", BINV, 1),
    ("Body mass index, kg/m2", "bmi", CONT, None),
    ("Admission: emergency", "admission_emergency", BINV, 1),
    ("Admission: elective", "admission_elective", BINV, 1),
    ("Admission: other", "admission_other", BINV, 1),
    ("CRRT", "crrt", BINV, 1),
    ("Invasive mechanical ventilation", "invasive_mechanical_ventilation", BINV, 1),
    ("Norepinephrine", "norepinephrine", BINV, 1),
    ("Vasopressin", "vasopressin", BINV, 1),
    ("Epinephrine", "epinephrine", BINV, 1),
    ("Charlson index = 0", "cci", BINV, ("eq", 0)),
    ("Charlson index = 1", "cci", BINV, ("eq", 1)),
    ("Charlson index = 2", "cci", BINV, ("eq", 2)),
    ("Charlson index >= 3", "cci", BINV, ("ge", 3)),
    ("Charlson index, missing", "cci", BINV, ("na", None)),
    ("APS III / APACHE-IVa score", "apsiii", CONT, None),
    ("Glasgow Coma Scale", "gcs_score", CONT, None),
    ("Urine output, mL/24 h", "urine_output", CONT, None),
]
for base, label in [("heart_rate", "Heart rate, /min"), ("sbp", "Systolic BP, mmHg"),
                    ("dbp", "Diastolic BP, mmHg"), ("mbp", "Mean BP, mmHg"),
                    ("respiratory_rate", "Respiratory rate, /min"),
                    ("temperature", "Temperature, C"), ("spo2", "SpO2, %"),
                    ("bun", "BUN, mg/dL"), ("creatinine", "Creatinine, mg/dL"),
                    ("wbc", "WBC, K/uL"), ("hemoglobin", "Hemoglobin, g/dL"),
                    ("platelet", "Platelet, K/uL"), ("sodium", "Sodium, mmol/L"),
                    ("potassium", "Potassium, mmol/L"), ("chloride", "Chloride, mmol/L"),
                    ("bicarbonate", "Bicarbonate, mmol/L"), ("glucose", "Glucose, mg/dL"),
                    ("calcium", "Calcium, mg/dL")]:
    ROWS += [(f"{label}, min", f"{base}_min", CONT, None),
             (f"{label}, max", f"{base}_max", CONT, None)]


def mask_for(df, var, cond):
    if cond == ("na", None):
        return df[var].isna()
    if isinstance(cond, tuple):
        return (df[var] == cond[1]) if cond[0] == "eq" else (df[var] >= cond[1])
    return df[var] == cond


def cells(df, var, kind, cond, T):
    ev, ne = df[df[T] == 1], df[df[T] == 0]
    if var not in df.columns:
        return "—", "—", np.nan
    if kind == CONT:
        a, b = ev[var].dropna(), ne[var].dropna()
        if len(a) < 2 or len(b) < 2:
            return "—", "—", np.nan
        p = st.ttest_ind(a, b, equal_var=False).pvalue
        return f"{a.mean():.1f} ± {a.std():.1f}", f"{b.mean():.1f} ± {b.std():.1f}", p
    me, mne = mask_for(ev, var, cond), mask_for(ne, var, cond)
    ct = np.array([[me.sum(), (~me).sum()], [mne.sum(), (~mne).sum()]])
    try:
        p = st.chi2_contingency(ct)[1]
    except Exception:
        p = np.nan
    return (f"{int(me.sum())} ({me.mean()*100:.1f})",
            f"{int(mne.sum())} ({mne.mean()*100:.1f})", p)


def pfmt(p):
    if p != p:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common)
    eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)

    hdr = (f"MIMIC-IV: safe transfer (n={int(mimic[T].sum()):,})",
           f"MIMIC-IV: no safe transfer (n={int((mimic[T]==0).sum()):,})", "p",
           f"eICU primary: safe transfer (n={int(ea[T].sum()):,})",
           f"eICU primary: no safe transfer (n={int((ea[T]==0).sum()):,})", "p")

    rows = []
    for label, var, kind, cond in ROWS:
        me, mne, mp = cells(mimic, var, kind, cond, T)
        ee, ene, ep = cells(ea, var, kind, cond, T)
        rows.append([label, me, mne, pfmt(mp), ee, ene, pfmt(ep)])

    df = pd.DataFrame(rows, columns=["Variable", *hdr])
    df.to_csv(OUT / "Table1.tsv", sep="\t", index=False)
    print(df.to_string(index=False))
    print()
    print("[SAVE]", OUT / "Table1.tsv")
    print(f"MIMIC-IV  n={len(mimic):,}  events={int(mimic[T].sum()):,} "
          f"({mimic[T].mean()*100:.1f}%)")
    print(f"eICU primary n={len(ea):,}  events={int(ea[T].sum()):,} "
          f"({ea[T].mean()*100:.1f}%)")
    print("Continuous: mean ± SD, Welch's t-test. Categorical: n (%), chi-squared test.")
    print("Percentages are out of the whole outcome group; see the explicit missing row for CCI.")


if __name__ == "__main__":
    main()
