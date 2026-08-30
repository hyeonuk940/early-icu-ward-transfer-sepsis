# -*- coding: utf-8 -*-
"""Distribution of demographics, predictors and outcome across the development set
and every evaluation cohort.

Table 1 contrasts outcome groups within a cohort. This table contrasts the cohorts
themselves, which is what a reader needs in order to judge how far each validation
population sits from the data the model was fitted on.

Standardised mean differences (SMD) are reported against the development set rather
than p-values: with tens of thousands of patients almost any difference reaches
significance, whereas the SMD measures how large it is. |SMD| > 0.10 is the usual
threshold for a non-negligible imbalance.

    continuous  SMD = (m1 - m2) / sqrt((sd1^2 + sd2^2) / 2)
    binary      SMD = (p1 - p2) / sqrt((p1(1-p1) + p2(1-p2)) / 2)

Cohort labels:
    primary      = eICU APACHE sepsis admission-diagnosis cohort (n=13,384, 200 hospitals)
    sensitivity  = eICU Sepsis-3 definition-concordant cohort   (n= 2,409,  84 hospitals)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import data_prep as dp

OUT = Path(__file__).resolve().parents[1] / "results"
CONT, BINV = "cont", "bin"

ROWS = [
    ("Primary outcome met", dp.TARGET_PRIMARY, BINV),
    ("Age, years", "age", CONT),
    ("Male", "male", BINV),
    ("Race: White", "race_white", BINV),
    ("Race: Black", "race_black", BINV),
    ("Race: Hispanic", "race_hispanic", BINV),
    ("Race: Asian", "race_asian", BINV),
    ("Race: Other", "race_others", BINV),
    ("Body mass index, kg/m2", "bmi", CONT),
    ("Admission: emergency", "admission_emergency", BINV),
    ("Admission: elective", "admission_elective", BINV),
    ("Admission: other", "admission_other", BINV),
    ("CRRT", "crrt", BINV),
    ("Invasive mechanical ventilation", "invasive_mechanical_ventilation", BINV),
    ("Norepinephrine", "norepinephrine", BINV),
    ("Vasopressin", "vasopressin", BINV),
    ("Epinephrine", "epinephrine", BINV),
    ("Charlson comorbidity index", "cci", CONT),
    ("APS III / APACHE-IVa score", "apsiii", CONT),
    ("Glasgow Coma Scale", "gcs_score", CONT),
    ("Urine output, mL/24 h", "urine_output", CONT),
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
    ROWS += [(f"{label}, min", f"{base}_min", CONT),
             (f"{label}, max", f"{base}_max", CONT)]


def cell(d, var, kind):
    x = d[var].dropna()
    if len(x) == 0:
        return "—", np.nan, np.nan
    if kind == CONT:
        return f"{x.mean():.1f} ± {x.std():.1f}", x.mean(), x.std()
    return f"{int(x.sum())} ({x.mean()*100:.1f})", x.mean(), np.nan


def smd(kind, m1, s1, m2, s2):
    if np.isnan(m1) or np.isnan(m2):
        return np.nan
    if kind == CONT:
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    else:
        pooled = np.sqrt((m1 * (1 - m1) + m2 * (1 - m2)) / 2)
    return np.nan if pooled == 0 else (m2 - m1) / pooled


def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common)
    eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}

    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    evals = [
        ("Temporal internal test (MIMIC-IV)", mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)),
        ("External: primary (eICU)", eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)),
        ("External: Sepsis-3 sensitivity (eICU)", eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)),
    ]

    rows = []
    for label, var, kind in ROWS:
        r = {"Variable": label}
        txt, m1, s1 = cell(dev, var, kind)
        r[f"Development (n={len(dev):,})"] = txt
        for name, d in evals:
            t2, m2, s2 = cell(d, var, kind)
            r[f"{name} (n={len(d):,})"] = t2
            r[f"SMD: {name.split(' (')[0]}"] = round(smd(kind, m1, s1, m2, s2), 3)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cohort_comparison.tsv", sep="\t", index=False)

    smd_cols = [c for c in df.columns if c.startswith("SMD")]
    w = max(len(v) for v in df.Variable)
    print(f"{'Variable':{w}s}" + "".join(f"{c.replace('SMD: ',''):>26s}" for c in smd_cols))
    print("-" * (w + 26 * len(smd_cols)))
    for _, r in df.iterrows():
        flag = "  <<" if any(abs(r[c]) > 0.10 for c in smd_cols if pd.notna(r[c])) else ""
        print(f"{r.Variable:{w}s}" + "".join(f"{r[c]:26.3f}" if pd.notna(r[c]) else f"{'—':>26s}"
                                            for c in smd_cols) + flag)

    print()
    for c in smd_cols:
        big = df[df[c].abs() > 0.10]
        top = big.reindex(big[c].abs().sort_values(ascending=False).index).head(6)
        print(f"  {c}:  {len(big)}/{len(df)} variables with |SMD| > 0.10")
        for _, r in top.iterrows():
            print(f"      {r.Variable:34s} {r[c]:+.3f}")
    print("\n  '<<' marks a variable exceeding |SMD| 0.10 in at least one evaluation cohort.")
    print("[SAVE]", OUT / "cohort_comparison.tsv")


if __name__ == "__main__":
    main()
