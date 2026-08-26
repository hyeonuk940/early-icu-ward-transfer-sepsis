# -*- coding: utf-8 -*-
"""Variable-level missingness for every predictor, in all three analysis cohorts.

Reported after physiologic-range screening, so the percentages match the values
underlying Table 1: measurements outside the fixed clinical bounds in data_prep
are set to missing before counting.

Every predictor with at least one missing value is listed. Sorting is by the
largest rate across cohorts rather than by one cohort, because several variables
are essentially complete in MIMIC-IV yet substantially missing in eICU-CRD
(Glasgow Coma Scale, SpO2, the severity score, the Charlson index); ordering by
MIMIC-IV alone pushes those to the bottom of the table.

Cohort labels:
    primary      = eICU APACHE sepsis admission-diagnosis cohort (n=13,384, 200 hospitals)
    sensitivity  = eICU Sepsis-3 definition-concordant cohort   (n= 2,409,  84 hospitals)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
import data_prep as dp

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common)
    eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY

    cohorts = [
        ("MIMIC-IV landmark", mimic),
        ("eICU primary", eicu[eicu.sepsis_admitdx == 1]),
        ("eICU Sepsis-3 sensitivity", eicu[eicu.sepsis3_primary == 1]),
    ]
    rates = {name: d[common].isna().mean() for name, d in cohorts}

    rows = []
    for v in common:
        r = {name: rates[name][v] * 100 for name, _ in cohorts}
        if max(r.values()) > 0:
            rows.append(dict(variable=v, **{k: round(x, 1) for k, x in r.items()}))
    df = pd.DataFrame(rows)
    df["_max"] = df[[c for c, _ in cohorts]].max(axis=1)
    df = df.sort_values("_max", ascending=False).drop(columns="_max").reset_index(drop=True)
    df.to_csv(OUT / "missingness_by_variable.csv", index=False)

    w = max(len(v) for v in df.variable)
    print(f"Cohort sizes:  " + " · ".join(f"{n} {len(d):,}" for n, d in cohorts))
    print(f"Predictors: {len(common)}   with any missing value: {len(df)}\n")
    print(f"  {'variable':{w}s}" + "".join(f"{n:>28s}" for n, _ in cohorts))
    print("  " + "-" * (w + 28 * len(cohorts)))
    for _, r in df.iterrows():
        print(f"  {r.variable:{w}s}" + "".join(f"{r[n]:27.1f}%" for n, _ in cohorts))

    print()
    for name, d in cohorts:
        s = rates[name]
        top = s.sort_values(ascending=False).head(1)
        print(f"  {name:26s} complete predictors {int((s == 0).sum()):2d}/{len(common)}   "
              f"highest: {top.index[0]} {top.iloc[0]*100:.1f}%")

    # differential missingness by outcome, where it is large enough to matter
    print("\n  Differential missingness by outcome (variables differing by >5 points):")
    any_flag = False
    for name, d in cohorts:
        ev, ne = d[d[T] == 1], d[d[T] == 0]
        for v in common:
            a, b = ev[v].isna().mean() * 100, ne[v].isna().mean() * 100
            if abs(a - b) > 5:
                print(f"    {name:26s} {v:22s} outcome {a:5.1f}%  vs  no outcome {b:5.1f}%")
                any_flag = True
    if not any_flag:
        print("    none")

    print("\n[SAVE]", OUT / "missingness_by_variable.csv")


if __name__ == "__main__":
    main()
