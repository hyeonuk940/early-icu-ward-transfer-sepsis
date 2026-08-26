"""
Reanalysis data preparation & physiologic-range screening.
- Loads MIMIC (development) and eICU (external) landmark feature CSVs.
- Defines the 56-feature predictor set (aligned names across both DBs).
- Physiologic plausibility ranges: values outside -> set to NaN at the
  individual-value level (NO whole-record deletion).
  Ranges are FIXED clinical bounds (not data-derived) so no leakage.
- Missing values are RETAINED for native-missing modeling.
Run: python data_prep.py  (prints shapes, alignment, out-of-range counts)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
MIMIC = DATA / "mimic_landmark_features_v1.csv"
EICU = DATA / "eicu_landmark_features_v1.csv"

TARGET_PRIMARY = "primary_safe_transfer"
TARGET_SECONDARY = "secondary_transfer_72h"

# non-feature columns (ids, outcomes, cohort flags, bookkeeping)
NON_FEATURES = {
    "subject_id", "hadm_id", "stay_id", "patientunitstayid",
    "patienthealthsystemstayid", "hospitalid",
    "primary_safe_transfer", "secondary_transfer_72h",
    "sepsis3_primary", "sepsis_admitdx",
    "icu_readmit_7d", "death_7d_post", "left_icu_alive", "went_to_ward",
    "icu_duration_hours", "anchor_year_group",
}

# Physiologic plausibility ranges (fixed clinical bounds). Outside -> NaN.
# min/max share the base variable's bounds.
PHYS_RANGES = {
    "heart_rate": (10, 300), "sbp": (20, 300), "dbp": (5, 225), "mbp": (10, 250),
    "respiratory_rate": (0, 80), "temperature": (25, 45), "spo2": (30, 100),
    "bun": (0, 250), "creatinine": (0, 40), "wbc": (0, 500),
    "hemoglobin": (2, 25), "platelet": (0, 2000), "chloride": (60, 160),
    "sodium": (90, 200), "bicarbonate": (2, 60), "potassium": (1, 12),
    "glucose": (10, 2000), "calcium": (2, 20),
    "bmi": (8, 100), "apsiii": (0, 299), "gcs_score": (3, 15),
    "urine_output": (0, 20000), "cci": (0, 40), "age": (18, 120),
}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def apply_physiologic_ranges(df, cols, report=False):
    """Set individual values outside fixed physiologic bounds to NaN. No row drop."""
    df = df.copy()
    counts = {}
    for c in cols:
        base = c
        for suf in ("_min", "_max"):
            if c.endswith(suf):
                base = c[: -len(suf)]
                break
        if base in PHYS_RANGES:
            lo, hi = PHYS_RANGES[base]
            mask = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
            n = int(mask.sum())
            if n:
                counts[c] = n
                df.loc[mask, c] = np.nan
    if report:
        tot = sum(counts.values())
        print(f"  out-of-range values set to NaN: {tot} across {len(counts)} cols")
        for c, n in sorted(counts.items(), key=lambda x: -x[1])[:12]:
            print(f"    {c:24s} {n}")
    return df


def load():
    mimic = pd.read_csv(MIMIC)
    eicu = pd.read_csv(EICU)
    fm = feature_columns(mimic)
    fe = feature_columns(eicu)
    common = [c for c in fm if c in fe]
    return mimic, eicu, fm, fe, common


if __name__ == "__main__":
    mimic, eicu, fm, fe, common = load()
    print(f"MIMIC: {mimic.shape}  |  eICU: {eicu.shape}")
    print(f"MIMIC features: {len(fm)}  eICU features: {len(fe)}  common: {len(common)}")
    only_m = sorted(set(fm) - set(fe)); only_e = sorted(set(fe) - set(fm))
    print(f"  MIMIC-only features: {only_m}")
    print(f"  eICU-only  features: {only_e}")
    print(f"\nPrimary outcome rate  MIMIC={mimic[TARGET_PRIMARY].mean():.3f}  "
          f"eICU(all)={eicu[TARGET_PRIMARY].mean():.3f}")
    print(f"  eICU sepsis3_primary n={int(eicu.sepsis3_primary.sum())} "
          f"rate={eicu.loc[eicu.sepsis3_primary==1, TARGET_PRIMARY].mean():.3f}")
    print(f"  eICU alt(admitdx)   n={int(eicu.sepsis_admitdx.sum())} "
          f"rate={eicu.loc[eicu.sepsis_admitdx==1, TARGET_PRIMARY].mean():.3f}")
    print(f"  eICU hospitals (alt): {eicu.loc[eicu.sepsis_admitdx==1,'hospitalid'].nunique()}")

    print("\n[MIMIC] physiologic-range screening:")
    m2 = apply_physiologic_ranges(mimic, fm, report=True)
    print("[eICU] physiologic-range screening:")
    e2 = apply_physiologic_ranges(eicu, fe, report=True)

    # missingness after screening (native-missing will handle these)
    miss_m = m2[fm].isna().mean().sort_values(ascending=False)
    print("\n[MIMIC] top missingness after screening:")
    print(miss_m.head(6).round(3).to_string())
