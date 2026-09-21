# -*- coding: utf-8 -*-
"""R3 — mutually exclusive 72-h outcome states + alternative-outcome sensitivity
(editor Major 2). Small read-only SQL pulls of existing cohort tables (no re-extraction).

States (landmark cohort, first ICU exit within 72 h of admission):
  1 ward_transfer_safe      : ICU->ward transfer, no ICU readmission or death <=7 d
  2 ward_transfer_event     : ICU->ward transfer followed by readmission/death <=7 d
  3 direct_discharge        : left hospital directly from ICU <=72 h (alive)
  4 other_disposition       : other non-ward disposition <=72 h (alive)
  5 icu_death_72h           : died in ICU <=72 h
  6 still_in_icu            : still in the ICU at 72 h
Alternative outcome definitions (LightGBM, locked HP, dev-refit, own labels):
  A ward transfer <=72 h irrespective of subsequent events
  B ward transfer without ICU readmission (death allowed)
  C ward transfer without death (readmission allowed)
  D ward transfer OR direct hospital discharge <=72 h (alive)
Outputs: results/r3_outcome_states.csv, results/r3_alt_outcomes.csv
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
def lgb(): return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED)

def pull_mimic():
    c = psycopg2.connect(host=os.environ.get("SEPSIS_DB_HOST", "localhost"), dbname="mimic4", user=os.environ.get("SEPSIS_DB_USER", "postgres"), password=os.environ["SEPSIS_DB_PASSWORD"])
    q = """select subject_id, icu_duration_hours, left_icu_alive, went_to_ward,
                  icu_readmit_7d, death_7d_post, dest_careunit, dest_eventtype
           from sepsis_work.mimic_cohort where in_landmark"""
    d = pd.read_sql(q, c); c.close(); return d

def pull_eicu():
    c = psycopg2.connect(host=os.environ.get("SEPSIS_DB_HOST", "localhost"), dbname="eicu", user=os.environ.get("SEPSIS_DB_USER", "postgres"), password=os.environ["SEPSIS_DB_PASSWORD"])
    q = """select patientunitstayid, icu_duration_hours, left_icu_alive, went_to_ward,
                  icu_readmit_7d, death_7d_post, unitdischargelocation, unitdischargestatus,
                  sepsis_admitdx, sepsis3_primary
           from sepsis_work.eicu_cohort
           where in_icu_at_24h=1 and sepsis_admitdx=1
             and patientunitstayid in (select patientunitstayid from sepsis_work.eicu_features
                                       where sepsis_admitdx=1)"""
    d = pd.read_sql(q, c); c.close(); return d

def states_mimic(d):
    exit72 = d.icu_duration_hours <= 72
    ward = d.went_to_ward.fillna(False).astype(bool)
    ev = (d.icu_readmit_7d.fillna(False).astype(bool) | d.death_7d_post.fillna(False).astype(bool))
    alive = d.left_icu_alive.fillna(False).astype(bool)
    disch = d.dest_eventtype.fillna("").str.lower().eq("discharge") | \
            d.dest_careunit.fillna("").str.contains("Discharge", case=False)
    s = pd.Series("still_in_icu", index=d.index)
    s[exit72 & ~alive] = "icu_death_72h"
    s[exit72 & alive & ward & ~ev] = "ward_transfer_safe"
    s[exit72 & alive & ward & ev] = "ward_transfer_event"
    s[exit72 & alive & ~ward & disch] = "direct_discharge"
    s[exit72 & alive & ~ward & ~disch] = "other_disposition"
    return s

def states_eicu(d):
    exit72 = d.icu_duration_hours <= 72
    ward = d.went_to_ward.fillna(False).astype(bool)
    ev = (d.icu_readmit_7d.fillna(False).astype(bool) | d.death_7d_post.fillna(False).astype(bool))
    alive = d.left_icu_alive.fillna(False).astype(bool)
    home = d.unitdischargelocation.fillna("").str.contains("Home|Death", case=False)
    disch = d.unitdischargelocation.fillna("").str.contains("Home", case=False)
    s = pd.Series("still_in_icu", index=d.index)
    s[exit72 & ~alive] = "icu_death_72h"
    s[exit72 & alive & ward & ~ev] = "ward_transfer_safe"
    s[exit72 & alive & ward & ev] = "ward_transfer_event"
    s[exit72 & alive & ~ward & disch] = "direct_discharge"
    s[exit72 & alive & ~ward & ~disch] = "other_disposition"
    return s

def main():
    dm = pull_mimic(); de = pull_eicu()
    sm = states_mimic(dm); se = states_eicu(de)
    order = ["ward_transfer_safe", "ward_transfer_event", "direct_discharge",
             "other_disposition", "icu_death_72h", "still_in_icu"]
    tab = pd.DataFrame({
        "MIMIC_n": sm.value_counts().reindex(order).fillna(0).astype(int),
        "MIMIC_pct": (sm.value_counts(normalize=True).reindex(order).fillna(0)*100).round(1),
        "eICU_primary_n": se.value_counts().reindex(order).fillna(0).astype(int),
        "eICU_primary_pct": (se.value_counts(normalize=True).reindex(order).fillna(0)*100).round(1)})
    print(tab)
    print("MIMIC total", len(dm), "| eICU primary total", len(de))
    tab.to_csv(OUT/"r3_outcome_states.csv")

    # ---- alternative outcome definitions ----
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}
    # merge SQL states into feature frames
    mimic = mimic.merge(dm[["subject_id"]].assign(state=sm.values), on="subject_id", how="left")
    eicu_p = eicu[eicu.sepsis_admitdx == 1].merge(
        de[["patientunitstayid"]].assign(state=se.values), on="patientunitstayid", how="left")
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    eicu_p = eicu_p.reset_index(drop=True)
    feats_full = common; feats_t = [c for c in common if c != "apsiii"]

    def labels(d):
        # ward transfer WITHIN 72 h = the two ward states from the mutually exclusive
        # state classification (editor: "transfer to a ward by 72 hours")
        ward72 = d.state.isin(["ward_transfer_safe", "ward_transfer_event"])
        return {
          "A_ward_transfer_any": ward72.astype(int),
          "B_transfer_no_readmit": (ward72 & ~d.icu_readmit_7d.fillna(0).astype(bool)).astype(int),
          "C_transfer_no_death": (ward72 & ~d.death_7d_post.fillna(0).astype(bool)).astype(int),
          "D_transfer_or_discharge": (ward72 | d.state.eq("direct_discharge")).astype(int),
          "primary (reference)": d[dp.TARGET_PRIMARY].astype(int)}
    rows = []
    Ldev, Lit, Lea = labels(dev), labels(itest), labels(eicu_p)
    for oname in Ldev:
        for mname, feats in [("full", feats_full), ("transportable", feats_t)]:
            m = lgb().fit(dev[feats], Ldev[oname])
            r = dict(outcome=oname, model=mname,
                     dev_rate=round(Ldev[oname].mean()*100, 1))
            for cn, d, L in [("internal", itest, Lit), ("eICU_primary", eicu_p, Lea)]:
                y = L[oname].values; p = m.predict_proba(d[feats])[:, 1]
                auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
                r[f"{cn}_auc"] = round(auc, 3); r[f"{cn}_ci"] = f"{lo:.3f}-{hi:.3f}"
                r[f"{cn}_rate"] = round(y.mean()*100, 1)
            rows.append(r)
            print(f"[alt {oname:24s} {mname:13s}] internal {r['internal_auc']} ({r['internal_ci']}) "
                  f"eICU {r['eICU_primary_auc']} ({r['eICU_primary_ci']})")
    pd.DataFrame(rows).to_csv(OUT/"r3_alt_outcomes.csv", index=False)
    print(f"[SAVE] {OUT/'r3_outcome_states.csv'} , r3_alt_outcomes.csv")

if __name__ == "__main__":
    main()
