# -*- coding: utf-8 -*-
"""R3 — three editor items in one pass:

A) Cohort overlap (Major 6): cross-tab of APACHE diagnosis-based vs Sepsis-3
   sensitivity membership within the eICU landmark population + characteristics.
B) Hospital-level heterogeneity drivers (Major 9): merge eicu_crd.hospital
   (numbeds, teaching, region; read-only lookup) with per-site AUCs (full +
   transportable), Spearman correlations, grouped random-effects pools,
   per-site calibration intercept summary.
C) MICE upgrade (Major 8): chained-equations iterative imputation with posterior
   sampling, m=5 imputations fit on the development set only, per-imputation
   LightGBM, predictions pooled by averaging.
Outputs: r3_overlap.csv, r3_hospital_drivers.csv, r3_mice_m5.csv
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
def lgb(seed=SEED): return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **LOCKED)

def dl_pool_sub(sd):
    if len(sd) < 3: return None
    a = sd.auc.values.clip(1e-4, 1-1e-4); se = sd.se.values
    yv = np.log(a/(1-a)); v = (se/(a*(1-a)))**2
    w = 1/v; yf = np.sum(w*yv)/np.sum(w)
    Q = np.sum(w*(yv-yf)**2); k = len(yv); C = np.sum(w)-np.sum(w**2)/np.sum(w)
    tau2 = max(0, (Q-(k-1))/C)
    wr = 1/(v+tau2); yb = np.sum(wr*yv)/np.sum(wr); sp = np.sqrt(1/np.sum(wr))
    inv = lambda z: 1/(1+np.exp(-z))
    return f"{inv(yb):.3f} ({inv(yb-1.96*sp):.3f}-{inv(yb+1.96*sp):.3f})", k

def main():
    mimic, eicu, fm, fe, common = dp.load()
    eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY

    # ================= A) overlap =================
    print("=== A) cohort overlap (eICU landmark) ===")
    ct = pd.crosstab(eicu.sepsis_admitdx, eicu.sepsis3_primary, margins=True)
    print(ct)
    both = eicu[(eicu.sepsis_admitdx == 1) & (eicu.sepsis3_primary == 1)]
    apache_only = eicu[(eicu.sepsis_admitdx == 1) & (eicu.sepsis3_primary == 0)]
    s3_only = eicu[(eicu.sepsis_admitdx == 0) & (eicu.sepsis3_primary == 1)]
    rows = []
    for nm, g in [("APACHE & Sepsis-3 (both)", both), ("APACHE only", apache_only),
                  ("Sepsis-3 only", s3_only)]:
        rows.append(dict(group=nm, n=len(g), hospitals=g.hospitalid.nunique(),
                         age=round(g.age.mean(), 1), male_pct=round(g.male.mean()*100, 1),
                         apache=round(g.apsiii.mean(), 1), vent_pct=round(g.invasive_mechanical_ventilation.mean()*100, 1),
                         cci=round(g.cci.mean(), 1), outcome_pct=round(g[T].mean()*100, 1)))
        print(rows[-1])
    ov = len(both)
    print(f"Sepsis-3 cohort inside APACHE cohort: {ov}/{int((eicu.sepsis3_primary==1).sum())} "
          f"({ov/(eicu.sepsis3_primary==1).sum()*100:.1f}%) | of APACHE cohort: "
          f"{ov}/{int((eicu.sepsis_admitdx==1).sum())} ({ov/(eicu.sepsis_admitdx==1).sum()*100:.1f}%)")
    pd.DataFrame(rows).to_csv(OUT/"r3_overlap.csv", index=False)

    # ================= B) hospital drivers =================
    print("\n=== B) hospital-level drivers ===")
    c = psycopg2.connect(host=os.environ.get("SEPSIS_DB_HOST", "localhost"), dbname="eicu", user=os.environ.get("SEPSIS_DB_USER", "postgres"), password=os.environ["SEPSIS_DB_PASSWORD"])
    hosp = pd.read_sql("select hospitalid, numbedscategory, teachingstatus, region from eicu_crd.hospital", c)
    c.close()
    for model_tag, f in [("full", OUT/"hospital_level_auc.csv"), ("transportable", OUT/"r3_transportable_hospital.csv")]:
        sd = pd.read_csv(f).merge(hosp, on="hospitalid", how="left")
        rs_n = spearmanr(sd.n, sd.auc); rs_ev = spearmanr(sd.event_rate, sd.auc)
        print(f"[{model_tag}] Spearman AUC~n rho={rs_n.correlation:.2f} p={rs_n.pvalue:.3f} | "
              f"AUC~event_rate rho={rs_ev.correlation:.2f} p={rs_ev.pvalue:.3f}")
        grows = []
        for var in ["teachingstatus", "region", "numbedscategory"]:
            for lev, g in sd.groupby(sd[var].fillna("Unknown").astype(str)):
                pool = dl_pool_sub(g)
                if pool:
                    grows.append(dict(model=model_tag, variable=var, level=lev, k=pool[1],
                                      median_auc=round(g.auc.median(), 3), pooled_auc_ci=pool[0],
                                      median_event_rate=round(g.event_rate.median(), 3)))
        gdf = pd.DataFrame(grows)
        print(gdf.to_string(index=False))
        gdf.to_csv(OUT/f"r3_hospital_drivers_{model_tag}.csv", index=False)
        if "cal_intercept" in sd.columns:
            ci_ = sd.cal_intercept.dropna()
            print(f"[{model_tag}] per-site calibration intercept: median {ci_.median():+.3f} "
                  f"IQR {ci_.quantile(.25):+.3f}..{ci_.quantile(.75):+.3f} range {ci_.min():+.2f}..{ci_.max():+.2f}")

    # ================= C) MICE m=5 =================
    print("\n=== C) MICE m=5 (posterior sampling, dev-fit only) ===")
    mimic = dp.apply_physiologic_ranges(mimic, common)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    feats = common
    M = 5
    preds = {nm: [] for nm in ["internal", "eICU_primary", "sepsis3"]}
    for m_i in range(M):
        imp = IterativeImputer(random_state=SEED + m_i, max_iter=10, sample_posterior=True).fit(dev[feats])
        Xd = pd.DataFrame(imp.transform(dev[feats]), columns=feats)
        mdl = lgb(SEED + m_i).fit(Xd, dev[T])
        for nm, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
            Xe = pd.DataFrame(imp.transform(d[feats]), columns=feats)
            preds[nm].append(mdl.predict_proba(Xe)[:, 1])
        print(f"  imputation {m_i+1}/{M} done")
    rows = []
    for nm, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
        p = np.mean(preds[nm], axis=0); y = d[T].values
        auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
        per_imp = [roc_auc_score(y, q) for q in preds[nm]]
        rows.append(dict(cohort=nm, auc_pooled=round(auc, 3), ci=f"{lo:.3f}-{hi:.3f}",
                         auc_per_imputation=";".join(f"{a:.3f}" for a in per_imp)))
        print(f"[MICE m=5 {nm:13s}] pooled AUC {auc:.3f} ({lo:.3f}-{hi:.3f}) per-imp {per_imp}")
    pd.DataFrame(rows).to_csv(OUT/"r3_mice_m5.csv", index=False)
    print(f"[SAVE] r3_overlap.csv, r3_hospital_drivers_*.csv, r3_mice_m5.csv")

if __name__ == "__main__":
    main()
