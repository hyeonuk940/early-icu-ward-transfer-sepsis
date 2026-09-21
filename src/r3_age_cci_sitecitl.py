# -*- coding: utf-8 -*-
"""External review round 2 — three verifications/sensitivity analyses.

1) Anchor-year offsets + AGE definition check (reviewer B):
   - offset = year(icu_intime) - anchor_year distribution -> results/r3_anchor_offset.csv
   - verify whether the stored age equals anchor_age (uncorrected) and quantify
     the corrected age (anchor_age + offset, capped at 91 per the >89 rule).
2) Age-corrected sensitivity (reviewer B): refit full + transportable models on the
   development set with corrected MIMIC age; evaluate internal/external AUC + slope.
   -> results/r3_age_sensitivity.csv
3) No-CCI transportable sensitivity (reviewer A): 54-feature model (no severity
   score, no CCI), locked HP, own development Youden threshold; discrimination,
   calibration, Brier, locked-threshold metrics. -> results/r3_transportable_nocci.csv
4) Site-level CITL (reviewer E): per eligible hospital, calibration-in-the-large
   (logistic intercept with slope fixed at 1 via offset) + O/E for full and
   transportable models. -> results/r3_site_citl.csv
DB access (read-only) via env vars SEPSIS_DB_HOST / SEPSIS_DB_USER / SEPSIS_DB_PASSWORD.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd, psycopg2
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci, cal_metrics, youden_threshold

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
def lgb(): return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED)

def conn(db):
    return psycopg2.connect(host=os.environ.get("SEPSIS_DB_HOST", "localhost"), dbname=db,
                            user=os.environ.get("SEPSIS_DB_USER", "postgres"),
                            password=os.environ["SEPSIS_DB_PASSWORD"])

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    test = {"2017 - 2019", "2020 - 2022"}
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)

    # ---------- 1) anchor offsets + age check ----------
    c = conn("mimic4")
    off = pd.read_sql("""
        select m.subject_id, m.age as stored_age, p.anchor_age,
               (extract(year from m.icu_intime) - p.anchor_year)::int as offset_yr
        from sepsis_work.mimic_cohort m join mimiciv_hosp.patients p using(subject_id)
        where m.in_landmark""", c)
    c.close()
    dist = off.groupby("offset_yr").size().rename("n").reset_index()
    dist["pct"] = (dist.n / dist.n.sum() * 100).round(2)
    dist.to_csv(OUT/"r3_anchor_offset.csv", index=False)
    same = (off.stored_age == off.anchor_age).mean()
    off["age_corrected"] = np.minimum(off.anchor_age + off.offset_yr, 91)
    diff = off.age_corrected - off.stored_age
    print(f"[age] stored_age == anchor_age in {same*100:.1f}% of patients")
    print(f"[age] corrected - stored: mean {diff.mean():.2f} y, median {diff.median():.0f}, "
          f">0 in {(diff>0).mean()*100:.1f}%, max {diff.max():.0f}")
    print(f"[age] stored mean {off.stored_age.mean():.1f} -> corrected mean {off.age_corrected.mean():.1f}")
    print(f"[age] offset: 0 in {(off.offset_yr==0).mean()*100:.1f}%, <=2 in {(off.offset_yr<=2).mean()*100:.1f}%")

    # ---------- 2) age-corrected sensitivity ----------
    m2 = mimic.merge(off[["subject_id", "age_corrected"]], on="subject_id", how="left")
    m2["age"] = m2["age_corrected"]
    dev0 = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    it0 = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    dev2 = m2[~m2.anchor_year_group.isin(test)].reset_index(drop=True)
    it2 = m2[m2.anchor_year_group.isin(test)].reset_index(drop=True)
    rows = []
    for tag, feats in [("full", common), ("transportable", [c_ for c_ in common if c_ != "apsiii"])]:
        m_orig = lgb().fit(dev0[feats], dev0[T])
        m_corr = lgb().fit(dev2[feats], dev2[T])
        for cname, d_o, d_c in [("internal", it0, it2), ("eICU_primary", ea, ea), ("sepsis3", ep, ep)]:
            po = m_orig.predict_proba(d_o[feats])[:, 1]
            pc = m_corr.predict_proba(d_c[feats])[:, 1]
            ya, yb = d_o[T].values, d_c[T].values
            so, _ = cal_metrics(ya, po); sc, _ = cal_metrics(yb, pc)
            rows.append(dict(model=tag, cohort=cname,
                             auc_orig=round(roc_auc_score(ya, po), 3),
                             auc_corrected=round(roc_auc_score(yb, pc), 3),
                             slope_orig=round(so, 3), slope_corrected=round(sc, 3)))
            print(f"[age-sens {tag:13s} {cname:12s}] AUC {rows[-1]['auc_orig']} -> {rows[-1]['auc_corrected']} "
                  f"| slope {rows[-1]['slope_orig']} -> {rows[-1]['slope_corrected']}")
    pd.DataFrame(rows).to_csv(OUT/"r3_age_sensitivity.csv", index=False)

    # ---------- 3) no-CCI transportable ----------
    feats_nc = [c_ for c_ in common if c_ not in ("apsiii", "cci")]
    print(f"\n[no-CCI] features: {len(feats_nc)}")
    mnc = lgb().fit(dev0[feats_nc], dev0[T])
    oof = cross_val_predict(lgb(), dev0[feats_nc], dev0[T], cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    thr_nc = round(float(youden_threshold(dev0[T].values, oof)), 3)
    devcv = roc_auc_score(dev0[T], oof)
    print(f"[no-CCI] dev-CV AUC {devcv:.3f}, threshold {thr_nc}")
    rows = []
    for cname, d in [("internal", it0), ("eICU_primary", ea), ("sepsis3", ep)]:
        p = mnc.predict_proba(d[feats_nc])[:, 1]; y = d[T].values
        auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
        s, i = cal_metrics(y, p)
        yh = (p >= thr_nc).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        rows.append(dict(cohort=cname, n=len(d), dev_cv_auc=round(devcv, 3), threshold=thr_nc,
                         auc=round(auc, 3), auc_ci=f"{lo:.3f}-{hi:.3f}",
                         slope=round(s, 3), intercept=round(i, 3),
                         brier=round(brier_score_loss(y, p), 3),
                         sens=round(tp/(tp+fn), 3), spec=round(tn/(tn+fp), 3),
                         ppv=round(tp/(tp+fp), 3), npv=round(tn/(tn+fn), 3)))
        print(f"[no-CCI {cname:12s}] AUC {rows[-1]['auc']} ({rows[-1]['auc_ci']}) slope {rows[-1]['slope']} "
              f"Brier {rows[-1]['brier']} sens {rows[-1]['sens']} spec {rows[-1]['spec']}")
    pd.DataFrame(rows).to_csv(OUT/"r3_transportable_nocci.csv", index=False)

    # ---------- 4) site-level CITL ----------
    mfull = lgb().fit(dev0[common], dev0[T])
    mt = lgb().fit(dev0[[c_ for c_ in common if c_ != "apsiii"]], dev0[T])
    ea2 = ea.copy()
    ea2["p_full"] = mfull.predict_proba(ea2[common])[:, 1]
    ea2["p_t"] = mt.predict_proba(ea2[[c_ for c_ in common if c_ != "apsiii"]])[:, 1]
    eps = 1e-6
    rows = []
    for hid, g in ea2.groupby("hospitalid"):
        y = g[T].values; n1 = int(y.sum()); n0 = len(g) - n1
        if len(g) < 25 or n1 < 5 or n0 < 5: continue
        rec = dict(hospitalid=hid, n=len(g), obs_rate=round(y.mean(), 3))
        for tag, pcol in [("full", "p_full"), ("transp", "p_t")]:
            p = np.clip(g[pcol].values, eps, 1-eps)
            lp = np.log(p/(1-p))
            citl = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                                offset=lp).fit().params[0])
            rec[f"citl_{tag}"] = round(citl, 3)
            rec[f"oe_{tag}"] = round(y.mean()/p.mean(), 3)
        rows.append(rec)
    sd = pd.DataFrame(rows)
    sd.to_csv(OUT/"r3_site_citl.csv", index=False)
    for tag in ["full", "transp"]:
        v = sd[f"citl_{tag}"]
        print(f"[site CITL {tag}] k={len(sd)} positive {(v>0).sum()}/{len(sd)} "
              f"median {v.median():+.3f} IQR {v.quantile(.25):+.3f}..{v.quantile(.75):+.3f} "
              f"| O/E>1: {(sd[f'oe_{tag}']>1).sum()}")

if __name__ == "__main__":
    main()
