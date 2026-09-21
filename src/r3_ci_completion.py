# -*- coding: utf-8 -*-
"""Round-3 external review — close remaining gaps.

1) Bootstrap 95% CIs for ALL remaining reported calibration slopes/intercepts
   (editor Minor 6, full closure):
   - Table 2 secondary models (Random Forest, XGBoost) x 3 cohorts, APS III-only (internal)
   - Table S22 no-CCI model (slope + intercept, 3 cohorts)
   - Tables S18/S19 subgroup slopes (full + transportable, 15 subgroups each)
   -> r3_slope_ci_table2.csv, r3_nocci_cal_ci.csv, r3_slope_ci_subgroups.csv
2) Exact locked-threshold decision-curve values (auditability):
   net benefit of the transportable model, in-cohort severity reference, and
   treat-all at thresholds 0.396/0.412/0.414 -> r3_dca_locked_nb.csv
3) corrected-age impact on the
   Table 1 age summary -> r3_age_table1_impact.csv (requires DB env vars).
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(SEED)

def cal(y, p):
    eps = 1e-6; pc = np.clip(p, eps, 1-eps)
    lg = np.log(pc/(1-pc)).reshape(-1, 1)
    m = LogisticRegression(solver="lbfgs").fit(lg, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])

def boot_cal(y, p, B=1000):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    s0, i0 = cal(y, p); ss, ii = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n): continue
        s, i = cal(y[idx], p[idx]); ss.append(s); ii.append(i)
    return (s0, np.percentile(ss, 2.5), np.percentile(ss, 97.5),
            i0, np.percentile(ii, 2.5), np.percentile(ii, 97.5))

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    T = dp.TARGET_PRIMARY
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    feats = common; featsT = [c for c in common if c != "apsiii"]
    featsNC = [c for c in common if c not in ("apsiii", "cci")]
    med = dev["apsiii"].median()

    lgbm = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[feats], dev[T])
    mt = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[featsT], dev[T])
    mnc = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED).fit(dev[featsNC], dev[T])
    rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]).fit(dev[feats], dev[T])
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=SEED,
                        eval_metric="logloss").fit(dev[feats], dev[T])
    aps = LogisticRegression().fit(dev[["apsiii"]].fillna(med), dev[T])
    COH = [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]

    # ---------- 1a) Table 2 secondary-model slope CIs ----------
    rows = []
    for mname, pred in [("Random Forest", lambda d: rf.predict_proba(d[feats])[:, 1]),
                        ("XGBoost", lambda d: xgb.predict_proba(d[feats])[:, 1]),
                        ("full", lambda d: lgbm.predict_proba(d[feats])[:, 1]),
                        ("transportable", lambda d: mt.predict_proba(d[featsT])[:, 1])]:
        for cname, d in COH:
            s, lo, hi, i, ilo, ihi = boot_cal(d[T].values, pred(d))
            rows.append(dict(model=mname, cohort=cname, slope=round(s, 3),
                             slope_ci=f"{lo:.3f}-{hi:.3f}", intercept=round(i, 3),
                             intercept_ci=f"{ilo:.3f}-{ihi:.3f}"))
            print(f"[T2 {mname:14s} {cname:12s}] slope {s:.3f} ({lo:.3f}-{hi:.3f})")
    s, lo, hi, i, ilo, ihi = boot_cal(itest[T].values, aps.predict_proba(itest[["apsiii"]].fillna(med))[:, 1])
    rows.append(dict(model="APS III alone", cohort="internal", slope=round(s, 3),
                     slope_ci=f"{lo:.3f}-{hi:.3f}", intercept=round(i, 3), intercept_ci=f"{ilo:.3f}-{ihi:.3f}"))
    print(f"[T2 APS III internal] slope {s:.3f} ({lo:.3f}-{hi:.3f})")
    pd.DataFrame(rows).to_csv(OUT/"r3_slope_ci_table2.csv", index=False)

    # ---------- 1b) no-CCI slope/intercept CIs ----------
    rows = []
    for cname, d in COH:
        s, lo, hi, i, ilo, ihi = boot_cal(d[T].values, mnc.predict_proba(d[featsNC])[:, 1])
        rows.append(dict(cohort=cname, slope=round(s, 3), slope_ci=f"{lo:.3f}-{hi:.3f}",
                         intercept=round(i, 3), intercept_ci=f"{ilo:.3f}-{ihi:.3f}"))
        print(f"[S22 {cname:12s}] slope {s:.3f} ({lo:.3f}-{hi:.3f}) int {i:.3f} ({ilo:.3f}-{ihi:.3f})")
    pd.DataFrame(rows).to_csv(OUT/"r3_nocci_cal_ci.csv", index=False)

    # ---------- 1c) subgroup slope CIs (S18 full / S19 transportable) ----------
    ea2 = ea.copy()
    ea2["p_full"] = lgbm.predict_proba(ea2[feats])[:, 1]
    ea2["p_t"] = mt.predict_proba(ea2[featsT])[:, 1]
    if "female" not in ea2.columns: ea2["female"] = 1 - ea2["male"]
    SUBS = [("Overall", ea2), ("MV: Yes", ea2[ea2.invasive_mechanical_ventilation == 1]),
            ("MV: No", ea2[ea2.invasive_mechanical_ventilation == 0]),
            ("Age <65", ea2[ea2.age < 65]), ("Age 65-79", ea2[(ea2.age >= 65) & (ea2.age < 80)]),
            ("Age >=80", ea2[ea2.age >= 80]), ("Male", ea2[ea2.male == 1]), ("Female", ea2[ea2.male == 0]),
            ("White", ea2[ea2.race_white == 1]), ("Black", ea2[ea2.race_black == 1]),
            ("Hispanic", ea2[ea2.race_hispanic == 1]), ("Asian", ea2[ea2.race_asian == 1]),
            ("CCI 0-1", ea2[ea2.cci <= 1]), ("CCI 2", ea2[ea2.cci == 2]), ("CCI >=3", ea2[ea2.cci >= 3])]
    rows = []
    for tag, pcol in [("full", "p_full"), ("transportable", "p_t")]:
        for nm, g in SUBS:
            s, lo, hi, i, ilo, ihi = boot_cal(g[T].values, g[pcol].values)
            rows.append(dict(model=tag, subgroup=nm, slope=round(s, 3), slope_ci=f"{lo:.3f}-{hi:.3f}"))
    pd.DataFrame(rows).to_csv(OUT/"r3_slope_ci_subgroups.csv", index=False)
    print(f"[subgroup slope CIs] {len(rows)} rows saved")

    # ---------- 2) exact locked-threshold DCA values ----------
    def nb(y, p, pt):
        yh = (p >= pt).astype(int); n = len(y)
        tp = ((yh == 1) & (y == 1)).sum(); fp = ((yh == 1) & (y == 0)).sum()
        return tp/n - fp/n*(pt/(1-pt))
    rows = []
    for cname, d in [("internal", itest), ("eICU_primary", ea)]:
        y = d[T].values
        pm = mt.predict_proba(d[featsT])[:, 1]
        if cname == "internal":
            pc = aps.predict_proba(d[["apsiii"]].fillna(med))[:, 1]; ref = "APS III LR (dev-fit)"
        else:
            lr = LogisticRegression().fit(d[["apsiii"]].fillna(d["apsiii"].median()), y)
            pc = lr.predict_proba(d[["apsiii"]].fillna(d["apsiii"].median()))[:, 1]
            ref = "APACHE-IVa LR (in-cohort)"
        for pt in [0.396, 0.412, 0.414]:
            rows.append(dict(cohort=cname, threshold=pt, reference=ref,
                             nb_transportable=round(nb(y, pm, pt), 4),
                             nb_reference=round(nb(y, pc, pt), 4),
                             nb_treat_all=round(y.mean()-(1-y.mean())*(pt/(1-pt)), 4)))
            print(f"[NB {cname} pt={pt}] model {rows[-1]['nb_transportable']} ref {rows[-1]['nb_reference']} all {rows[-1]['nb_treat_all']}")
    pd.DataFrame(rows).to_csv(OUT/"r3_dca_locked_nb.csv", index=False)

    # ---------- 3) INTERNAL: corrected-age Table 1 impact ----------
    try:
        import psycopg2
        c = psycopg2.connect(host=os.environ.get("SEPSIS_DB_HOST", "localhost"), dbname="mimic4",
                             user=os.environ.get("SEPSIS_DB_USER", "postgres"),
                             password=os.environ["SEPSIS_DB_PASSWORD"])
        off = pd.read_sql("""select m.subject_id,
                 (extract(year from m.icu_intime) - p.anchor_year)::int as offset_yr, p.anchor_age
                 from sepsis_work.mimic_cohort m join mimiciv_hosp.patients p using(subject_id)
                 where m.in_landmark""", c)
        c.close()
        m2 = mimic.merge(off, on="subject_id", how="left")
        m2["age_corr"] = np.minimum(m2.anchor_age + m2.offset_yr, 91)
        rows = []
        for grp, g in [("event", m2[m2[T] == 1]), ("non-event", m2[m2[T] == 0]), ("overall", m2)]:
            rows.append(dict(group=grp,
                stored_median=f"{g.age.median():.0f} ({g.age.quantile(.25):.0f}-{g.age.quantile(.75):.0f})",
                corrected_median=f"{g.age_corr.median():.0f} ({g.age_corr.quantile(.25):.0f}-{g.age_corr.quantile(.75):.0f})",
                stored_mean=round(g.age.mean(), 1), corrected_mean=round(g.age_corr.mean(), 1)))
            print(f"[age T1 {grp:9s}] stored {rows[-1]['stored_median']} -> corrected {rows[-1]['corrected_median']}")
        pd.DataFrame(rows).to_csv(OUT/"r3_age_table1_impact.csv", index=False)
    except KeyError:
        print("[age T1] SEPSIS_DB_PASSWORD not set — skipped (internal-only analysis)")

if __name__ == "__main__":
    main()
