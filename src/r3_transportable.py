# -*- coding: utf-8 -*-
"""R3 Editor comment — severity-score-free transportable model (co-primary).

55-feature LightGBM (all common features EXCEPT the severity-score variable),
same locked hyperparameters as the primary model (prespecified, disclosed).
Own Youden threshold from 5-fold CV predictions within the development set.

Outputs (results/):
  r3_transportable_performance.csv : AUC/CI, calibration slope+intercept (+boot CI),
                                     Brier, sens/spec/PPV/NPV at locked threshold
                                     for internal / eICU primary / Sepsis-3
  r3_transportable_subgroups.csv   : subgroup AUC/CI, slope, FPR/FNR (eICU primary)
  r3_transportable_hospital.csv    : per-site AUC; DL pooled + PI printed
  r3_transportable_dca.csv         : net-benefit curves (internal + eICU primary)
  r3_scores_distribution.csv       : APS III vs APACHE-IVa distribution table
figures/: figR3_cal_transportable.png, figR3_dca_transportable.png, figR3_scores.png
Also: DeLong full-model vs transportable vs native severity comparator.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci, cal_metrics, youden_threshold
from delong_compare import delong
from hospital_level import hanley_se

SEED = 42
LOCKED = dict(learning_rate=0.05, min_child_samples=50, n_estimators=300, num_leaves=15)
OUT = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400, "savefig.bbox": "tight"})
rng = np.random.default_rng(SEED)

def lgb(): return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **LOCKED)

def boot_cal(y, p, B=1000):
    """bootstrap CI for calibration slope & intercept"""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    s0, i0 = cal_metrics(y, p)
    ss, ii = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n): continue
        s, i = cal_metrics(y[idx], p[idx])
        ss.append(s); ii.append(i)
    return (s0, np.percentile(ss, 2.5), np.percentile(ss, 97.5),
            i0, np.percentile(ii, 2.5), np.percentile(ii, 97.5))

def thr_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    ppv = tp/(tp+fp) if tp+fp else np.nan; npv = tn/(tn+fn) if tn+fn else np.nan
    return sens, spec, ppv, npv

def net_benefit(y, p, thr):
    y = np.asarray(y); n = len(y); out = []
    for pt in thr:
        yh = (p >= pt).astype(int)
        tp = np.sum((yh == 1) & (y == 1)); fp = np.sum((yh == 1) & (y == 0))
        out.append(tp/n - fp/n*(pt/(1-pt)))
    return np.array(out)

def dl_pool(sd):
    a = sd.auc.values.clip(1e-4, 1-1e-4); se = sd.se.values
    yv = np.log(a/(1-a)); v = (se/(a*(1-a)))**2
    w = 1/v; yf = np.sum(w*yv)/np.sum(w)
    Q = np.sum(w*(yv-yf)**2); k = len(yv); C = np.sum(w)-np.sum(w**2)/np.sum(w)
    tau2 = max(0, (Q-(k-1))/C)
    wr = 1/(v+tau2); yb = np.sum(wr*yv)/np.sum(wr); sp = np.sqrt(1/np.sum(wr))
    from scipy.stats import t
    tc = t.ppf(0.975, k-2) if k > 2 else 1.96
    inv = lambda z: 1/(1+np.exp(-z))
    return dict(k=k, pooled=inv(yb), lo=inv(yb-1.96*sp), hi=inv(yb+1.96*sp),
                pi_lo=inv(yb-tc*np.sqrt(tau2+sp**2)), pi_hi=inv(yb+tc*np.sqrt(tau2+sp**2)),
                I2=max(0, (Q-(k-1))/Q)*100)

def main():
    mimic, eicu, fm, fe, common = dp.load()
    mimic = dp.apply_physiologic_ranges(mimic, common); eicu = dp.apply_physiologic_ranges(eicu, common)
    test = {"2017 - 2019", "2020 - 2022"}
    dev = mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest = mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea = eicu[eicu.sepsis_admitdx == 1].reset_index(drop=True)
    ep = eicu[eicu.sepsis3_primary == 1].reset_index(drop=True)
    T = dp.TARGET_PRIMARY
    feats_full = common
    feats_t = [c for c in common if c != "apsiii"]
    print(f"transportable feature count: {len(feats_t)} (full {len(feats_full)})")

    # ---- fit both models on dev ----
    mfull = lgb().fit(dev[feats_full], dev[T])
    mt = lgb().fit(dev[feats_t], dev[T])

    # dev 5-fold CV AUC + Youden threshold for transportable model
    oof = cross_val_predict(lgb(), dev[feats_t], dev[T], cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    dev_cv_auc = roc_auc_score(dev[T], oof)
    thr_t = round(float(youden_threshold(dev[T].values, oof)), 3)
    print(f"[transportable] dev 5-fold CV AUC={dev_cv_auc:.3f}  locked Youden threshold={thr_t}")

    # ---- performance table ----
    rows = []
    for nm, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
        p = mt.predict_proba(d[feats_t])[:, 1]; y = d[T].values
        auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
        s, slo, shi, i, ilo, ihi = boot_cal(y, p)
        sens, spec, ppv, npv = thr_metrics(y, p, thr_t)
        rows.append(dict(cohort=nm, n=len(d), auc=round(auc, 3), auc_ci=f"{lo:.3f}-{hi:.3f}",
                         slope=round(s, 3), slope_ci=f"{slo:.3f}-{shi:.3f}",
                         intercept=round(i, 3), intercept_ci=f"{ilo:.3f}-{ihi:.3f}",
                         brier=round(brier_score_loss(y, p), 3), threshold=thr_t,
                         sens=round(sens, 3), spec=round(spec, 3), ppv=round(ppv, 3), npv=round(npv, 3)))
        print(f"[T {nm:13s}] AUC {auc:.3f} ({lo:.3f}-{hi:.3f}) slope {s:.3f} ({slo:.3f}-{shi:.3f}) "
              f"int {i:.3f} ({ilo:.3f}-{ihi:.3f}) Brier {rows[-1]['brier']} "
              f"sens {sens:.3f} spec {spec:.3f} ppv {ppv:.3f} npv {npv:.3f}")
    pd.DataFrame(rows).to_csv(OUT/"r3_transportable_performance.csv", index=False)

    # also full-model calibration CIs (editor: CIs for ALL slopes/intercepts)
    frows = []
    for nm, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
        p = mfull.predict_proba(d[feats_full])[:, 1]; y = d[T].values
        s, slo, shi, i, ilo, ihi = boot_cal(y, p)
        frows.append(dict(cohort=nm, slope=round(s, 3), slope_ci=f"{slo:.3f}-{shi:.3f}",
                          intercept=round(i, 3), intercept_ci=f"{ilo:.3f}-{ihi:.3f}"))
        print(f"[FULL {nm:11s}] slope {s:.3f} ({slo:.3f}-{shi:.3f})  int {i:.3f} ({ilo:.3f}-{ihi:.3f})")
    pd.DataFrame(frows).to_csv(OUT/"r3_full_calibration_ci.csv", index=False)

    # ---- native severity comparator + DeLong ----
    print("\n[native severity comparator (raw APACHE-IVa rank in eICU; APS III LR internal)]")
    med = dev["apsiii"].median()
    aps_lr = LogisticRegression().fit(dev[["apsiii"]].fillna(med), dev[T])
    dl_rows = []
    for nm, d in [("internal", itest), ("eICU_primary", ea), ("sepsis3", ep)]:
        y = d[T].values
        pf = mfull.predict_proba(d[feats_full])[:, 1]
        pt = mt.predict_proba(d[feats_t])[:, 1]
        if nm == "internal":
            pc = aps_lr.predict_proba(d[["apsiii"]].fillna(med))[:, 1]
            mask = np.ones(len(d), bool); comp_label = "APS III (LR, dev-fit)"
        else:
            mask = d["apsiii"].notna().values
            pc = -d.loc[mask, "apsiii"].values  # raw score rank, native
            comp_label = "APACHE-IVa (raw, native)"
        ym = y[mask]
        auc_c = roc_auc_score(ym, pc if nm != "internal" else pc[mask])
        lo, hi = bootstrap_ci(ym, pc if nm != "internal" else pc[mask])
        pT, pF = pt[mask], pf[mask]
        pcm = pc if nm != "internal" else pc[mask]
        a1, b1, p1 = delong(ym, pT, pcm); d1 = a1 - b1
        a2, b2, p2 = delong(ym, pF, pcm); d2 = a2 - b2
        a3, b3, p3 = delong(y, pt, pf); d3 = a3 - b3
        print(f"  [{nm:13s}] comparator({comp_label}) AUC={auc_c:.3f} ({lo:.3f}-{hi:.3f}) n_eval={mask.sum()}")
        print(f"     transportable-vs-comp dAUC={d1:+.3f} p={p1:.2e} | full-vs-comp dAUC={d2:+.3f} p={p2:.2e} "
              f"| transportable-vs-full (all pts) dAUC={d3:+.3f} p={p3:.2e}")
        dl_rows.append(dict(cohort=nm, comparator=comp_label, comp_auc=round(auc_c, 3),
                            comp_ci=f"{lo:.3f}-{hi:.3f}", n_eval=int(mask.sum()),
                            d_transp_vs_comp=round(d1, 3), p_transp_vs_comp=p1,
                            d_full_vs_comp=round(d2, 3), p_full_vs_comp=p2,
                            d_transp_vs_full=round(d3, 3), p_transp_vs_full=p3))
    pd.DataFrame(dl_rows).to_csv(OUT/"r3_native_comparator_delong.csv", index=False)

    # ---- score distributions (APS III vs APACHE-IVa) ----
    srows = []
    for nm, s in [("MIMIC dev (APS III)", dev["apsiii"]), ("MIMIC internal test (APS III)", itest["apsiii"]),
                  ("eICU primary (APACHE-IVa)", ea["apsiii"]), ("eICU Sepsis-3 (APACHE-IVa)", ep["apsiii"])]:
        v = s.dropna()
        srows.append(dict(cohort=nm, n=len(s), missing_pct=round(s.isna().mean()*100, 1),
                          mean=round(v.mean(), 1), sd=round(v.std(), 1), median=v.median(),
                          q1=v.quantile(.25), q3=v.quantile(.75), min=v.min(), max=v.max()))
        print(f"[score {nm:30s}] mean {v.mean():.1f}±{v.std():.1f} med {v.median():.0f} "
              f"IQR {v.quantile(.25):.0f}-{v.quantile(.75):.0f} range {v.min():.0f}-{v.max():.0f} "
              f"miss {s.isna().mean()*100:.1f}%")
    pd.DataFrame(srows).to_csv(OUT/"r3_scores_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.arange(0, 205, 5)
    ax.hist(dev["apsiii"].dropna(), bins=bins, density=True, alpha=0.55, color="#2166ac",
            label="APS III — MIMIC-IV development")
    ax.hist(ea["apsiii"].dropna(), bins=bins, density=True, alpha=0.55, color="#b2182b",
            label="APACHE-IVa — primary eICU external cohort")
    ax.set_xlabel("Severity-score value"); ax.set_ylabel("Density")
    ax.legend(fontsize=9.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"figR3_scores.png"); plt.close(fig)
    print(f"[FIG] {FIG/'figR3_scores.png'}")

    # ---- calibration figure (transportable) ----
    def cal_bins(y, p, nb=10):
        df = pd.DataFrame({"y": y, "p": p}); df["bin"] = pd.qcut(df.p, nb, labels=False, duplicates="drop")
        g = df.groupby("bin").agg(mp=("p", "mean"), ob=("y", "mean"), n=("y", "size"))
        se = np.sqrt(g.ob*(1-g.ob)/g.n)
        return g.assign(lo=(g.ob-1.96*se).clip(0), hi=(g.ob+1.96*se).clip(upper=1))
    panels = [("(a) Temporal internal test", itest), ("(b) Primary eICU external cohort", ea),
              ("(c) Sepsis-3 sensitivity cohort", ep)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, (nm, d) in zip(axes, panels):
        p = mt.predict_proba(d[feats_t])[:, 1]; y = d[T].values
        g = cal_bins(y, p)
        ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, label="Perfect calibration")
        ax.errorbar(g.mp, g.ob, yerr=[g.ob-g.lo, g.hi-g.ob], fmt="-o", color="#2166ac",
                    ms=5, lw=1.5, capsize=2.5, label="Transportable model")
        s, i = cal_metrics(y, p)
        ax.text(0.03, 0.965, f"Slope {s:.3f}\nIntercept {i:.3f}\nBrier {brier_score_loss(y,p):.3f}",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted probability")
        ax.set_title(nm, fontsize=11); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed event frequency"); axes[1].legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG/"figR3_cal_transportable.png"); plt.close(fig)
    print(f"[FIG] {FIG/'figR3_cal_transportable.png'}")

    # ---- DCA (transportable vs native comparator) ----
    thr_grid = np.linspace(0.01, 0.99, 99)
    dca_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (nm, d) in zip(axes, [("(a) Temporal internal test", itest), ("(b) Primary eICU external cohort", ea)]):
        y = d[T].values
        pm = mt.predict_proba(d[feats_t])[:, 1]
        if nm.startswith("(a)"):
            pc = aps_lr.predict_proba(d[["apsiii"]].fillna(med))[:, 1]; comp = "APS III (logistic regression)"
        else:
            lr_nat = LogisticRegression().fit(d[["apsiii"]].fillna(d["apsiii"].median()), y)
            pc = lr_nat.predict_proba(d[["apsiii"]].fillna(d["apsiii"].median()))[:, 1]
            comp = "APACHE-IVa (native logistic regression)"
        nb_m = net_benefit(y, pm, thr_grid); nb_c = net_benefit(y, pc, thr_grid)
        er = y.mean(); nb_all = er-(1-er)*(thr_grid/(1-thr_grid))
        better = (nb_m > nb_c) & (nb_m > np.maximum(nb_all, 0))
        sup = [round(t_, 2) for t_ in thr_grid[better]]
        runs = []
        for t_ in sup:
            if runs and abs(t_ - runs[-1][1] - 0.01) < 1e-9:
                runs[-1][1] = t_
            else:
                runs.append([t_, t_])
        run_str = ", ".join((f"{a:.2f}" if a == b else f"{a:.2f}-{b:.2f}") for a, b in runs)
        print(f"[DCA transportable {nm}] superior thresholds (contiguous runs, 0.01 grid): {run_str}")
        for i_, t_ in enumerate(thr_grid):
            dca_rows.append(dict(cohort=nm, threshold=round(t_, 2), nb_model=nb_m[i_],
                                 nb_comp=nb_c[i_], nb_all=nb_all[i_]))
        ax.plot(thr_grid, nb_m, color="#2166ac", lw=1.9, label="Transportable model")
        ax.plot(thr_grid, nb_c, color="#b2182b", lw=1.7, label=comp)
        ax.plot(thr_grid, nb_all, "--", color="0.55", lw=1.4, label="Treat all")
        ax.axhline(0, color="black", lw=1.1, label="Treat none")
        ax.set_ylim(-0.05, max(nb_all.max(), nb_m.max())*1.12); ax.set_xlim(0, 1)
        ax.set_xlabel("Threshold probability"); ax.set_title(nm, fontsize=11)
        ax.legend(fontsize=9.5); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Net benefit")
    fig.tight_layout(); fig.savefig(FIG/"figR3_dca_transportable.png"); plt.close(fig)
    pd.DataFrame(dca_rows).to_csv(OUT/"r3_transportable_dca.csv", index=False)
    print(f"[FIG] {FIG/'figR3_dca_transportable.png'}")

    # ---- subgroups (eICU primary, transportable) ----
    ea2 = ea.copy(); ea2["p"] = mt.predict_proba(ea2[feats_t])[:, 1]
    def sub_eval(name, g):
        y = g[T].values; p = g["p"].values
        if y.sum() < 10 or (1-y).sum() < 10: return None
        auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
        s, _ = cal_metrics(y, p)
        yh = (p >= thr_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        return dict(subgroup=name, n=len(g), events=int(y.sum()), auc=round(auc, 3),
                    ci=f"{lo:.3f}-{hi:.3f}", slope=round(s, 3),
                    fpr=round(fp/(fp+tn), 3), fnr=round(fn/(fn+tp), 3))
    subs = [("Overall", ea2),
            ("MV: Yes", ea2[ea2.invasive_mechanical_ventilation == 1]),
            ("MV: No", ea2[ea2.invasive_mechanical_ventilation == 0]),
            ("Age <65", ea2[ea2.age < 65]), ("Age 65-79", ea2[(ea2.age >= 65) & (ea2.age < 80)]),
            ("Age >=80", ea2[ea2.age >= 80])]
    if "male" in ea2.columns:
        subs += [("Male", ea2[ea2.male == 1]), ("Female", ea2[ea2.male == 0])]
    for rc, lab in [("race_white", "White"), ("race_black", "Black"),
                    ("race_hispanic", "Hispanic"), ("race_asian", "Asian")]:
        if rc in ea2.columns:
            subs.append((lab, ea2[ea2[rc] == 1]))
    if "cci" in ea2.columns:
        subs += [("CCI 0-1", ea2[ea2.cci <= 1]), ("CCI 2", ea2[ea2.cci == 2]), ("CCI >=3", ea2[ea2.cci >= 3])]
    srows = [r for r in (sub_eval(n, g) for n, g in subs) if r]
    for r in srows:
        print(f"[subgroup {r['subgroup']:10s}] n={r['n']:5d} AUC {r['auc']} ({r['ci']}) slope {r['slope']} "
              f"FPR {r['fpr']} FNR {r['fnr']}")
    pd.DataFrame(srows).to_csv(OUT/"r3_transportable_subgroups.csv", index=False)

    # ---- hospital-level (transportable) ----
    hr = []
    for hid, g in ea2.groupby("hospitalid"):
        y = g[T].values; n1 = int(y.sum()); n0 = len(g)-n1
        if len(g) >= 25 and n1 >= 5 and n0 >= 5:
            auc = roc_auc_score(y, g["p"].values)
            hr.append(dict(hospitalid=hid, n=len(g), events=n1, event_rate=n1/len(g),
                           auc=auc, se=hanley_se(auc, n1, n0)))
    sd = pd.DataFrame(hr)
    meta = dl_pool(sd)
    print(f"\n[transportable hospital-level] k={meta['k']} median {sd.auc.median():.3f} "
          f"IQR {sd.auc.quantile(.25):.3f}-{sd.auc.quantile(.75):.3f} range {sd.auc.min():.3f}-{sd.auc.max():.3f} "
          f"sites<0.60: {(sd.auc<0.6).sum()}")
    print(f"  pooled {meta['pooled']:.3f} ({meta['lo']:.3f}-{meta['hi']:.3f}) "
          f"PI {meta['pi_lo']:.3f}-{meta['pi_hi']:.3f} I2 {meta['I2']:.0f}%")
    sd.sort_values("auc").to_csv(OUT/"r3_transportable_hospital.csv", index=False)
    print("[SAVE] all r3_transportable outputs")

if __name__ == "__main__":
    main()
