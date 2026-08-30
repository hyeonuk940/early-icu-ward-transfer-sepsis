"""Hospital-level transportability on the eICU primary cohort.
Locked LightGBM (MIMIC dev). Per-hospital AUC (n>=25, >=5 events & >=5 non-events),
Hanley-McNeil SE, DerSimonian-Laird random-effects pooling on logit-AUC scale ->
pooled AUC + 95% CI + 95% PREDICTION INTERVAL. Per-site event rate & calibration intercept.
Labeled exploratory. Also leave-one-hospital-out is approximated by per-site eval of the
MIMIC-trained model (external internal-external CV)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import data_prep as dp
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"

def hanley_se(auc,n1,n0):
    Q1=auc/(2-auc); Q2=2*auc*auc/(1+auc)
    var=(auc*(1-auc)+(n1-1)*(Q1-auc*auc)+(n0-1)*(Q2-auc*auc))/(n1*n0)
    return np.sqrt(max(var,1e-9))

def cal_intercept(y,p):
    eps=1e-6; p=np.clip(p,eps,1-eps); logit=np.log(p/(1-p)).reshape(-1,1)
    lr=LogisticRegression(solver="lbfgs").fit(logit,y); return float(lr.intercept_[0])

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    T=dp.TARGET_PRIMARY
    model=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],dev[T])

    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True).copy()
    ea["p"]=model.predict_proba(ea[feats])[:,1]
    print(f"eICU alt cohort n={len(ea)} across {ea.hospitalid.nunique()} hospitals")

    rows=[]
    for hid,g in ea.groupby("hospitalid"):
        y=g[T].values; n=len(g); n1=int(y.sum()); n0=n-n1
        if n>=25 and n1>=5 and n0>=5:
            auc=roc_auc_score(y,g["p"].values)
            rows.append(dict(hospitalid=hid,n=n,events=n1,event_rate=n1/n,
                             auc=auc,se=hanley_se(auc,n1,n0),
                             cal_intercept=cal_intercept(y,g["p"].values)))
    sd=pd.DataFrame(rows)
    print(f"sites with n>=25 & >=5 events/non-events: {len(sd)}")
    print(f"  per-site AUC: median {sd.auc.median():.3f}  IQR {sd.auc.quantile(.25):.3f}-{sd.auc.quantile(.75):.3f}"
          f"  range {sd.auc.min():.3f}-{sd.auc.max():.3f}")
    print(f"  per-site event rate: median {sd.event_rate.median():.3f} "
          f"range {sd.event_rate.min():.3f}-{sd.event_rate.max():.3f}")
    print(f"  sites AUC<0.6: {(sd.auc<0.6).sum()}  |  AUC>0.75: {(sd.auc>0.75).sum()}")

    # DerSimonian-Laird random effects on logit(AUC)
    a=sd.auc.values.clip(1e-4,1-1e-4); se=sd.se.values
    y=np.log(a/(1-a)); v=(se/(a*(1-a)))**2                      # logit + delta-method var
    w=1/v; ybar_fixed=np.sum(w*y)/np.sum(w)
    Q=np.sum(w*(y-ybar_fixed)**2); k=len(y); C=np.sum(w)-np.sum(w**2)/np.sum(w)
    tau2=max(0,(Q-(k-1))/C)
    wr=1/(v+tau2); ybar=np.sum(wr*y)/np.sum(wr); se_pool=np.sqrt(1/np.sum(wr))
    from scipy.stats import t,norm
    ci=(ybar-1.96*se_pool, ybar+1.96*se_pool)
    tcrit=t.ppf(0.975,k-2) if k>2 else 1.96
    pi=(ybar-tcrit*np.sqrt(tau2+se_pool**2), ybar+tcrit*np.sqrt(tau2+se_pool**2))
    inv=lambda z:1/(1+np.exp(-z))
    print(f"\n[Random-effects meta, k={k} hospitals]")
    print(f"  pooled AUC = {inv(ybar):.3f}  95% CI {inv(ci[0]):.3f}-{inv(ci[1]):.3f}")
    print(f"  tau^2(logit)={tau2:.4f}  95% PREDICTION INTERVAL {inv(pi[0]):.3f}-{inv(pi[1]):.3f}")
    print(f"  I^2 = {max(0,(Q-(k-1))/Q)*100:.0f}%")
    sd.sort_values("auc").to_csv(OUT/"hospital_level_auc.csv",index=False)
    print(f"[SAVE] {OUT/'hospital_level_auc.csv'}  (forest-plot data)")

if __name__=="__main__": main()
