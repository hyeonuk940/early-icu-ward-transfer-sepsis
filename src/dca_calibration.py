"""Decision-curve analysis (exploratory) + calibration.
Locked LightGBM vs APS III vs treat-all/treat-none, internal + eICU alt.
Net benefit across thresholds; range where model beats all comparators.
Calibration: 10 quantile bins (mean predicted vs observed) + slope/intercept/Brier.
DCA is EXPLORATORY: relative harms of FP/FN not quantified (stated in text)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from lightgbm import LGBMClassifier
import data_prep as dp
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"

def net_benefit(y,p,thr):
    y=np.asarray(y); n=len(y); nb=[]
    for pt in thr:
        if pt>=1: nb.append(0.0); continue
        yh=(p>=pt).astype(int); tp=np.sum((yh==1)&(y==1)); fp=np.sum((yh==1)&(y==0))
        nb.append(tp/n - fp/n*(pt/(1-pt)))
    return np.array(nb)

def cal_bins(y,p,nb=10):
    df=pd.DataFrame({"y":y,"p":p}); df["bin"]=pd.qcut(df.p,nb,labels=False,duplicates="drop")
    g=df.groupby("bin").agg(mean_pred=("p","mean"),obs=("y","mean"),n=("y","size")).reset_index()
    return g

def cal_metrics(y,p):
    eps=1e-6; pc=np.clip(p,eps,1-eps); lg=np.log(pc/(1-pc)).reshape(-1,1)
    m=LogisticRegression(solver="lbfgs").fit(lg,y); return float(m.coef_[0][0]),float(m.intercept_[0])

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; ydev=dev[T]; med=dev["apsiii"].median()
    lgbm=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],ydev)
    aps=LogisticRegression().fit(dev[["apsiii"]].fillna(med),ydev)
    thr=np.linspace(0.01,0.99,99); calrows=[]; dcarows=[]
    for nm,d in [("internal",itest),("primary",ea)]:
        y=d[T].values; pm=lgbm.predict_proba(d[feats])[:,1]; pa=aps.predict_proba(d[["apsiii"]].fillna(med))[:,1]
        nb_m=net_benefit(y,pm,thr); nb_a=net_benefit(y,pa,thr)
        er=y.mean(); nb_all=er-(1-er)*(thr/(1-thr))
        better=(nb_m>nb_a)&(nb_m>np.maximum(nb_all,0))
        rng=(thr[better].min(),thr[better].max()) if better.any() else (np.nan,np.nan)
        cs,ci=cal_metrics(y,pm); br=brier_score_loss(y,pm)
        print(f"[{nm}] cal slope={cs:.3f} intercept={ci:.3f} Brier={br:.3f} | "
              f"DCA: LGB>APSIII&defaults over pt {rng[0]:.2f}-{rng[1]:.2f}; "
              f"NB@0.4: LGB={nb_m[np.argmin(abs(thr-0.4))]:.3f} APSIII={nb_a[np.argmin(abs(thr-0.4))]:.3f}")
        cb=cal_bins(y,pm); cb["cohort"]=nm; calrows.append(cb)
        for i,t in enumerate(thr):
            dcarows.append(dict(cohort=nm,threshold=round(t,2),nb_model=nb_m[i],nb_apsiii=nb_a[i],
                                nb_all=nb_all[i],nb_none=0.0))
    pd.concat(calrows).to_csv(OUT/"calibration_bins.csv",index=False)
    pd.DataFrame(dcarows).to_csv(OUT/"dca_netbenefit.csv",index=False)
    print(f"\n[SAVE] {OUT/'calibration_bins.csv'} , dca_netbenefit.csv (plot-ready)")
    print("DCA labeled exploratory; FP/FN relative harms not quantified (text).")

if __name__=="__main__": main()
