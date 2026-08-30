# -*- coding: utf-8 -*-
"""Clean Table 2: locked LightGBM + comparators, internal & external,
AUC(CI), calibration slope, Brier, sens/spec at each model's own development threshold.

Each model gets its OWN Youden threshold from 5-fold out-of-fold predictions on the
DEVELOPMENT set only -- the same procedure nested_cv.py uses -- and that threshold is
written to the output so the operating point is explicit. A single shared cut-point would
not be comparable across models: predicted-probability scales differ (a 1-variable logistic
model concentrates probabilities near the base rate, a 56-variable GBM spreads them toward
0/1), so one threshold puts each model at a different operating point.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import data_prep as dp
from nested_cv import bootstrap_ci, locked_threshold
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"

def cal_slope(y,p):
    eps=1e-6;pc=np.clip(p,eps,1-eps);lg=np.log(pc/(1-pc)).reshape(-1,1)
    return float(LogisticRegression(solver="lbfgs").fit(lg,y).coef_[0][0])
def ss(y,p,thr):
    yh=(p>=thr).astype(int);tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return tp/(tp+fn) if tp+fn else np.nan, tn/(tn+fp) if tn+fp else np.nan
def youden(y,p):
    fpr,tpr,thr=roc_curve(y,p); return float(thr[np.argmax(tpr-fpr)])
def dev_threshold(est,X,y):
    """Youden cut-point from 5-fold out-of-fold predictions on the development set.
    cv=5 (plain int, unshuffled) mirrors nested_cv.py exactly -- that script is the source
    of the locked LightGBM threshold, so the resampling scheme must match or this table
    would disagree with results/headline_nested_cv.csv all over again."""
    oof=cross_val_predict(est,X,y,cv=5,method="predict_proba",n_jobs=-1)[:,1]
    return youden(y,oof)
def row(nm,y,p,thr):
    a=roc_auc_score(y,p);lo,hi=bootstrap_ci(y,p);sn,sp=ss(y,p,thr)
    return dict(model=nm,auc=f"{a:.3f} ({lo:.3f}-{hi:.3f})",cal_slope=f"{cal_slope(y,p):.3f}",
               brier=f"{brier_score_loss(y,p):.3f}",sens=f"{sn:.3f}",spec=f"{sp:.3f}",
               threshold=f"{thr:.3f}")

def main():
    mimic,eicu,fm,fe,common=dp.load();feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats);eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ep=eicu[eicu.sepsis3_primary==1].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY;ydev=dev[T];med=dev["apsiii"].median()

    # (name, unfitted-estimator factory, design-matrix extractor)
    X_full=lambda d: d[feats]
    X_aps =lambda d: d[["apsiii"]].fillna(med)
    specs=[("LightGBM (primary)",
            lambda: LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED), X_full),
           ("Random Forest",
            lambda: Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                              ("clf",RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1))]), X_full),
           ("XGBoost",
            lambda: XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=4,
                                  random_state=SEED,eval_metric="logloss"), X_full),
           ("APS III alone",
            lambda: LogisticRegression(), X_aps)]

    models={}
    print("Development-derived operating points (Youden on 5-fold OOF, development set only):")
    for nm,factory,X in specs:
        thr=dev_threshold(factory(),X(dev),ydev)
        models[nm]=(factory().fit(X(dev),ydev), X, thr)
        print(f"  {nm:22s} threshold={thr:.3f}")
    lgb_thr=models["LightGBM (primary)"][2]
    thr_locked=locked_threshold()
    if abs(lgb_thr-thr_locked)>0.001:
        print(f"  [WARN] LightGBM threshold {lgb_thr:.3f} != locked {thr_locked:.3f} "
              f"-- nested_cv.py and downstream scripts must be re-checked.")

    rows=[]
    for coh,d in [("Internal test (MIMIC-IV)",itest),("External: Sepsis-3 sensitivity (eICU)",ep),
                  ("External: primary (eICU)",ea)]:
        for nm,(mdl,X,thr) in models.items():
            r=row(nm,d[T].values,mdl.predict_proba(X(d))[:,1],thr);r["cohort"]=coh;rows.append(r)
    df=pd.DataFrame(rows)[["cohort","model","auc","cal_slope","brier","sens","spec","threshold"]]
    df.to_csv(OUT/"table2_performance.csv",index=False)
    print()
    print(df.to_string(index=False))
    print("\n[SAVE]",OUT/"table2_performance.csv")
    print("NOTE: sens/spec are reported at each model's own development-derived threshold; "
          "thresholds are listed so the operating point is explicit.")

if __name__=="__main__": main()
