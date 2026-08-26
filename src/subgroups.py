"""Subgroup discrimination + calibration + error rates.
Locked LightGBM + locked threshold. eICU alternative cohort (representative) & internal test.
Reports per subgroup: n, events, AUC(95%CI), cal slope/intercept, FPR, FNR at locked threshold.
Subgroups: mechanical ventilation, age band, sex, race, CCI band."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci, locked_threshold
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15); THR=locked_threshold()
OUT=Path(__file__).resolve().parents[1]/"results"

def cal(y,p):
    if len(np.unique(y))<2: return np.nan,np.nan
    eps=1e-6; pc=np.clip(p,eps,1-eps); lg=np.log(pc/(1-pc)).reshape(-1,1)
    m=LogisticRegression(solver="lbfgs").fit(lg,y); return float(m.coef_[0][0]),float(m.intercept_[0])

def err(y,p,thr):
    yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    fpr=fp/(fp+tn) if (fp+tn) else np.nan; fnr=fn/(fn+tp) if (fn+tp) else np.nan
    return fpr,fnr

def summarize(df,feats,T,label):
    rows=[]
    def add(name,mask):
        d=df[mask]; y=d[T].values
        if len(d)<20 or len(np.unique(y))<2:
            rows.append(dict(cohort=label,subgroup=name,n=len(d),events=int(y.sum()),auc=np.nan)); return
        p=d["p"].values; auc=roc_auc_score(y,p); lo,hi=bootstrap_ci(y,p)
        cs,ci=cal(y,p); fpr,fnr=err(y,p,THR)
        rows.append(dict(cohort=label,subgroup=name,n=len(d),events=int(y.sum()),
                         auc=round(auc,3),auc_ci=f"{lo:.3f}-{hi:.3f}",cal_slope=round(cs,3),
                         cal_intercept=round(ci,3),fpr=round(fpr,3),fnr=round(fnr,3)))
    add("ALL",np.ones(len(df),bool))
    add("MechVent=yes",df.invasive_mechanical_ventilation==1)
    add("MechVent=no",df.invasive_mechanical_ventilation==0)
    add("age<65",df.age<65); add("age65-79",(df.age>=65)&(df.age<80)); add("age>=80",df.age>=80)
    add("male",df.male==1); add("female",df.male==0)
    add("race_white",df.race_white==1); add("race_black",df.race_black==1)
    add("race_hispanic",df.race_hispanic==1); add("race_asian",df.race_asian==1)
    add("CCI0-1",df.cci<=1); add("CCI2",df.cci==2); add("CCI>=3",df.cci>=3)
    return rows

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True).copy()
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True).copy()
    T=dp.TARGET_PRIMARY
    model=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],dev[T])
    itest["p"]=model.predict_proba(itest[feats])[:,1]
    ea["p"]=model.predict_proba(ea[feats])[:,1]
    rows=summarize(itest,feats,T,"internal_test")+summarize(ea,feats,T,"primary")
    res=pd.DataFrame(rows)
    print(res[res.cohort=="primary"].to_string(index=False))
    res.to_csv(OUT/"subgroups.csv",index=False)
    print(f"\n[SAVE] {OUT/'subgroups.csv'}")
    # flag safety-critical subgroups
    ea_r=res[(res.cohort=='primary')&(res.auc.notna())]
    print("\nSafety-critical (external AUC<0.65):")
    print(ea_r[ea_r.auc<0.65][['subgroup','n','auc','auc_ci','cal_slope','fpr','fnr']].to_string(index=False))

if __name__=="__main__": main()
