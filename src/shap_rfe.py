"""SHAP-based recursive feature elimination -> parsimonious model.
Within nested logic: SHAP ranking + RFE evaluated by dev 5-fold CV AUC; smallest subset
within 0.01 of max CV AUC selected. Report parsimonious internal + external AUC.
Primary composite outcome, temporal dev."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shap
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"
def lgb(): return LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED)
def cvauc(X,y): return roc_auc_score(y,cross_val_predict(lgb(),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba",n_jobs=-1)[:,1])

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=list(common)
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ep=eicu[eicu.sepsis3_primary==1].reset_index(drop=True); ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; ydev=dev[T]

    cur=list(feats); steps=[]
    while len(cur)>=1:
        auc=cvauc(dev[cur],ydev); steps.append((len(cur),auc,list(cur)))
        m=lgb().fit(dev[cur],ydev)
        sv=shap.TreeExplainer(m).shap_values(dev[cur])
        sv=sv[1] if isinstance(sv,list) and len(sv)>1 else sv
        imp=np.abs(sv).mean(0); worst=cur[int(np.argmin(imp))]
        if len(cur)==1: break
        cur.remove(worst)
    aucs=np.array([s[1] for s in steps]); mx=aucs.max()
    # smallest subset within 0.01 of max
    ok=[s for s in steps if s[1]>=mx-0.01]; best=min(ok,key=lambda s:s[0])
    k,cvA,sub=best
    print(f"max dev-CV AUC={mx:.3f}; parsimonious k={k} (CV {cvA:.3f})")
    print("selected features:",sub)
    m=lgb().fit(dev[sub],ydev)
    perf=[]
    for nm,d in [("internal",itest),("eICU primary",ep),("eICU alt",ea)]:
        auc=roc_auc_score(d[T],m.predict_proba(d[sub])[:,1])
        perf.append({"cohort":nm,"n_features":k,"parsimonious_auc":round(auc,3)})
        print(f"  parsimonious {nm:14s} AUC={auc:.3f}")
    # full-model external for contrast
    mf=lgb().fit(dev[feats],ydev)
    for nm,d in [("internal",itest),("eICU primary",ep),("eICU alt",ea)]:
        for row in perf:
            if row["cohort"]==nm: row["full_auc"]=round(roc_auc_score(d[T],mf.predict_proba(d[feats])[:,1]),3)
    print(f"  (full 56-var eICU alt AUC={roc_auc_score(ea[T],mf.predict_proba(ea[feats])[:,1]):.3f})")
    pd.DataFrame(perf).to_csv(OUT/"parsimonious_performance.csv",index=False)
    print(f"[SAVE] {OUT/'parsimonious_performance.csv'}")
    pd.DataFrame([(s[0],s[1]) for s in steps],columns=["n_features","dev_cv_auc"]).to_csv(OUT/"shap_rfe_curve.csv",index=False)
    pd.Series(sub,name="parsimonious_features").to_csv(OUT/"parsimonious_features.csv",index=False)
    print(f"[SAVE] {OUT/'shap_rfe_curve.csv'} , parsimonious_features.csv")

if __name__=="__main__": main()
