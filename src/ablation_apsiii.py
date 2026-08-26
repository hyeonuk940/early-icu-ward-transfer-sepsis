"""APS III overlap ablation: does ML add beyond the severity score?
Four models, primary composite outcome, temporal dev + eval on both external cohorts.
  A: APS III alone (LR)
  B: raw clinical variables WITHOUT apsiii (LightGBM native)
  C: apsiii + non-overlapping variables only (treatments/demographics/comorbidity + labs
     not contained in APS III: platelet, calcium, chloride, bmi)
  D: full 56-feature model (LightGBM native)
Reports dev 5-fold CV AUC + external AUCs; DeLong of D vs A and C vs A externally.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier
import data_prep as dp
from delong_compare import delong
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"

# APS III physiologic components -> overlapping (exclude from set C)
APS_OVERLAP_BASE = {"heart_rate","sbp","dbp","mbp","respiratory_rate","temperature","spo2",
                    "bun","creatinine","wbc","hemoglobin","sodium","potassium","glucose",
                    "bicarbonate","gcs_score","urine_output","age"}
def is_overlap(col):
    b=col
    for s in ("_min","_max"):
        if col.endswith(s): b=col[:-len(s)]; break
    return b in APS_OVERLAP_BASE

def cv_auc(X,y,model_fn):
    p=cross_val_predict(model_fn(),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                        method="predict_proba",n_jobs=-1)[:,1]
    return roc_auc_score(y,p)

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ep=eicu[eicu.sepsis3_primary==1].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; ydev=dev[T]; med=dev["apsiii"].median()

    raw_noaps=[c for c in feats if c!="apsiii"]
    nonoverlap=["apsiii"]+[c for c in feats if c!="apsiii" and not is_overlap(c)]
    print("set C (apsiii + non-overlap) vars:",nonoverlap)

    lgbf=lambda: LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED)
    specs={
      "A_APSIII_alone":  (["apsiii"], lambda: LogisticRegression(), True),
      "B_raw_no_APSIII": (raw_noaps, lgbf, False),
      "C_APSIII_nonoverlap": (nonoverlap, lgbf, False),
      "D_full":          (feats, lgbf, False),
    }
    # fit + predict store for DeLong
    preds={}; rows=[]
    for name,(cols,fn,is_lr) in specs.items():
        def prep(d):
            X=d[cols].copy()
            if is_lr: X=X.fillna(med)   # LR needs complete
            return X
        dev_auc=cv_auc(prep(dev),ydev,fn)
        model=fn().fit(prep(dev),ydev)
        r={"model":name,"n_vars":len(cols),"dev_cv_auc":round(dev_auc,3)}
        preds[name]={}
        for nm,d in [("internal",itest),("eicu_primary",ep),("eicu_alt",ea)]:
            p=model.predict_proba(prep(d))[:,1]; preds[name][nm]=(d[T].values,p)
            r[nm+"_auc"]=round(roc_auc_score(d[T],p),3)
        rows.append(r)
        print(f"  {name:22s} vars={len(cols):2d} devCV={dev_auc:.3f} "
              f"int={r['internal_auc']:.3f} eICUprim={r['eicu_primary_auc']:.3f} eICUalt={r['eicu_alt_auc']:.3f}")

    print("\nDeLong vs APS III alone (external cohorts):")
    for comp in ["C_APSIII_nonoverlap","D_full","B_raw_no_APSIII"]:
        for nm in ["eicu_primary","eicu_alt"]:
            y,pC=preds[comp][nm]; _,pA=preds["A_APSIII_alone"][nm]
            a1,a2,pv=delong(y,pC,pA)
            print(f"  {comp:22s} vs APSIII [{nm:12s}]: {a1:.3f} vs {a2:.3f} diff{a1-a2:+.3f} p={pv:.4f}")

    pd.DataFrame(rows).to_csv(OUT/"ablation_apsiii.csv",index=False)
    print(f"\n[SAVE] {OUT/'ablation_apsiii.csv'}")

if __name__=="__main__": main()
