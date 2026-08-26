"""Missing-data strategy sensitivity.
Primary composite outcome, temporal dev, eval internal + eICU alt.
Strategies:
  native      : LightGBM native NaN (PRIMARY)
  complete    : listwise deletion (comparability w/ original)
  mice        : IterativeImputer fit on dev-train only, applied unchanged
  indicators  : native + binary missingness indicators
Shows how the missing-data choice affects transportability.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import data_prep as dp
from nested_cv import bootstrap_ci
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"
def lgb(): return LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED)

def evalset(model,X,y):
    p=model.predict_proba(X)[:,1]; lo,hi=bootstrap_ci(y,p)
    return roc_auc_score(y,p),lo,hi

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; ydev=dev[T]
    rows=[]
    def rec(name,model,prep):
        r={"strategy":name}
        for nm,d in [("internal",itest),("primary",ea)]:
            a,lo,hi=evalset(model,prep(d),d[T]); r[nm+"_auc"]=round(a,3); r[nm+"_ci"]=f"{lo:.3f}-{hi:.3f}"
        rows.append(r); print(f"  {name:12s} internal {r['internal_auc']} ({r['internal_ci']})  "
                              f"primary {r['primary_auc']} ({r['primary_ci']})")

    # native (primary)
    m=lgb().fit(dev[feats],ydev); rec("native",m,lambda d:d[feats])

    # complete-case
    cc=dev.dropna(subset=feats)
    m=lgb().fit(cc[feats],cc[T])
    rec("complete",m,lambda d:d[feats])   # eval on all (NaN passed to native tree anyway)
    print(f"    (complete-case dev n={len(cc)}/{len(dev)})")

    # MICE (fit on dev only)
    imp=IterativeImputer(random_state=SEED,max_iter=10,sample_posterior=False).fit(dev[feats])
    Xdev_i=pd.DataFrame(imp.transform(dev[feats]),columns=feats)
    m=lgb().fit(Xdev_i,ydev)
    rec("mice",m,lambda d:pd.DataFrame(imp.transform(d[feats]),columns=feats))

    # missingness indicators
    def add_ind(d):
        ind=d[feats].isna().astype(int); ind.columns=[c+"_miss" for c in feats]
        return pd.concat([d[feats].reset_index(drop=True),ind.reset_index(drop=True)],axis=1)
    Xdev_ind=add_ind(dev)
    m=lgb().fit(Xdev_ind,ydev)
    rec("indicators",m,add_ind)

    pd.DataFrame(rows).to_csv(OUT/"missing_data_sensitivity.csv",index=False)
    print(f"\n[SAVE] {OUT/'missing_data_sensitivity.csv'}")

if __name__=="__main__": main()
