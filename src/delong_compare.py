"""Locked LightGBM vs APS III alone — DeLong test on each cohort.
Also checks how much the composite outcome's death/readmission component drives AUC
(predict secondary observed-transfer target for comparison)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
import data_prep as dp

SEED = 42
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def _fastdelong(preds,m):
    n=preds.shape[1]-m; k=preds.shape[0]; pos=preds[:,:m]; neg=preds[:,m:]
    tx=np.empty([k,m]);ty=np.empty([k,n]);tz=np.empty([k,m+n])
    for r in range(k):
        tx[r]=_midrank(pos[r]);ty[r]=_midrank(neg[r]);tz[r]=_midrank(preds[r])
    aucs=tz[:,:m].sum(1)/m/n-(m+1)/2/n
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n; return aucs,cov
def delong(y,pa,pb):
    y=np.asarray(y); order=np.argsort(-y,kind="mergesort"); m=int(y.sum())
    preds=np.vstack((pa,pb))[:,order]; aucs,cov=_fastdelong(preds,m)
    l=np.array([[1.,-1.]]); var=float(l@cov@l.T)
    if var<=0: return aucs[0],aucs[1],1.0
    z=(aucs[0]-aucs[1])/np.sqrt(var); return float(aucs[0]),float(aucs[1]),float(2*scipy.stats.norm.sf(abs(z)))

LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)

def run(TARGET, label):
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    Xdev,ydev=dev[feats],dev[TARGET]
    model=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(Xdev,ydev)
    med=dev["apsiii"].median()
    aps=LogisticRegression().fit(dev[["apsiii"]].fillna(med),ydev)
    print(f"\n===== TARGET = {label} =====")
    for nm,d in [("MIMIC internal test",itest),
                 ("eICU primary(Sepsis-3)",eicu[eicu.sepsis3_primary==1]),
                 ("eICU primary",eicu[eicu.sepsis_admitdx==1])]:
        pm=model.predict_proba(d[feats])[:,1]; pa=aps.predict_proba(d[["apsiii"]].fillna(med))[:,1]
        a_m,a_a,pval=delong(d[TARGET],pm,pa)
        print(f"  {nm:24s} LGB={a_m:.3f}  APSIII={a_a:.3f}  diff={a_m-a_a:+.3f}  DeLong p={pval:.4f}")

if __name__=="__main__":
    run(dp.TARGET_PRIMARY,"primary_safe_transfer (composite)")
    run(dp.TARGET_SECONDARY,"secondary_transfer_72h (observed, no safety)")
