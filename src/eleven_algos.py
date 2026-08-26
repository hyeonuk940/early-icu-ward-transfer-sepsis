"""11-algorithm comparison, de-emphasized & exploratory.
Fair common preprocessing (median impute + standardize) so non-native models run.
Dev 5-fold CV AUC + external AUCs; Holm-adjusted DeLong vs top model (multiplicity correction).
Primary composite outcome, temporal dev split."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import data_prep as dp
from delong_compare import delong
SEED=42; OUT=Path(__file__).resolve().parents[1]/"results"

def models():
    return {
      "LR":LogisticRegression(max_iter=5000,C=1.0),
      "RF":RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1),
      "GNB":GaussianNB(),
      "DT":DecisionTreeClassifier(max_depth=5,random_state=SEED),
      "KNN":KNeighborsClassifier(n_neighbors=21),
      "MLP":MLPClassifier(hidden_layer_sizes=(50,),max_iter=500,random_state=SEED),
      "AdaBoost":AdaBoostClassifier(n_estimators=200,random_state=SEED),
      "SVM":SVC(probability=True,random_state=SEED),
      "XGBoost":XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=4,random_state=SEED,eval_metric="logloss"),
      "LightGBM":LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=15,min_child_samples=50,random_state=SEED,verbose=-1),
      "CatBoost":CatBoostClassifier(iterations=500,depth=6,learning_rate=0.05,random_state=SEED,verbose=0,allow_writing_files=False),
    }

def holm(pvals):
    idx=np.argsort(pvals); adj=np.empty(len(pvals)); m=len(pvals); prev=0
    for r,i in enumerate(idx):
        val=(m-r)*pvals[i]; prev=max(prev,val); adj[i]=min(prev,1.0)
    return adj

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; ydev=dev[T]
    pre=lambda clf:Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("clf",clf)])
    rows=[]; extpred={}
    for name,clf in models().items():
        pipe=pre(clf)
        cvp=cross_val_predict(pipe,dev[feats],ydev,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                              method="predict_proba",n_jobs=-1)[:,1]
        devcv=roc_auc_score(ydev,cvp)
        pipe.fit(dev[feats],ydev)
        pi=pipe.predict_proba(itest[feats])[:,1]; pe=pipe.predict_proba(ea[feats])[:,1]
        extpred[name]=pe
        rows.append(dict(model=name,dev_cv_auc=round(devcv,3),
                         internal_auc=round(roc_auc_score(itest[T],pi),3),
                         primary_auc=round(roc_auc_score(ea[T],pe),3)))
        print(f"  {name:10s} devCV={devcv:.3f} int={rows[-1]['internal_auc']:.3f} ext={rows[-1]['primary_auc']:.3f}")
    res=pd.DataFrame(rows).sort_values("dev_cv_auc",ascending=False).reset_index(drop=True)
    top=res.iloc[0]["model"]
    print(f"\n  top by dev-CV: {top}")
    # Holm-adjusted DeLong (external) vs top
    others=[m for m in res.model if m!=top]; pv=[]
    ya=ea[T].values
    for m in others:
        _,_,p=delong(ya,extpred[top],extpred[m]); pv.append(p)
    padj=holm(np.array(pv))
    dl=pd.DataFrame({"reference":top,"compared":others,"external_delong_p":np.round(pv,4),
                     "holm_adj_p":np.round(padj,4)})
    print(dl.to_string(index=False))
    res.to_csv(OUT/"eleven_algos.csv",index=False); dl.to_csv(OUT/"eleven_algos_delong_holm.csv",index=False)
    print(f"\n[SAVE] {OUT/'eleven_algos.csv'} , eleven_algos_delong_holm.csv")
    print("NOTE: exploratory; no algorithm selected on test/external data (selection = dev-CV only).")

if __name__=="__main__": main()
