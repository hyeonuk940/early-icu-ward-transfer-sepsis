"""(1) 48/72/96h horizon sensitivity; (2) competing-risks CIF.
Horizon composite outcome recomputed from exported columns (no re-query):
  primary_H = (icu_duration_hours<=H) & left_icu_alive & went_to_ward
              & (icu_readmit_7d==0) & (death_7d_post==0)
Competing risks (eICU alt): time=icu_duration_hours (time to ICU exit),
  event 1=ICU->ward transfer, 2=in-ICU death / non-ward exit (competing).
  Aalen-Johansen CIF of ward transfer by predicted-risk quartile at 72h.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from lifelines import AalenJohansenFitter
import data_prep as dp
from nested_cv import bootstrap_ci
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
OUT=Path(__file__).resolve().parents[1]/"results"

def horizon_target(df,H):
    return ((df.icu_duration_hours<=H)&(df.left_icu_alive==1)&(df.went_to_ward==1)
            &(df.icu_readmit_7d==0)&(df.death_7d_post==0)).astype(int)

def main():
    mimic,eicu,fm,fe,common=dp.load(); feats=common
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ep=eicu[eicu.sepsis3_primary==1].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)

    print("=== (1) Horizon sensitivity (48/72/96h), locked-HP LightGBM ===")
    rows=[]
    for H in (48,72,96):
        yd=horizon_target(dev,H)
        model=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],yd)
        r={"horizon_h":H,"dev_event_rate":round(yd.mean(),3)}
        for nm,d in [("internal",itest),("primary",ea),("sensitivity",ep)]:
            yy=horizon_target(d,H); p=model.predict_proba(d[feats])[:,1]
            auc=roc_auc_score(yy,p); lo,hi=bootstrap_ci(yy,p)
            r[nm+"_auc"]=round(auc,3); r[nm+"_ci"]=f"{lo:.3f}-{hi:.3f}"; r[nm+"_rate"]=round(yy.mean(),3)
        rows.append(r)
        print(f"  {H}h: dev_rate={r['dev_event_rate']:.3f} | internal {r['internal_auc']} "
              f"| primary {r['primary_auc']} | sensitivity {r['sensitivity_auc']} "
              f"(primary event rate {r['primary_rate']})")
    pd.DataFrame(rows).to_csv(OUT/"horizon_sensitivity.csv",index=False)
    print(f"  [SAVE] {OUT/'horizon_sensitivity.csv'}")

    print("\n=== (2) Competing-risks CIF by predicted-risk quartile (eICU alt, 72h) ===")
    # locked model on primary composite for risk scoring
    ydev=horizon_target(dev,72)
    model=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],ydev)
    e=ea.copy(); e["p"]=model.predict_proba(e[feats])[:,1]
    # event coding: 1=ward transfer (alive & ward), 2=competing (death/non-ward exit)
    e["etype"]=np.where((e.left_icu_alive==1)&(e.went_to_ward==1),1,2)
    e["t"]=e.icu_duration_hours.clip(lower=0.1)
    e["q"]=pd.qcut(e["p"],4,labels=["Q1(low)","Q2","Q3","Q4(high)"])
    print(f"  n={len(e)}  ward-transfer events={int((e.etype==1).sum())}  competing={int((e.etype==2).sum())}")
    cif72={}
    for q,g in e.groupby("q"):
        ajf=AalenJohansenFitter(calculate_variance=False)
        ajf.fit(g["t"],g["etype"],event_of_interest=1)
        cif=ajf.cumulative_density_
        val=float(cif[cif.index<=72].iloc[-1,0]) if (cif.index<=72).any() else float(cif.iloc[0,0])
        cif72[str(q)]=val
        print(f"    {q}: n={len(g)}  CIF(ward transfer by 72h) = {val:.3f}")
    pd.Series(cif72,name="CIF_ward_72h").to_csv(OUT/"cif_by_riskquartile.csv")
    print(f"  [SAVE] {OUT/'cif_by_riskquartile.csv'}")
    print("  (monotone increase across risk quartiles = predicted risk tracks actual transfer incidence"
          " while treating death as a competing event)")

if __name__=="__main__": main()
