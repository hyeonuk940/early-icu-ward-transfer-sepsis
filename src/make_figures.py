"""Generate presentation figures (PNG, 300 dpi) into reanalysis/figures/."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import shap
import data_prep as dp
SEED=42; LOCKED=dict(learning_rate=0.05,min_child_samples=50,n_estimators=300,num_leaves=15)
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results"; FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":11,"savefig.bbox":"tight"})

def load_models():
    mimic,eicu,fm,fe,common=dp.load(); feats=list(common)
    mimic=dp.apply_physiologic_ranges(mimic,feats); eicu=dp.apply_physiologic_ranges(eicu,feats)
    test={"2017 - 2019","2020 - 2022"}
    dev=mimic[~mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    itest=mimic[mimic.anchor_year_group.isin(test)].reset_index(drop=True)
    ea=eicu[eicu.sepsis_admitdx==1].reset_index(drop=True)
    T=dp.TARGET_PRIMARY; med=dev["apsiii"].median()
    lg=LGBMClassifier(random_state=SEED,n_jobs=-1,verbose=-1,**LOCKED).fit(dev[feats],dev[T])
    aps=LogisticRegression().fit(dev[["apsiii"]].fillna(med),dev[T])
    return feats,T,med,dev,itest,ea,lg,aps

def fig_calibration():
    cb=pd.read_csv(RES/"calibration_bins.csv")
    plt.figure(figsize=(5.2,5.2)); plt.plot([0,1],[0,1],"k--",alpha=.5,label="Perfect")
    for nm,c in [("internal","#2166ac"),("primary","#b2182b")]:
        g=cb[cb.cohort==nm]; plt.plot(g.mean_pred,g.obs,"o-",color=c,label=nm)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration (LightGBM)"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(FIG/"calibration.png"); plt.close(); print("[FIG] calibration.png")

def fig_dca():
    d=pd.read_csv(RES/"dca_netbenefit.csv")
    for nm in ["internal","primary"]:
        g=d[d.cohort==nm]
        plt.figure(figsize=(5.2,4.6))
        plt.plot(g.threshold,g.nb_model,color="#2166ac",lw=2,label="LightGBM")
        plt.plot(g.threshold,g.nb_apsiii,color="#b2182b",lw=2,label="APS III")
        plt.plot(g.threshold,g.nb_all,color="gray",ls="--",label="Treat all")
        plt.axhline(0,color="k",lw=1,label="Treat none")
        plt.ylim(-0.05,g.nb_model.max()+0.05); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
        plt.title(f"Decision curve ({nm}) — exploratory"); plt.legend(); plt.grid(alpha=.3)
        plt.savefig(FIG/f"dca_{nm}.png"); plt.close(); print(f"[FIG] dca_{nm}.png")

def fig_forest():
    s=pd.read_csv(RES/"hospital_level_auc.csv").sort_values("auc").reset_index(drop=True)
    plt.figure(figsize=(5,7)); y=np.arange(len(s))
    plt.errorbar(s.auc,y,xerr=1.96*s.se,fmt="o",ms=2.5,lw=.6,color="#333",alpha=.7)
    plt.axvline(0.730,color="#2166ac",lw=2,label="Pooled 0.730")
    plt.axvspan(0.626,0.814,color="#2166ac",alpha=.12,label="95% prediction interval")
    plt.axvline(0.5,color="gray",ls=":",lw=1)
    plt.xlabel("Hospital-specific AUC"); plt.ylabel(f"Hospitals (n={len(s)}, sorted)")
    plt.title("Transportability across eICU hospitals"); plt.legend(loc="lower right"); plt.grid(alpha=.3,axis="x")
    plt.yticks([]); plt.savefig(FIG/"hospital_forest.png"); plt.close(); print("[FIG] hospital_forest.png")

def fig_rfe():
    c=pd.read_csv(RES/"shap_rfe_curve.csv").sort_values("n_features")
    plt.figure(figsize=(5.6,4.2)); plt.plot(c.n_features,c.dev_cv_auc,"o-",color="#2166ac",ms=3)
    plt.axvline(19,color="#b2182b",ls="--",label="Parsimonious (19)")
    plt.xlabel("Number of features"); plt.ylabel("Development 5-fold CV AUC")
    plt.title("SHAP-RFE feature reduction"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(FIG/"shap_rfe.png"); plt.close(); print("[FIG] shap_rfe.png")

def fig_cif():
    c=pd.read_csv(RES/"cif_by_riskquartile.csv",index_col=0)
    plt.figure(figsize=(5,4.2)); plt.bar(c.index.astype(str),c.iloc[:,0],color="#2166ac")
    plt.ylabel("CIF ward transfer by 72h"); plt.xlabel("Predicted-risk quartile")
    plt.title("Competing-risks CIF (death competing)"); plt.grid(alpha=.3,axis="y")
    plt.savefig(FIG/"cif_quartile.png"); plt.close(); print("[FIG] cif_quartile.png")

def fig_shap(feats,dev,lg):
    sv=shap.TreeExplainer(lg).shap_values(dev[feats])
    sv=sv[1] if isinstance(sv,list) and len(sv)>1 else sv
    plt.figure(); shap.summary_plot(sv,dev[feats],max_display=15,show=False,plot_size=(7,6))
    plt.title("SHAP feature importance (development)"); plt.savefig(FIG/"shap_summary.png"); plt.close()
    print("[FIG] shap_summary.png")

def main():
    feats,T,med,dev,itest,ea,lg,aps=load_models()
    fig_calibration(); fig_dca()
    fig_forest(); fig_rfe(); fig_cif(); fig_shap(feats,dev,lg)
    print("all figures ->",FIG)

if __name__=="__main__": main()
