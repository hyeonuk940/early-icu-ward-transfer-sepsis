# -*- coding: utf-8 -*-
"""Figure 1 — patient-selection flowchart (per-patient numbers). Saves figures/figure1_flowchart.png."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
FIG=Path(__file__).resolve().parents[1]/"figures"
plt.rcParams.update({"font.family":"Arial","savefig.dpi":300,"savefig.bbox":"tight"})

fig,ax=plt.subplots(figsize=(14,8)); ax.set_xlim(0,15); ax.set_ylim(0.5,10); ax.axis("off")
BLUE="#2166ac"; GREY="#555555"; RED="#b2182b"

def box(x,y,w,h,text,fc="#eef4fb",ec=BLUE,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",fc=fc,ec=ec,lw=1.4))
    ax.text(x,y,text,ha="center",va="center",fontsize=fs,fontweight=("bold" if bold else "normal"))
def excl(x,y,w,text):
    ax.add_patch(FancyBboxPatch((x-w/2,y-0.42),w,0.84,boxstyle="round,pad=0.02,rounding_size=0.06",fc="#fbeeee",ec=RED,lw=1.0))
    ax.text(x,y,text,ha="center",va="center",fontsize=8.2)
def varrow(x,y0,y1): ax.add_patch(FancyArrowPatch((x,y0),(x,y1),arrowstyle="-|>",mutation_scale=14,lw=1.4,color=GREY))
def harrow(x0,x1,y): ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=11,lw=1.0,color=RED))

ax.text(3.0,9.7,"MIMIC-IV (development)",ha="center",fontsize=13,fontweight="bold",color=BLUE)
ax.text(10.2,9.7,"eICU-CRD (external validation)",ha="center",fontsize=13,fontweight="bold",color=BLUE)

# ---- MIMIC (x=3) ----
box(3.0,9.0,3.8,0.8,"First ICU episode per patient\n(subject_id): 65,355",bold=True)
varrow(3.0,8.6,8.0); harrow(4.9,5.15,8.3); excl(6.6,8.3,2.9,"Sepsis-3 not met by 24 h\nexcluded: 39,752")
box(3.0,7.6,3.8,0.7,"Met Sepsis-3 by 24 h: 25,603")
varrow(3.0,7.25,6.6); harrow(4.9,5.15,6.85); excl(6.6,6.85,2.9,"Not alive & in ICU at 24 h\nexcluded: 2,778")
box(3.0,6.2,3.8,0.9,"24-hour landmark cohort\n22,825",fs=11,bold=True,fc="#cfe0f2")
varrow(3.0,5.75,5.15)
box(3.0,4.55,4.4,1.15,"Primary outcome:\nsafe ICU-to-ward transfer 24–72 h,\nno readmission/death ≤7 d\n9,239 (40.5%)   |   Secondary (ICU exit ≤72 h): 49.1%",fs=8.6)

# ---- eICU (x=10.2) ----
box(10.2,9.0,4.0,0.8,"First ICU stay per patient\n(uniquepid): 139,306",bold=True)
varrow(10.2,8.6,8.0); harrow(12.2,12.45,8.3); excl(13.6,8.3,2.4,"Age < 18\nexcluded: 508")
box(10.2,7.6,4.0,0.7,"Adults (age ≥ 18): 138,798")
varrow(10.2,7.25,6.75)
box(10.2,6.35,4.2,0.55,"Two sepsis definitions applied",fs=9.5,fc="#f2f2f2",ec=GREY)
ax.add_patch(FancyArrowPatch((10.2,6.05),(8.3,5.25),arrowstyle="-|>",mutation_scale=13,lw=1.3,color=GREY))
ax.add_patch(FancyArrowPatch((10.2,6.05),(12.1,5.25),arrowstyle="-|>",mutation_scale=13,lw=1.3,color=GREY))
box(8.3,4.55,3.0,1.2,"Sepsis-3 sensitivity\n(antibiotic + culture\n+ SOFA ≥ 2 by 24 h)\n2,409  |  84 hospitals",fs=8.6,fc="#cfe0f2")
box(12.1,4.55,3.2,1.2,"Primary\n(APACHE sepsis\nadmission diagnosis)\n13,384  |  200 hospitals",fs=8.6,fc="#cfe0f2",bold=True)

ax.text(7.5,1.4,"Cohort unit = one first ICU episode per patient. In-hospital deaths were retained and handled within the "
        "outcome definition.\nMIMIC-IV is adult-only (no patients removed by the age criterion). See Supplementary "
        "Table S5 for the full audit and outcome reconciliation.",ha="center",fontsize=8.4,style="italic",color=GREY)

plt.savefig(FIG/"figure1_flowchart.png"); plt.close(); print("SAVED",FIG/"figure1_flowchart.png")
