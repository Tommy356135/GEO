#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/"results/paper_phase7_revision/fig6_qualitative"
CASES=[("Case 1: Transmission wires","pointcloud_752",3),
       ("Case 2: Poles","pointcloud_1033",5),
       ("Case 3: Fence","pointcloud_1308",7)]
NAMES=["Ground","Vegetation","Buildings","Transmission Wires","Distribution Wires",
       "Poles","Transmission Towers","Fence","Vehicle"]
COLORS=np.array([[.55,.42,.28],[.20,.55,.25],[.72,.30,.25],[.90,.72,.10],
                 [.95,.48,.08],[.35,.25,.60],[.65,.18,.55],[.15,.55,.58],[.20,.35,.75]])
mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif"],
 "font.size":8,"axes.titlesize":8,"figure.facecolor":"white","axes.facecolor":"white",
 "savefig.facecolor":"white","pdf.fonttype":42})
rng=np.random.default_rng(42)
fig,axes=plt.subplots(3,4,figsize=(7.16,7.0),constrained_layout=True)
titles=["Ground truth","Baseline","Exp3 Single-Fusion","Exp4 Dual-Fusion"]
reports=[]
for r,(case,stem,target) in enumerate(CASES):
    with np.load(FIG/"artifacts"/f"{stem}_val_predictions.npz",allow_pickle=False) as z:
        xyz=z["coordinates"].astype(np.float32)*.1
        arrays=[z["labels"],z["baseline"],z["exp3"],z["exp4"]]
    t=arrays[0]==target; lo=np.percentile(xyz[t,:2],1,axis=0)-5; hi=np.percentile(xyz[t,:2],99,axis=0)+5
    crop=(xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1])&(arrays[0]!=-100)
    ids=np.flatnonzero(crop); target_ids=np.flatnonzero(crop&t)
    other=np.setdiff1d(ids,target_ids,assume_unique=False); budget=max(0,70000-len(target_ids))
    if len(other)>budget: other=rng.choice(other,budget,replace=False)
    plot_ids=np.sort(np.concatenate([target_ids,other]))
    for col,(title,arr) in enumerate(zip(titles,arrays)):
        ax=axes[r,col]; ax.scatter(xyz[plot_ids,0],xyz[plot_ids,1],c=COLORS[arr[plot_ids]],
            s=.28,linewidths=0,rasterized=True)
        ax.set_xlim(*[lo[0],hi[0]]); ax.set_ylim(*[lo[1],hi[1]]); ax.set_aspect("equal")
        ax.set_title(title if r==0 else ""); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#666"); s.set_linewidth(.45)
        if col==0: ax.set_ylabel(f"{case}\n{stem}",fontsize=8)
    reports.append({"case":case,"tile":stem,"target_class":NAMES[target],
                    "target_voxels":int(t.sum()),"crop_voxels":int(crop.sum()),
                    "plotted_voxels":int(len(plot_ids)),"split":"val"})
handles=[plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=COLORS[i],
         markeredgecolor="none",markersize=5,label=n) for i,n in enumerate(NAMES)]
fig.legend(handles=handles,loc="outside lower center",ncol=5,frameon=False,fontsize=7)
fig.savefig(FIG/"fig6_val_qualitative_comparison.pdf",bbox_inches="tight")
fig.savefig(FIG/"fig6_val_qualitative_comparison.png",dpi=300,bbox_inches="tight")
plt.close(fig)
(FIG/"plot_report.json").write_text(json.dumps(reports,indent=2)+"\n")
