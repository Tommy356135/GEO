#!/usr/bin/env python3
"""Val-only complexity benchmark and qualitative figure generator."""
from __future__ import annotations
import csv, gc, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_phase7_revision"
FIG = OUT / "fig6_qualitative"
ART = FIG / "artifacts"

MODELS = {
    "Exp0": ("Baseline", ROOT/"results/baseline_strong_block50_merged_amp_safe/checkpoint_best.pth"),
    "Exp1": ("Geometry-Concat", ROOT/"results/exp1_geometry_concat_final_v2_resume_fp32val/checkpoint_best.pth"),
    "Exp2": ("Geometry-Encoder", ROOT/"results/exp2_geometry_encoder_final/checkpoint_best.pth"),
    "Exp3": ("Single-Fusion", ROOT/"results/exp3_single_fusion_final/checkpoint_best.pth"),
    "Exp4": ("Dual-Fusion", ROOT/"results/exp4_geomorph_seg_final/checkpoint_best.pth"),
}
CASES = [
    ("Case 1: Transmission wires", "pointcloud_752", 3, "xy"),
    ("Case 2: Poles", "pointcloud_1033", 5, "xy"),
    ("Case 3: Fence", "pointcloud_1308", 7, "xy"),
]
CLASS_NAMES = ["Ground","Vegetation","Buildings","Transmission Wires",
               "Distribution Wires","Poles","Transmission Towers","Fence","Vehicle"]
COLORS = np.array([
    [0.55,0.42,0.28], [0.20,0.55,0.25], [0.72,0.30,0.25],
    [0.90,0.72,0.10], [0.95,0.48,0.08], [0.35,0.25,0.60],
    [0.65,0.18,0.55], [0.15,0.55,0.58], [0.20,0.35,0.75],
])

def load_tile(stem):
    p=ROOT/"datasets/processed/val"/f"{stem}.npz"
    with np.load(p,allow_pickle=False) as z:
        return z["coordinates"].astype(np.int32),z["features"].astype(np.float32),z["labels"].astype(np.int64)

def load_geom(stem):
    p=ROOT/"datasets/geometry_features/val"/f"{stem}_geometry.npz"
    with np.load(p,allow_pickle=False) as z:
        idx=z["point_indices"].astype(np.int64); g=z["geometry_normalized"].astype(np.float32)
    if not np.array_equal(idx,np.arange(len(idx))): raise RuntimeError(f"bad val geometry index: {stem}")
    return g

def main():
    import torch
    from tools.train_geomorph_experiment import model_factory
    from engine.sliding_window_inference import sliding_window_inference

    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    tiles={stem:load_tile(stem) for _,stem,_,_ in CASES}
    geom={stem:load_geom(stem) for _,stem,_,_ in CASES}
    warm_stem="pointcloud_8"; warm=load_tile(warm_stem); warm_geom=load_geom(warm_stem)
    predictions={}
    rows=[]
    for exp,(name,ckpt_path) in MODELS.items():
        model=model_factory(exp).to(device)
        checkpoint=torch.load(ckpt_path,map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
        params=sum(p.numel() for p in model.parameters())
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Uniform val-only warm-up; excluded from timing and memory.
        sliding_window_inference(model,*warm[:2],device,block_size_voxels=500,
            stride_voxels=250,num_classes=9,use_amp=False,
            geometry=None if exp=="Exp0" else warm_geom)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times=[]; windows=0
        for _,stem,_,_ in CASES:
            c,f,l=tiles[stem]
            torch.cuda.synchronize(); t0=time.perf_counter()
            out=sliding_window_inference(model,c,f,device,block_size_voxels=500,
                stride_voxels=250,num_classes=9,use_amp=False,
                geometry=None if exp=="Exp0" else geom[stem])
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
            if not out["logits_finite"]: raise RuntimeError(f"nonfinite val logits {exp} {stem}")
            times.append(elapsed); windows+=out["window_count"]
            if exp in {"Exp0","Exp3","Exp4"}:
                predictions[(exp,stem)]=out["averaged_logits"].argmax(1).numpy().astype(np.int16)
        peak=torch.cuda.max_memory_allocated()
        rows.append({
            "Experiment":exp,"Method":name,"Params":params,"Trainable parameters":trainable,
            "GPU memory (GiB)":f"{peak/2**30:.4f}",
            "Inference time (validation, s/tile)":f"{np.mean(times):.4f}",
            "Inference time std (s/tile)":f"{np.std(times,ddof=1):.4f}",
            "Validation tiles benchmarked":len(CASES),"Validation windows":windows,
            "Precision":"FP32","Checkpoint epoch":checkpoint.get("completed_epoch",checkpoint.get("epoch")),
            "Benchmark scope":"val-only representative tiles; model load and warm-up excluded",
        })
        del model,checkpoint; torch.cuda.empty_cache(); gc.collect()
    with (OUT/"table_complexity.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Save compact val-only prediction artifacts for reproducibility.
    for _,stem,_,_ in CASES:
        c,_,l=tiles[stem]
        np.savez_compressed(ART/f"{stem}_val_predictions.npz",coordinates=c,labels=l,
            baseline=predictions[("Exp0",stem)],exp3=predictions[("Exp3",stem)],
            exp4=predictions[("Exp4",stem)])

    selection_report=[]
    for case,stem,target,view in CASES:
        c,_,labels=tiles[stem]; xyz=c.astype(np.float32)*0.1
        target_mask=labels==target
        target_xy=xyz[target_mask,:2]
        lo=np.percentile(target_xy,1,axis=0)-5; hi=np.percentile(target_xy,99,axis=0)+5
        crop=(xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1])&(labels!=-100)
        selection_report.append({"case":case,"tile":stem,"target_class":CLASS_NAMES[target],
            "target_voxels":int(target_mask.sum()),"crop_voxels":int(crop.sum()),
            "crop_bounds_xy_m":[lo.tolist(),hi.tolist()],"split":"val"})
    (FIG/"selection_report.json").write_text(json.dumps(selection_report,indent=2)+"\n")
    print(json.dumps({"status":"complete","complexity":rows,"cases":selection_report},indent=2))

if __name__=="__main__": main()
