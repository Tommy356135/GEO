# Command Index

Commands below are transcribed from existing entry points and are **not
executed by this package**. Run from `<PROJECT_ROOT>` in the recorded
environment.

| Stage | Command | Status |
|---|---|---|
| Preprocess train/val only | `python tools/prepare_eclair_minkowski.py --config configs/dataset/eclair_minkowski.yaml --splits train,val` | CLI confirmed |
| Build 50 m block indices | `python tools/build_eclair_block_index.py --block-size-m 50 --processed-root datasets/processed --output-root datasets/block_indices_50m` | CLI confirmed; implementation also expects held-out split, so Manual confirmation required before restricted use |
| Merge train blocks | `python tools/build_eclair_block50_merged_train_index.py` | entry point confirmed |
| Geometry train cache | `python tools/prepare_eclair_geometry_features.py --all-train --output-root datasets/geometry_features` | CLI confirmed |
| Geometry val cache | `python tools/prepare_eclair_geometry_features.py --all-val --output-root datasets/geometry_features` | CLI confirmed |
| Exp0 training | `python -c "from pathlib import Path; from tools.train_baseline_strong_block import run; raise SystemExit(run(Path('configs/train/baseline_strong_block50_merged_amp_safe.yaml')))"` | callable confirmed; Manual confirmation required because no dedicated CLI wrapper is retained |
| Exp1 formal historical resume | `python tools/train_exp1_resume_fp32val.py` | exact historical entry point; hard-coded prior checkpoint/output paths |
| Exp2 training | `python tools/train_geomorph_experiment.py --experiment exp2 --output-dir results/exp2_geometry_encoder_final --epochs 100` | CLI confirmed |
| Exp3 training | `python tools/train_geomorph_experiment.py --experiment exp3 --output-dir results/exp3_single_fusion_final --epochs 100` | CLI confirmed |
| Exp4 training | `python tools/train_geomorph_experiment.py --experiment exp4 --output-dir results/exp4_geomorph_seg_final --epochs 100` | CLI confirmed; paper name GeoMorph-DF |
| Held-out geometry cache | `python tools/build_test_geometry_cache.py` | exact entry point; final-evaluation stage only |
| Validation/final evaluation | integrated in formal training entry points | no separate general CLI located |
| Parameter counts | `python tools/report_geomorph_model_complexity.py` | existing entry point; writes legacy-named internal output |
| Representative complexity/val predictions | `python results/paper_phase7_revision/run_val_revision.py` | existing fixed-path script; Manual confirmation required |
| Figure 6 | `python results/paper_phase7_revision/plot_val_qualitative.py` | existing fixed-path script |
| Figures 1–5 | Manual confirmation required | generation scripts NOT LOCATED |
| Tables 1–4 | Manual confirmation required | generation scripts NOT LOCATED |
| Tables 5–6 | inspect existing Phase7 script and frozen CSV | no standalone table-only CLI |

The final held-out evaluation must occur only after the validation-selected
best checkpoint is fixed. Commands that train or evaluate are documentation,
not actions performed during Phase 9B.
