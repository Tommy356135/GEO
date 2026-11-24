#!/usr/bin/env python3
"""Build deterministic, independent geometry caches for selected train tiles."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from geometry.geometry_pipeline import compute_multiscale_geometry, load_simple_yaml
from geometry.frozen_features import FEATURE_NAMES, finalize_geometry_features, normalize_geometry, load_frozen_normalization

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()

def atomic_save(path: Path, arrays: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)

def validate_cache(source: Path, out: Path) -> tuple[bool, str]:
    if not out.exists(): return False, 'missing'
    try:
        with np.load(source, allow_pickle=False) as s, np.load(out, allow_pickle=False) as d:
            n=len(s['coordinates']); meta=json.loads(str(d['metadata']))
            if meta.get('source_npz_sha256') != sha256(source): return False, 'source_hash_mismatch'
            if len(d['point_indices']) != n or not np.array_equal(d['point_indices'], np.arange(n,dtype=np.int64)): return False, 'point_index_mismatch'
            if d['geometry_raw_small'].shape != (n,6) or d['geometry_raw_large'].shape != (n,6) or d['geometry_normalized'].shape != (n,12): return False, 'shape_error'
            if not all(np.isfinite(d[k]).all() for k in ('geometry_raw_small','geometry_raw_large','geometry_normalized')): return False, 'nonfinite'
        return True, 'valid'
    except Exception as e: return False, f'invalid:{e}'

def build_tile(source: Path, out: Path, split: str, chunk_size: int, overwrite: bool, resume: bool=False) -> dict:
    if out.exists() and not overwrite:
        ok, reason=validate_cache(source,out)
        if ok and resume: return {'tile_name': source.name, 'status': 'skipped_valid', 'cache': str(out), 'cache_size_bytes': out.stat().st_size}
        if ok and not resume: raise RuntimeError(f'cache exists; use --resume or --overwrite: {out}')
        raise RuntimeError(f'cache validation failed ({reason}): {out}')
    started=time.perf_counter()
    with np.load(source, allow_pickle=False) as d:
        coordinates=np.asarray(d['coordinates'], dtype=np.int32)
    # Translation and scale do not affect covariance/KNN ordering.
    points=coordinates.astype(np.float64)*0.1
    tree=cKDTree(points)
    n=len(points)
    small_raw=np.empty((n,6), dtype=np.float32); large_raw=np.empty((n,6), dtype=np.float32)
    small_valid=np.empty(n, dtype=bool); large_valid=np.empty(n, dtype=bool)
    for start in range(0,n,chunk_size):
        stop=min(start+chunk_size,n); q=np.arange(start,stop,dtype=np.int64)
        values=compute_multiscale_geometry(points,q,[16,64],minimum_valid_neighbors=8,epsilon=1e-12,exclude_query_point=False,batch_size=16384,direction_consistency_scales=[16,64],tree=tree)
        small,sv=finalize_geometry_features(values[16]); large,lv=finalize_geometry_features(values[64])
        small_raw[start:stop]=small.astype(np.float32); large_raw[start:stop]=large.astype(np.float32)
        small_valid[start:stop]=sv; large_valid[start:stop]=lv
    norm=load_frozen_normalization()
    normalized=normalize_geometry(np.concatenate([small_raw.astype(np.float64),large_raw.astype(np.float64)],axis=1),norm).astype(np.float32)
    metadata={
        'tile_name':source.name,'split':split,'point_count':n,'source_npz_sha256':sha256(source),
        'source_coordinates_sha256':hashlib.sha256(coordinates.tobytes()).hexdigest(),
        'geometry_definition_version':'geomorph_geometry_v2','geometry_config_sha256':sha256(Path('configs/geometry/geomorph_features.yaml')),
        'invalid_policy_sha256':sha256(Path('configs/geometry/geometry_invalid_policy.yaml')),'normalization_sha256':sha256(Path('configs/geometry/geomorph_normalization.json')),
        'feature_names':list(FEATURE_NAMES),'small_k':16,'large_k':64,'dtype':'float32','creation_time':'2026-07-18T00:00:00Z','code_version_or_git_state':'UNBORN_HEAD-deterministic-cache-v1'}
    tmp=out.with_suffix(out.suffix+'.tmp')
    atomic_save(tmp,{'point_indices':np.arange(n,dtype=np.int64),'geometry_raw_small':small_raw,'geometry_raw_large':large_raw,'geometry_normalized':normalized,'geometry_valid_small':small_valid,'geometry_valid_large':large_valid,'metadata':np.asarray(json.dumps(metadata,sort_keys=True))})
    ok, reason=validate_cache(source,tmp)
    if not ok: raise RuntimeError(f'post-write validation failed ({reason})')
    tmp.replace(out)
    return {'tile_name':source.name,'status':'complete','point_count':n,'cache':str(out),'runtime_seconds':time.perf_counter()-started,'small_valid_fraction':float(small_valid.mean()),'large_valid_fraction':float(large_valid.mean()),'cache_size_bytes':out.stat().st_size}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--selection',type=Path,default=Path('results/geomorph_phase4a/cache_smoke_tile_selection.json')); ap.add_argument('--output-root',type=Path,default=Path('datasets/geometry_features_smoke')); ap.add_argument('--chunk-size',type=int,default=8192); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--all-train',action='store_true'); ap.add_argument('--all-val',action='store_true'); ap.add_argument('--tiles',nargs='*'); args=ap.parse_args()
    split='val' if args.all_val else 'train'
    if args.all_train or args.all_val:
        names=sorted(p.name for p in (Path('datasets/processed')/split).glob('*.npz'))
    else:
        selection=json.loads(args.selection.read_text()); names=[r['tile_name'] for r in selection['tiles']]
    if args.tiles: names=[n for n in names if n in set(args.tiles)]
    rows=[]; progress=Path('results/geomorph_phase4c/val_cache_progress.csv') if args.all_val else (Path('results/geomorph_phase4b/train_cache_progress.csv') if args.all_train else None)
    if progress and not progress.exists(): progress.parent.mkdir(parents=True,exist_ok=True); progress.write_text('tile_index,tile_name,point_count,runtime_seconds,cache_size_bytes,status\n')
    for i,name in enumerate(names):
        source=Path('datasets/processed')/split/name; out=args.output_root/split/f"{source.stem}_geometry.npz"; started=time.perf_counter()
        try: row=build_tile(source,out,split,args.chunk_size,args.overwrite,args.resume)
        except Exception as e:
            row={'tile_name':name,'status':'failed','exception':repr(e),'elapsed_seconds':time.perf_counter()-started}; rows.append(row)
            if progress: progress.open('a').write(f"{i},{name},,,,'failed'\n")
            (args.output_root/'cache_build_failure.json').write_text(json.dumps({'failed_tile':name,'rows_completed':rows,'exception':repr(e)},indent=2)+'\n'); raise
        rows.append(row); print(name,row['status'],flush=True)
        if progress:
            progress.open('a').write(f"{i},{name},{row.get('point_count','')},{row.get('runtime_seconds','')},{row.get('cache_size_bytes','')},{row['status']}\n")
    (args.output_root/f'{split}_cache_build_summary.json').write_text(json.dumps({'tiles':rows,'geometry_definition_version':'geomorph_geometry_v2','source_split':split},indent=2)+'\n')
    return 0
if __name__=='__main__': raise SystemExit(main())
