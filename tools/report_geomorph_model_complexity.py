#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
from models.geomorph import GeometryConcatMinkUNet, GeometryEncodedInputMinkUNet, GeometrySingleFusionMinkUNet, GeoMorphMinkUNet
def count(m): return sum(p.numel() for p in m.parameters())
def main():
 kwargs={'num_classes':9,'dimension':3,'stem_channels':32,'planes':(32,64,128,256,256,128,96,96),'layers':(2,3,4,6,2,2,2,2)}
 models=[('B0','Baseline-Strong',MinkUNet34C(in_channels=6,**kwargs),6,0,'none'),('B1','Geometry-Concat',GeometryConcatMinkUNet(**kwargs),18,0,'input'),('B2','Geometry-Encoder',GeometryEncodedInputMinkUNet(**kwargs),22,16,'none'),('B3','Geometry-SingleFusion',GeometrySingleFusionMinkUNet(**kwargs),6,16,'encoder1'),('Ours','GeoMorph-Seg',GeoMorphMinkUNet(**kwargs),6,16,'encoder1,encoder2')]
 base=count(models[0][2]); rows=[]
 for ident,name,m,inp,emb,loc in models:
  total=count(m); rows.append({'model_id':ident,'model_name':name,'input_baseline_dim':6,'geometry_dim':12 if ident!='B0' else 0,'geometry_embedding_dim':emb,'fusion_locations':loc,'total_parameters':total,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'additional_parameters_vs_baseline':total-base,'relative_increase_percent':100*(total-base)/base})
 out=Path('results/geomorph_phase4a/model_complexity.csv'); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 return 0
if __name__=='__main__': raise SystemExit(main())
