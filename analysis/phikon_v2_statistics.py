import os
"""Patient-cluster bootstrap of Phikon-v2 versus the fine-tuned CNN-based models (95% and Bonferroni 99.875%) and versus Phikon."""
import json, os, glob, zipfile, numpy as np, pandas as pd
# run from the repository root: python analysis/phikon_v2_statistics.py (results/*.zip are unpacked on first use)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.join("results", os.path.splitext(os.path.basename(_z))[0])):
        zipfile.ZipFile(_z).extractall("results")
S=[42,43,44,45,46]; R='results'
FT=["MobileNetV2","EfficientNetB0","Hybrid_EffNetB0_Transformer","EffNetB0_TokenMLP"]
csv=pd.read_csv(f'{R}/phikon/protocolA_seed42_external_predictions.csv'); y=csv.y_true.values; pat=csv.patient_id.values
u=np.unique(pat); idx=[np.where(pat==k)[0] for k in u]
def boot(ca,cb,B,seed=0):
    rng=np.random.default_rng(seed); d=np.empty(B)
    for b in range(B):
        s=np.concatenate([idx[k] for k in rng.integers(0,len(u),len(u))]); d[b]=ca[s].mean()-cb[s].mean()
    return 100*d
def corr(p): return (p.argmax(1)==y).astype(float)
v2=lambda P,s: corr(np.load(f'results/phikon_v2_results/Phikon-v2/protocol{P}_seed{s}_external_probs.npy'))
cnn=lambda P,s,m: (pd.read_csv(f'{R}/protocol{P}/seed{s}/{m}/external_predictions.csv').y_pred.values==y).astype(float)
ph=lambda P,s: (pd.read_csv(f'{R}/phikon/protocol{P}_seed{s}_external_predictions.csv').y_pred.values==y).astype(float)
out={}
for P in 'AB':
    lo95=[];lob=[];dd=[]
    for s in S:
        a=v2(P,s)
        for m in FT:
            b=cnn(P,s,m); d=boot(a,b,20000,seed=s)
            dd.append(100*(a.mean()-b.mean())); lo95.append(np.percentile(d,2.5)); lob.append(np.percentile(d,100*0.05/40/2))
    print(P,'v2-CNN diff %.1f-%.1f; 95%% lower %.1f-%.1f; Bonferroni lower %.1f-%.1f'%(min(dd),max(dd),min(lo95),max(lo95),min(lob),max(lob)))
    r=[]
    for s in S:
        a=v2(P,s); b=ph(P,s); d=boot(a,b,2000,seed=s); r.append((100*(a.mean()-b.mean()),np.percentile(d,2.5),np.percentile(d,97.5)))
    print(P,'v2-Phikon',[tuple(round(x,1) for x in t) for t in r])

# summary of the Phikon-v2 frozen-feature baseline (Table 7)
for P in 'AB':
    D=[json.load(open(f'results/phikon_v2_results/Phikon-v2/protocol{P}_seed{s}.json')) for s in S]
    f=lambda k,d: np.array([x[d][k] for x in D])*100
    print(P,'internal %.2f ± %.2f; external %.1f ± %.1f; balanced %.1f; normal recall %.2f; patient-class %.1f'%(f('accuracy','internal').mean(),f('accuracy','internal').std(ddof=1),
          f('accuracy','external').mean(),f('accuracy','external').std(ddof=1),f('balanced_accuracy','external').mean(),np.mean([x['external']['recall']['lung_n'] for x in D]),f('patient_class_accuracy','external').mean()))
