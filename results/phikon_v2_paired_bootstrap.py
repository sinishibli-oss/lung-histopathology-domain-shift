"""Patient-cluster bootstrap of Phikon-v2 versus the fine-tuned CNN-based models (95% and Bonferroni 99.875%) and versus Phikon."""
import json,numpy as np,pandas as pd
S=[42,43,44,45,46]; R='results'  # run from the repository root after unzipping results/phikon_v2_results.zip inside results/
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
