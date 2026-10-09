"""Results figures: confusion matrices (internal and external, Protocol A), Protocol A versus B, internal versus external accuracy.
Run from the repository root:  python analysis/make_result_figures.py   (writes to figures/)"""
# Unpack the stored results (results/*.zip) on first use and run from the repository root
import os, zipfile, glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
for _z in glob.glob(os.path.join("results", "*.zip")):
    if not os.path.isdir(os.path.splitext(_z)[0]):
        zipfile.ZipFile(_z).extractall("results")
os.makedirs("figures", exist_ok=True)
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#8a8a85","axes.labelcolor":"#2b2b29","xtick.color":"#55554f","ytick.color":"#55554f"})
R="results/protocolB"; S=[42,43,44,45,46]
N=["MobileNetV2","EfficientNetB0","Hybrid_EffNetB0_Transformer","EffNetB0_TokenMLP"]
LAB=["MobileNetV2","EfficientNetB0","Hybrid","TokenMLP"]
COL=["#2a78d6","#E69F00","#1baf7a","#4d4d4d"]; MK=["o","s","^","D"]; INK="#2b2b29"; MUTED="#8a8a85"
I={(s,n):json.load(open(f"{R}/seed{s}/{n}/internal.json")) for s in S for n in N}
E={(s,n):json.load(open(f"{R}/seed{s}/{n}/external.json")) for s in S for n in N}
from matplotlib.patches import Rectangle
def vheat(a,M,cmap,vmax=1.0):
    from matplotlib import cm as _cm
    cm_=matplotlib.colormaps[cmap] if isinstance(cmap,str) else cmap
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            a.add_patch(Rectangle((c-0.5,r-0.5),1,1,facecolor=cm_(M[r,c]/vmax),edgecolor="white",linewidth=0.6))
    a.set_xlim(-0.5,M.shape[1]-0.5); a.set_ylim(M.shape[0]-0.5,-0.5); a.set_aspect("equal")
RA="results/protocolA"; NA=N
IA={(s,n):json.load(open(f"{RA}/seed{s}/{n}/internal.json")) for s in S for n in NA}
EA={(s,n):json.load(open(f"{RA}/seed{s}/{n}/external.json")) for s in S for n in NA}

# ---------- Figure: internal accuracy, Protocol A vs Protocol B (paired by seed) ----------
fig,ax=plt.subplots(figsize=(7.2,3.9))
for j,n in enumerate(N):
    vb=np.array([100*I[s,n]["accuracy"] for s in S])
    if n in NA:
        va=np.array([100*IA[s,n]["accuracy"] for s in S]); xa,xb=j-0.17,j+0.17
        for k in range(5): ax.plot([xa,xb],[va[k],vb[k]],color="#c9c9c4",lw=0.9,zorder=1)
        ax.scatter([xa]*5,va,s=30,marker=MK[j],facecolor="white",edgecolor=COL[j],linewidth=1.3,zorder=3)
        ax.errorbar(xa-0.13,va.mean(),yerr=va.std(ddof=1),fmt="_",color=INK,ms=10,mew=2,elinewidth=1.2,capsize=3,zorder=4)
        ax.text(xa-0.2,va.mean(),f"{va.mean():.2f}",va="center",ha="right",fontsize=7.6,color=INK)
        ax.text(j,99.95,("Δ "+f"{vb.mean()-va.mean():+.2f}".replace("-","−")),ha="center",fontsize=7.8,color=MUTED)
    else: xb=j
    ax.scatter([xb]*5,vb,s=30,marker=MK[j],color=COL[j],edgecolor="white",linewidth=0.8,zorder=3)
    ax.errorbar(xb+0.13,vb.mean(),yerr=vb.std(ddof=1),fmt="_",color=INK,ms=10,mew=2,elinewidth=1.2,capsize=3,zorder=4)
    ax.text(xb+0.2,vb.mean(),f"{vb.mean():.2f}",va="center",fontsize=7.6,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(LAB); ax.set_xlim(-0.65,3.6); ax.set_ylim(93.4,100.5)
ax.set_ylabel("Internal test accuracy (%)"); ax.grid(axis="y",color="#e6e6e1",lw=0.8); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker="o",ls="",mfc="white",mec=INK,ms=5.5,label="Protocol A (image-level), seeds 42–46"),
   Line2D([],[],marker="o",ls="",color=MUTED,ms=5.5,label="Protocol B (source-group), seeds 42–46"),
   Line2D([],[],color="#c9c9c4",lw=0.9,label="Same seed"),
   Line2D([],[],marker="_",ls="",color=INK,ms=10,mew=2,label="Mean ± SD")]
ax.legend(handles=h,loc="upper center",bbox_to_anchor=(0.5,-0.11),frameon=False,fontsize=7.4,ncol=4)
fig.tight_layout(); fig.savefig("figures/fig_groupsplit.pdf"); fig.savefig("figures/fig_groupsplit.png",dpi=600); plt.close(fig)

# ---------- Figure: internal vs external, and external confusion matrices ----------
fig=plt.figure(figsize=(7.4,6.4)); gs=fig.add_gridspec(2,4,height_ratios=[1.45,1],hspace=0.42,wspace=0.35)
ax=fig.add_subplot(gs[0,:])
for j,n in enumerate(N):
    xi=[100*I[s,n]["accuracy"] for s in S]; ye=[100*E[s,n]["accuracy"] for s in S]
    ax.scatter(xi,ye,s=40,color=COL[j],marker=MK[j],edgecolor="white",linewidth=0.8,label=LAB[j],zorder=3)
    if n in NA: ax.scatter([100*IA[s,n]["accuracy"] for s in S],[100*EA[s,n]["accuracy"] for s in S],s=40,marker=MK[j],facecolor="white",edgecolor=COL[j],linewidth=1.3,zorder=3)
FM="results/phikon"
for P,fc in [("B","#4a3aa7"),("A","white")]:
    xs=[100*json.load(open(f"{FM}/protocol{P}_seed{s}.json"))["internal"]["accuracy"] for s in S]
    ys=[100*json.load(open(f"{FM}/protocol{P}_seed{s}.json"))["external"]["accuracy"] for s in S]
    ax.scatter(xs,ys,s=90,marker="*",facecolor=fc,edgecolor="#4a3aa7",linewidth=1.2,zorder=4,label="Phikon (frozen)" if P=="B" else None)
ax.scatter([],[],s=40,marker="o",facecolor="white",edgecolor=MUTED,linewidth=1.5,label="open: Protocol A")
ax.scatter([],[],s=40,marker="o",color=MUTED,edgecolor="white",label="filled: Protocol B")
ax.axhline(40.5,ls="--",color=MUTED,lw=1); ax.text(100.2,41.6,"majority-class rate 40.5%",ha="right",fontsize=7.5,color=MUTED)
ax.set_xlabel("Internal LC25000 test accuracy (%)"); ax.set_ylabel("LungHist700 accuracy (%)")
ax.set_xlim(93.5,100.4); ax.set_ylim(38,98); ax.grid(color="#e6e6e1",lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="upper left",frameon=False,fontsize=7.2,ncol=3)
ax.set_title("(a) Internal versus external accuracy",loc="left",fontsize=9,color=INK)
cmap=LinearSegmentedColormap.from_list("blue",["#f4f8fd","#9ec5f4","#3987e5","#184f95","#0d366b"])
cls=["aca","n","scc"]
for j,n in enumerate(N):
    a=fig.add_subplot(gs[1,j]); cm=np.sum([np.array(E[s,n]["confusion_matrix"]) for s in S],0).astype(float); cm=cm/cm.sum(1,keepdims=True)
    vheat(a,cm,cmap)
    for r in range(3):
        for c in range(3): a.text(c,r,f"{cm[r,c]:.2f}",ha="center",va="center",fontsize=7.5,color="white" if cm[r,c]>0.55 else INK)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(cls,fontsize=7.5); a.set_yticklabels(cls if j==0 else [],fontsize=7.5)
    a.set_title(LAB[j],fontsize=8.2,color=INK); a.set_xlabel("Predicted",fontsize=7.5)
    if j==0: a.set_ylabel("True",fontsize=7.5)
    for sp in a.spines.values(): sp.set_visible(False)
    a.tick_params(length=0)
fig.text(0.07,0.405,"(b) External confusion matrices, row-normalised, pooled over five partitions",fontsize=9,color=INK)
fig.savefig("figures/fig_ext_group.pdf",bbox_inches="tight"); fig.savefig("figures/fig_ext_group.png",dpi=600,bbox_inches="tight"); plt.close(fig)

# ---------- Figures: Protocol A confusion matrices pooled over five seeds ----------
def cmfig(D,fname,title_fn,n_lab):
    with plt.rc_context({"font.size":9}):
        fig,axs=plt.subplots(1,len(NA),figsize=(7.4,2.3))
        cl=["Adeno.","Benign","Squam."] if n_lab=="int" else ["Adeno.","Normal","Squam."]
        for j,n in enumerate(NA):
            a=axs[j]; cm=np.sum([np.array(D[s,n]["confusion_matrix"]) for s in S],0); rn=cm/cm.sum(1,keepdims=True)
            vheat(a,rn,"Blues")
            for r in range(3):
                for c in range(3): a.text(c,r,f"{cm[r,c]:,}",ha="center",va="center",fontsize=7.4,color="white" if rn[r,c]>0.55 else INK)
            a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(cl,fontsize=7.0); a.set_yticklabels(cl if j==0 else [],fontsize=7.0)
            a.set_title(title_fn(n,LAB[j]),fontsize=8.0); a.set_xlabel("Predicted",fontsize=8.5)
            if j==0: a.set_ylabel("True",fontsize=8.5)
            for sp in a.spines.values(): sp.set_visible(False)
            a.tick_params(length=0)
        fig.tight_layout(); fig.savefig("figures/"+fname+".pdf",bbox_inches="tight"); fig.savefig("figures/"+fname+".png",dpi=600,bbox_inches="tight"); plt.close(fig)
cmfig(IA,"fig_confusion",lambda n,l:f"{l} ({np.mean([100*IA[s,n]['accuracy'] for s in S]):.2f}%)","int")
cmfig(EA,"fig_ext_confusion",lambda n,l:f"{l} ({np.mean([100*EA[s,n]['accuracy'] for s in S]):.1f}%)","ext")
print("ok")
