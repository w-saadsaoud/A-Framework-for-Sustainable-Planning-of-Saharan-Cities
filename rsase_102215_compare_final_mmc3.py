"""
compare_std_vs_rccam_final.py
==============================
Correction : expansion favorable = CODES_ZONE NaN
On utilise class_zone == 'expansion favorable' pour le masque eligible.
"""

import os, json, csv
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import uniform_filter
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              cohen_kappa_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE     = r"Chemain données"
RASTERS  = {
    1990: os.path.join(BASE, "01_DONNEES", "Rasters", "Urban_1990.tif"),
    2010: os.path.join(BASE, "01_DONNEES", "Rasters", "Urban_2010.tif"),
    2025: os.path.join(BASE, "01_DONNEES", "Rasters", "Urban_2025.tif"),
}
SHP_PATH = os.path.join(BASE, "01_DONNEES", "Shapefiles",
                         "contrainte_emreinte1.shp")
OUT_DIR  = os.path.join(BASE, "04_SCRIPTS_PEER_REVIEW", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_YEARS   = 15
RADIUS    = 1
N_NEW_OBS = 858   # 6499 - 5641

print("=" * 70)
print("STANDARD CA-MARKOV  vs  RC-CAM  ---  Version finale")
print("expansion favorable = eligible  |  autres = restreint")
print("=" * 70)

# ── 1. Rasters ────────────────────────────────────────────────────────
print("\n[1/6] Chargement des rasters ...")
rasters = {}; meta = None; transform = None; crs = None
for yr, path in RASTERS.items():
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float32)
        rasters[yr] = np.where(d > 0, 1, 0).astype(np.uint8)
        if meta is None:
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
        print(f"  {yr}: urban={int(rasters[yr].sum())} px")

H, W     = rasters[1990].shape
total_px = H * W
observed = rasters[2025].flatten().astype(int)
base2010 = rasters[2010].flatten()

# ── 2. CODESZone depuis class_zone ───────────────────────────────────
print("\n[2/6] Rasterisation CODESZone ...")
gdf = gpd.read_file(SHP_PATH)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

# Masque eligible : class_zone == 'expansion favorable'
# Masque restreint : class_zone in ['expansion interdite','reconversion possible']
shapes_elig = [
    (geom, 1)
    for geom, cl in zip(gdf.geometry, gdf['class_zone'])
    if cl == 'expansion favorable'
]
shapes_rest = [
    (geom, 1)
    for geom, cl in zip(gdf.geometry, gdf['class_zone'])
    if cl in ['expansion interdite', 'reconversion possible']
]
shapes_reconv = [
    (geom, 1)
    for geom, cl in zip(gdf.geometry, gdf['class_zone'])
    if cl == 'reconversion possible'
]

eligible_raster   = rasterize(shapes_elig,  out_shape=(H,W),
                               transform=transform, fill=0, dtype=np.uint8)
restricted_raster = rasterize(shapes_rest,  out_shape=(H,W),
                               transform=transform, fill=0, dtype=np.uint8)
reconv_raster     = rasterize(shapes_reconv,out_shape=(H,W),
                               transform=transform, fill=0, dtype=np.uint8)
# Reconversion : traitement partiel possible (w=0.8 dans l'article)
# Pour la comparaison binaire, on l'inclut dans restreint

n_elig  = int(eligible_raster.sum())
n_rest  = int(restricted_raster.sum())
n_reconv= int(reconv_raster.sum())
print(f"  Pixels eligibles   (expansion favorable)   : {n_elig:,}")
print(f"  Pixels restreints  (expansion interdite)   : {n_rest - n_reconv:,}")
print(f"  Pixels reconversion(reconversion possible)  : {n_reconv:,}")
print(f"  Pixels hors SHP                            : {total_px-n_elig-n_rest:,}")

if n_elig == 0:
    raise SystemExit("ERREUR : toujours 0 pixels eligibles")

# Sauvegarder
cz_path = os.path.join(OUT_DIR, "codeszone_final.tif")
with rasterio.open(cz_path,'w',driver='GTiff',height=H,width=W,count=1,
                   dtype=np.uint8,crs=crs,transform=transform) as dst:
    dst.write(eligible_raster,1)
print(f"  Masque eligible sauvegarde : {cz_path}")

# ── 3. Matrice Markov ─────────────────────────────────────────────────
print("\n[3/6] Matrice de transition (1990-2010) ...")
u90 = rasters[1990].flatten(); u10 = rasters[2010].flatten()
P_nu2u = float(u10[u90==0].mean())
P_u2u  = float(u10[u90==1].mean())
print(f"  P(non-urbain->urbain) = {P_nu2u:.4f}")
print(f"  P(urbain->urbain)     = {P_u2u:.4f}")

# ── 4. Simulation ─────────────────────────────────────────────────────
print("\n[4/6] Simulation 2010->2025 ...")

def simulate(initial, mask, label):
    state = initial.copy().astype(float)
    for t in range(N_YEARS):
        infl = uniform_filter(state, size=2*RADIUS+1)
        prob = np.where(state==1, P_u2u,
                        P_nu2u*(0.5+0.5*infl))
        prob = np.clip(prob, 0, 1)
        if mask is not None:
            prob = prob * mask  # zones restreintes -> prob=0
        state = np.where(prob >= 1.0, 1, state)
    print(f"  {label}: min={prob.min():.4f} max={prob.max():.4f} "
          f"mean={prob.mean():.4f}  pixels_actifs={(prob>0).sum():,}")
    return prob

prob_std = simulate(rasters[2010].astype(float), None,
                    "Standard CA-Markov")
prob_rc  = simulate(rasters[2010].astype(float),
                    eligible_raster.astype(float), "RC-CAM")

# ── 5. Calibration du seuil ───────────────────────────────────────────
print(f"\n[5/6] Calibration seuils (cible ~{N_NEW_OBS} nouveaux px) ...")

def find_thresh(prob_surf, label):
    pf = prob_surf.flatten()
    bt,bd,bn = 1.0,9999,0
    for t in np.arange(0.001,0.999,0.001):
        n = int(((pf>=t)&(base2010==0)).sum())
        d = abs(n-N_NEW_OBS)
        if d < bd:
            bd=d; bt=round(t,3); bn=n
    print(f"  {label}: seuil={bt}  nouveaux={bn} (cible={N_NEW_OBS})")
    return bt

thresh_std = find_thresh(prob_std, "Standard CA-Markov")
thresh_rc  = find_thresh(prob_rc,  "RC-CAM")

# ── 6. Metriques ──────────────────────────────────────────────────────
print("\n[6/6] Calcul des metriques ...")

def compute(prob_surf, threshold, label):
    pf   = prob_surf.flatten()
    pred = np.where(pf>=threshold, 1, base2010).astype(int)
    tn,fp,fn,tp = confusion_matrix(observed,pred,labels=[0,1]).ravel()
    kappa  = cohen_kappa_score(observed, pred)
    oa     = (tp+tn)/total_px
    prec   = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1     = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0
    fom    = tp/(fn+tp+fp) if (fn+tp+fp)>0 else 0.0
    auc    = roc_auc_score(observed, pf)
    new_u  = np.maximum(pred-base2010, 0)
    tot    = int(new_u.sum())
    rci    = 100.0*(new_u*eligible_raster.flatten()).sum()/tot if tot>0 else 0.0
    fp_r   = int((new_u*restricted_raster.flatten()).sum())
    print(f"\n  === {label} (seuil={threshold}) ===")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"  FP dans zones restreintes : {fp_r:,}")
    print(f"  Nouveaux pixels predits   : {tot:,}")
    print(f"  Kappa={kappa:.4f}  F1={f1:.4f}  OA={oa:.4f}")
    print(f"  FoM={fom:.4f}  AUC={auc:.4f}")
    print(f"  Precision={prec:.4f}  Recall={recall:.4f}  RCI={rci:.1f}%")
    return dict(label=label, thresh=threshold,
                TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
                FP_restricted=fp_r, NewUrban=tot,
                Kappa=round(kappa,4),F1=round(f1,4),OA=round(oa,4),
                FoM=round(fom,4),AUC=round(auc,4),
                Precision=round(prec,4),Recall=round(recall,4),
                RCI=round(rci,1))

m_std = compute(prob_std, thresh_std, "Standard CA-Markov")
m_rc  = compute(prob_rc,  thresh_rc,  "RC-CAM")

# ── Tableau ───────────────────────────────────────────────────────────
print("\n"+"="*70)
print("TABLEAU COMPARATIF  ---  TABLE 5 FINALE")
print("="*70)
pairs = [
    ("Kappa Coefficient",      "Kappa"),
    ("F1-Score",               "F1"),
    ("Overall Accuracy",       "OA"),
    ("Figure of Merit",        "FoM"),
    ("AUC (ROC)",              "AUC"),
    ("Precision (urban)",      "Precision"),
    ("Recall (urban)",         "Recall"),
    ("RCI (%)",                "RCI"),
    ("FP in restricted zones", "FP_restricted"),
]
print(f"  {'Metrique':<28} {'Std CA-Markov':>14} {'RC-CAM':>10} {'Delta':>10}")
print("-"*66)
rows = []
for lbl,key in pairs:
    vs=m_std[key]; vr=m_rc[key]
    d = f"{(vr-vs)/abs(vs)*100:+.1f}%" if isinstance(vs,(int,float)) and vs!=0 else "--"
    print(f"  {lbl:<28} {str(vs):>14} {str(vr):>10} {d:>10}")
    rows.append([lbl,vs,vr,d])
print("="*70)

# Sauvegarde
json_p = os.path.join(OUT_DIR,"comparison_final.json")
with open(json_p,"w") as f:
    json.dump({"standard":m_std,"rccam":m_rc},f,indent=2)
csv_p = os.path.join(OUT_DIR,"comparison_table5_final.csv")
with open(csv_p,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["Metric","Standard CA-Markov","RC-CAM","Delta"])
    for r in rows: w.writerow(r)

# Figure ROC
fig,ax=plt.subplots(figsize=(8,7))
fpr_s,tpr_s,_=roc_curve(observed,prob_std.flatten())
fpr_r,tpr_r,_=roc_curve(observed,prob_rc.flatten())
ax.plot(fpr_s,tpr_s,color='steelblue',lw=2.5,
        label=f"Standard CA-Markov (AUC={m_std['AUC']:.4f})")
ax.plot(fpr_r,tpr_r,color='crimson',lw=2.5,
        label=f"RC-CAM (AUC={m_rc['AUC']:.4f})")
ax.plot([0,1],[0,1],'k--',lw=1.2,label='Random classifier')
ax.set_xlabel('False Positive Rate',fontsize=13)
ax.set_ylabel('True Positive Rate',fontsize=13)
ax.set_title('ROC Comparison: Standard CA-Markov vs RC-CAM\n'
             '(2010-2025 validation, CODESZone from contrainte_emreinte1.shp)',
             fontsize=13,fontweight='bold')
ax.legend(fontsize=12,loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01]); ax.grid(alpha=0.3)
plt.tight_layout()
roc_p=os.path.join(OUT_DIR,"roc_comparison_final.png")
plt.savefig(roc_p,dpi=300,bbox_inches='tight'); plt.close()

print(f"\n  JSON : {json_p}")
print(f"  CSV  : {csv_p}")
print(f"  ROC  : {roc_p}")
print("="*70+"  TERMINE  "+"="*70)
