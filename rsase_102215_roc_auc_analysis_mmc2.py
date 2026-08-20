#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROC/AUC ANALYSIS FOR CA-MARKOV URBAN GROWTH MODEL
==================================================
Computes receiver operating characteristic (ROC) curves and area under
the curve (AUC) using the REAL transition probability surface generated
by the CA-Markov validation run (2010-2025).

Scientific rationale
--------------------
A CA-Markov model produces a binary output (urban / non-urban). To derive
a genuine probability surface for ROC analysis, we run the model in
VALIDATION mode (2010->2025, 15 annual steps) and accumulate, at each
step, the combined transition probability p_combined(i,j):

    p_combined = 0.20 * P_markov
               + 0.30 * neighbourhood_density
               + 0.50 * CODESZone_weight

The MAXIMUM p_combined reached by each non-urban pixel over the 15-year
period is used as its urban transition score. This surface represents
the model's genuine estimate of each pixel's propensity to urbanise and
is directly comparable to probability scores used in the ROC literature
(Pontius & Schneider 2001; Eastman et al. 2005).

This approach avoids the methodological error of smoothing a binary
prediction map with a Gaussian filter, which inflates AUC artificially
and is not interpretable in the standard probabilistic sense.

Usage
-----
    # Step 1: run ca_markov_urban_growth.py first to generate
    #          transition_matrix.json and verify data paths
    # Step 2:
    conda activate base
    python roc_auc_analysis.py

Outputs (written to RESULTS_DIR)
---------------------------------
    probability_surface_2025.tif   - Real transition probability map
    roc_auc_figure.png             - ROC + Precision-Recall curves
    threshold_analysis.csv         - Metrics at each decision threshold
    roc_auc_metrics.json           - All computed metrics (JSON)

References
----------
    Eastman et al. (2005) ROC and LCM. IDRISI manual.
    Pontius & Schneider (2001) Land-cover change model validation.
    Fawcett (2006) An introduction to ROC analysis.

Author  : [Author name(s)]
Version : 1.0
License : MIT
"""

import os
import sys
import json
import warnings
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from rasterio import features
from pathlib import Path
from scipy import ndimage
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION -- must match ca_markov_urban_growth.py
# =============================================================================

BASE_DIR = r"D:\Ouargla_CA_Project_fin_wahid"   # <-- adapt to your machine

RASTER_DIR  = os.path.join(BASE_DIR, "01_DONNEES", "Rasters")
SHP_DIR     = os.path.join(BASE_DIR, "01_DONNEES", "Shapefiles")
RESULTS_DIR = os.path.join(BASE_DIR, "04_SCRIPTS_PEER_REVIEW", "outputs")

RASTER_FILES = {
    1990: "Urban_1990.tif",
    2000: "Urban_2000.tif",
    2010: "Urban_2010.tif",
    2025: "Urban_2025.tif",
}

SHP_CONSTRAINT = "contrainte_emreinte1.shp"
SHP_PERIMETER  = "perimetre_urbain_futur.shp"

# Validation run parameters (2010 -> 2025)
VALIDATION_YEARS   = 15          # 2010 -> 2025
CODES_ZONE_WEIGHTS = {0: 1.0, 1: 0.0, 2: 0.8}  # codes present: 1.0, 2.0
EXPANSION_THRESHOLD = 0.15

# Sampling cap for ROC computation (to manage memory)
MAX_SAMPLE_PIXELS = 200_000

np.random.seed(42)   # reproducibility

# =============================================================================
# DATA LOADING (shared with ca_markov_urban_growth.py)
# =============================================================================

def find_raster(year):
    filename = RASTER_FILES.get(year)
    if filename:
        path = os.path.join(RASTER_DIR, filename)
        if os.path.exists(path):
            return path
    matches = list(Path(RASTER_DIR).glob(f"*{year}*.tif"))
    return str(matches[0]) if matches else None


def load_rasters(years):
    rasters, profile = {}, None
    for year in years:
        path = find_raster(year)
        if path is None:
            print(f"  [WARNING] Raster not found for year {year}")
            continue
        with rasterio.open(path) as src:
            rasters[year] = src.read(1).astype(np.uint8)
            if profile is None:
                profile = src.profile.copy()
                profile.update(dtype=np.float32, compress='lzw')
        print(f"  {year}: {rasters[year].shape}  "
              f"urban={int(np.sum(rasters[year]==1))} px")
    return rasters, profile


def rasterize_codes_zone(shp_path, shape, transform, weights=None):
    """
    CODESZone weight raster - correct 3-zone interpretation:
      NaN (uncoded) = Zone 0 Favourable  -> weight 1.0 (default)
      1.0           = Zone 1 Restricted  -> weight 0.0
      2.0           = Zone 2 Reconversion-> weight 0.8
    """
    if not os.path.exists(shp_path):
        return np.ones(shape, dtype=np.float32)
    gdf = gpd.read_file(shp_path)
    if 'CODES_ZONE' not in gdf.columns:
        return np.ones(shape, dtype=np.float32)

    weight_raster = np.ones(shape, dtype=np.float32)

    z1 = gdf[gdf['CODES_ZONE'] == 1.0]
    if not z1.empty:
        m1 = features.rasterize([(g,1) for g in z1.geometry],
            out_shape=shape, transform=transform, fill=0, dtype=np.uint8)
        weight_raster[m1 == 1] = 0.0

    z2 = gdf[gdf['CODES_ZONE'] == 2.0]
    if not z2.empty:
        m2 = features.rasterize([(g,1) for g in z2.geometry],
            out_shape=shape, transform=transform, fill=0, dtype=np.uint8)
        mask2 = (m2 == 1) & (weight_raster != 0.0)
        weight_raster[mask2] = 0.8

    return weight_raster
def rasterize_perimeter(shp_path, shape, transform):
    gdf = gpd.read_file(shp_path)
    return features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=shape, transform=transform,
        fill=0, dtype=np.uint8
    )


def compute_transition_matrix(rasters, t0, t1):
    a = (rasters[t0] > 0).astype(np.int8).ravel()
    b = (rasters[t1] > 0).astype(np.int8).ravel()
    P_raw = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            P_raw[i, j] = np.sum((a == i) & (b == j))
    row_sums = P_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return P_raw / row_sums


def average_transition_matrix(rasters):
    epochs = [(1990, 2000), (2000, 2010)]
    matrices = [compute_transition_matrix(rasters, t0, t1)
                for t0, t1 in epochs
                if t0 in rasters and t1 in rasters]
    if not matrices:
        return np.array([[0.990, 0.010], [0.020, 0.980]])
    return np.mean(np.stack(matrices), axis=0)

# =============================================================================
# VALIDATION SIMULATION WITH PROBABILITY SURFACE OUTPUT
# =============================================================================

def run_validation_with_probability(initial_state, transition_matrix,
                                    expansion_mask, weight_raster,
                                    n_years, threshold):
    """
    Run CA-Markov in VALIDATION mode (2010->2025) and return both
    the final binary prediction AND the maximum transition probability
    reached by each pixel during the simulation.

    The probability surface is defined as:

        prob(i, j) = max over t in [1..n_years] of p_combined(i, j, t)

    where p_combined is the weighted combination of the Markov component,
    the Moore neighbourhood density, and the CODESZone weight.

    For pixels that were already urban at t=0 (initial_state == 1), we
    assign prob = 1.0 (they are urban with certainty).
    For pixels outside the expansion mask, prob = 0.0 (structurally
    forbidden by regulatory constraints).

    Parameters
    ----------
    initial_state     : ndarray (H, W) float32  binary map at t=0
    transition_matrix : ndarray (2, 2)
    expansion_mask    : ndarray (H, W) bool
    weight_raster     : ndarray (H, W) float32
    n_years           : int
    threshold         : float

    Returns
    -------
    prediction    : ndarray (H, W) uint8    0=non-urban, 255=urban
    prob_surface  : ndarray (H, W) float32  probability in [0, 1]
    """
    p_markov   = float(transition_matrix[0, 1])
    state      = initial_state.astype(np.float32).copy()
    kernel     = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0

    # Accumulate maximum probability for non-urban pixels
    prob_surface = np.where(initial_state > 0, 1.0, 0.0).astype(np.float32)

    for year_step in range(n_years):
        # Neighbourhood density
        neighbour_density = ndimage.convolve(
            state, kernel, mode='constant', cval=0.0) / 8.0

        # Combined probability surface
        # Multiplicative formula: CODESZone weight as multiplier
        p_base     = (0.50 * p_markov + 0.50 * neighbour_density)
        p_combined = p_base * weight_raster

        # Update probability surface: take maximum over time steps
        # Only for pixels eligible for expansion
        eligible = (state == 0) & expansion_mask
        prob_surface = np.where(
            eligible,
            np.maximum(prob_surface, p_combined),
            prob_surface
        )

        # Transition rule
        np.random.seed(year_step * 7 + 13)
        rng_draw = np.random.random(state.shape).astype(np.float32)
        new_urban = eligible & (p_combined > threshold) & (rng_draw < p_combined)
        state[new_urban] = 1.0

    # Pixels outside expansion mask keep prob = 0
    prob_surface = np.where(expansion_mask | (initial_state > 0),
                            prob_surface, 0.0)

    prediction = (state > 0.5).astype(np.uint8) * 255
    return prediction, prob_surface.astype(np.float32)

# =============================================================================
# ROC / AUC COMPUTATION
# =============================================================================

def compute_roc_metrics(y_true, y_scores, max_samples=MAX_SAMPLE_PIXELS):
    """
    Compute ROC curve, AUC, Precision-Recall curve and discrimination
    metrics from binary labels and continuous scores.

    Sampling is applied when the number of pixels exceeds max_samples
    to keep memory usage reasonable, while preserving the class ratio
    via stratified sampling.

    Parameters
    ----------
    y_true    : ndarray (N,) int    ground truth (0/1)
    y_scores  : ndarray (N,) float  probability scores in [0, 1]
    max_samples : int

    Returns
    -------
    metrics : dict
    """
    # Stratified sampling
    if len(y_true) > max_samples:
        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]
        n_pos   = min(len(idx_pos), max_samples // 2)
        n_neg   = min(len(idx_neg), max_samples - n_pos)
        idx = np.concatenate([
            np.random.choice(idx_pos, n_pos, replace=False),
            np.random.choice(idx_neg, n_neg, replace=False)
        ])
        y_true_s   = y_true[idx]
        y_scores_s = y_scores[idx]
        print(f"  Stratified sample: {n_pos} urban + {n_neg} non-urban pixels")
    else:
        y_true_s, y_scores_s = y_true, y_scores

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_s, y_scores_s)
    roc_auc = float(auc(fpr, tpr))

    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true_s, y_scores_s)
    avg_precision = float(average_precision_score(y_true_s, y_scores_s))

    # Optimal threshold (Youden's J statistic)
    youden_j   = tpr - fpr
    best_idx   = int(np.argmax(youden_j))
    best_thr   = float(thresholds[best_idx])

    # Metrics at optimal threshold
    y_pred_opt = (y_scores_s >= best_thr).astype(int)
    cm         = confusion_matrix(y_true_s, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_opt      = float(f1_score(y_true_s, y_pred_opt))
    bal_acc     = float((sensitivity + specificity) / 2.0)

    # Threshold analysis table
    thr_table = []
    for thr_val in np.arange(0.05, 1.00, 0.05):
        idx_t = np.searchsorted(thresholds[::-1], thr_val)
        idx_t = min(idx_t, len(fpr) - 1)
        thr_table.append({
            "threshold"  : round(float(thr_val), 2),
            "FPR"        : round(float(fpr[-(idx_t+1)]), 4),
            "TPR"        : round(float(tpr[-(idx_t+1)]), 4),
            "specificity": round(float(1 - fpr[-(idx_t+1)]), 4),
        })

    metrics = {
        "n_pixels_total"       : int(len(y_true)),
        "n_pixels_urban_obs"   : int(np.sum(y_true == 1)),
        "n_pixels_nonurban_obs": int(np.sum(y_true == 0)),
        "roc_auc"              : roc_auc,
        "average_precision"    : avg_precision,
        "optimal_threshold_youden": best_thr,
        "at_optimal_threshold" : {
            "sensitivity" : sensitivity,
            "specificity" : specificity,
            "f1_score"    : f1_opt,
            "balanced_accuracy": bal_acc,
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
        },
        "roc_curve"  : {"fpr": fpr.tolist(),  "tpr": tpr.tolist()},
        "pr_curve"   : {"precision": precision.tolist(),
                        "recall"   : recall.tolist()},
        "threshold_table": thr_table,
        "note": (
            "Probability scores derived from the maximum combined "
            "transition probability (p_combined) accumulated over "
            "the CA-Markov validation run (2010-2025). "
            "Stratified random sampling applied for ROC computation."
        )
    }
    return metrics, fpr, tpr, precision, recall


# =============================================================================
# FIGURES
# =============================================================================

def plot_roc_figure(metrics, fpr, tpr, precision, recall, save_path):
    """Generate publication-quality ROC and Precision-Recall figure."""
    roc_auc       = metrics["roc_auc"]
    avg_precision = metrics["average_precision"]
    best_thr      = metrics["optimal_threshold_youden"]
    best_idx      = int(np.argmin(np.abs(
        np.array(metrics["roc_curve"]["fpr"]) +
        np.array(metrics["roc_curve"]["tpr"]) * 0 -
        fpr)))   # recompute for plot marker

    # Recompute best_idx directly on fpr/tpr
    youden    = np.array(tpr) - np.array(fpr)
    best_idx  = int(np.argmax(youden))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        "ROC / AUC Analysis - CA-Markov Urban Growth Model\n"
        "Ouargla, Algeria  |  Validation period: 2010-2025",
        fontsize=13, fontweight='bold'
    )

    # --- Panel 1: ROC curve ---
    ax1.plot(fpr, tpr, color='#d73027', lw=2.5,
             label=f"CA-Markov (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random classifier (AUC = 0.5)")
    ax1.fill_between(fpr, tpr, alpha=0.10, color='#d73027')
    ax1.plot(fpr[best_idx], tpr[best_idx], 'ro', markersize=9,
             label=f"Optimal threshold = {best_thr:.3f}\n"
                   f"(Youden's J = {tpr[best_idx]-fpr[best_idx]:.3f})")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=11)
    ax1.set_title("Receiver Operating Characteristic Curve", fontsize=12,
                  fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Performance annotation
    opt = metrics["at_optimal_threshold"]
    ax1.text(0.55, 0.20,
             f"At optimal threshold:\n"
             f"Sensitivity = {opt['sensitivity']:.3f}\n"
             f"Specificity = {opt['specificity']:.3f}\n"
             f"F1-score    = {opt['f1_score']:.3f}\n"
             f"Bal. Acc.   = {opt['balanced_accuracy']:.3f}",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.85, edgecolor='gray'))

    # --- Panel 2: Precision-Recall curve ---
    baseline = metrics["n_pixels_urban_obs"] / metrics["n_pixels_total"]
    ax2.plot(recall, precision, color='#4575b4', lw=2.5,
             label=f"CA-Markov (AP = {avg_precision:.4f})")
    ax2.axhline(baseline, color='gray', lw=1.5, linestyle='--',
                label=f"Baseline (class ratio = {baseline:.4f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ROC figure saved: {save_path}")


def plot_probability_surface(prob_surface, obs_2025, save_path):
    """Plot the probability surface alongside the observed 2025 map."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle(
        "Probability Surface vs Observed Urban Extent (2025)\n"
        "Probability = max transition probability over 2010-2025 simulation",
        fontsize=12, fontweight='bold'
    )

    im1 = ax1.imshow(obs_2025, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax1.set_title("Observed urban extent 2025\n(ground truth)", fontsize=11,
                  fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04,
                 label='Urban (1) / Non-urban (0)')

    im2 = ax2.imshow(prob_surface, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title("CA-Markov transition probability surface\n"
                  "(genuine model score for ROC)", fontsize=11, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                 label='Transition probability')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Probability surface figure saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("ROC/AUC ANALYSIS - CA-MARKOV URBAN GROWTH MODEL")
    print("Validation run: 2010 -> 2025")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load rasters
    # ------------------------------------------------------------------
    print("\n[1/6] Loading rasters ...")
    rasters, profile = load_rasters([1990, 2000, 2010, 2025])

    if 2010 not in rasters or 2025 not in rasters:
        sys.exit("[ERROR] Both 2010 (initial state) and 2025 (ground truth) are required.")

    shape     = rasters[2025].shape
    transform = profile['transform']

    # ------------------------------------------------------------------
    # 2. Load regulatory constraints
    # ------------------------------------------------------------------
    print("\n[2/6] Loading CODESZone constraints ...")
    shp_c = os.path.join(SHP_DIR, SHP_CONSTRAINT)
    shp_p = os.path.join(SHP_DIR, SHP_PERIMETER)

    if os.path.exists(shp_c):
        weight_raster = rasterize_codes_zone(shp_c, shape, transform)
    else:
        print(f"  [WARNING] {shp_c} not found. Using uniform weights.")
        weight_raster = np.ones(shape, dtype=np.float32)

    if os.path.exists(shp_p):
        perimeter = rasterize_perimeter(shp_p, shape, transform)
    else:
        print(f"  [WARNING] {shp_p} not found. No perimeter restriction.")
        perimeter = np.ones(shape, dtype=np.uint8)

    expansion_mask = (perimeter == 1) & (weight_raster > 0)
    print(f"  Eligible pixels: {int(np.sum(expansion_mask))}")

    # ------------------------------------------------------------------
    # 3. Calibrate transition matrix
    # ------------------------------------------------------------------
    print("\n[3/6] Calibrating Markov transition matrix ...")
    P = average_transition_matrix(rasters)
    print(f"  P(non-urban -> urban)  = {P[0,1]:.4f}")
    print(f"  P(urban    -> urban)   = {P[1,1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Validation simulation (2010 -> 2025) + probability surface
    # ------------------------------------------------------------------
    print(f"\n[4/6] Running validation simulation "
          f"(2010 -> 2025, {VALIDATION_YEARS} years) ...")

    initial_2010 = (rasters[2010] > 0).astype(np.float32)

    prediction_2025, prob_surface = run_validation_with_probability(
        initial_state     = initial_2010,
        transition_matrix = P,
        expansion_mask    = expansion_mask,
        weight_raster     = weight_raster,
        n_years           = VALIDATION_YEARS,
        threshold         = EXPANSION_THRESHOLD
    )

    print(f"  Probability surface: min={prob_surface.min():.4f}, "
          f"max={prob_surface.max():.4f}, "
          f"mean={prob_surface.mean():.4f}")

    # Save probability surface raster
    prob_path = os.path.join(RESULTS_DIR, "probability_surface_2025.tif")
    prob_profile = profile.copy()
    prob_profile.update(dtype=np.float32)
    with rasterio.open(prob_path, 'w', **prob_profile) as dst:
        dst.write(prob_surface, 1)
    print(f"  Probability surface saved: {prob_path}")

    # ------------------------------------------------------------------
    # 5. ROC/AUC computation
    # ------------------------------------------------------------------
    print("\n[5/6] Computing ROC/AUC metrics ...")

    obs_2025 = (rasters[2025] > 0).astype(int)
    y_true   = obs_2025.ravel()
    y_scores = prob_surface.ravel()

    print(f"  Total pixels : {len(y_true):,}")
    print(f"  Urban (obs)  : {int(np.sum(y_true==1)):,} "
          f"({np.sum(y_true==1)/len(y_true)*100:.2f} %)")
    print(f"  Non-urban    : {int(np.sum(y_true==0)):,}")

    metrics, fpr, tpr, precision, recall = compute_roc_metrics(
        y_true, y_scores, max_samples=MAX_SAMPLE_PIXELS)

    print(f"\n  === ROC/AUC RESULTS ===")
    print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")
    print(f"  Average Precision    : {metrics['average_precision']:.4f}")
    print(f"  Optimal threshold    : {metrics['optimal_threshold_youden']:.4f}")
    opt = metrics["at_optimal_threshold"]
    print(f"  Sensitivity          : {opt['sensitivity']:.4f}")
    print(f"  Specificity          : {opt['specificity']:.4f}")
    print(f"  F1-score             : {opt['f1_score']:.4f}")
    print(f"  Balanced Accuracy    : {opt['balanced_accuracy']:.4f}")

    # Honest interpretation
    roc_auc = metrics['roc_auc']
    if roc_auc >= 0.90:
        interp = "Excellent discrimination (AUC >= 0.90)"
    elif roc_auc >= 0.80:
        interp = "Good discrimination (AUC >= 0.80)"
    elif roc_auc >= 0.70:
        interp = "Acceptable discrimination (AUC >= 0.70)"
    else:
        interp = "Poor discrimination (AUC < 0.70) -- review model calibration"
    metrics["interpretation"] = interp
    print(f"\n  Interpretation: {interp}")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print("\n[6/6] Saving outputs ...")

    # Metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, "roc_auc_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Threshold table CSV
    thr_path = os.path.join(RESULTS_DIR, "threshold_analysis.csv")
    pd.DataFrame(metrics["threshold_table"]).to_csv(thr_path, index=False)
    print(f"  Threshold table saved: {thr_path}")

    # Figures
    roc_fig_path   = os.path.join(RESULTS_DIR, "roc_auc_figure.png")
    prob_fig_path  = os.path.join(RESULTS_DIR, "probability_surface_figure.png")
    plot_roc_figure(metrics, fpr, tpr, precision, recall, roc_fig_path)
    plot_probability_surface(prob_surface, obs_2025, prob_fig_path)

    print("\n" + "=" * 70)
    print("ROC/AUC ANALYSIS COMPLETE")
    print(f"ROC-AUC = {roc_auc:.4f}  |  {interp}")
    print(f"Results written to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
