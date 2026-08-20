#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-MARKOV URBAN GROWTH MODEL WITH CODESZone REGULATORY CONSTRAINTS
===================================================================
Simulates urban expansion from 2025 to 2045 using a Cellular Automata
Markov (CA-Markov) framework constrained by regulatory zoning data
(CODESZone) and a defined urban perimeter shapefile.

Study area : Ouargla agglomeration, Algeria
Resolution  : 30 m (Sentinel-2 / Landsat)
Projection  : UTM Zone 31N (EPSG:32631)

Methodology :
  - Markov transition matrix calibrated on 1990-2000-2010 land-cover epochs
  - Moore 3x3 neighbourhood for spatial contiguity
  - CODESZone regulatory constraints applied as hard/soft boundaries
      code 0 -> Favourable zone  (weight 1.0, expansion PERMITTED)
      code 1 -> Restricted zone  (weight 0.0, expansion FORBIDDEN)
      code 2 -> Reconversion zone (weight 0.8, expansion PERMITTED)
  - Expansion restricted to the official future urban perimeter

Usage :
    conda activate base
    python ca_markov_urban_growth.py

Outputs (written to RESULTS_DIR) :
    prediction_2045.tif           - Predicted urban raster 2045
    transition_matrix.json        - Calibrated Markov transition matrix
    simulation_metrics.json       - Surface statistics 2025 vs 2045
    urban_growth_map.png          - Publication-quality figure

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
import matplotlib.patches as mpatches
from rasterio import features
from pathlib import Path
from scipy import ndimage

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - edit these paths before running
# =============================================================================

BASE_DIR = r"D:\Ouargla_CA_Project_fin_wahid"   # <-- adapt to your machine

RASTER_DIR  = os.path.join(BASE_DIR, "01_DONNEES", "Rasters")
SHP_DIR     = os.path.join(BASE_DIR, "01_DONNEES", "Shapefiles")
RESULTS_DIR = os.path.join(BASE_DIR, "04_SCRIPTS_PEER_REVIEW", "outputs")

# Input raster filenames (binary urban / non-urban, value 1 = urban)
RASTER_FILES = {
    1990: "Urban_1990.tif",
    2000: "Urban_2000.tif",
    2010: "Urban_2010.tif",
    2025: "Urban_2025.tif",
}

# Regulatory constraint shapefile (must contain field 'CODES_ZONE')
SHP_CONSTRAINT = "contrainte_emreinte1.shp"

# Future urban perimeter shapefile
SHP_PERIMETER  = "perimetre_urbain_futur.shp"

# Simulation parameters
SIM_YEARS      = 20          # years to simulate (2025 -> 2045)
CODES_ZONE_WEIGHTS = {0: 1.0, 1: 0.0, 2: 0.8}  # codes present: 1.0, 2.0   # weight per CODESZone code
EXPANSION_THRESHOLD = 0.15   # Requires 2+ urban neighbours - prevents blob growth   # probability threshold for new urban transition

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_raster(year):
    """Return the full path of the urban raster for a given year."""
    filename = RASTER_FILES.get(year)
    if filename:
        path = os.path.join(RASTER_DIR, filename)
        if os.path.exists(path):
            return path
    # Fallback: glob search
    matches = list(Path(RASTER_DIR).glob(f"*{year}*.tif"))
    if matches:
        return str(matches[0])
    return None


def load_rasters(years):
    """
    Load binary urban rasters for a list of years.

    Returns
    -------
    rasters : dict  {year: ndarray (H, W) uint8}
    profile : dict  rasterio profile of the first successfully loaded file
    """
    rasters = {}
    profile = None

    for year in years:
        path = find_raster(year)
        if path is None:
            print(f"[WARNING] Raster not found for year {year}")
            continue
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.uint8)
            rasters[year] = data
            if profile is None:
                profile = src.profile.copy()
                profile.update(dtype=np.uint8, compress='lzw')
        n_urban = int(np.sum(data == 1))
        print(f"  {year}: {data.shape}, {n_urban} urban pixels "
              f"({n_urban / data.size * 100:.2f} %)")

    return rasters, profile


def rasterize_shapefile(shp_path, shape, transform, burn_value=1, dtype=np.uint8):
    """
    Rasterize a vector file onto the reference raster grid.

    Parameters
    ----------
    shp_path   : str    path to shapefile
    shape      : tuple  (H, W) of the target raster
    transform  : affine transform from rasterio profile
    burn_value : scalar value to burn for all features
    dtype      : numpy dtype of output array

    Returns
    -------
    ndarray (H, W)
    """
    gdf = gpd.read_file(shp_path)
    out = features.rasterize(
        [(geom, burn_value) for geom in gdf.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=dtype
    )
    return out, gdf


def rasterize_codes_zone(shp_path, shape, transform, weights=None):
    """
    Build CODESZone weight raster with correct 3-zone interpretation:
      NaN (uncoded) = Zone 0 - Favourable  -> weight 1.0  (DEFAULT)
      1.0           = Zone 1 - Restricted  -> weight 0.0
      2.0           = Zone 2 - Reconversion-> weight 0.8

    Areas not covered by any polygon inherit Zone 0 (weight=1.0).
    This is the scientifically correct interpretation: only explicitly
    coded zones 1 and 2 override the favourable default.
    """
    if not os.path.exists(shp_path):
        print(f"  [WARNING] {shp_path} not found - all pixels favourable")
        return np.ones(shape, dtype=np.float32)

    gdf = gpd.read_file(shp_path)
    if 'CODES_ZONE' not in gdf.columns:
        print(f"  [WARNING] CODES_ZONE field not found - all pixels favourable")
        return np.ones(shape, dtype=np.float32)

    # Start: everything is Zone 0 (favourable, weight=1.0)
    weight_raster = np.ones(shape, dtype=np.float32)

    # Apply Zone 1 (restricted, weight=0.0)
    z1 = gdf[gdf['CODES_ZONE'] == 1.0]
    if not z1.empty:
        m1 = features.rasterize(
            [(g, 1) for g in z1.geometry],
            out_shape=shape, transform=transform,
            fill=0, dtype=np.uint8)
        weight_raster[m1 == 1] = 0.0
        print(f"  Zone 1 (Restricted):   {int(np.sum(m1==1)):,} px -> weight=0.0")

    # Apply Zone 2 (reconversion, weight=0.8) — only where not Zone 1
    z2 = gdf[gdf['CODES_ZONE'] == 2.0]
    if not z2.empty:
        m2 = features.rasterize(
            [(g, 1) for g in z2.geometry],
            out_shape=shape, transform=transform,
            fill=0, dtype=np.uint8)
        mask2 = (m2 == 1) & (weight_raster != 0.0)
        weight_raster[mask2] = 0.8
        print(f"  Zone 2 (Reconversion): {int(np.sum(mask2)):,} px -> weight=0.8")

    n_z0 = int(np.sum(weight_raster == 1.0))
    print(f"  Zone 0 (Favourable):   {n_z0:,} px -> weight=1.0 (default)")

    return weight_raster
def compute_transition_matrix(rasters, t_start, t_end):
    """
    Compute the 2x2 Markov transition matrix between two epochs.

    Parameters
    ----------
    rasters  : dict {year: ndarray}
    t_start  : int  start year
    t_end    : int  end year

    Returns
    -------
    P : ndarray (2, 2)  row-normalised transition matrix
        P[i, j] = P(class j | class i)
    """
    a = (rasters[t_start] > 0).astype(np.int8).ravel()
    b = (rasters[t_end]   > 0).astype(np.int8).ravel()

    P_raw = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            P_raw[i, j] = np.sum((a == i) & (b == j))

    row_sums = P_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P_raw / row_sums
    return P


def average_transition_matrix(rasters):
    """
    Return the average transition matrix across all available consecutive epochs.
    Epochs used: 1990-2000, 2000-2010  (calibration period).
    """
    epochs = [(1990, 2000), (2000, 2010)]
    matrices = []
    for t0, t1 in epochs:
        if t0 in rasters and t1 in rasters:
            P = compute_transition_matrix(rasters, t0, t1)
            matrices.append(P)
            print(f"  Transition matrix {t0}-{t1}:")
            print(f"    P(non-urban -> non-urban) = {P[0,0]:.4f}")
            print(f"    P(non-urban -> urban)     = {P[0,1]:.4f}")
            print(f"    P(urban    -> non-urban)  = {P[1,0]:.4f}")
            print(f"    P(urban    -> urban)      = {P[1,1]:.4f}")

    if not matrices:
        print("[WARNING] No calibration epochs found. Using default matrix.")
        return np.array([[0.990, 0.010], [0.020, 0.980]])

    return np.mean(np.stack(matrices), axis=0)


# =============================================================================
# CA-MARKOV SIMULATION (vectorised)
# =============================================================================

def run_ca_markov(initial_state, transition_matrix, expansion_mask,
                  weight_raster, n_years, threshold):
    """
    Run vectorised CA-Markov simulation.

    At each annual step, for every non-urban pixel inside the authorised
    expansion zone, the transition probability is:

        p = 0.20 * P(non->urban)        [Markov component]
          + 0.30 * neighbourhood_density [CA component]
          + 0.50 * CODESZone_weight      [regulatory component]

    A pixel converts to urban when p > threshold.
    Urban pixels are permanent (no de-urbanisation).

    Parameters
    ----------
    initial_state      : ndarray (H, W) float32  binary urban map (0/1)
    transition_matrix  : ndarray (2, 2)
    expansion_mask     : ndarray (H, W) bool  True = expansion permitted
    weight_raster      : ndarray (H, W) float32  CODESZone weights
    n_years            : int
    threshold          : float

    Returns
    -------
    state : ndarray (H, W) uint8  predicted urban map (0=non-urban, 255=urban)
    """
    p_markov = float(transition_matrix[0, 1])   # P(non-urban -> urban)
    state    = initial_state.astype(np.float32).copy()

    # Kernel for Moore 3x3 neighbourhood density
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0   # exclude central pixel
    n_neighbours = 8.0

    for year_offset in range(n_years):
        # --- Neighbourhood density (fraction of urban neighbours) ---
        neighbour_density = ndimage.convolve(state, kernel, mode='constant', cval=0.0)
        neighbour_density /= n_neighbours

        # --- Combined probability (multiplicative CODESZone weight) ---
        # CODESZone weight acts as a MULTIPLIER (not additive component):
        #   Zone 0 (w=1.0): full probability allowed
        #   Zone 1 (w=0.0): expansion forbidden (p=0)
        #   Zone 2 (w=0.8): probability reduced by 20%
        # Base probability from Markov + neighbourhood only
        p_base = (0.50 * p_markov + 0.50 * neighbour_density)
        p_combined = p_base * weight_raster

        # --- Stochastic transition rule ---
        # Prevents contiguous blob growth by adding randomness:
        # A pixel transitions if p > threshold AND random draw < p_combined.
        # This preserves the dispersed urban pattern characteristic of Ouargla.
        np.random.seed(year_step * 7 + 13)  # reproducible per step
        rng_draw = np.random.random(state.shape).astype(np.float32)
        new_urban = ((state == 0) & expansion_mask
                     & (p_combined > threshold)
                     & (rng_draw < p_combined))
        state[new_urban] = 1.0

        # Urban pixels are permanent
        # (no change needed; state already contains 1 for urban)

    return (state > 0.5).astype(np.uint8) * 255


# =============================================================================
# OUTPUT & FIGURES
# =============================================================================

def save_prediction(prediction, profile, path):
    """Write prediction raster to disk."""
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(prediction, 1)
    print(f"  Prediction raster saved: {path}")


def save_metrics(rasters, prediction, path):
    """Compute and save surface statistics."""
    surf_2025 = int(np.sum(rasters[2025] == 1))
    surf_2045 = int(np.sum(prediction == 255))
    growth    = surf_2045 - surf_2025
    growth_pct = round(growth / surf_2025 * 100, 2) if surf_2025 > 0 else 0

    metrics = {
        "urban_pixels_2025" : surf_2025,
        "urban_pixels_2045" : surf_2045,
        "new_urban_pixels"  : growth,
        "growth_rate_pct"   : growth_pct,
        "pixel_area_m2"     : 900,           # 30m x 30m
        "urban_area_2025_km2": round(surf_2025 * 900 / 1e6, 3),
        "urban_area_2045_km2": round(surf_2045 * 900 / 1e6, 3),
    }

    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Simulation metrics saved: {path}")
    print(f"    Urban area 2025 : {metrics['urban_area_2025_km2']} km2")
    print(f"    Urban area 2045 : {metrics['urban_area_2045_km2']} km2")
    print(f"    Growth 2025-2045: {growth_pct} %")
    return metrics


def generate_map(rasters, prediction, weight_raster, perimeter, path):
    """Generate and save a publication-quality urban growth map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle(
        "Urban Growth Simulation 2025-2045 - Ouargla, Algeria\n"
        "CA-Markov Model with CODESZone Regulatory Constraints",
        fontsize=13, fontweight='bold'
    )

    cmap_urban   = plt.cm.colors.ListedColormap(['#F5F5DC', '#B22222'])  # sand, red
    cmap_zones   = plt.cm.RdYlGn

    # Panel 1 - Urban extent 2025
    display_2025 = (rasters[2025] > 0).astype(int)
    axes[0].imshow(display_2025, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Urban extent 2025', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2 - CODESZone regulatory constraints
    im2 = axes[1].imshow(weight_raster, cmap=cmap_zones, vmin=0, vmax=1)
    axes[1].set_title('CODESZone regulatory weights\n'
                       '(green=permitted, red=restricted)',
                       fontsize=12, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Weight')

    # Panel 3 - Predicted urban 2045
    urban_2025   = (rasters[2025] > 0)
    pred_binary  = (prediction == 255)
    new_expansion = pred_binary & ~urban_2025

    composite = np.zeros(prediction.shape, dtype=int)
    composite[urban_2025]   = 1   # existing urban
    composite[new_expansion] = 2  # new urban 2025-2045

    cmap3 = plt.cm.colors.ListedColormap(['#D3D3D3', '#1E90FF', '#FF4500'])
    axes[2].imshow(composite, cmap=cmap3, vmin=0, vmax=2)
    axes[2].set_title('Predicted urban extent 2045', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    legend_patches = [
        mpatches.Patch(color='#D3D3D3', label='Non-urban'),
        mpatches.Patch(color='#1E90FF', label='Urban 2025'),
        mpatches.Patch(color='#FF4500', label='New urban 2025-2045'),
    ]
    axes[2].legend(handles=legend_patches, loc='lower right', fontsize=9,
                   framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Map saved: {path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("CA-MARKOV URBAN GROWTH MODEL - Ouargla, Algeria")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load rasters
    # ------------------------------------------------------------------
    print("\n[1/5] Loading urban rasters ...")
    rasters, profile = load_rasters([1990, 2000, 2010, 2025])

    if len(rasters) < 3:
        sys.exit("[ERROR] At least 3 epochs are required (1990, 2000, 2010).")
    if 2025 not in rasters:
        sys.exit("[ERROR] The 2025 raster (initial state) is required.")

    shape     = rasters[2025].shape
    transform = profile['transform']

    # ------------------------------------------------------------------
    # 2. Load and rasterize regulatory constraints
    # ------------------------------------------------------------------
    print("\n[2/5] Loading CODESZone regulatory constraints ...")
    shp_constraint_path = os.path.join(SHP_DIR, SHP_CONSTRAINT)
    shp_perimeter_path  = os.path.join(SHP_DIR, SHP_PERIMETER)

    # CODESZone weights
    if os.path.exists(shp_constraint_path):
        weight_raster = rasterize_codes_zone(shp_constraint_path, shape, transform)
        print(f"  Pixels permitted (weight > 0) : {int(np.sum(weight_raster > 0))}")
        print(f"  Pixels forbidden (weight = 0) : {int(np.sum(weight_raster == 0))}")
    else:
        print(f"  [WARNING] Constraint shapefile not found: {shp_constraint_path}")
        print("  Using uniform weight (all expansion permitted).")
        weight_raster = np.ones(shape, dtype=np.float32)

    # Urban perimeter mask
    if os.path.exists(shp_perimeter_path):
        perimeter, _ = rasterize_shapefile(shp_perimeter_path, shape, transform)
        print(f"  Urban perimeter: {int(np.sum(perimeter == 1))} pixels")
    else:
        print(f"  [WARNING] Perimeter shapefile not found: {shp_perimeter_path}")
        print("  Expansion permitted everywhere.")
        perimeter = np.ones(shape, dtype=np.uint8)

    # Combined expansion mask:
    # a pixel is eligible only if it is inside the perimeter AND weight > 0
    expansion_mask = (perimeter == 1) & (weight_raster > 0)
    print(f"  Eligible pixels for new urbanisation: {int(np.sum(expansion_mask))}")

    # ------------------------------------------------------------------
    # 3. Calibrate Markov transition matrix
    # ------------------------------------------------------------------
    print("\n[3/5] Calibrating Markov transition matrix ...")
    transition_matrix = average_transition_matrix(rasters)

    matrix_path = os.path.join(RESULTS_DIR, "transition_matrix.json")
    with open(matrix_path, 'w') as f:
        json.dump({
            "calibration_epochs": ["1990-2000", "2000-2010"],
            "matrix": transition_matrix.tolist(),
            "labels": ["non-urban", "urban"],
            "note": "Row i -> probability of transitioning from class i to each class j"
        }, f, indent=2)
    print(f"  Transition matrix saved: {matrix_path}")

    # ------------------------------------------------------------------
    # 4. Run CA-Markov simulation
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running CA-Markov simulation (2025 -> 2045, {SIM_YEARS} years) ...")
    initial_state = (rasters[2025] > 0).astype(np.float32)

    prediction_2045 = run_ca_markov(
        initial_state      = initial_state,
        transition_matrix  = transition_matrix,
        expansion_mask     = expansion_mask,
        weight_raster      = weight_raster,
        n_years            = SIM_YEARS,
        threshold          = EXPANSION_THRESHOLD
    )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    print("\n[5/5] Saving results ...")

    pred_path    = os.path.join(RESULTS_DIR, "prediction_2045.tif")
    metrics_path = os.path.join(RESULTS_DIR, "simulation_metrics.json")
    map_path     = os.path.join(RESULTS_DIR, "urban_growth_map.png")

    save_prediction(prediction_2045, profile, pred_path)
    save_metrics(rasters, prediction_2045, metrics_path)
    generate_map(rasters, prediction_2045, weight_raster, perimeter, map_path)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"Results written to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
