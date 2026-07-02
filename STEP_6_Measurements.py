#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:20:05 2026

@author: cecilieandre
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom

from scipy.ndimage import (
    zoom,
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    map_coordinates,
    convolve,
    gaussian_filter,
    uniform_filter
)

from skimage.measure import label
from skimage.morphology import ball, footprint_rectangle, binary_closing
from skimage.restoration import denoise_nl_means, denoise_wavelet, denoise_tv_chambolle
from sklearn.cluster import KMeans
from skimage.restoration import denoise_bilateral
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

# Run the code in terminal
NAME = sys.argv[1]


# Run the code yourself
#NAME = "00075" 
SCAN_ID = NAME

DATA_DIR = f"./Results/{NAME}/cache"
DATA_DIR_VOLUME = f"./Results/{NAME}"
OUT_DIR = f"./Results_seg/{NAME}"
OUT_DIR_CACHE = f"./Results_seg/{NAME}/cache"
folder  = f"./Scans/{NAME}"
OSSICLES = f"./Seg/seg_auto_{NAME}.nii"
OSSICLES_RE = f"./Results_seg/{NAME}/cache/seg_auto_{NAME}_resampled.nii"
PIC_DIR = f"./Results_seg/{NAME}/Picture/"


VISUAL_SANITY = False
SHOW_VISUALS = False
Debug = False
CSV =  True
no_pca = True

# Radius is calculated for cylinder - relations of tissue
radius_mm_xy = 5

CSV_PATH = f"./feature_table_reference_radius_proof_of_concept_cylinder_radius_mm_xy_{radius_mm_xy}.csv"


MASK_file = os.path.join(OUT_DIR_CACHE, "ossicles_kmean_bone_5_no_PCA.nii.gz")

if not os.path.exists(MASK_file) or os.path.getsize(MASK_file) == 0:
    print(f"ossicles_kmean_bone_5_no_PCA is missing {NAME} – stops.")
    sys.exit(1)
    

# ============================================================
# VISUAL HELPER
# ============================================================

def show_overlay(volume, mask, title, z=None, cmap="Reds"):
    if not (VISUAL_SANITY and SHOW_VISUALS):
        return

    if z is None:
        z = volume.shape[0] // 2

    for dz in [-75, -25, 25, 75]:
        zz = z + dz
        if 0 <= zz < volume.shape[0]:
            plt.figure(figsize=(6, 6))
            plt.imshow(volume[zz], cmap="gray")
            plt.imshow(mask[zz], cmap=cmap, alpha=0.4)
            plt.title(title)
            plt.axis("off")
            plt.show()

# ============================================================
# LOAD DATA - step 1
# ============================================================

body = np.load(os.path.join(DATA_DIR, "body_mask.npy"))
volume = np.load(os.path.join(DATA_DIR, "volume.npy"))

# ============================================================
# BASIC MASKS
# ============================================================


# ROI 
roi_2 = nib.load(os.path.join(OUT_DIR_CACHE, "roi_mask_bone_final_none_PCA.nii.gz"))
ROI_no_pca = np.transpose(roi_2.get_fdata() > 0, (2, 1, 0))


# Ossicles no PCA
Ossicles_2 = nib.load(os.path.join(OUT_DIR_CACHE, "ossicles_kmean_bone_5_no_PCA.nii.gz"))
Ossicles_no_pca = np.transpose(Ossicles_2.get_fdata() > 0, (2, 1, 0))

# Kmean 
mask_Km = nib.load(os.path.join(OUT_DIR_CACHE, "kmeans_4class_roi_bone_5.nii.gz"))
kmeans_volume = np.transpose(mask_Km.get_fdata(), (2,1,0)).astype(np.uint8)



# Find gennemsnitlig intensitet for hver label
label_means = []

for lab in range(1,6):

    mask = kmeans_volume == lab

    if np.any(mask):
        label_means.append(volume[mask].mean())
    else:
        label_means.append(np.inf)

label_means = np.array(label_means)

# Sorter efter intensitet
order = np.argsort(label_means)


order = np.argsort(label_means)

AIR   = order[0] + 1
SOFT = order[1] + 1
TRAB1 = order[2] + 1
TRAB2  = order[3] + 1
CORB  = order[4] + 1

print("Cluster order:")
print("AIR :", AIR)
print("SOFT:", SOFT)
print("TRAB 1:", TRAB1)
print("TRAB 2:", TRAB2)
print("CORB:", CORB)

kmean_air   = kmeans_volume == AIR
kmean_soft = kmeans_volume == SOFT
kmean_trab1 = kmeans_volume == TRAB1
kmean_trab2  = kmeans_volume == TRAB2
kmean_bone  = kmeans_volume == CORB



# Header information
ref = nib.load(os.path.join(DATA_DIR_VOLUME, "CBCT_volume.nii.gz"))
voxel_volume = np.prod(ref.header.get_zooms())
voxel_size = ref.header.get_zooms()[0] 



# Visual sanity check
if VISUAL_SANITY == True:
    
    show_overlay(volume, kmean_air, "kmean_air")
    show_overlay(volume, kmean_soft, "kmean_soft")
    show_overlay(volume, kmean_trab1, "kmean_trab 1")
    show_overlay(volume, kmean_trab2, "kmean_trab 2")
    show_overlay(volume, kmean_bone, "kmean_bone")


total_scan_volume = body.sum() + (~body).sum()

if Ossicles_no_pca.sum() == total_scan_volume:
    Ossicles_found = np.zeros_like(body)
else:
    Ossicles_found = Ossicles_no_pca


        
# ============================================================
# STEP – VOLUMETRIC FEATURES NO PCA
# ============================================================


CONTACT_TOL = 2
bone_seg = (kmean_bone | kmean_trab1 | kmean_trab2)

roi_mask_wo_bone_npca = ROI_no_pca & ~bone_seg

cavity_roi_npca = kmean_air & ROI_no_pca
soft_roi_npca   = kmean_soft & ROI_no_pca
bone_roi_npca   = binary_dilation(bone_seg, ball(1), iterations=CONTACT_TOL) & roi_mask_wo_bone_npca

V_cavity_npca = cavity_roi_npca.sum()
V_soft_npca   = soft_roi_npca.sum()
V_bone_npca   = bone_roi_npca.sum()
V_interstitial_npca = (ROI_no_pca & ~bone_seg).sum()

V_oss_npca   = Ossicles_found.sum()
Vol_oss_npca = V_oss_npca * voxel_volume


if V_soft_npca == 0:
    n_comp_npca = 0
    V_largest_npca = 0
    V_soft_conn_npca = 0
else:
    labels_npca, n_comp_npca = label(soft_roi_npca, connectivity=3, return_num=True)

    sizes = np.bincount(labels_npca.ravel())
    sizes[0] = 0
    V_largest_npca = sizes.max()

    oss_d = binary_dilation(Ossicles_found, ball(1), iterations=CONTACT_TOL)
    hit_labels = np.unique(labels_npca[oss_d])
    hit_labels = hit_labels[hit_labels != 0]

    soft_conn_mask = np.isin(labels_npca, hit_labels)
    V_soft_conn_npca = soft_conn_mask.sum()


features_npca = {
    "R_soft_cavity_fill (NPCA)"       : V_soft_npca / max(V_cavity_npca, 1),
    "R_soft_interstitial (NPCA)"      : V_soft_npca / max(V_interstitial_npca, 1),
    "R_soft_bone_ratio (NPCA)"        : V_soft_npca / max(V_bone_npca, 1),
    "R_air_fraction (NPCA)"           : V_cavity_npca / max(V_cavity_npca + V_soft_npca, 1),

    "R_oss_all_soft_ratio (NPCA)"     : V_oss_npca / max(V_soft_npca, 1),
    "R_oss_air_ratio (NPCA)"          : V_oss_npca / max(V_cavity_npca, 1),
    "R_oss_touch_soft_ratio (NPCA)"   : V_oss_npca / max(V_soft_conn_npca, 1),

    "V_soft_voxels (NPCA)"            : int(V_soft_npca),
    "V_cavity_voxels (NPCA)"          : int(V_cavity_npca),
    "V_bone_voxels (NPCA)"            : int(V_bone_npca),
    "V_ossicle_found (NPCA)"          : int(V_oss_npca),
    "Volume_ossicle_found (NPCA)"     : Vol_oss_npca,

}

# CONTACT
bone_touch_npca = binary_dilation(bone_seg, ball(1), iterations=CONTACT_TOL)
air_touch_npca  = binary_dilation(cavity_roi_npca, ball(1), iterations=CONTACT_TOL)

bone_touch_npca &= ~air_touch_npca
air_touch_npca  &= ~bone_touch_npca

V_bone_contact_npca = (soft_roi_npca & bone_touch_npca).sum()
V_air_contact_npca  = (soft_roi_npca & air_touch_npca).sum()

features_npca.update({
    "R_bone_soft_contact (NPCA)"      : V_bone_contact_npca / max(V_soft_npca, 1),
    "R_air_soft_contact (NPCA)"       : V_air_contact_npca / max(V_soft_npca, 1),
    "R_bone_air_ratio (NPCA)"         : V_bone_contact_npca / max(V_bone_contact_npca + V_air_contact_npca, 1),
    "R_soft_to_total (NPCA)"          : V_soft_npca / max(V_soft_npca + V_cavity_npca + V_bone_npca, 1),
    "V_bone_contact_voxels (NPCA)"    : int(V_bone_contact_npca),
    "V_air_contact_voxels (NPCA)"     : int(V_air_contact_npca),
})



# ============================================================
# STEP 8 – SAVE FEATURES TO CSV
# ============================================================

features = {}
features.update(features_npca)

if CSV:

    # row indeholder allerede scan_id
    row = {"scan_id": SCAN_ID}
    row.update(features)
    
    if os.path.exists(CSV_PATH):
        
        df = pd.read_csv(CSV_PATH)
        
        if row["scan_id"] in df["scan_id"].values:
            df.loc[df["scan_id"] == row["scan_id"], row.keys()] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
    else:
        df = pd.DataFrame([row])
        
        
    # SORTER EFTER scan_id
    df["scan_id"] = df["scan_id"].astype(str)
    df = df.sort_values("scan_id")
    
    
    
    df.to_csv(CSV_PATH, index=False)
    
    print("\n=== FEATURES ===")
    for k, v in row.items():
        print(f"{k:25s}: {v}")





