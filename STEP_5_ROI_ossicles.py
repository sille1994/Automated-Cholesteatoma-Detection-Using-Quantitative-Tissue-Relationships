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
#SCAN_ID = NAME

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
SAVE = True
CSV =  True
no_pca = True

# Radius is calculated for cylinder - relations of tissue
radius_mm_z = 7.5
radius_mm_xy = 5

    
if not os.path.exists(OSSICLES) or os.path.getsize(OSSICLES) == 0:
    print(f"seg_auto_{NAME}.nii is missing – stops.")
    sys.exit(1)
#    

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if not os.path.exists(OUT_DIR_CACHE):
    os.makedirs(OUT_DIR_CACHE)

if not os.path.exists(PIC_DIR):
    os.makedirs(PIC_DIR)
    
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

seg = np.load(os.path.join(DATA_DIR, "seg_final.npy"))
body = np.load(os.path.join(DATA_DIR, "body_mask.npy"))
volume = np.load(os.path.join(DATA_DIR, "volume.npy"))

# ============================================================
# BASIC MASKS
# ============================================================


# Big mask
mask_b = nib.load(os.path.join(OUT_DIR, "mask_small_c.nii.gz"))
mask_big = np.transpose(mask_b.get_fdata() > 0, (2, 1, 0))


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

# Segmentation into air, bone and soft tissue 
air_raw = ((seg == 0) & body).astype(bool)
bone = ((seg == 3)).astype(bool)
soft = ((seg == 2) & body).astype(bool)


assert air_raw.any(), "No air inside temporal ROI"
assert bone.any(), "No bone inside temporal ROI"

if VISUAL_SANITY == True:
    show_overlay(volume, air_raw, "Air (raw)")
    show_overlay(volume, bone, "Bone (raw)")
    show_overlay(volume, soft, "Soft (raw)")

if VISUAL_SANITY == True:
    show_overlay(volume, mask_big, "Mask Big")


# Visual sanity check
if VISUAL_SANITY == True:
    
    show_overlay(volume, kmean_air, "kmean_air")
    show_overlay(volume, kmean_soft, "kmean_soft")
    show_overlay(volume, kmean_trab1, "kmean_trab 1")
    show_overlay(volume, kmean_trab2, "kmean_trab 2")
    show_overlay(volume, kmean_bone, "kmean_bone")





if os.path.exists(OSSICLES_RE):

    oss = nib.load(OSSICLES_RE)

    oss_data = oss.get_fdata()

    oss_pipeline = np.transpose(
        oss_data,
        (2,1,0)
    ).astype(np.uint8)

else:

    oss = nib.load(OSSICLES)

    ref_shape = ref.shape
    oss_data = oss.get_fdata()

    scale = (
        ref_shape[0] / oss_data.shape[0],
        ref_shape[1] / oss_data.shape[1],
        ref_shape[2] / oss_data.shape[2],
    )

    oss_resampled = zoom(
        oss_data,
        scale,
        order=0
    )

    oss_nii = nib.Nifti1Image(
        oss_resampled.astype(np.uint8),
        ref.affine,
        ref.header
    )

    nib.save(oss_nii, OSSICLES_RE)

    oss_pipeline = np.transpose(
        oss_resampled,
        (2,1,0)
    ).astype(np.uint8)


malleus = oss_pipeline == 4
incus   = oss_pipeline == 5
stapes  = oss_pipeline == 6
    
# ============================================================
# CENTER
# ============================================================
    
# Center of ossicles
ossicles_only = malleus 
    
coords_ossicles = np.argwhere(ossicles_only)
center = coords_ossicles.mean(axis=0).astype(int)


# Calculate the voxels distance
radius_vox_xy = int(5 / voxel_size)
z_margin_vox = int(15 / voxel_size)
        
# Extracts the dimensions of your 3D CBCT volume.
z_dim, y_dim, x_dim = volume.shape
    
# store the coordinate (x,y,z) of every voxel
zz, yy, xx = np.meshgrid(
    np.arange(z_dim),
    np.arange(y_dim),
    np.arange(x_dim),
    indexing="ij"
)
    
# With "ij" the arrays follow matrix indexing:
# Otherwise it would use volume[z,y,x]
# which is correct for medical volumes.
    
# Calculates the distance from every voxel to the center in the XY plane.
# Euclidean distance formula
dist_xy = np.sqrt(
    ((yy - center[1])**2) +
    ((xx - center[2])**2)
)
    
# Making the cylinder mask based on defined distance
cylinder_mask = dist_xy <= radius_vox_xy
    
# Calcualte the distance
z_margin_vox = int(radius_mm_z / voxel_size)
    
# The distance from the caudalt
z_max = min(z_dim, center[0] + z_margin_vox)
    
# The distance from crinial
z_min = max(0, center[0] - z_margin_vox)
    
# Making sure that 1 cm from center
z_mask = (zz >= z_min) & (zz <= z_max)
    
# Take only the mask
roi_mask_1 = cylinder_mask & z_mask
roi_mask = roi_mask_1 
 

type_os_mi = False
incus_dil = binary_dilation(incus, ball(1), iterations=20)
placement_right = roi_mask & incus_dil

if placement_right.any():
    
    # Ossicles
    ossicles_only = malleus | incus 
    type_os_mi = True
else:
    # Ossicles
    ossicles_only = malleus 
        
        
print()
print(f"For ID: {NAME}")
print(" -  Mal + Inc: ", type_os_mi)
print()


if VISUAL_SANITY == True:
    show_overlay(volume, ossicles_only, "ossicles_only ROI")


ROI_path = f"./Results_seg/{NAME}/cache/roi_mask_bone_final_none_PCA.nii.gz"

if os.path.exists(ROI_path):

    # load hvis den findes
    roi_mask_d_scaling = nib.load(ROI_path)
    roi_mask_d = np.transpose(
        roi_mask_d_scaling.get_fdata(), (2,1,0)
    ).astype(np.uint8)
else:
    
    # ============================================================
    # DIMENSIONS
    # ============================================================
    
    z_dim, y_dim, x_dim = volume.shape

    # ============================================================
    # CENTER
    # ============================================================
    

    coords_ossicles = np.argwhere(ossicles_only)
    
    if len(coords_ossicles) > 0:
        center = coords_ossicles.mean(axis=0).astype(int)
    else:
        print("Ossicles missing → fallback to cavity")
        coords_air = np.argwhere(kmean_air)
        center = coords_air.mean(axis=0).astype(int)
        
    # Calculate the voxels distance
    radius_vox_xy = int(radius_mm_xy / voxel_size)
    z_margin_vox = int(radius_mm_z / voxel_size)
    
    # Extracts the dimensions of your 3D CBCT volume.
    z_dim, y_dim, x_dim = volume.shape
    
    # store the coordinate (x,y,z) of every voxel
    zz, yy, xx = np.meshgrid(
        np.arange(z_dim),
        np.arange(y_dim),
        np.arange(x_dim),
        indexing="ij"
    )
    
    # With "ij" the arrays follow matrix indexing:
    # Otherwise it would use volume[z,y,x]
    # which is correct for medical volumes.
    
    # Calculates the distance from every voxel to the center in the XY plane.
    # Euclidean distance formula
    dist_xy = np.sqrt(
        ((yy - center[1])**2) +
        ((xx - center[2])**2)
    )
    
    # Making the cylinder mask based on defined distance
    cylinder_mask = dist_xy <= radius_vox_xy
    
    # Calcualte the distance
    z_margin_vox = int(radius_mm_z / voxel_size)
    
    # The distance from the caudalt
    z_max = min(z_dim, center[0] + z_margin_vox)
    
    # The distance from crinial
    z_min = max(0, center[0] - z_margin_vox)
    
    # Making sure that 1 cm from center
    z_mask = (zz >= z_min) & (zz <= z_max)
    
    # Take only the mask
    roi_mask_1 = cylinder_mask & z_mask
    
    #
    roi_mask_2 = roi_mask_1
    
    # Keep inside
    roi_mask = roi_mask_2 & mask_big
        

    # kernel to  nabo + nabo² 
    kernel = np.ones((3,3,3), dtype=int)
    
    # Bone segmentation
    bone_seg = kmean_bone | kmean_trab1  |kmean_trab2
    
    
    bone_neighbour_count = convolve(
        bone_seg.astype(int),
        kernel,
        mode="constant",
        cval=0
    )
    
    roi_mask_e = binary_erosion(roi_mask, ball(1),iterations=5)

    roi_mask_n = roi_mask_e & (bone_neighbour_count < 7 )
    
    labels_roi_mask, n_components = label(roi_mask_n, connectivity=3, return_num=True)
    sizes = np.bincount(labels_roi_mask.ravel())
    sizes[0] = 0  # ignorer baggrund
    
    largest_label = sizes.argmax()
    
    roi_mask_1 = labels_roi_mask == largest_label
    
    
    roi_mask_d = roi_mask_1.copy()
    
    
        
    for _ in range(6):
        grown = binary_dilation(roi_mask_d, ball(1))
        grown = grown & (~bone_seg)
        roi_mask_d = roi_mask_d | grown
        
        
    roi_mask_d = roi_mask_d & ((kmeans_volume > 0))
    
    
    
    if VISUAL_SANITY:
        show_overlay(volume, roi_mask_d, "Spherical middle ear ROI")
        
        
    if SAVE == True:
            
        mask_xyz = np.transpose(roi_mask_d, (2,1,0)).astype(np.uint8)
        assert mask_xyz.shape == ref.shape, "Shape mismatch!"
            
        nii = nib.Nifti1Image(mask_xyz, ref.affine, ref.header)
        nii.set_data_dtype(np.uint8)
        
        out_path = os.path.join(OUT_DIR_CACHE, "roi_mask_bone_final_none_PCA.nii.gz")
        nib.save(nii, out_path)
            
        print("Saved roi_mask_bone_final_none_PCA as NIfTI")



roi_final = roi_mask_d


bone_seg = kmean_bone | kmean_trab1  |kmean_trab2

# Known volumes (mm³)


Malleus = np.array([11.9-1.4, 11.9+1.4])
Incus   = np.array([13.1-1.1, 13.1+1.1])
Stapes  = np.array([1.24-0.13,  1.24+0.13])

# Total ossicl volume interval
Ossicle_vol_mm = np.array([
    Malleus[0] + Incus[0] + Stapes[0],
    Malleus[1] + Incus[1] + Stapes[1]
])

# Convert to voxels
Ossicle_vox = np.round(Ossicle_vol_mm / voxel_volume).astype(int)

print("Expected ossicle voxel range:", Ossicle_vox)

# Tolerance
MIN_VOX = int(Ossicle_vox[0] * 0)
MAX_VOX = int(Ossicle_vox[1] * 1.5)

print("Voxel (min):", MIN_VOX)
print("Voxel (max):", MAX_VOX)


roi_current = binary_erosion(roi_final,ball(1),iterations=10)
found_ossicles = False
ossicles_kmean = np.zeros_like(roi_final)


roi_filled = binary_fill_holes(roi_current)
new_voxels = roi_filled & (~roi_current)

labels_new, n_components = label(new_voxels, connectivity=3, return_num=True)

print("\nInitial fill:")
print("Connected components:", n_components)

for i in range(1, n_components + 1):
    comp = labels_new == i
    comp = comp & bone_seg
    size = comp.sum()

    hits = comp & ossicles_only

    if hits.any()  and size > MIN_VOX and size < MAX_VOX:
        print(f"Component {i}: size={size} → overlaps ossicles")
        ossicles_kmean |= comp          
    else:
        print(f"Component {i}: size={size}")


if ossicles_kmean.sum() == 0:
    print("\nTrying multiscale detection...\n")

    for step in range(3):

        scale = round(0.30 + step * 0.05, 2)
        
        print(f"\n--- Scaling step {step} | scale={scale} ---")
        
        # Downscale
        roi_ds = zoom(
            roi_current.astype(float),
            (scale, scale, scale),
            order=0
        ).astype(bool)
        
        # Closing + fill
        roi_ds_closed = roi_ds.copy()
        
        for _ in range(4):
            closed = binary_closing(roi_ds_closed, ball(8))
            added = closed & (~roi_ds_closed)
            roi_ds_closed |= added
            
        roi_ds_closed = binary_fill_holes(roi_ds_closed)
        
        # Upscale
        scale_factors = (
            roi_current.shape[0] / roi_ds_closed.shape[0],
            roi_current.shape[1] / roi_ds_closed.shape[1],
            roi_current.shape[2] / roi_ds_closed.shape[2],
        )
        
        roi_zoomed = zoom(
            roi_ds_closed.astype(float),
            scale_factors,
            order=0
        ).astype(bool)
        
        # New voxels
        new_voxels = roi_zoomed & (~roi_current)
        
        labels_new, n_components = label(new_voxels, connectivity=3, return_num=True)
        
        for n in range(1, n_components + 1):
            comp = labels_new == n
            
            comp = bone_seg & comp
            size = comp.sum()
            hits = comp & ossicles_only
            
            if hits.any() and size > MIN_VOX and size < MAX_VOX:
                print(f"Component {n}: size={size} → overlaps ossicles")
                ossicles_kmean |= comp   #  
                
        # ----------------------------------------------------
        # STOP 
        # ----------------------------------------------------
        if ossicles_kmean.sum() > 0:
            print("\n Ossicles detected")
            break


labels_bone = label(ossicles_kmean, connectivity=3)
sizes_bone = np.bincount(labels_bone.flat)
sizes_bone[0] = 0
ossicles_kmean_final = labels_bone == sizes_bone.argmax()


        
if SAVE == True:
        mask_xyz = np.transpose(ossicles_kmean_final, (2,1,0)).astype(np.uint8)
        assert mask_xyz.shape == ref.shape, "Shape mismatch!"
        
        nii = nib.Nifti1Image(mask_xyz, ref.affine, ref.header)
        nii.set_data_dtype(np.uint8)
        
        out_path = os.path.join(OUT_DIR_CACHE, "ossicles_kmean_bone_5_no_PCA.nii.gz")
        nib.save(nii, out_path)
        
        print("Saved ossicles_kmean_bone_5_no_PCA NIfTI")
    

