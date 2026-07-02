#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
# NAME = "00075"
SCAN_ID = NAME

DATA_DIR = f"./Results/{NAME}/cache"
DATA_DIR_VOLUME = f"./Results/{NAME}"
OUT_DIR = f"./Results_seg/{NAME}"
OUT_DIR_CACHE = f"./Results_seg/{NAME}/cache"
folder  = f"./Skanninger/{NAME}"
OSSICLES = f"./Seg/seg_auto_{NAME}.nii"
OSSICLES_RE = f"./Results_seg/{NAME}/cache/seg_auto_{NAME}_resampled.nii"
PIC_DIR = f"./Results_seg/{NAME}/Picture/"


VISUAL_SANITY = False
SHOW_VISUALS = False
Debug = True
SAVE = True
CSV =  False

# Cylinder
radius_mm_cyl = 15
z_margin_mm = 5


# Radius is calculated for cylinder - relations of tissue
radius_mm_z = 7.5
radius_mm_xy = 5


MASK_file = os.path.join(DATA_DIR, "bone_scaled_BEFORE_STEP_7.nii.gz")
#
if not os.path.exists(MASK_file) or os.path.getsize(MASK_file) == 0:
    print(f"bone_scaled_BEFORE_STEP_7 is missing for {NAME} – stops.")
    sys.exit(1)
#    
if not os.path.exists(OSSICLES) or os.path.getsize(OSSICLES) == 0:
    print(f"seg_auto_{NAME}.nii is missing for – stops.")
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
# LOAD DATA
# ============================================================

seg = np.load(os.path.join(DATA_DIR, "seg_final.npy"))
body = np.load(os.path.join(DATA_DIR, "body_mask.npy"))
volume = np.load(os.path.join(DATA_DIR, "volume.npy"))

# ============================================================
# BASIC MASKS
# ============================================================

# Small mask
mask_s = nib.load(os.path.join(DATA_DIR, "bone_scaled_BEFORE_STEP_7.nii.gz"))
mask_small = np.transpose(mask_s.get_fdata() > 0, (2, 1, 0))

# Big mask
mask_b = nib.load(os.path.join(OUT_DIR, "mask_small_c.nii.gz"))
mask_big = np.transpose(mask_b.get_fdata() > 0, (2, 1, 0))

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
    show_overlay(volume, mask_small, "Mask Small")
    show_overlay(volume, mask_big, "Mask Big")


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
print("For ID: ", SCAN_ID)
print(" -  Mal + Inc: ", type_os_mi)
print()


if VISUAL_SANITY == True:
    show_overlay(volume, ossicles_only, "ossicles_only ROI")



Kmean_path = f"./Results_seg/{NAME}/cache/kmeans_4class_roi_bone_5.nii.gz"



if os.path.exists(Kmean_path):
    # Kmeans mask
    mask_Km = nib.load(os.path.join(OUT_DIR_CACHE, "kmeans_4class_roi_bone_5.nii.gz"))
    kmeans_volume = np.transpose(mask_Km.get_fdata(), (2,1,0)).astype(np.uint8)
else:

    

    # Center of ossicles
    coords_ossicles = np.argwhere(ossicles_only)
    center = coords_ossicles.mean(axis=0)
    center = center.astype(int)
    
    radius_vox = int(radius_mm_cyl / voxel_size)  #isotrop voxel
    
    # Calculate the voxels distance
    radius_vox_x = radius_vox
    radius_vox_y = radius_vox
    
    
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
    cylinder_mask = dist_xy <= radius_vox
    
    # Calcualte the distance
    z_margin_vox = int(z_margin_mm / voxel_size)
    
    # The distance from the caudalt
    z_min = z_margin_vox
    
    # The distance from crinial
    z_max = z_dim - z_margin_vox
    
    # Making sure that 2 cm are taken of superior and inferior
    z_mask = (zz >= z_min) & (zz <= z_max)
    
    # Take only the mask
    roi_mask_1 = cylinder_mask & z_mask
    
    roi_mask = roi_mask_1 & mask_big
    
    
    if VISUAL_SANITY == True:
        show_overlay(volume, roi_mask_1, "cylinder ROI 1")
        show_overlay(volume, roi_mask,   "cylinder ROI")

    
    # ============================================================
    # DENOISE CYLINDRICAL ROI
    # ============================================================

    p1, p99 = np.percentile(volume, [1, 99])
    volume_scaled = (volume - p1) / (p99 - p1)

    empty_mask = np.zeros_like(volume)


    if VISUAL_SANITY == True:
        show_overlay(volume, empty_mask, "volume")
        show_overlay(volume_scaled, empty_mask, "volume_scaled")

    # Find bounding box around the ROI
    coords = np.argwhere(roi_mask)

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    # Crop the region
    sub_volume = volume_scaled[zmin:zmax, ymin:ymax, xmin:xmax]
    roi_sub = roi_mask[zmin:zmax, ymin:ymax, xmin:xmax]

    # Denoise only the cropped region (slice-by-slice)
    denoised_sub = sub_volume.copy()



    volume_smooth = gaussian_filter(sub_volume, sigma=1)


    # 2. Downsample
    scale = 0.80
    volume_ds = zoom(volume_smooth, scale, order=1)

    # 3. TV 
    TV_WEIGHT = 0.1

    volume_tv = denoise_tv_chambolle(
            volume_ds,
            weight=TV_WEIGHT,
            channel_axis=None
        )

    # 4. Upsample 
    scale_factors = (
            volume_smooth.shape[0] / volume_tv.shape[0],
            volume_smooth.shape[1] / volume_tv.shape[1],
            volume_smooth.shape[2] / volume_tv.shape[2],
    )

    denoised_sub = zoom(volume_tv, scale_factors, order=1)



    # Put the denoised cylinder back into the full volume
    denoised = volume_scaled.copy()

    sub_region = denoised[zmin:zmax, ymin:ymax, xmin:xmax]

    sub_region[roi_sub] = denoised_sub[roi_sub]

    denoised[zmin:zmax, ymin:ymax, xmin:xmax] = sub_region



    # Visual sanity check
    if VISUAL_SANITY == True:
        show_overlay(volume, empty_mask, "Before denoised")
        show_overlay(denoised, empty_mask, "Denoised cylinder")

    
    # ============================================================
    # KMEANS WITH MULTI-FEATURE SPACE (FIXED VERSION)
    # ============================================================


    # Only take values in ROI
    values = denoised[roi_mask]
    
    # Coordinates
    coords = np.array(np.where(roi_mask)).astype(np.float32)
    
    Z = coords[0] / volume.shape[0]
    Y = coords[1] / volume.shape[1]
    X = coords[2] / volume.shape[2]
    
    # Local context
    local_mean_1 = uniform_filter(denoised, size=5)[roi_mask]
    local_mean_2 = uniform_filter(denoised, size=10)[roi_mask]
    
    # Feature weighting
    features = np.column_stack([
        values,              # intensity
        0.8 * local_mean_1,    # denoised context
        0.8 * local_mean_2,    # denoised context

    ])
    
    # Scaling values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # KMeans with five clusters 
    kmeans = KMeans(
        n_clusters=5,
        init="random", 
        n_init=1,
        random_state=0
    )
    
    labels = kmeans.fit_predict(features_scaled)
    
    # Prepare volume
    kmeans_volume = np.zeros(volume.shape, dtype=np.uint8)
    
    # Making sure that there is no 0 values for segmentation
    kmeans_volume[roi_mask] = labels + 1
    

    if SAVE == True:
        
        mask_xyz = np.transpose(kmeans_volume, (2,1,0)).astype(np.uint8)
        assert mask_xyz.shape == ref.shape, "Shape mismatch!"
        
        nii = nib.Nifti1Image(mask_xyz, ref.affine, ref.header)
        nii.set_data_dtype(np.uint8)
        
        out_path = os.path.join(OUT_DIR_CACHE, "kmeans_4class_roi_bone_5.nii.gz")
        nib.save(nii, out_path)
        
        print("Saved kmeans_4class_roi_bone_5 NIfTI")


    # Visual sanity check
    if VISUAL_SANITY == True:
        show_overlay(volume, roi_mask, "cylinder ROI")
        show_overlay(volume, kmeans_volume, "KMeans result")



# Find the average itensity for each label
label_means = []

for lab in range(1,6):

    mask = kmeans_volume == lab

    if np.any(mask):
        label_means.append(volume[mask].mean())
    else:
        label_means.append(np.inf)

label_means = np.array(label_means)

# Sort after intensity
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


# Visual sanity check
if VISUAL_SANITY == True:
    
    show_overlay(volume, kmean_air, "kmean_air")
    show_overlay(volume, kmean_soft, "kmean_soft")
    show_overlay(volume, kmean_trab1, "kmean_trab 1")
    show_overlay(volume, kmean_trab2, "kmean_trab 2")
    show_overlay(volume, kmean_bone, "kmean_bone")


    
if Debug:
    # --- slice ---
    coords_ossicles = np.argwhere(ossicles_only)
    
    if len(coords_ossicles) > 0:
        center = coords_ossicles.mean(axis=0).astype(int)
    else:
        coords_air = np.argwhere(kmean_air)
        center = coords_air.mean(axis=0).astype(int)
        
    slice_idx = center[0]
    
    # --- means ---
    mean_air   = label_means[AIR-1]
    mean_soft = label_means[SOFT-1]
    mean_trab1 = label_means[TRAB1-1]
    mean_trab2  = label_means[TRAB2-1]
    mean_bone  = label_means[CORB-1]
    
    # --- loop ---
    for name, mask, mean in [
            ("bone_kmean_air", kmean_air, mean_air),
            ("bone_kmean_soft", kmean_soft, mean_soft),
            ("bone_kmean_trab1", kmean_trab1, mean_trab1),
            ("bone_kmean_trab2", kmean_trab2, mean_trab2),
            ("bone_kmean_bone", kmean_bone, mean_bone),
            ]:
        
        plt.figure(figsize=(6,6))
        plt.imshow(volume[slice_idx], cmap="gray")
        plt.imshow(mask[slice_idx], alpha=0.4)
        
        plt.title(f"{name} | mean: {mean:.2f}")
        plt.axis("off")
        
        out_path = os.path.join(PIC_DIR, f"{name}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()


        
        
