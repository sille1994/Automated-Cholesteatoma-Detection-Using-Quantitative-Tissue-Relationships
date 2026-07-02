
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Imports
# ============================================================

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import (
    zoom,
    binary_fill_holes
)


from skimage.morphology import ball, binary_closing



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
OSSICLES = f"./Seg/seg_auto_{NAME}.nii"

VISUAL_SANITY = False
SHOW_VISUALS = False
Debug = False
SAVE = True
    
# Checking if the code is already run
OUTPUT_FILE = os.path.join(OUT_DIR, "mask_small_c.nii.gz")

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    print(f"Output already exists for {NAME} — stopping")
    sys.exit(1)


MASK_file = os.path.join(DATA_DIR, "bone_scaled_BEFORE_STEP_7.nii.gz")

if not os.path.exists(MASK_file) or os.path.getsize(MASK_file) == 0:
    print(f"bone_scaled_BEFORE_STEP_7 mangler for {NAME} – stopper.")
    sys.exit(1)
    
if not os.path.exists(OSSICLES) or os.path.getsize(OSSICLES) == 0:
    print(f"seg_auto_{NAME}.nii mangler – stopper.")
    sys.exit(1)
    

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if not os.path.exists(OUT_DIR_CACHE):
    os.makedirs(OUT_DIR_CACHE)

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


mask_small_c = mask_small.copy()


for i in range (0, 2, 1): 
          
    # 1. Scaling
      
    scale =  round( 0.30 + i * 0.05 ,2)
        
    mask_small_c_ds = zoom(
            mask_small_c.astype(float),
            (scale, scale, scale),
            order=0
        ).astype(bool)

    mask_small_c_ds_closed = mask_small_c_ds.copy()
    for _ in range(4):
        closed = binary_closing(mask_small_c_ds_closed, ball(10))
        mask_small_c_ds_closed = binary_fill_holes(closed)
            
    scale_factors = (
            mask_small_c.shape[0] / mask_small_c_ds_closed.shape[0],
            mask_small_c.shape[1] / mask_small_c_ds_closed.shape[1],
            mask_small_c.shape[2] / mask_small_c_ds_closed.shape[2],
        )
            
    mask_small_c = zoom(
            mask_small_c_ds_closed.astype(float),
            scale_factors,
            order=0
        ).astype(bool)
            
     
    show_overlay(
          volume,
          mask_small_c,
          title=f"STEP 4 — SCALING DOWN IN SAME COORDINATE SYSTEM . SCALING FACOTR {scale}"
      )

if SAVE == True:
        
    mask_xyz = np.transpose(mask_small_c, (2,1,0)).astype(np.uint8)
    assert mask_xyz.shape == ref.shape, "Shape mismatch!"
        
    nii = nib.Nifti1Image(mask_xyz, ref.affine, ref.header)
    nii.set_data_dtype(np.uint8)
        
    out_path = os.path.join(OUT_DIR, "mask_small_c.nii.gz")
    nib.save(nii, out_path)
        
    print("Saved mask_small_c NIfTI")
    
    
    mask_xyz = np.transpose(volume, (2,1,0)).astype(np.float32)
    assert mask_xyz.shape == ref.shape, "Shape mismatch!"
        
    nii = nib.Nifti1Image(mask_xyz, ref.affine, ref.header)
    nii.set_data_dtype(np.float32)
        
    out_path = os.path.join(OUT_DIR, "CBCT_volume.nii.gz")
    nib.save(nii, out_path)
        
    print("Saved volume NIfTI")






