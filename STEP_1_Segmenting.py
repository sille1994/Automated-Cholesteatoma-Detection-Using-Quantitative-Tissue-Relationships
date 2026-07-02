#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# CBCT → 3D SEGMENTING → NIFTI + RTSTRUCT
# ============================================================

import os
import shutil
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
from skimage.morphology import remove_small_objects
from sklearn.cluster import MiniBatchKMeans
# from rt_utils import RTStructBuilder
from scipy.ndimage import gaussian_filter, zoom, uniform_filter
import sys
# ============================================================
# USER INPUT
# ============================================================

# Run the code in terminal
NAME = sys.argv[1]

# Run the code yourself
#NAME = "00075"

DICOM_ROOT = f"./Scans/{NAME}"
CACHE_DIR = f"./Results/{NAME}/cache"

CBCT_NIFTI_PATH = f"./Results/{NAME}/CBCT_volume.nii.gz"
OUT_NIFTI = f"./Results/{NAME}/CBCT_segmentation_LABELMAP.nii.gz"
OUT_RTSTRUCT = f"./Results/{NAME}/CBCT_RTSTRUCT.dcm"
RT_FOLDER = f"./Results/{NAME}/RTSTRUCT_SERIE"

AIR_THRESHOLD = 5
TV_WEIGHT = 0.05
SPATIAL_WEIGHT = 0.25

MIN_RED = 5000
MIN_CYAN = 2000

os.makedirs(CACHE_DIR, exist_ok=True)


SHOW_VISUALS = False
DICOM_FILES = True


#%%

def show_overlay(volume, mask, title, z=None, cmap="Reds"):
        if not SHOW_VISUALS:
            return
        if z is None:
            z = volume.shape[0] // 2
        for dz in [-75, -25, 25, 75]:
            plt.figure(figsize=(6, 6))
            plt.imshow(volume[z + dz], cmap="gray")
            plt.imshow(mask[z + dz], cmap=cmap, alpha=0.4)
            plt.title(title)
            plt.axis("off")
            plt.show()




#%%

# ============================================================
# 1. FIND + FILTER CBCT SERIE
# ============================================================


if DICOM_FILES == True:
    dicom_files = []
    for root, _, files in os.walk(DICOM_ROOT):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))

    assert dicom_files, "No DICOM-filer found"

    dicom_files = sorted(
        dicom_files,
        key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
        )

    first_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    SERIES_UID = first_ds.SeriesInstanceUID

    dicom_files = [
    f for f in dicom_files
        if pydicom.dcmread(f, stop_before_pixels=True).SeriesInstanceUID == SERIES_UID
        ]

    print("Number CBCT slices:", len(dicom_files))

# ============================================================
# 2. LOAD / CACHE CBCT VOLUME (Z,Y,X)
# ============================================================

vol_path = f"{CACHE_DIR}/volume.npy"

if os.path.exists(vol_path):
    volume = np.load(vol_path)
else:
    volume = np.stack([
        pydicom.dcmread(f).pixel_array
        for f in dicom_files
    ]).astype(np.float32)
    np.save(vol_path, volume)

# ============================================================
# 2.5 BUILD + CACHE CBCT_VOLUME.NII.GZ  (GROUND TRUTH)
# ============================================================

if not os.path.exists(CBCT_NIFTI_PATH):

    ds0 = pydicom.dcmread(dicom_files[0])

    px, py = map(float, ds0.PixelSpacing)
    pz = float(ds0.SliceThickness)

    # DICOM volume (Z,Y,X) → NIFTI (X,Y,Z)
    cbct_nifti_data = np.transpose(volume, (2, 1, 0))

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = px
    affine[1, 1] = py
    affine[2, 2] = pz

    cbct_nii = nib.Nifti1Image(cbct_nifti_data, affine)
    cbct_nii.header.set_xyzt_units("mm")
    cbct_nii.header['cal_min'] = float(cbct_nifti_data.min())
    cbct_nii.header['cal_max'] = float(cbct_nifti_data.max())

    nib.save(cbct_nii, CBCT_NIFTI_PATH)
    print("CBCT_volume.nii.gz made")

else:
    print("CBCT_volume.nii.gz already found (cache)")
    cbct_nii = nib.load(CBCT_NIFTI_PATH)

# ============================================================
# 3. NORMALIZE + BODY MASK (CACHE)
# ============================================================

norm_path = f"{CACHE_DIR}/volume_norm.npy"
mask_path = f"{CACHE_DIR}/body_mask.npy"

if os.path.exists(norm_path):
    volume_norm = np.load(norm_path)
    body_mask = np.load(mask_path)
else:
    vmin, vmax = volume.min(), volume.max()
    volume_norm = 100 * (volume - vmin) / (vmax - vmin)
    body_mask = volume_norm > AIR_THRESHOLD
    np.save(norm_path, volume_norm)
    np.save(mask_path, body_mask)

# ============================================================
# STEP 4 - DENOISING
# ============================================================

diff_path = f"{CACHE_DIR}/volume_diff.npy"


print("STEP 4 - Denoising")


if os.path.exists(diff_path):
    volume_diff = np.load(diff_path)
else:
    
    empty_mask = np.zeros_like(volume)
    

    # 1. Gaussian smoothing
    volume_smooth = gaussian_filter(volume_norm, sigma=1)

    show_overlay(volume_norm, empty_mask, title="STEP 4 - Denoising (Before filtering)")
    show_overlay(volume_smooth, empty_mask, title="STEP 4 - Denoising (Gaussian)")



    # 2. Downsample
    scale = 0.75
    volume_ds = zoom(volume_smooth, scale, order=1)

    # 3. TV denoising
    volume_tv = denoise_tv_chambolle(
        volume_ds,
        weight=TV_WEIGHT,
        channel_axis=None
    )

    # 4. Upsample 
    scale_factors = (
        volume_norm.shape[0] / volume_tv.shape[0],
        volume_norm.shape[1] / volume_tv.shape[1],
        volume_norm.shape[2] / volume_tv.shape[2],
        )

    volume_tv_up = zoom(volume_tv, scale_factors, order=1)

    # 5. Saml med original baggrund
    volume_diff = volume_norm.copy()
    volume_diff[body_mask] = volume_tv_up[body_mask]

    show_overlay(volume_diff, empty_mask, title="STEP 4 - Denoising (Gaussian and TV)")
    
    np.save(diff_path, volume_diff)


# ============================================================
# 5. SPATIAL K-MEANS
# ============================================================


print("STEP 5 - K-Means")


# 1. Coordinates for body voxels
coords = np.array(np.where(body_mask)).astype(np.float32)

Z = coords[0] / volume_diff.shape[0]
Y = coords[1] / volume_diff.shape[1]
X = coords[2] / volume_diff.shape[2]


# 2. Local intensity context
local_mean = uniform_filter(volume_diff, size=3)


# 3. Feature vector
features = np.column_stack([
    volume_diff[body_mask],
    local_mean[body_mask],
    SPATIAL_WEIGHT * Z,
    SPATIAL_WEIGHT * Y,
    SPATIAL_WEIGHT * X,
])


# 4. MiniBatch K-means
kmeans = MiniBatchKMeans(
    n_clusters=3,
    batch_size=500_000,
    n_init=10,
    random_state=0
)


labels = kmeans.fit_predict(features)


# 5. Sort clusters by intensity
centers = kmeans.cluster_centers_

order = np.argsort(
    centers[:,0] + centers[:,1]   # intensity + local_mean
)


# 6. Reassign cluster labels
seg_body = np.zeros_like(labels, dtype=np.uint8)

for new, old in enumerate(order):
    seg_body[labels == old] = new + 1


# 7. Insert into full volume
seg = np.zeros_like(volume_norm, dtype=np.uint8)
seg[body_mask] = seg_body


# 8. (Optional) save segmentation
#np.save(seg_path, seg)


# ============================================================
# 6. POST-PROCESSING 
# ============================================================


print("STEP 6 - Post-processing")


seg_final_path = f"{CACHE_DIR}/seg_final.npy"


seg_clean = seg.copy()

mask_red = remove_small_objects(seg == 1, MIN_RED, connectivity=3)
seg_clean[(seg == 1) & (~mask_red)] = 2

mask_cyan = remove_small_objects(seg_clean == 3, MIN_CYAN, connectivity=3)
seg_clean[(seg_clean == 3) & (~mask_cyan)] = 2

seg_clean[seg_clean == 1] = 0
seg_final = seg_clean
np.save(seg_final_path, seg_final)



# ============================================================
# 7. NIFTI LABELMAP (MATCHER CBCT_VOLUME)
# ============================================================
seg_nifti_data = np.transpose(seg_final, (2, 1, 0))  # (X,Y,Z)

seg_nii = nib.Nifti1Image(
    seg_nifti_data.astype(np.uint8),
    affine=cbct_nii.affine,
    header=cbct_nii.header
)

seg_nii.set_data_dtype(np.uint8)
seg_nii.header.set_intent("label")

nib.save(seg_nii, OUT_NIFTI)
print(" NIFTI labelmap saved:", OUT_NIFTI)

# ============================================================
# 8 GEOMETRY VALIDATION
# ============================================================

print("\n=== GEOMETRY CHECK ===")
print("CBCT shape:", cbct_nii.shape)
print("SEG  shape:", seg_nii.shape)
print("CBCT spacing:", cbct_nii.header.get_zooms())
print("SEG  spacing:", seg_nii.header.get_zooms())
print("Affine equal:", np.allclose(cbct_nii.affine, seg_nii.affine))




