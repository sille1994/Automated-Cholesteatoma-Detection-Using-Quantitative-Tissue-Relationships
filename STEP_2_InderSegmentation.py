"""
Batch-runner for temporal bone + EAM pipeline

"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import (
    zoom,
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    map_coordinates
)
from skimage.measure import label
from skimage.morphology import ball, binary_closing


# ============================================================
# SETTINGS  
# ============================================================

# Run the code in terminal
NAME = sys.argv[1]

# Run the code yourself
#NAME = "00075" 



BASE_DIR = f"./Results/{NAME}"
CACHE_DIR = f"{BASE_DIR}/cache"
SEG_PATH = f"{CACHE_DIR}/seg_final.npy"
BODY_MASK_PATH = f"{CACHE_DIR}/body_mask.npy"
CBCT_PATH = f"{BASE_DIR}/CBCT_volume.nii.gz"
PIC_DIR = f"./Results_seg/{NAME}/Picture/"
SHOW_VISUALS = False

os.makedirs(CACHE_DIR, exist_ok=True)


# Checking if the code is already run
OUTPUT_FILE = os.path.join(CACHE_DIR, "bone_scaled_BEFORE_STEP_7.nii.gz")

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    print(f"Output already exists for {NAME} — skipping.")
    sys.exit(0)



# ============================================================
# LOAD DATA
# ============================================================
    
seg_final = np.load(SEG_PATH)
body_mask = np.load(BODY_MASK_PATH)
    
cbct_nii = nib.load(CBCT_PATH)
cbct_vol = np.transpose(cbct_nii.get_fdata(), (2, 1, 0))
    
air = seg_final == 0 
soft = seg_final == 2 
bone = seg_final == 3 
    
assert body_mask.shape == air.shape, "Body mask shape mismatch"


# ============================================================
# VISUALIZATION
# ============================================================

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
            

# ============================================================
# SAVE
# ============================================================

def save_npy_and_nii(mask, name):
        #np.save(f"{CACHE_DIR}/{name}.npy", mask.astype(np.uint8))

        mask_nii = np.transpose(mask.astype(np.uint8), (2, 1, 0))
        nii = nib.Nifti1Image(
            mask_nii,
            affine=cbct_nii.affine,
            header=cbct_nii.header
        )
        nii.set_data_dtype(np.uint8)
        nii.header.set_intent("label")
        

        nib.save(nii, f"{CACHE_DIR}/{name}.nii.gz")
#        nib.save(nii, f"./Resultater_seg/{NAME}/Picture/{name}.nii.gz")



# Checking if the code is already run
OUTPUT_FILE = os.path.join(CACHE_DIR, "bone_scaled_BEFORE_STEP_7.nii.gz")

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    print(f"Output already exists for {NAME} — skipping STEP 1–7.")

else:
    
    # ============================================================
    # STEP 1 – ISOLATE TEMPORAL / SKULL BONE
    # ============================================================
    
    bone_filled = binary_fill_holes(bone) & body_mask

    labels_bone = label(bone_filled, connectivity=3)
    sizes_bone = np.bincount(labels_bone.flat)
    sizes_bone[0] = 0
    skull_bone = labels_bone == sizes_bone.argmax()
    
    
    print("STEP 1 – ISOLATE TEMPORAL / SKULL BONE")
    show_overlay(cbct_vol, skull_bone, title="STEP 1 – ISOLATE TEMPORAL / SKULL BONE")
    
    save_npy_and_nii(skull_bone, "skull_bone")

    # ============================================================
    # STEP 2 – MULTISCALE CLOSING
    # ============================================================
    #down_factor = 0.15
    
    down_factor = 0.20
    
    skull_ds = zoom(
        skull_bone.astype(float),
        (down_factor, down_factor, down_factor),
        order=0
    ).astype(bool)
    
    skull_ds_closed = skull_ds.copy()
    for _ in range(4):
        skull_ds_closed = binary_closing(skull_ds_closed, ball(4))
        
    scale_factors = (
        skull_bone.shape[0] / skull_ds_closed.shape[0],
        skull_bone.shape[1] / skull_ds_closed.shape[1],
        skull_bone.shape[2] / skull_ds_closed.shape[2],
    )   
    
    temporal_shell = zoom(
        skull_ds_closed.astype(float),
        scale_factors,
        order=0
    ).astype(bool)
    
    print("STEP 2 – MULTISCALE CLOSING")
    show_overlay(cbct_vol, temporal_shell, title="STEP 2 – MULTISCALE CLOSING")
    

    
    # ============================================================
    # STEP 3 – FILL TEMPORAL
    # ============================================================
    
    temporal_filled = binary_fill_holes(temporal_shell) & body_mask

    print("STEP 3 – FILL TEMPORAL")
    show_overlay(cbct_vol, temporal_filled, title="STEP 3 – FILL TEMPORAL")
    
    save_npy_and_nii(temporal_filled, "temporal_filled")

    # ============================================================
    # STEP 4 – SCALING DOWN IN SAME COORDINATESYSTEM
    # ============================================================
    
    temporal_scaled = np.copy(temporal_filled)  
    
    for i in range (0, 14, 1): 
        
        
        
        # 1. Scaling
        scale = round( 0.30 + i * 0.05 ,2)
        
        print(f"STEP 4 — SCALING DOWN IN SAME COORDINATE SYSTEM. SCALING FACOTR {scale}")
        
        
        # 2. Find bounding box for temporal bone
        coords = np.where(temporal_scaled)
        
        zmin, zmax = coords[0].min(), coords[0].max()
        ymin, ymax = coords[1].min(), coords[1].max()
        xmin, xmax = coords[2].min(), coords[2].max()
        
        roi = temporal_scaled[
            zmin:zmax+1,
            ymin:ymax+1,
            xmin:xmax+1
        ]
        
        
        # 3. Find center
        coords_roi = np.array(np.where(roi))
        center = coords_roi.mean(axis=1)
        
        # 4. Create coordinate grid (ONLY ROI)
        Z, Y, X = np.indices(roi.shape)
        
        Zc = Z - center[0]
        Yc = Y - center[1]
        Xc = X - center[2]
        
        
        Zs = Zc / scale + center[0]
        Ys = Yc / scale + center[1]
        Xs = Xc / scale + center[2]
        
        # 5. Sample scaled mask
        roi_scaled = map_coordinates(
            roi.astype(float),
            [Zs, Ys, Xs],
            order=0
        ).astype(bool)
        
        # 6. Place scaled ROI back in full volume
        temporal_filled_scaled = np.zeros_like(temporal_scaled)
        
        temporal_filled_scaled[
            zmin:zmax+1,
            ymin:ymax+1,
            xmin:xmax+1
        ] = roi_scaled
        
        
        # 7. Find air inside scaled temporal bone
        air_temporal = air & temporal_filled_scaled
        
        # 8. Add air to temporal ROI
        temporal_filled_with_air = temporal_filled | air_temporal
            
        # 9. Visualize
        
        show_overlay(
            cbct_vol,
            temporal_filled_with_air,
            title=f"STEP 4 — SCALING DOWN IN SAME COORDINATE SYSTEM . SCALING FACOTR {scale}"
        )

        # ============================================================
        # STEP 5 – MULTISCALE CLOSING
        # ============================================================

        
        down_factor = 0.20
        
        skull_ds = zoom(
            temporal_filled_with_air.astype(float),
            (down_factor, down_factor, down_factor),
            order=0
        ).astype(bool)

        skull_ds_closed = skull_ds.copy()
        for _ in range(4):
            skull_ds_closed = binary_closing(skull_ds_closed, ball(4))
            
        scale_factors = (
            skull_bone.shape[0] / skull_ds_closed.shape[0],
            skull_bone.shape[1] / skull_ds_closed.shape[1],
            skull_bone.shape[2] / skull_ds_closed.shape[2],
        )
            
        temporal_shell_2 = zoom(
            skull_ds_closed.astype(float),
            scale_factors,
            order=0
        ).astype(bool)
            
        print(f"STEP 5 – {i+1}. MULTISCALE CLOSING")
        show_overlay(cbct_vol, temporal_shell_2, title=f"STEP  {5 + 3 * i} – {i+1}. MULTISCALE CLOSING")
        
        # ============================================================
        # STEP 6 – FILL TEMPORAL
        # ============================================================

        temporal_scaled = binary_fill_holes(temporal_shell_2) & body_mask
        
        print(f"STEP 6 – {i+1}. FILL TEMPORAL")
        show_overlay(cbct_vol, temporal_scaled, title=f"STEP  {6 + 3 * i} – {i+1}. FILL TEMPORAL")

    # ============================================================
    # STEP 7 – EXPAND AIR TO DETECT SOFT TISSUE AND AIR TISSUE
    # ============================================================

    save_npy_and_nii(temporal_scaled, "bone_scaled_BEFORE_STEP_7")
    
    # THIS STEP WAS NOT USED AND THEREFOR DELETED




