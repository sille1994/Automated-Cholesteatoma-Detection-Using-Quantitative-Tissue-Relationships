#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master batch runner

Runs all pipeline scripts sequentially for every patient.
Each script is executed in a fresh Python process.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# SETTINGS
SCAN_DIR = "./Scans"
RESULT_DIR = "./Results"
RESULT_SEG_DIR = "./Results_seg"

# Skip steps that are already completed
SKIP_COMPLETED = True

# Force rerun of every step
FORCE_RERUN = False


USE_FILTER = False

valid_patients = {
    # Eksempel:
    # 63, 75, 76
}

PIPELINE = [
    "STEP_1_Segmenting.py",
    "STEP_2_InnerSegmentation.py",
    "STEP_3_Mask_filler.py",
    "STEP_4_ROI_Kmean.py",
    "STEP_5_ROI_Ossicles.py",
    "STEP_6_Measurements.py",
]



# OUTPUT FILE THAT DEFINES EACH STEP AS COMPLETED

STEP_OUTPUTS = {
    "STEP_1_Segmenting.py":
        lambda p: os.path.join(RESULT_DIR, p, "cache", "segmentation.nii.gz"),

    "STEP_2_InnerSegmentation.py":
        lambda p: os.path.join(RESULT_SEG_DIR, p, "cache", "body_mask.nii.gz"),

    "STEP_3_Mask_filler.py":
        lambda p: os.path.join(RESULT_SEG_DIR, p, "cache", "body_mask_filled.nii.gz"),

    "STEP_4_ROI_Kmean.py":
        lambda p: os.path.join(RESULT_SEG_DIR, p, "cache", "kmeans_5class_roi_bone.nii.gz"),

    "STEP_5_ROI_Ossicles.py":
        lambda p: os.path.join(RESULT_SEG_DIR, p, "cache", "roi_mask_bone_final_none_PCA.nii.gz"),

    "STEP_6_Measurements.py":
        lambda p: os.path.join(
            RESULT_SEG_DIR,
            p,
            "feature_table_reference_radius_5_cylinder_allmost_all_filtered.csv"
        ),
}


LOG_FILE = "Pipeline_log.txt"

# CREATE OUTPUT DIRECTORIES

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_SEG_DIR, exist_ok=True)

# CHECK THAT PIPELINE EXISTS

for script in PIPELINE:
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Cannot find {script}")


# LOG FUNCTION

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# FIND PATIENTS

folders = sorted([
    f for f in os.listdir(SCAN_DIR)
    if os.path.isdir(os.path.join(SCAN_DIR, f))
])

log("=" * 70)
log("PIPELINE START")
log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 70)
log(f"Found {len(folders)} patient folders")
log("")

# STATISTICS

n_finished = 0
n_failed = 0
n_skipped = 0

pipeline_start = time.time()


# LOOP OVER PATIENTS

for folder in folders:

    try:
        patient = int(folder)
    except ValueError:
        log(f"Skipping {folder} (not numeric)")
        n_skipped += 1
        continue

    if USE_FILTER and patient not in valid_patients:
        log(f"Skipping {folder} (filtered)")
        n_skipped += 1
        continue

    log("")
    log("=" * 70)
    log(f"PATIENT {folder}")
    log("=" * 70)

    patient_failed = False

    patient_start = time.time()
    
    
    
    # SKIP ENTIRE PATIENT IF FINAL OUTPUT EXISTS
    if SKIP_COMPLETED and not FORCE_RERUN:

        final_output = STEP_OUTPUTS["STEP_6_Measurements.py"](folder)

        if os.path.exists(final_output):

            log("Patient already completed")
            n_skipped += 1
            continue

    # RUN PIPELINE
    for script in PIPELINE:


        # SKIP IF STEP ALREADY COMPLETED
        if SKIP_COMPLETED and not FORCE_RERUN:

            output_file = STEP_OUTPUTS[script](folder)

            if os.path.exists(output_file):
                
                log(f"Skipping {script}")
                log(f"Found: {output_file}")
                continue
            
            else:

                log(f"Output missing: {output_file}")
            


        log(f"Running {script}")

        log(f"Command: {sys.executable} {script} {folder}")
        
        start = time.time()

        try:

            with open(LOG_FILE, "a") as logfile:

                result = subprocess.run(
                    [sys.executable, script, folder],
                    stdout=logfile,
                    stderr=logfile,
                    text=True
                )

            runtime = round(time.time() - start, 2)
            

            if result.returncode != 0:

                log(f"FAILED ({runtime} s)")
                patient_failed = True
                break

            else:

                log(f"Finished ({runtime} s)")

        except Exception as e:

            log(f"ERROR: {e}")
            patient_failed = True
            break

    total_patient_time = round(time.time() - patient_start, 2)

    if patient_failed:
        n_failed += 1
        log(f"PATIENT {folder} FAILED ({total_patient_time} s)")
    else:
        n_finished += 1
        log(f"PATIENT {folder} DONE ({total_patient_time} s)")



# SUMMARY

total_runtime = round(time.time() - pipeline_start, 2)

log("")
log("=" * 70)
log("PIPELINE FINISHED")
log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 70)

log(f"Finished patients : {n_finished}")
log(f"Failed patients   : {n_failed}")
log(f"Skipped patients  : {n_skipped}")
log(f"Total runtime     : {total_runtime:.1f} seconds")

log("=" * 70)