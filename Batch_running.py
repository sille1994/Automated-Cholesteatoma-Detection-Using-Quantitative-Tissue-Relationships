#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for Ossicle_ROI_v2.py
Runs the script for all folders in ./Results
Optionally filters patients (0,1,98) using hardcoded list
"""

import os
import subprocess
import time
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

RESULT_DIR = "Results"


SCRIPT = "NAME_THE_PYHTON_FILE_YOU_WANT_TO RUN.py"
LOG_FILE = "NAME_THE_LOG_FILE.txt"

# E.G.
#SCRIPT = "Test_ROI_OS.py"
#LOG_FILE = "Test_ROI_OS.txt"

USE_FILTER = False   

# ============================================================
# HARDCODET LISTE
# ============================================================

valid_patients = {
    .....
}

# E.G.
# valid_patients = {
#    63, 75
# }

# ============================================================
# LOG FUNCTION
# ============================================================

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ============================================================
# MAIN
# ============================================================

def main():

    log("\n==========================================")
    log("BATCH START")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("==========================================\n")

    # Find folders
    folders = sorted([
        f for f in os.listdir(RESULT_DIR)
        if os.path.isdir(os.path.join(RESULT_DIR, f))
    ])

    log(f"Found {len(folders)} folders\n")

    for folder in folders:

        # ====================================================
        # FILTER (OPTIONAL)
        # ====================================================

        if USE_FILTER:
            try:
                patient_id = int(folder)
            except:
                log(f"Skipping {folder} (not numeric)")
                continue

            if patient_id not in valid_patients:
                log(f"Skipping {folder} (not in the group")
                continue

        # ====================================================
        # RUN
        # ====================================================

        start_time = time.time()

        log("------------------------------------------------")
        log(f"Running scan: {folder}")
        log("------------------------------------------------")

        try:

            with open(LOG_FILE, "a") as log_file:

                subprocess.run(
                    ["python", SCRIPT, folder],
                    stdout=log_file,
                    stderr=log_file
                )

            runtime = round(time.time() - start_time, 2)

            log(f"Finished {folder} in {runtime} sec\n")

        except Exception as e:

            runtime = round(time.time() - start_time, 2)

            log(f"ERROR while running {folder}")
            log(str(e))
            log(f"Runtime before crash: {runtime} sec\n")

    log("\n==========================================")
    log("BATCH FINISHED")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("==========================================\n")


if __name__ == "__main__":
    main()