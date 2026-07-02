# Automatic Cholesteatoma Feature Extraction from CBCT

This repository contains a fully automated pipeline for extracting
radiological features from cone-beam CT (CBCT) scans of the temporal
bone. The pipeline combines anatomical segmentation, automatic ossicle
segmentation, region-of-interest (ROI) generation, tissue
classification, and feature extraction for cholesteatoma research.

The workflow was developed as part of a Master's thesis investigating
automatic detection of middle ear cholesteatoma using CBCT.

------------------------------------------------------------------------

# Pipeline Overview

The pipeline consists of six sequential processing steps:

  -----------------------------------------------------------------------
  Step                   Description
  ---------------------- ------------------------------------------------
  **STEP 1**             Segment the CBCT scan into air, bone and soft
                         tissue using unsupervised clustering.

  **STEP 2**             Generate a temporal bone mask by morphological
                         processing of the skull segmentation.

  **STEP 3**             Refine and fill the temporal bone mask.

  **STEP 4**             Create a cylindrical ROI around the ossicles and
                         classify tissue inside the ROI using K-means
                         clustering.

  **STEP 5**             Detect the ossicles and generate the final
                         middle ear ROI.

  **STEP 6**             Extract volumetric and ratio-based radiological
                         features and export them to a CSV file.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Repository Structure

``` text
Scans/
    00001/
    00002/
    ...

Seg/
    seg_auto_00001.nii
    seg_auto_00002.nii
    ...

Results/
Results_seg/

STEP_1_Segmenting.py
STEP_2_InderSegmentation.py
STEP_3_Mask_filler.py
STEP_4_ROI_Kmean.py
STEP_5_ROI_ossicles.py
STEP_6_Measurements.py

Batch_running.py
```

------------------------------------------------------------------------

# Requirements

Python 3.10 or newer is recommended.

Required packages:

``` text
numpy
scipy
scikit-image
scikit-learn
pandas
matplotlib
nibabel
pydicom
SimpleITK
```

Install them with:

``` bash
pip install numpy scipy scikit-image scikit-learn pandas matplotlib nibabel pydicom SimpleITK
```

------------------------------------------------------------------------

# Automatic Ossicle Segmentation

This pipeline relies on the automatic temporal bone segmentation model
developed by the Auditory Biophysics Laboratory (ABL), Western
University.

Before running this repository:

1.  Install Docker Desktop.
2.  Pull the ABL Docker image.
3.  Generate automatic ossicle segmentations.
4.  Place the generated segmentation files in the **Seg/** folder.

The segmentation files should follow the naming convention:

``` text
Seg/
    seg_auto_00075.nii
    seg_auto_00102.nii
```

Example Docker command:

``` bash
docker pull uwoabl/temporal-bone-segmentation:latest
```

------------------------------------------------------------------------

# Input Data

Organize the CBCT scans as:

``` text
Scans/

    00075/
        DICOM files...

    00102/
        DICOM files...
```

Each patient folder should contain a single CBCT DICOM series.

------------------------------------------------------------------------

# Running the Pipeline

Run the scripts in the following order:

``` text
STEP_1_Segmenting.py
STEP_2_InderSegmentation.py
STEP_3_Mask_filler.py
STEP_4_ROI_Kmean.py
STEP_5_ROI_ossicles.py
STEP_6_Measurements.py
```

Each script accepts the patient ID:

``` bash
python STEP_1_Segmenting.py 00075
```

------------------------------------------------------------------------

# Batch Processing

To process multiple scans automatically, edit:

``` python
SCRIPT = "STEP_1_Segmenting.py"
```

inside:

``` text
Batch_running.py
```

Then execute:

``` bash
python Batch_running.py
```

The batch runner launches a new Python process for every patient. This
automatically releases memory after each case and prevents memory
accumulation during large batch analyses.

Filtering can optionally be enabled to process only a selected subset of
patients.

------------------------------------------------------------------------

# Output

The pipeline generates:

``` text
Results/
```

-   Intermediate processing files
-   Cached volumes
-   Tissue segmentations

``` text
Results_seg/
```

-   ROI masks
-   K-means tissue classifications
-   Detected ossicles
-   Debug figures
-   Intermediate masks

``` text
feature_table_reference_*.csv
```

-   Final quantitative radiological features

------------------------------------------------------------------------

# Adjustable Parameters

Several parameters can easily be modified inside the scripts, including:

-   ROI radius
-   Cylinder dimensions
-   Number of K-means clusters
-   Visualization options
-   Debug mode
-   CSV export
-   Batch filtering

Intermediate files are cached to avoid recomputation when rerunning the
pipeline.

------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite the associated
Master's thesis and acknowledge the Auditory Biophysics Laboratory
(Western University) for providing the automatic ossicle segmentation
model.

------------------------------------------------------------------------

# Future Improvements

Possible future additions include:

-   Pipeline overview figure
-   Example input and output data
-   Docker installation guide
-   Example dataset
-   Performance benchmarks
