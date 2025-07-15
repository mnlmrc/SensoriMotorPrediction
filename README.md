This repository contains Python and MATLAB code related to the SensoriMotorPrediction (SMP) project in which we investigate how probabilistic cues are incorporated into feedback control mechanisms. 

Toolbox dependencies to use the code:

MATLAB R2020a (version used for fMRI data preprocessing), SPM12 (https://www.fil.ion.ucl.ac.uk/spm), Dataframe toolbox (https://github.com/jdiedrichsen/dataframe), ...

# **fMRI data**

The raw fMRI data are stored following the Brain Imaging Data Structure (BIDS). Participants' information are stored in the participants.tsv file. 

## Preprocessing

For preprocessing, we used the MATLAB functions smp2_anat (structural images), smp2_func (EPI images and fieldmap correction) 
and smp2_glm (first-level GLM) in sensori-motor-prediction/smp2.

### Structural images

- Unzip T1w images: ` smp2_anat('BIDS:move_unzip_raw_anat', 'sn', sn) ` (sn->subject number from the 'sn' field in participants.tsv)
- LPI reslicing: ` smp2_anat('ANAT:reslice_LPI', 'sn', sn) `
- Re-centering to anterior commissure (AC): ` smp2_anat('ANAT:centre_AC', 'sn', sn) `
- Brain tissue segmentation: ` smp2_anat('ANAT:segment', 'sn', sn) `
- Surface reconstruction in fsaverage space:  ` smp2_anat('SURF:reconall', 'sn', sn) `
- Resampling surface(s) from fsaverage to fs_LR space: ` smp2_anat('SURF:fs2wb', 'sn', sn) `

More on surface reconstruction pipelines [here](https://diedrichsenlab.github.io/guides/06_surface_analysis.html).

### First-level GLM: 
...

Using the `betas` module, the beta coefficients estimated for each regressor and the contrasts from 1st level GLM 
can be stored in a single CIFTI file using the make_cifti_betas function from [imaging_pipelines](https://github.com/mnlmrc/imaging_pipelines):

- beta coefficient estimated for each regressor: `python betas.py save_betas_cifti --sn <subject number>>`
- Save contrasts to a single cifti file: `python betas.py save_contrasts_cifti --sn <subject number>>`

Both the above steps require first to create masks for each hemisphere (see ROI definition).

### ROI definition

Surface-based definition of cortical ROIs is implemented in the `rois` module using the `Rois` class 
from [imaging_pipelines](https://github.com/mnlmrc/imaging_pipelines), which in turn uses the functions and classes 
from the `atlas_map` module from [Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion):

- Save NIFTI files of cortical ROI masks plus NIFTI hemisphere masks: 
    `python rois.py make_cortical_rois --sn <subject number>`

### Activation

- 


