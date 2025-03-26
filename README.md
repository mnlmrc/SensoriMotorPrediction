This repository contains Python and MATLAB code related to the SensoriMotorPrediction (SMP) project in which we investigate how probabilistic cues are incorporated into feedback control mechanisms. 

Toolbox dependencies to use the code:

MATLAB R2020a (version used for fMRI data preprocessing)
SPM12 (https://www.fil.ion.ucl.ac.uk/spm)
Dataframe toolbox (https://github.com/jdiedrichsen/dataframe)
...

# **fMRI data**

The raw fMRI data are stored following the Brain Imaging Data Structure (BIDS). Participants information are stored in the participants.tsv file. 

## Preprocessing

For preprocessing, we used the MATLAB functions smp2_anat (structural images), smp2_func (EPI images and fieldmap correction) and smp2_glm (first-level GLM) in sensori-motor-prediction/smp2.

### Structural images

- Unzip T1w images: ` smp2_anat('BIDS:move_unzip_raw_anat', 'sn', sn) ` (sn->subject number from the 'sn' field in participants.tsv)
- LPI reslicing: ` smp2_anat('ANAT:reslice_LPI', 'sn', sn) `
- Re-centering to anterior commissure (AC): ` smp2_anat('ANAT:centre_AC', 'sn', sn) `
- Brain tissue segmentation: ` smp2_anat('ANAT:segment', 'sn', sn) `
- Surface reconstruction in fsaverage space:  ` smp2_anat('SURF:reconall', 'sn', sn) `
- Resampling surface(s) from fsaverage to fs_LR space: ` smp2_anat('SURF:fs2wb', 'sn', sn) `

More on surface reconstruction pipelines [here](https://diedrichsenlab.github.io/guides/06_surface_analysis.html).

First-level GLM: ...

ROI definition: ...

Time
