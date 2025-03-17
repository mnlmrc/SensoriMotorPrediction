This repository contains Python and MATLAB code related to the SensoriMotorPrediction (SMP) project in which we investigate how probabilistic cues are incorporated into feedback control mechanisms. 

Toolbox dependencies to use the code:

MATLAB R2020a (version used for fMRI data preprocessing)
SPM12 (https://www.fil.ion.ucl.ac.uk/spm)
Dataframe toolbox (https://github.com/jdiedrichsen/dataframe)
...

# **fMRI data**

The raw fMRI data are stored following the Brain Imaging Data Structure (BIDS). Participants information are stored in the participants.tsv file. 

## Preprocessing
### Structural images

- Unzip T1w images

```
smp2_anat('BIDS:move_unzip_raw_anat', 'sn', sn)
```

First-level GLM: ...

ROI definition: ...

Time
