This repository contains Python and MATLAB code related to the SensoriMotorPrediction (SMP) project in which we 
investigate how probabilistic predictions are incorporated into feedback control mechanisms. Within the project, the 
`experiment=smp2` refer to Experiment 1 and `experiment=smp0` to Experiment 2. Wherever required to run the code, the
`<subject number>` for a given participant can be found in the `participants.tsv` file. 

Toolbox dependencies to use the code:

MATLAB R2020a (version used for fMRI data preprocessing), SPM12 (https://www.fil.ion.ucl.ac.uk/spm), 
Dataframe toolbox (https://github.com/jdiedrichsen/dataframe), ...

# **Behavioural data**

- `python force.py single_trial <subject number>`: calculate behavioural metrics, including RT, average force response 
200-400ms from the perturbation (Fig. 1c) and mean deviation (Fig. 1e). 

This step produces a `.tsv` file with one row per trial. The columns `thumb0, ..., pinkie0` contain the average force 
1.5s before the perturbation. The columns `thumb1, ..., pinkie1` contain the average force 200-400ms from the 
perturbation. Individual-participant behavioural datasets pooled together in a single `.tsv` file can be found at 
`data/behavioural/smp2_force_single_trial.tsv`.

# **fMRI data**

The raw fMRI data are stored following the Brain Imaging Data Structure (BIDS). Participants' information are stored in 
the `participants.tsv` file. 

## Preprocessing

For preprocessing, we used the MATLAB functions smp2_anat (structural images), smp2_func (EPI images and fieldmap 
correction) and smp2_glm (first-level GLM) in sensori-motor-prediction.

### ROI definition

- `python rois.py make_cortical_rois --sn <subject number>`: save ROI (see Fig. 2a) and hemisphere masks to NIFTI files 

### Univariate activation

- `python betas.py make_contrasts_cifti --sn <subject number>`: save cortical contrasts to CIFTI file with condition in 
the rows and voxels in the columns

- `python betas.py roi_contrasts`: calculate average activation for each ROI using the individual-participant CIFTI 
files from the previous step and save it to .tsv file. This step produces the file at 
`data/univariate_activation/ROI.con.avg.tsv`

### PCM model definition

All the representational models are impletemented in the `pcm_models` module:

- `python pcm_models.py plan`: save preparation models (Fig. 3a)
- `python pcm_models.py exec`: save execution models (Fig. 6a)

This step produces `.p` files containing a list of second moment (G) matrix for each model. The G matrix can be 
translated to squared Euclidean distances using the PCM function `pcm.G_to_dist` (see Eq. 5) 

### PCM model fitting

PCM models are fitted to the beta coefficients from 1st level GLM, separately for preparation and execution. First, the 
beta coefficients and the residuals of each participants are saved to CIFTI files:

- `python betas.py make_betas_cifti --sn <subject number>`: save cortical beta coefficients from 1st-level GLM to CIFTI 
file with condition and runs in the rows 
and voxels in the columns

- `python betas.py make_residuals_cifti --sn <subject number>`: save residuals timeseries to CIFTI file with time in the 
rows and voxel in the columns

The residual timeseries is needed to perform multivariate pre-whitening (see Eq. 2). Then, PCM models are fitted to the
pre-whitened neural activity patterns:

- `python pcm_cortical.py rois_planning`: fit preparation models to beta coefficient estimated for response preparation
in each ROI
- `python pcm_cortical.py rois_execution`: fit preparation models to beta coefficient estimated for response execution

These steps produce: 1) a `G_obs.<epoch>.glm12.<Hem>.<roi>.npy` file for each ROI with the observed G matrix in 
each participant (a copy of these files is stored in `data/encoding/`); 2) a `theta_in.<epoch>.glm12.<Hem>.<roi>.p` with 
the log-weight of each model in each participant (see Eq. 12). The weights of each model are stored in 
`data/encoding/component_model.BOLD.tsv`.

### PCM correlations

- Preparation-execution correlation in BOLD: `python pcm_cortical.py correlation_plan-exec`





