This repository contains Python and MATLAB code related to the SensoriMotorPrediction (SMP) project in which we 
investigate how probabilistic predictions are incorporated into feedback control mechanisms. Within the project, 
`experiment=smp2` refer to Experiment 1 and `experiment=smp0` to Experiment 2. Wherever required to run the code, the
`<subject number>` for a given participant can be found in the `participants.tsv` file. 

Toolbox dependencies to use the code:

MATLAB R2020a (version used for fMRI data preprocessing), SPM12 (https://www.fil.ion.ucl.ac.uk/spm), 
Dataframe toolbox (https://github.com/jdiedrichsen/dataframe), RWLS toolbox (https://github.com/jdiedrichsen/rwls), 
surfAnalysis (https://github.com/DiedrichsenLab/surfAnalysis), surfAnalysisPy 
(https://github.com/DiedrichsenLab/surfAnalysisPy), nitools (https://github.com/DiedrichsenLab/nitools), 
Functional_Fusion (https://github.com/DiedrichsenLab/Functional_Fusion), PcmPy 
(https://github.com/DiedrichsenLab/PcmPy), imaging_pipelines (https://github.com/mnlmrc/imaging_pipelines)

The `data` folder in the this repository contains the minimal dataset required to reproduce the results and figures 
presented in the article [Sensory expectations and prediction error during feedback control in the human brain](https://www.biorxiv.org/content/10.64898/2026.01.19.700321v1). 
Participants' information is stored in the `participants.tsv` file. The `notebooks` folder contains jupyter notebooks 
reproducing each figure for the mininal dataset in the `data` folder. Additional data will be provided upon requested to 
the authors. 

# **Behavioural data**

- `python force.py single_trial <subject number>`: calculate behavioural metrics, including RT, average force response 
200-400ms from the perturbation (Fig. 1c) and mean deviation (Fig. 1e). 

This step produces a `.tsv` file with one row per trial. The columns `thumb0, ..., pinkie0` contain the average force 
1.5s before the perturbation. The columns `thumb1, ..., pinkie1` contain the average force 200-400ms from the 
perturbation. Individual-participant behavioural datasets pooled together in a single `.tsv` file can be found at 
`data/behavioural/smp2_force_single_trial.tsv`.

# **fMRI data**

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

## **PCM**

### model definition

All the representational models are impletemented in the `pcm_models` module:

- `python pcm_models.py plan`: save preparation models (Fig. 3a)
- `python pcm_models.py exec`: save execution models (Fig. 6a)

This step produces `.p` files containing a list with the second moment (G) matrix for each model. The G matrix can be 
translated to squared Euclidean distances using the PCM function `pcm.G_to_dist` (see Eq. 5) 

### PCM model fitting

PCM models are fitted to the beta coefficients from 1st level GLM. The beta coefficients and the residuals of each 
participants need first to be saved as CIFTI files:

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
in each ROI

These steps produce: 1) a `G_obs.<epoch>.glm12.<Hem>.<roi>.npy` file for each ROI with the observed G matrix in 
each participant (a copy of these files is stored in `data/encoding/`); 2) a `theta_in.<epoch>.glm12.<Hem>.<roi>.p` with 
the log-weight of each model in each participant (see Eq. 12). The standardised weights of each model are stored in 
`data/encoding/component_model.BOLD.tsv`. The procedure is similar for the LFPs and spiking activity:

- `python pcm_lfp.py continuous --<roi>`: fit PCM models for preparation to the LFPs aligned with cue presentation and 
perturbation onset
- `python pcm_spk.py continuous --<roi>`: fit PCM models for preparation to the spiking activity aligned with cue 
presentation and perturbation onset

The resulting standardised weight for each roi are stored in `data/encoding/weight.lfp.<roi>.plan.npy` and 
`data/encoding/weight.lfp.<roi>.plan.npy`. The frequencies of interest used for LFP extraction can be found in 
`data/LFPs/cfg.mat`. 

### PCM correlations

- `python pcm_cortical.py correlation_plan-exec`: preparation-execution correlation in BOLD activity (see Fig. 4)
- `python pcm_cortical.py correlation_cue-finger`: sensory input-expectation correlation in BOLD activity (see Fig. 7g)

These steps produce `theta_in.corr_<plan-exec/cue-finger>.glm12.<Hem>.<roi>.p`,  
`theta_gr.corr_<plan-exec/cue-finger>.glm12.<Hem>.<roi>.p` and `r_bootstrap.corr_<plan-exec/cue-finger>.<Hem>.<roi>.npy`
files that contain the necessary information to calculate the individual and group correlation estimates (see black dots 
and dashed red line, respectively, in Fig. 4 and 7g) and well the fSNR and the confidence interval using the 
`extract_mle_corr` and `bootstrap_summary` functions in `imaging_pipelines`. All resulting correlation estimates and 
confidence intervals are stored in `data/correlations/correlations.BOLD.tsv`. A similar approach was used to calculate 
the correlation between sensory input and expectation in EMG, LFPs and spiking activity (see Fig. 7f,h,i):

- `python pcm_emg.py correlation_cue-finger`: sensory input-expectation correlation in EMG activity (see Fig. 7f)
- `python pcm_lfp.py correlation_cue-dir`: sensory input-expectation correlation in LFPs activity (see Fig. 7h)
- `python pcm_spk.py correlation_cue-dir`: sensory input-expectation correlation in spiking activity (see Fig. 7i)

These steps work similarly as for neuroimaging data. All resulting correlation estimates and confidence intervals are 
stored in `data/correlations/correlations.EMG.tsv`, `data/correlations/correlations.lfp.tsv`, and
`data/correlations/correlations.spk.tsv`.



