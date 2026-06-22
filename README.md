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
reproducing each figure from the mininal dataset in the `data` folder. Additional data will be provided upon requested to 
the authors. 




