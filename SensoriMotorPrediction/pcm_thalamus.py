import PcmPy as pcm
import os
import SensoriMotorPrediction.globals as gl
import numpy as np
import nitools as nt
import nibabel as nb
import pickle
import pandas as pd
from SensoriMotorPrediction.util import load_model
import imaging_pipelines.model as md
from imaging_pipelines.util import bootstrap_correlation
from SensoriMotorPrediction.pcm_models import find_model
from joblib import Parallel, delayed

def component_model(sns, glm, epoch, n_jobs=8, experiment='smp2', atlas='Thalamus',):

    # load model
    M, comp_names, regr_interest = load_model(epoch)
    
    # make cifti lists
    glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}')
    cifti_img = [os.path.join(glm_path, f'subj{sn}', f'beta.thalamus.dscalar.nii') for sn in sns]
    res_img = [os.path.join(glm_path, f'subj{sn}', 'residual.thalamus.dtseries.nii') for sn in sns]
    pcm_path = os.path.join(gl.baseDir, 'smp2', gl.pcmDir)
    os.makedirs(pcm_path, exist_ok=True)

    # make roi dict
    roi_path = os.path.join(gl.baseDir, 'smp2', gl.roiDir)
    rois = gl.rois[atlas]

    print(f'doing pcm in thalamus rois for {epoch}...')

    for H in gl.Hem:
        roi_dict = {roi: [os.path.join(roi_path, f'subj{sn}', f'{atlas}.{H}.{roi}.nii') for sn in sns] for roi in rois}

        # run PCM across rois
        PCM = md.PcmRois(cifti_imgs=cifti_img, 
                        res_imgs=res_img, 
                        M=M,
                        structnames=['ThalamusLeft', 'ThalamusRight'],
                        roi_names=rois,
                        roi_dict=roi_dict, 
                        regressor_mapping=gl.regressor_mapping, 
                        regr_interest=regr_interest, 
                        n_jobs=n_jobs)

        # Component model
        res_comp_model = PCM.run_parallel_pcm_across_rois()

        # Model family
        _, mcidx = find_model(M, 'component')
        do_model_family = mcidx > 0
        if do_model_family:
            res_model_family = PCM.fit_model_family_across_rois('component', comp_names=comp_names)

        for roi in rois:
            r = res_comp_model['roi'].index(roi)

            if do_model_family:
                res_model_family['T'][r].to_pickle(os.path.join(pcm_path, f'T.model_family.Thalamus.glm{glm}.{H}.{roi}.p'))
                f = open(os.path.join(pcm_path, f'theta.model_family.Thalamus.glm{glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res_model_family['theta'][r], f)

            res_comp_model['T_in'][r].to_pickle(os.path.join(pcm_path, f'T_in.Thalamus.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_cv'][r].to_pickle(os.path.join(pcm_path, f'T_cv.Thalamus.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_gr'][r].to_pickle(os.path.join(pcm_path, f'T_gr.Thalamus.glm{glm}.{H}.{roi}.p'))

            np.save(os.path.join(pcm_path, f'G_obs.{epoch}.Thalamus.glm{glm}.{H}.{roi}.npy'), res_comp_model['G_obs'][r])

            f = open(os.path.join(pcm_path, f'theta_in.Thalamus.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_in'][r], f)

            f = open(os.path.join(pcm_path, f'theta_cv.Thalamus.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_cv'][r], f)

            f = open(os.path.join(pcm_path, f'theta_gr.Thalamus.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_gr'][r], f)