import PcmPy as pcm
import os
import SensoriMotorPrediction.globals as gl
import numpy as np
import nibabel as nb
import pickle
import pandas as pd
import imaging_pipelines.model as md
from SensoriMotorPrediction.pcm_models import find_model 


def pcm_rois(sns, glm, epoch, label=None, n_jobs=6, experiment='smp2'):

    if epoch=='plan': #, 'regr_out_preact_ols', 'regr_out_preact_cv', 'regr_out_preact_ancova']:
        regr_interest = [0, 1, 2, 3, 4]
        comp_names = ['expectation', 'uncertainty']
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan.p'), "rb")
    # elif epoch=='warp':
    #     regr_interest = [0, 1, 2, 3, 4]
    #     f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.warp.p'), "rb")
    elif epoch=='exec':
        regr_interest = [5, 6, 7, 8, 9, 10, 11, 12,]
        comp_names = ['sensory input', 'expectation', 'surprise']
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.exec.p'), "rb")
    else:
        pass
    
    M = pickle.load(f)
    
    # make cifti lists
    glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}')
    cifti_img = [os.path.join(glm_path, f'subj{sn}', f'beta{(f".{label}" if label is not None else "")}.dscalar.nii') for sn in sns]
    res_img = [os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii') for sn in sns]

    # make roi dict
    roi_path = os.path.join(gl.baseDir, 'smp2', gl.roiDir)
    atlas = 'ROI'
    rois=gl.rois[atlas]

    print(f'doing pcm for {epoch}, label {label}')

    for H in gl.Hem:
        roi_dict = {roi: [os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii') for sn in sns] for roi in rois}

        # run PCM across rois
        PCM = md.PcmRois(cifti_imgs=cifti_img, 
                        res_imgs=res_img, 
                        M=M,
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

            path = os.path.join(gl.baseDir, 'smp2', gl.pcmDir)
            os.makedirs(path, exist_ok=True)

            if do_model_family:
                res_model_family['T'][r].to_pickle(os.path.join(path, f'T.model_family.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res_model_family['theta'][r], f)

            res_comp_model['T_in'][r].to_pickle(os.path.join(path, f'T_in.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'))

            np.save(os.path.join(path, f'G_obs.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.npy'), res_comp_model['G_obs'][r])

            f = open(os.path.join(path, f'theta_in.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_in'][r], f)
            f = open(os.path.join(path, f'theta_cv.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_cv'][r], f)
            f = open(os.path.join(path, f'theta_gr.{epoch}{(f".{label}" if label is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_gr'][r], f)
            


def regress_out_preactivation(sn, glm, method='ancova'):

    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'smp2', 'participants.tsv'), sep='\t')
    FuncRuns = pinfo[pinfo.sn==sn].reset_index()['FuncRuns'][0].split('.')

    glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'subj{sn}')
    
    print(f'participant {sn}, loading betas and force...')
    cifti_img = nb.load(os.path.join(glm_path, 'beta.dscalar.nii'))
    B = cifti_img.get_fdata()
    #B = B[:, ~np.isnan(B).all(axis=0)]

    # extract reginfo to match with force df
    reginfo = pd.read_csv(os.path.join(glm_path, f'subj{sn}_reginfo.tsv'), sep='\t')
    reginfo.name = reginfo.name.str.replace(' ', '')
    #reginfo = pd.DataFrame({'reginfo': reginfo})
    #tmp = reginfo['reginfo'].str.split('.', expand=True)
    #reginfo['cond'] = tmp[0]
    reginfo['BN'] = reginfo['run']
    tmp = reginfo['name'].str.split(',', expand=True)
    reginfo['cue'] = tmp[0]
    
    # select prep regressors only
    B = B[reginfo.name.isin(gl.cues)]
    reginfo = reginfo[reginfo.name.isin(gl.cues)]

    # load force
    dat = pd.read_csv(os.path.join(gl.baseDir, 'smp2', gl.behavDir, 'behaviour.block.cue.tsv'), sep='\t')
    dat_s = dat[(dat.sn==sn) & (dat.BN.astype(str).isin(FuncRuns))].reset_index()
    force_df = reginfo.merge(dat_s[['BN', 'cue', 'index0', 'ring0', 'diff']], on=['BN','cue'], how='left')
    cond_vec = force_df['cue'].map(gl.regressor_mapping)
    Z = pcm.indicator(cond_vec)
    F = force_df['diff'].to_numpy()[:, None]

    # calc residuals
    print(f'Calculating residuals with {method} regression')
    if method=='ols':
        F = np.column_stack([np.ones(F.shape[0]), F])
        W, _, _, _ = np.linalg.lstsq(F, B, rcond=None)
        B_hat = F @ W
        B_res = B - B_hat
        label = 'regr_out_preact_ols'
    if method=='cv':
        F = np.column_stack([np.ones(F.shape[0]), F])
        part_vec = reginfo.run.to_numpy()
        B_res = _regress_out_preactivation_cv(B, F, part_vec)
        label = 'regr_out_preact_cv'
    if method=='ancova':
        F = np.c_[F, Z]
        W = np.linalg.pinv(F) @ B #np.linalg.lstsq(F, B, rcond=None) # use ols formula
        B_hat = np.outer(F[:, 0], W[0]) #F[:, 0] @ W[0].T
        B_res = B - B_hat
        label = 'regr_out_preact_ancova'

    # save residuals
    row_axis = nb.cifti2.ScalarAxis(reginfo.name + '.' + reginfo['run'].astype(str))
    brain_axis = cifti_img.header.get_axis(1)
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(dataobj=B_res, header=header)
    nb.save(cifti, glm_path + '/' + f'beta.{label}.dscalar.nii')

    # save coefficients
    row_axis = nb.cifti2.ScalarAxis(np.arange(F.shape[1]))
    brain_axis = cifti_img.header.get_axis(1)
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(dataobj=W, header=header)
    nb.save(cifti, glm_path + '/' + f'W.{label}.dscalar.nii')


def _regress_out_preactivation_cv(B, F, part_vec):
    """
    Cross-validated regression of force out of betas using leave-one-block-out OLS.

    Parameters
    ----------
    B : array, shape (n_samples, n_voxels)
        Cortical betas.
    F : array, shape (n_samples, n_features)
        Force features.
    part_vec : array, shape (n_samples,)
        Block labels.

    Returns
    -------
    B_resid : array, shape (n_samples, n_voxels)
        Residual betas after removing the cross-validated force prediction.
    B_pred : array, shape (n_samples, n_voxels)
        Cross-validated force-predicted component.
    """

    B_hat = np.full_like(B, np.nan)

    for part in np.unique(part_vec):
        train_idx = part_vec != part
        test_idx = part_vec == part

        F_train = F[train_idx]
        B_train = B[train_idx]
        F_test = F[test_idx]

        # OLS fit on training blocks
        W, _, _, _ = np.linalg.lstsq(F_train, B_train, rcond=None)

        # Predict held-out block
        B_hat[test_idx] = F_test @ W

    return B - B_hat
