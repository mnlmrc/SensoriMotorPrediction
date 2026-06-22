import PcmPy as pcm
import os
import SensoriMotorPrediction.globals as gl
import numpy as np
import nitools as nt
import nibabel as nb
import pickle
import pandas as pd
import imaging_pipelines.model as md
from imaging_pipelines.util import bootstrap_correlation
from SensoriMotorPrediction.pcm_models import find_model
from joblib import Parallel, delayed
from AnatSearchlight import searchlight as sl


def searchlight(sns, glm, epoch='plan', experiment='smp2'):

    def _fit_indiv(data, M=None, cond_vec=None, part_vec=None):

        # remove nans
        data = data[:, ~np.isnan(data).all(axis=0)]

        # retrieve model params
        Mc, idxc = find_model(M, 'component')
        n_param_c = Mc.n_param

        # make individual dataset
        obs_des = {'cond_vec': cond_vec, 'part_vec': part_vec}
        Y = pcm.dataset.Dataset(data, obs_descriptors=obs_des)

        # fit model to individual dataset 
        try:      
            _, theta_in = pcm.fit_model_individ(Y, Mc, fit_scale=False, fixed_effect='block', verbose=False)
            weight = np.exp(theta_in).reshape(-1)
        except:
            weight = np.full(n_param_c + 1, np.nan, dtype=float)
        return weight

    def _mean(data):

        return np.nanmean(data)

    # load model
    M, comp_names, regr_interest = load_model(epoch)

    nargout = len(comp_names) + 1

    # loop through Hem and subjects
    for H in gl.Hem:
        weight = np.zeros((len(sns), nargout, 32492))
        for s, sn in enumerate(sns):
            print(f'starting participant {sn}...')
            glm_path = os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}')
            roi_path = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}')

            # load searchlight
            print('loading searchlight...')
            SL = sl.load(os.path.join(roi_path, f'searchlight.{H}.h5'))

            # load betas residuals
            print('loading and prewhitening betas...')
            beta_cifti = nb.load(os.path.join(glm_path, 'beta.dscalar.nii'))
            beta_vol = nt.volume_from_cifti(beta_cifti)
            res_vol = nb.load(os.path.join(glm_path, 'ResMS.nii'))
            beta_pw = beta_vol.get_fdata() / np.sqrt(res_vol.get_fdata()[:, :, :, None])

            # load reginfo and select regressors for session
            reginfo = pd.read_csv(os.path.join(glm_path, f'subj{sn}_reginfo.tsv'), sep='\t')
            cond_vec = reginfo.name.map(gl.regressor_mapping).to_numpy()
            part_vec = reginfo.run.to_numpy()
            idx = np.isin(cond_vec, regr_interest)
            func_args = {'cond_vec': cond_vec[idx], 'part_vec': part_vec[idx], 'M': M}

            # run searchlight in parallel
            beta_pw_vol = nb.Nifti2Image(beta_pw[:, :, :, idx], affine=beta_vol.affine, header=beta_vol.header)
            weight[s] = SL.run_parallel(beta_pw_vol, _fit_indiv, func_args, nargout=nargout).T

        # distance trained to gifti
        gifti = nt.make_func_gifti(np.nanmean(weight, axis=0).T, anatomical_struct=SL.structure, column_names=comp_names + ['noise'])

        # define path to surface
        nb.save(gifti, os.path.join(gl.baseDir, experiment, gl.surfDir, f'searchlight.component_model.glm{glm}.{epoch}.{H}.func.gii'))


def component_model(sns, glm, epoch, method=None, n_jobs=6, experiment='smp2', atlas=None, struct=gl.struct):

    # load model
    M, comp_names, regr_interest = load_model(epoch)
    
    # make cifti lists
    glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}')
    cifti_img = [os.path.join(glm_path, f'subj{sn}', f'beta{(f".{method}" if method is not None else "")}.dscalar.nii') for sn in sns]
    res_img = [os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii') for sn in sns]
    pcm_path = os.path.join(gl.baseDir, 'smp2', gl.pcmDir)
    os.makedirs(pcm_path, exist_ok=True)

    # make roi dict
    roi_path = os.path.join(gl.baseDir, 'smp2', gl.roiDir)
    rois = gl.rois[atlas]

    print(f'doing pcm for {epoch}, label {method}')

    for H in gl.Hem:
        roi_dict = {roi: [os.path.join(roi_path, f'subj{sn}', f'{atlas}.{H}.{roi}.nii') for sn in sns] for roi in rois}

        # run PCM across rois
        PCM = md.PcmRois(cifti_imgs=cifti_img, 
                        res_imgs=res_img, 
                        M=M,
                        structnames=struct,
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
                res_model_family['T'][r].to_pickle(os.path.join(pcm_path, f'T.model_family.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'))
                f = open(os.path.join(pcm_path, f'theta.model_family.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res_model_family['theta'][r], f)

            res_comp_model['T_in'][r].to_pickle(os.path.join(pcm_path, f'T_in.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_cv'][r].to_pickle(os.path.join(pcm_path, f'T_cv.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'))
            res_comp_model['T_gr'][r].to_pickle(os.path.join(pcm_path, f'T_gr.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'))

            np.save(os.path.join(pcm_path, f'G_obs.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.npy'), res_comp_model['G_obs'][r])

            f = open(os.path.join(pcm_path, f'theta_in.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_in'][r], f)

            f = open(os.path.join(pcm_path, f'theta_cv.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res_comp_model['theta_cv'][r], f)

            f = open(os.path.join(pcm_path, f'theta_gr.{epoch}{(f".{method}" if method is not None else "")}.glm{glm}.{H}.{roi}.p'), 'wb')
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
    reginfo['BN'] = reginfo['run']
    tmp = reginfo['name'].str.split(',', expand=True)
    reginfo['cue'] = tmp[0]
    
    # select prep regressors only
    B = B[reginfo.name.isin(gl.cues)]
    reginfo = reginfo[reginfo.name.isin(gl.cues)]

    # load force
    dat = pd.read_csv(os.path.join(gl.baseDir, 'smp2', gl.behavDir, 'behaviour.block.cue.tsv'), sep='\t')
    dat_s = dat[(dat.sn==sn) & (dat.BN.astype(str).isin(FuncRuns))].reset_index()
    force_df = reginfo.merge(dat_s[['BN', 'cue', 'index0', 'ring0', 'diff0']], on=['BN','cue'], how='left')
    cond_vec = force_df['cue'].map(gl.regressor_mapping)
    part_vec = reginfo.run.to_numpy()
    Z = pcm.indicator(cond_vec)
    #F = force_df['diff0'].to_numpy()[:, None]

    # calc residuals
    print(f'Calculating residuals with {method} regression')
    if method=='ols':
        F = force_df[['diff0']].to_numpy()
        F = F - F.mean(axis=0, keepdims=True)
        X = np.c_[np.ones(F.shape[0]), F]
        W = np.linalg.pinv(X) @ B
        B_hat = X @ W
        B_res = B - B_hat
    if method=='ancova':
        F = force_df[['diff0']].to_numpy()
        F = F - F.mean(axis=0, keepdims=True)
        X = np.c_[F, Z] 
        W = np.linalg.pinv(X) @ B 
        B_hat = np.outer(F[:, 0], W[0]) #F[:, 0] @ W[0].T
        #B_hat = F @ W[:2]
        B_res = B - B_hat
    if method=='ancova_cv':
        F = force_df[['diff0']].to_numpy()
        F = F - F.mean(axis=0, keepdims=True)
        X = np.c_[F, Z]
        B_res, W = _regress_out_ancova_cv(B, F, Z, part_vec)

    # save residuals
    row_axis = nb.cifti2.ScalarAxis(reginfo.name + '.' + reginfo['run'].astype(str))
    brain_axis = cifti_img.header.get_axis(1)
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(dataobj=B_res, header=header)
    nb.save(cifti, glm_path + '/' + f'beta.{method}.dscalar.nii')

    # save coefficients
    row_axis = nb.cifti2.ScalarAxis(np.arange(X.shape[1]))
    brain_axis = cifti_img.header.get_axis(1)
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(dataobj=W, header=header)
    nb.save(cifti, glm_path + '/' + f'W.{method}.dscalar.nii')


def _regress_out_ancova_cv(B, F, Z, part_vec):
    """
    Leave-one-partition-out ANCOVA regression.

    For each partition p, estimates the effect of F on B using all
    *other* partitions (jointly with Z), then applies the correction
    to partition p. This avoids leaking partition p's data into its
    own correction, keeping the partitions independent for crossnobis.

    Parameters
    ----------
    B        : (n_obs, n_vox)   data matrix
    F        : (n_obs, n_f)     nuisance regressors to remove
    Z        : (n_obs, n_cond)  condition indicators (kept in model, not removed)
    part_vec : (n_obs,)         integer partition labels

    Returns
    -------
    B_cv  : (n_obs, n_vox)  corrected data
    W_avg : (n_f, n_vox)    regression weights for F averaged across partitions
    """
    if F.ndim==1:
        F = F[:, None]
    n_f      = F.shape[1]
    B_cv     = B.copy()
    parts    = np.unique(part_vec)
    W_folds  = []

    for p in parts:
        train = part_vec != p   # all partitions except p
        test  = part_vec == p   # partition p

        # Estimate W_F on training data only (joint model accounts for Z)
        X_train = np.c_[F[train], Z[train]]
        W_train = np.linalg.pinv(X_train) @ B[train]

        # Apply only F's contribution to the test partition
        B_cv[test] = B[test] - F[test] @ W_train[:n_f]

        W_folds.append(W_train)

    W_avg = np.mean(W_folds, axis=0) 

    return B_cv, W_avg


def _correlation_masks(B, n_part, corr):

    if corr=='plan-exec':
        i_x = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] # plan index
        r_x = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # plan ring
        i_y = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0] # exec index
        r_y = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1] # exec ring
    elif corr=='cue-finger':
        i_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        r_x = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        i_y = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        r_y = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    i_x_sum = np.array(i_x).sum()
    r_x_sum = np.array(r_x).sum()
    i_y_sum = np.array(i_y).sum()
    r_y_sum = np.array(r_y).sum()

    mask_x = {'i': np.array(i_x * n_part, dtype=bool),
              'r': np.array(r_x * n_part, dtype=bool)}
    mask_y = {'i': np.array(i_y * n_part, dtype=bool),
              'r': np.array(r_y * n_part, dtype=bool)}

    xi = B[mask_x['i']].reshape(n_part, i_x_sum, -1).mean(axis=1)
    xr = B[mask_x['r']].reshape(n_part, r_x_sum, -1).mean(axis=1)
    yi = B[mask_y['i']].reshape(n_part, i_y_sum, -1).mean(axis=1)
    yr = B[mask_y['r']].reshape(n_part, r_y_sum, -1).mean(axis=1)

    x = xi - xr
    y = yi - yr

    return x, y


def correlation(sns, glm, rois, corr='plan-exec', experiment='smp2'):

    # define paths
    glm_path = os.path.join(gl.baseDir, experiment, f'glm{glm}')
    roi_path = os.path.join(gl.baseDir, experiment, gl.roiDir)
    pcm_path = os.path.join(gl.baseDir, experiment, gl.pcmDir)

    # load correlation model
    f = open(os.path.join(pcm_path, f'M.corr.p'), "rb")
    Mflex = pickle.load(f)

    for H in gl.Hem:
        for roi in ['M1', 'S1']:
            N = len(sns)
            Y = list()

            # loop over participants
            for s, sn in enumerate(sns):
                print(f'doing ROI.{H}.{roi}, participant {sn}...')

                # load betas, residuals and roi masks
                betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii'))

                residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii'))
                
                # prewhiten betas
                betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)

                # extract cond_vec and part_vec
                reginfo = np.char.split(betas.header.get_axis(0).name, sep='.')
                part_vec = np.array([int(r[1]) for r in reginfo])
                n_part = len(np.unique(part_vec))
                obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                           'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}

                # mask correlation terms
                x, y = _correlation_masks(betas_prewhitened, n_part, corr)
                x -= x.mean(axis=-1, keepdims=True)
                y -= y.mean(axis=-1, keepdims=True)

                data = np.r_[x, y]                
                Y.append(pcm.dataset.Dataset(data, obs_descriptors=obs_des))
            
            # estimate MLE correlations
            T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=False)
            T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)

            # save results
            T_in.to_pickle(os.path.join(pcm_path, f'T_in.corr_{corr}.glm{glm}.{H}.{roi}.p'))
            T_gr.to_pickle(os.path.join(pcm_path, f'T_gr.corr_{corr}.glm{glm}.{H}.{roi}.p'))
            f = open(os.path.join(pcm_path, f'theta_in.corr_{corr}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(theta_in, f)
            f = open(os.path.join(pcm_path, f'theta_gr.corr_{corr}.glm{glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(theta_gr, f)

            # do bootstrap
            B = 1000
            S = len(Y)
            rng = np.random.default_rng(0)
            indeces = rng.integers(0, S, size=(B, S))
            results = Parallel(n_jobs=16, backend='loky')(
                delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces)
            r_bootstrap = np.array([r for r in results if r is not None])
            n_disc = len(results) - len(r_bootstrap)
            print(f'ROI.{H}.{roi}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')
            np.save(os.path.join(pcm_path, f'r_bootstrap.corr_{corr}.glm{glm}.{H}.{roi}.npy'), r_bootstrap)






