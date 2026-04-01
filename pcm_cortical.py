import warnings
import time
from globals import regressor_mapping
import argparse
import pickle
from itertools import combinations
import rsatoolbox as rsa
import PcmPy as pcm
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
from pcm_models import find_model
import globals as gl
import pandas as pd
import numpy as np
import nibabel as nb
import os
import nitools as nt
import sys
import Functional_Fusion.atlas_map as am
from imaging_pipelines import model as md
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from imaging_pipelines.util import bootstrap_correlation, extract_mle_corr

warnings.filterwarnings("ignore")

def pcm_rois(M, epoch, args):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    cifti_img = 'beta.dscalar.nii'
    roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    subj_ids = [f'subj{sn}' for sn in args.sns]
    PCM = md.PcmRois(subj_ids, M, glm_path, cifti_img,
             roi_path=roi_path,
             roi_imgs=roi_imgs,
             regressor_mapping=gl.regressor_mapping,
             regr_interest=[0, 1, 2, 3, 4] if epoch == 'plan' else [5, 6, 7, 8, 9, 10, 11, 12,],
             res_img='residual.dtseries.nii',
             n_jobs=args.n_jobs)
    res = PCM.run_parallel_pcm_across_rois()

    for H in Hem:
        for roi in rois:
            r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

            path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
            os.makedirs(path, exist_ok=True)

            res['T_in'][r].to_pickle(os.path.join(path, f'T_in.{epoch}.glm{args.glm}.{H}.{roi}.p'))
            res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.{epoch}.glm{args.glm}.{H}.{roi}.p'))
            res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.{epoch}.glm{args.glm}.{H}.{roi}.p'))

            np.save(os.path.join(path, f'G_obs.{epoch}.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])

            f = open(os.path.join(path, f'theta_in.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_in'][r], f)
            f = open(os.path.join(path, f'theta_cv.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_cv'][r], f)
            f = open(os.path.join(path, f'theta_gr.{epoch}.glm{args.glm}.{H}.{roi}.p'), 'wb')
            pickle.dump(res['theta_gr'][r], f)

def pcm_searchlight(M, epoch, args):
    Hem = ['L', 'R']
    structnames = ['CortexLeft', 'CortexRight']
    subj_ids = [f'subj{sn}' for sn in args.sns]
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    cifti_img = 'beta.dscalar.nii'
    res_img = 'ResMS.nii' #'residual.dtseries.nii'
    searchlight_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    surf_path = os.path.join(gl.baseDir, args.experiment, gl.wbDir)
    for h, H in enumerate(Hem):
        SL = md.PcmSearchlight(
            M=M,
            sns=subj_ids,
            glm_path=glm_path,
            cifti_img=cifti_img,
            res_img=res_img,
            searchlight_path=searchlight_path,
            structnames=structnames[h],
            regressor_mapping=gl.regressor_mapping,
            regr_interest=[0, 1, 2, 3, 4] if epoch == 'plan' else [5, 6, 7, 8, 9, 10, 11, 12, ],
            n_jobs=args.n_jobs
        )
        Mc, idx_c = find_model(M, 'component')
        n_centre = SL.n_centre
        if epoch=='exec':
            Mf, idx_f = find_model(M, 'feature')
            n_param_f = Mf.n_param
            param_f = np.full((n_centre, n_param_f), np.nan)
        n_param_c = Mc.n_param
        param_c = np.full((n_centre, n_param_c), np.nan)
        # n_centre = 2
        # SL.n_centre = n_centre
        var_tot = np.full((n_centre, len(SL.sns)), np.nan)
        distance = np.full((n_centre, len(SL.sns)), np.nan)
        G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good = SL.run_seachlight_parallel()
        for c in range(SL.n_centre):
            if good[c]:
                G = G_obs[c]
                var_tot[c] = np.trace(G, axis1=1, axis2=2)
                distance[c] = np.array([pcm.G_to_dist(G[s]).mean() for s in range(len(SL.sns))])
                param_c[c] = theta_gr[c][idx_c][:n_param_c]
                if epoch == 'exec':
                    param_f[c] = theta_gr[c][idx_f][:n_param_f]

        var_expl = np.exp(param_c)

        # trace to gifti
        data = var_tot
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=args.sns)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.var_tot.{epoch}.{H}.func.gii'))

        # distance to gifti
        data = distance
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=args.sns)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.distance.{epoch}.{H}.func.gii'))

        # var_expl to gifti
        data = var_expl
        column_names = ['cue', 'uncertainty'] if epoch=='plan' else ['finger', 'cue', 'surprise']
        gifti = nt.make_func_gifti(data, anatomical_struct=structnames[h], column_names=column_names)
        nb.save(gifti, os.path.join(surf_path, f'searchlight.var_expl.{epoch}.{H}.func.gii'))

        # correlation
        if epoch=='exec':
            theta2 = param_f ** 2
            covariance = param_f[:, 1] * param_f[:, 2]
            stds = np.sqrt((theta2[:, 0] + theta2[:, 1]) * theta2[:, 2])
            correlation = covariance / stds
            column_names = ['correlation']
            gifti = nt.make_func_gifti(correlation, anatomical_struct=structnames[h], column_names=column_names)
            nb.save(gifti, os.path.join(surf_path, f'searchlight.correlation.{epoch}.{H}.func.gii'))

def main(args):
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    pcm_path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
    behav_path = os.path.join(gl.baseDir, args.experiment, gl.behavDir)
    Hem = ['L', 'R']
    subj_ids = [f'subj{sn}' for sn in args.sns]
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp']
    roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
    cifti_img = 'beta.dscalar.nii'
    res_img = 'ResMS.nii'
    if args.what == 'tessel_execution':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        pcm_tessel(M, 'exec', args)
    if args.what == 'tessel_planning':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        pcm_tessel(M, 'plan', args)
    if args.what == 'searchlight_planning':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        pcm_searchlight(M[:-1], 'plan', args) # last model is always ceil -  not need here
    if args.what == 'searchlight_execution':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        pcm_searchlight(M[:-1], 'exec', args) # last model is always ceil -  not need here
    if args.what == 'rois_planning':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        pcm_rois(M, 'plan', args)
    if args.what == 'rois_execution':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        pcm_rois(M, 'exec', args)
    if args.what == 'model_family_rois_planning':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        PCM = md.PcmRois(subj_ids, M, glm_path, cifti_img, roi_path=roi_path, roi_imgs=roi_imgs, n_jobs=args.n_jobs,
                 regressor_mapping=gl.regressor_mapping, regr_interest=[0, 1, 2, 3, 4], res_img='residual.dtseries.nii')
        res = PCM.fit_model_family_across_rois('component', comp_names=['cue', 'uncertainty'])
        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')
                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)
                res['T'][r].to_pickle(os.path.join(path, f'T.model_family.plan.glm{args.glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta'][r], f)
    if args.what == 'model_family_rois_execution':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        PCM = md.PcmRois(subj_ids, M, glm_path, cifti_img, roi_path=roi_path, roi_imgs=roi_imgs, n_jobs=args.n_jobs,
                 regressor_mapping=gl.regressor_mapping, regr_interest=[5, 6, 7, 8, 9, 10, 11, 12], res_img='residual.dtseries.nii')
        res = PCM.fit_model_family_across_rois('component', comp_names=['finger', 'cue', 'surprise'],
                                             basecomp=np.eye(8)[None, :, :])
        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')
                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)
                res['T'][r].to_pickle(os.path.join(path, f'T.model_family.exec.glm{args.glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta'][r], f)
    if args.what == 'G_obs_plan-exec':
        for H in Hem:
            for roi in rois:
                print(f'doing {H},{roi}')
                G_obs = np.zeros((len(subj_ids), 13, 13))
                for s, sn in enumerate(subj_ids):
                    betas = nb.load(os.path.join(glm_path, sn, 'beta.dscalar.nii'))
                    residuals = nb.load(os.path.join(glm_path, sn, 'residual.dtseries.nii'))
                    mask = nb.load(os.path.join(roi_path, sn, f'ROI.{H}.{roi}.nii'))
                    betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)
                    reginfo = np.char.split(betas.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])
                    obs_des = {'cond_vec': cond_vec,
                               'part_vec': part_vec}
                    Y = pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des)
                    G_obs[s], _ = pcm.est_G_crossval(Y.measurements, Y.obs_descriptors['cond_vec'],
                                                     Y.obs_descriptors['part_vec'],
                                                     X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']))
                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)
                np.save(os.path.join(path, f'G_obs.plan-exec.glm{args.glm}.{H}.{roi}.npy'), G_obs)
    if args.what == 'correlation_plan-exec':
        rng = np.random.default_rng(0) # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for H in Hem:
            for roi in rois:
                N = len(args.sns)
                Y = list()
                r = roi_imgs.index(f'ROI.{H}.{roi}.nii')
                print(f'doing...ROI.{H}.{roi}')
                G_obs = np.zeros((N, 2, 2))
                for s, sn in enumerate(args.sns):
                    betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                    residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii'))
                    mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                    betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)
                    reginfo = np.char.split(betas.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])
                    n_part = len(np.unique(part_vec))
                    mask_plan = {'i': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * n_part, dtype=bool),
                                 'r': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * n_part, dtype=bool)}
                    mask_exec = {'i': np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool),
                                 'r': np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1] * n_part, dtype=bool)}
                    plani = betas_prewhitened[mask_plan['i']].reshape(n_part, 2, -1).mean(axis=1)
                    planr = betas_prewhitened[mask_plan['r']].reshape(n_part, 2, -1).mean(axis=1)
                    execi = betas_prewhitened[mask_exec['i']].reshape(n_part, 4, -1).mean(axis=1)
                    execr = betas_prewhitened[mask_exec['r']].reshape(n_part, 4, -1).mean(axis=1)
                    plan = plani - planr
                    exec = execi - execr
                    beta_corr = np.r_[plan - plan.mean(axis=-1, keepdims=True), exec - exec.mean(axis=-1, keepdims=True)]
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                    Y.append(pcm.dataset.Dataset(beta_corr, obs_descriptors=obs_des))
                    beta_corr_mean = np.c_[beta_corr[0::2].mean(axis=0), beta_corr[1::2].mean(axis=0)]
                    G_obs[s] = (beta_corr_mean.T @ beta_corr_mean) / beta_corr_mean.shape[0] #pcm.est_G(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],Y[s].obs_descriptors['part_vec'])

                np.save(os.path.join(pcm_path, f'G_obs.corr_plan-exec.glm{args.glm}.{H}.{roi}.npy'), G_obs)
                T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=False)
                T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
                T_in.to_pickle(os.path.join(pcm_path, f'T_in.corr_plan-exec.glm{args.glm}.{H}.{roi}.p'))
                T_gr.to_pickle(os.path.join(pcm_path, f'T_gr.corr_plan-exec.glm{args.glm}.{H}.{roi}.p'))

                f = open(os.path.join(pcm_path, f'theta_in.corr_plan-exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_in, f)
                f = open(os.path.join(pcm_path, f'theta_gr.corr_plan-exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_gr, f)

                # do bootstrap
                B = 1000
                S = len(Y)
                indeces = rng.integers(0, S, size=(B, S))
                results = Parallel(n_jobs=16, backend='loky')(
                    delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces)
                r_bootstrap = np.array([r for r in results if r is not None])
                n_disc = len(results) - len(r_bootstrap)
                print(f'ROI.{H}.{roi}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')
                np.save(os.path.join(pcm_path, f'r_bootstrap.corr_plan-exec.{H}.{roi}.npy'), r_bootstrap)
    if args.what == 'correlation_cue-finger':
        rng = np.random.default_rng(0)  # seed for reproducibility
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for H in Hem:
            for roi in rois:
                N = len(args.sns)
                Y = list()
                r = roi_imgs.index(f'ROI.{H}.{roi}.nii')
                print(f'doing...ROI.{H}.{roi}')
                G_obs = np.zeros((N, 2, 2))
                for s, sn in enumerate(args.sns):
                    betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                    residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii'))
                    mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                    betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)
                    reginfo = np.char.split(betas.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])
                    n_part = len(np.unique(part_vec))
                    mask_cue = {'i': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * n_part, dtype=bool),
                                 'r': np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] * n_part, dtype=bool)}
                    mask_finger = {'i': np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool),
                                 'r': np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1] * n_part, dtype=bool)}
                    plani = betas_prewhitened[mask_cue['i']].reshape(n_part, 2, -1).mean(axis=1)
                    planr = betas_prewhitened[mask_cue['r']].reshape(n_part, 2, -1).mean(axis=1)
                    execi = betas_prewhitened[mask_finger['i']].reshape(n_part, 4, -1).mean(axis=1)
                    execr = betas_prewhitened[mask_finger['r']].reshape(n_part, 4, -1).mean(axis=1)
                    plan = plani - planr
                    exec = execi - execr
                    beta_corr = np.r_[plan - plan.mean(axis=-1, keepdims=True), exec - exec.mean(axis=-1, keepdims=True)]
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                    Y.append(pcm.dataset.Dataset(beta_corr, obs_descriptors=obs_des))
                    beta_corr_mean = np.c_[beta_corr[0::2].mean(axis=0), beta_corr[1::2].mean(axis=0)]
                    G_obs[s] = (beta_corr_mean.T @ beta_corr_mean) / beta_corr_mean.shape[0]
                    #G_obs[s], _ = pcm.est_G(Y[s].measurements,
                     #                                Y[s].obs_descriptors['cond_vec'],
                      #                               Y[s].obs_descriptors['part_vec'])

                np.save(os.path.join(pcm_path, f'G_obs.corr_cue-finger.glm{args.glm}.{H}.{roi}.npy'), G_obs)
                T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=False)
                T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
                T_in.to_pickle(os.path.join(pcm_path, f'T_in.corr_cue-finger.glm{args.glm}.{H}.{roi}.p'))
                T_gr.to_pickle(os.path.join(pcm_path, f'T_gr.corr_cue-finger.glm{args.glm}.{H}.{roi}.p'))

                f = open(os.path.join(pcm_path, f'theta_in.corr_cue-finger.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_in, f)
                f = open(os.path.join(pcm_path, f'theta_gr.corr_cue-finger.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_gr, f)

                # do bootstrap
                B = 1000
                S = len(Y)
                indeces = rng.integers(0, S, size=(B, S))
                results = Parallel(n_jobs=16, backend='loky')(
                    delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces
                )
                r_bootstrap = np.array([r for r in results if r is not None])
                n_disc = len(results) - len(r_bootstrap)
                print(f'ROI.{H}.{roi}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')

                np.save(os.path.join(pcm_path, f'r_bootstrap.corr_cue-finger.{H}.{roi}.npy'), r_bootstrap)
    if args.what == 'correlation_cue-cue':
        rng = np.random.default_rng(0)  # seed for reproducibility
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for H in Hem:
            for roi in rois:
                N = len(args.sns)
                Y = list()
                r = roi_imgs.index(f'ROI.{H}.{roi}.nii')
                print(f'doing...ROI.{H}.{roi}')
                G_obs = np.zeros((N, 2, 2))
                for s, sn in enumerate(args.sns):
                    betas = nb.load(os.path.join(glm_path, f'subj{sn}', 'beta.dscalar.nii'))
                    residuals = nb.load(os.path.join(glm_path, f'subj{sn}', 'residual.dtseries.nii'))
                    mask = nb.load(os.path.join(roi_path, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                    betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)
                    reginfo = np.char.split(betas.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])
                    n_part = len(np.unique(part_vec))
                    mask_cue_plan = {'i': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * n_part, dtype=bool),
                                 'r': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * n_part, dtype=bool)}
                    mask_cue_exec = {'i': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * n_part, dtype=bool),
                                 'r': np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] * n_part, dtype=bool)}
                    plani = betas_prewhitened[mask_cue_plan['i']].reshape(n_part, 2, -1).mean(axis=1)
                    planr = betas_prewhitened[mask_cue_plan['r']].reshape(n_part, 2, -1).mean(axis=1)
                    execi = betas_prewhitened[mask_cue_exec['i']].reshape(n_part, 2, -1).mean(axis=1)
                    execr = betas_prewhitened[mask_cue_exec['r']].reshape(n_part, 2, -1).mean(axis=1)
                    plan = plani - planr
                    exec = execi - execr
                    beta_corr = np.r_[plan - plan.mean(axis=-1, keepdims=True), exec - exec.mean(axis=-1, keepdims=True)]
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                    Y.append(pcm.dataset.Dataset(beta_corr, obs_descriptors=obs_des))
                    G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                                     Y[s].obs_descriptors['cond_vec'],
                                                     Y[s].obs_descriptors['part_vec'])

                np.save(os.path.join(pcm_path, f'G_obs.corr_cue-cue.glm{args.glm}.{H}.{roi}.npy'), G_obs)
                T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=False)
                T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
                T_in.to_pickle(os.path.join(pcm_path, f'T_in.corr_cue-cue.glm{args.glm}.{H}.{roi}.p'))
                T_gr.to_pickle(os.path.join(pcm_path, f'T_gr.corr_cue-cue.glm{args.glm}.{H}.{roi}.p'))

                f = open(os.path.join(pcm_path, f'theta_in.corr_cue-cue.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_in, f)
                f = open(os.path.join(pcm_path, f'theta_gr.corr_cue-cue.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_gr, f)

                # do bootstrap
                B = 1000
                S = len(Y)
                indeces = rng.integers(0, S, size=(B, S))
                results = Parallel(n_jobs=16, backend='loky')(
                    delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces
                )
                r_bootstrap = np.array([r for r in results if r is not None])
                n_disc = len(results) - len(r_bootstrap)
                print(f'ROI.{H}.{roi}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')

                np.save(os.path.join(pcm_path, f'r_bootstrap.corr_cue-cue.{H}.{roi}.npy'), r_bootstrap)
    if args.what == 'G_obs_null-potent':
        scaler = StandardScaler(with_mean=True, with_std=False)
        for H in Hem:
            for roi in rois:
                G_nogo_null = np.zeros((len(subj_ids), 5, 5))
                G_nogo_pot = np.zeros_like(G_nogo_null)
                G_go_null = np.zeros((len(subj_ids), 8, 8))
                G_go_pot = np.zeros_like(G_go_null)
                G_null = np.zeros((len(subj_ids), 13, 13))
                G_pot = np.zeros_like(G_null)
                for s, sn in enumerate(args.sns):
                    subj_id = f'subj{sn}'
                    betas = nb.load(os.path.join(glm_path, subj_id, 'beta.dscalar.nii'))
                    residuals = nb.load(os.path.join(glm_path, subj_id, 'residual.dtseries.nii'))
                    force = pd.read_csv(os.path.join(behav_path, subj_id, f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
                    force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
                    force['cue'] = force['cue'].map(gl.cue_mapping)
                    force['cond_vec'] = (force['cue'] + ',' + force['stimFinger']).str.replace(',nogo', '')
                    force = force.groupby(['BN', 'cond_vec', 'GoNogo'], as_index=False).mean(numeric_only=True).reset_index()
                    cond_vec = force['cond_vec'].map(gl.regressor_mapping).to_numpy()
                    part_vec = force['BN'].to_numpy()
                    mask_go = force['GoNogo'] == 'go'
                    mask_nogo = force['GoNogo'] == 'nogo'
                    part_vec_go = force[mask_go].BN.to_numpy()
                    force_go = force.loc[mask_go, ['index1', 'ring1', ]].to_numpy()
                    print(f'doing participant {sn}, {H}, {roi}')
                    mask = nb.load(os.path.join(roi_path, subj_id, f'ROI.{H}.{roi}.nii'))
                    betas_prewhitened = md.calc_prewhitened_betas(betas, residuals, mask)
                    beta_go = betas_prewhitened[mask_go]
                    beta_go_s = scaler.fit_transform(beta_go)

                    reg = Ridge(alpha=beta_go.shape[0] * 10, fit_intercept=True)
                    reg.fit(beta_go_s, force_go)
                    force_go_hat = reg.predict(beta_go_s)
                    B = reg.coef_.T

                    U, S, Vt = np.linalg.svd(B, full_matrices=False)
                    r = (S > 1e-8).sum()
                    U_pot = U[:, :r]
                    P_pot = U_pot @ U_pot.T
                    P_null = np.eye(B.shape[0]) - P_pot

                    betas_s = scaler.transform(betas_prewhitened)
                    beta_pot = scaler.inverse_transform(betas_s @ P_pot)
                    beta_null = scaler.inverse_transform(betas_s @ P_null)

                    beta_go_pot = scaler.inverse_transform(beta_go_s @ P_pot)
                    beta_go_null = scaler.inverse_transform(beta_go_s @ P_null)

                    beta_nogo = betas_prewhitened[mask_nogo]
                    beta_nogo_s = scaler.transform(beta_nogo)
                    beta_nogo_pot = scaler.inverse_transform(beta_nogo_s @ P_pot)
                    beta_nogo_null = scaler.inverse_transform(beta_nogo_s @ P_null)

                    force_nogo = force[mask_nogo]

                    G_null[s], _ = pcm.est_G_crossval(beta_null,
                                                         cond_vec,
                                                         part_vec,
                                                         X=pcm.indicator(part_vec))
                    G_pot[s], _ = pcm.est_G_crossval(beta_pot,
                                                        cond_vec,
                                                        part_vec,
                                                        X=pcm.indicator(part_vec))
                    G_nogo_null[s], _ = pcm.est_G_crossval(beta_nogo_null,
                                                   cond_vec[mask_nogo],
                                                   part_vec[mask_nogo],
                                                   X=pcm.indicator(part_vec[mask_nogo]))
                    G_nogo_pot[s], _ = pcm.est_G_crossval(beta_nogo_pot,
                                                  cond_vec[mask_nogo],
                                                  part_vec[mask_nogo],
                                                  X=pcm.indicator(part_vec[mask_nogo]))
                    G_go_null[s], _ = pcm.est_G_crossval(beta_go_null,
                                                        cond_vec[mask_go],
                                                        part_vec[mask_go],
                                                        X=pcm.indicator(part_vec[mask_go]))
                    G_go_pot[s], _ = pcm.est_G_crossval(beta_go_pot,
                                                       cond_vec[mask_go],
                                                       part_vec[mask_go],
                                                       X=pcm.indicator(part_vec[mask_go]))
                np.save(os.path.join(pcm_path, f'G_obs.null.plan-exec.glm{args.glm}.{H}.{roi}.npy'), G_null)
                np.save(os.path.join(pcm_path, f'G_obs.pot.plan-exec.glm{args.glm}.{H}.{roi}.npy'), G_pot)
                np.save(os.path.join(pcm_path, f'G_obs.null.plan.glm{args.glm}.{H}.{roi}.npy'), G_nogo_null)
                np.save(os.path.join(pcm_path, f'G_obs.pot.plan.glm{args.glm}.{H}.{roi}.npy'), G_nogo_pot)
                np.save(os.path.join(pcm_path, f'G_obs.null.exec.glm{args.glm}.{H}.{roi}.npy'), G_go_null)
                np.save(os.path.join(pcm_path, f'G_obs.pot.exec.glm{args.glm}.{H}.{roi}.npy'), G_go_pot)
    if args.what == 'pcm2tsv':
        components = {
            'plan': ['expectation', 'uncertainty'],
            'exec': ['sensory input', 'expectation', 'surprise']
        }
        pcm_dict = {
            'epoch': [],
            'Hem': [],
            'roi': [],
            'weight': [],
            'noise': [],
            'weight_sum': [],
            'BF': [],
            'component': [],
            'participant_id': []
        }
        for epoch in ['plan', 'exec']:
            Mc, idxc = find_model(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{epoch}.p'), 'component')
            n_param_c = Mc.n_param
            MF = pcm.model.ModelFamily(Mc.Gc,
                                       comp_names=components[epoch],
                                       basecomponents=np.eye(8)[None, :, :] if epoch=='exec' else None)
            for H in Hem:
                for roi in rois:
                    f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'theta_in.{epoch}.glm{args.glm}.{H}.{roi}.p'), "rb")
                    param = pickle.load(f)
                    param_c = param[idxc][:n_param_c]
                    noise = np.exp(param[idxc][-1])
                    T = pd.read_pickle(
                        os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                     f'T.model_family.{epoch}.glm{args.glm}.{H}.{roi}.p'))
                    c_bf = MF.component_bayesfactor(T.likelihood, method='AIC', format='DataFrame')
                    c_bf = pd.melt(c_bf, var_name='component', value_name='BF')
                    weight_sum = np.exp(param_c).sum(axis=0)
                    weight = np.exp(param_c).reshape(-1)
                    pcm_dict['epoch'].extend([epoch] * weight.size)
                    pcm_dict['roi'].extend([roi] * weight.size)
                    pcm_dict['Hem'].extend([H] * weight.size)
                    pcm_dict['weight'].extend(weight)
                    pcm_dict['weight_sum'].extend(np.concatenate([weight_sum] * n_param_c))
                    pcm_dict['noise'].extend(np.concatenate([noise] * n_param_c))
                    pcm_dict['BF'].extend(c_bf['BF'].to_numpy())
                    pcm_dict['component'].extend(c_bf['component'].to_numpy())
                    pcm_dict['participant_id'].extend(args.sns * len(components[epoch]))

        df = pd.DataFrame(pcm_dict)
        df.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'component_model.BOLD.tsv'), sep='\t', index=False)
    if args.what == 'corr2tsv':
        corrs = ['plan-exec', 'cue-finger']
        corr_dict = {
            'r_indiv': [],
            'r_group': [],
            'SNR': [],
            'corr': [],
            'ci_lo': [],
            'ci_hi': [],
            'Hem': [],
            'roi': [],
            'participant_id': []
        }
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for corr in corrs:
            for H in Hem:
                for roi in rois:
                    f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'theta_in.corr_{corr}.glm{args.glm}.{H}.{roi}.p'), 'rb')
                    theta = pickle.load(f)[0]
                    r_bootstrap = np.load(
                        os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'r_bootstrap.corr_{corr}.{H}.{roi}.npy'))
                    f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'theta_gr.corr_{corr}.glm{args.glm}.{H}.{roi}.p'), 'rb')
                    theta_g = pickle.load(f)[0]

                    r_group, r_indiv, SNR = extract_mle_corr(Mflex, theta, theta_g)
                    (ci_lo, ci_hi), _, _ = bootstrap_summary(r_bootstrap, alpha=0.025)

                    corr_dict['r_indiv'].extend(r_indiv)
                    corr_dict['r_group'].extend(r_group)
                    corr_dict['ci_lo'].extend([ci_lo] * len(args.sns))
                    corr_dict['ci_hi'].extend([ci_hi] * len(args.sns))
                    corr_dict['SNR'].extend(SNR)
                    corr_dict['corr'].extend([corr] * len(args.sns))
                    corr_dict['participant_id'].extend(args.sns)
                    corr_dict['Hem'].extend([H] * len(args.sns))
                    corr_dict['roi'].extend([roi] * len(args.sns))
        df_corr = pd.DataFrame(corr_dict)
        df_corr.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'correlations.BOLD.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--n_tessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')