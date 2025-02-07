import argparse
import os.path

import PcmPy as pcm
import scipy

import globals as gl
import pandas as pd
import numpy as np
import os
import subprocess
import nibabel as nb
import nitools as nt

import matplotlib.pyplot as plt

from betas import get_roi


def make_Z_all(experiment='smp2', sn=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    # Extract percentage and finger information from the "name" column
    reginfo['percentage'] = reginfo['name'].str.extract(r'(\d+%)')[0]
    reginfo['finger'] = reginfo['name'].str.extract(r',(index|ring)')[0].fillna('nogo')

    # Define unique percentages and fingers for one-hot encoding
    unique_percentages = ['0%', '25%', '50%', '75%', '100%']
    unique_fingers = ['index', 'ring', 'nogo']

    # Initialize the design matrix
    Z = np.zeros((len(reginfo), len(unique_percentages) + len(unique_fingers)), dtype=int)

    # Fill in the design matrix
    for i, row in reginfo.iterrows():
        # Percentage columns
        percentage_idx = unique_percentages.index(row['percentage'])
        Z[i, percentage_idx] = 1

        # Finger columns
        finger_idx = unique_fingers.index(row['finger']) + len(unique_percentages)
        Z[i, finger_idx] = 1

    return Z


def make_Z_cue(experiment='smp2', sn=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    # Extract percentage and finger information from the "name" column
    reginfo['percentage'] = reginfo['name'].str.extract(r'(\d+%)')[0]
    reginfo['finger'] = reginfo['name'].str.extract(r',(index|ring)')[0].fillna('nogo')

    # Define unique percentages and fingers for one-hot encoding
    unique_percentages = ['0%', '25%', '50%', '75%', '100%']

    # Initialize the design matrix
    Z = np.zeros((len(reginfo), len(unique_percentages)), dtype=int)

    # Fill in the design matrix
    for i, row in reginfo.iterrows():
        # Percentage columns
        percentage_idx = unique_percentages.index(row['percentage'])
        Z[i, percentage_idx] = 1

    return Z


def FixedModel(name, Z):
    G = np.matmul(Z, Z.T)
    M = pcm.model.FixedModel(name, G)

    return M, G


def get_tessel_betas(experiment=None, sn=None, atlas=None, Hem=None, idx=None, glm=None):
    R = get_roi(experiment, sn, Hem, f'label-{idx}', atlas=atlas)

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    betas = list()
    for n_regr in np.arange(0, reginfo.shape[0]):
        vol = nb.load(
            os.path.join(gl.baseDir, 'smp2', f'{gl.glmDir}{glm}', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
        beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        betas.append(beta)

    betas = np.array(betas)
    betas = betas[:, ~np.all(np.isnan(betas), axis=0)]

    assert betas.ndim == 2

    return betas


if __name__ == '__main__':

    Z_stimFinger = np.zeros((13, 3))
    Z_stimFinger[0:5, 0] = 1
    Z_stimFinger[5:9, 1] = 1
    Z_stimFinger[9:13, 2] = 1

    Z_cue = np.zeros((13, 5))
    Z_cue[[0, 9], 0] = 1
    Z_cue[[1, 5, 10], 1] = 1
    Z_cue[[2, 6, 11], 2] = 1
    Z_cue[[3, 7, 12], 3] = 1
    Z_cue[[4, 8], 4] = 1

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='Icosahedron-1002')
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == '_get_tessel_betas':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for idx in range(924):
                print(f'Hemisphere: {H}, tessel:{idx + 1}')
                betas = get_tessel_betas(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    idx=idx + 1,
                    atlas=args.atlas,
                    glm=args.glm,
                )
    if args.what == 'save_rois_pcm_plan+exec':

        M = []
        M.append(pcm.FixedModel('null', np.eye(13)))
        M.append(pcm.FixedModel('stimFinger', Z_stimFinger @ Z_stimFinger.T))
        M.append(pcm.FixedModel('cue', Z_cue @ Z_cue.T))
        M.append(
            pcm.ComponentModel('stimFinger+cue', np.array([Z_stimFinger @ Z_stimFinger.T, Z_cue @ Z_cue.T])))
        M.append(pcm.FreeModel('ceil', 13))  # Noise ceiling model

        snS = [102, 103, 104, 106, 107]

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(snS)

                G_hat_betas = np.zeros((N, 13, 13))
                Y = list()

                for s, sn in enumerate(snS):
                    reginfo = pd.read_csv(
                        os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                     f'subj{sn}_reginfo.tsv'), sep='\t')

                    betas = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                                 f'subj{sn}', f'ROI.{H}.{roi}.beta.npy'))
                    res = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                               f'subj{sn}', f'ROI.{H}.{roi}.res.npy'))

                    betas_prewhitened = betas / res

                    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
                    part_vec = reginfo.run

                    obs_des = {'cond_vec': cond_vec,
                               'part_vec': part_vec}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))

                    G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                           Y[s].obs_descriptors['part_vec'],
                                                           X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_in.exec+plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')
                T_cv.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_cv.exec+plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')
                T_gr.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_gr.exec+plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')

    if args.what == 'save_rois_pcm_plan':

        Z_base = np.eye(5)  # Identity matrix
        A = []
        for i in range(5):
            for j in range(5):
                if i != j:  # Only modify off-diagonal elements
                    Z_new = Z_base.copy()
                    Z_new[i, j] = 1  # Set one off-diagonal element to 1
                    A.append(Z_new)
        A = np.stack(A, axis=0)

        assert A.ndim == 3

        np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                             f'features.plan.npy'), A)

        M = []
        M.append(pcm.FixedModel('null', np.eye(5)))
        M.append(pcm.FeatureModel('cue', A))
        M.append(pcm.FreeModel('ceil', 5))  # Noise ceiling model

        snS = [102, 103, 104, 106, 107]

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(snS)

                G_hat_betas = np.zeros((N, 5, 5))
                Y = list()

                for s, sn in enumerate(snS):
                    reginfo = pd.read_csv(
                        os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                     f'subj{sn}_reginfo.tsv'), sep='\t')

                    betas = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                                 f'subj{sn}', f'ROI.{H}.{roi}.beta.npy'))
                    res = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                               f'subj{sn}', f'ROI.{H}.{roi}.res.npy'))

                    betas_prewhitened = betas / res

                    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
                    part_vec = reginfo.run

                    idx = cond_vec.isin([0, 1, 2, 3, 4])

                    obs_des = {'cond_vec': cond_vec[idx],
                               'part_vec': part_vec[idx]}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

                    G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                           Y[s].obs_descriptors['part_vec'],
                                                           X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True,
                                                              fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_in.plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')
                T_cv.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_cv.plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')
                T_gr.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_gr.plan.glm{args.glm}.{H}.{roi}.tsv'),
                            sep='\t')

                np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'theta_in.plan.glm{args.glm}.{H}.{roi}.npy'), theta_in)
                np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                     f'theta_in.plan.glm{args.glm}.{H}.{roi}.npy'), theta_cv)
                np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                     f'theta_in.plan.glm{args.glm}.{H}.{roi}.npy'), theta_gr)

