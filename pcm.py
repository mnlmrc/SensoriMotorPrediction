import argparse
import os.path

import PcmPy as pcm
import scipy

import pickle

import warnings
warnings.filterwarnings("ignore")

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

    # planning models


    # execution models


    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='Icosahedron-1002')
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'save_rois_execution':

        C = pcm.centering(8)

        v_fingerID = C @ np.array([1, 1, 1, 1, -1, -1, -1, -1])
        v_cue = C @ np.array([-1, 0, 1, 2, -2, -1, 0, 1, ])
        v_cert = C @ np.array([0.1875, .25, 0.1875, 0, 0, 0.1875, .25, 0.1875])  # variance of a Bernoulli distribution
        v_surprise = C @ -np.log2(np.array([.25, .5, .75, 1, 1, .75, .5, .25]))  # with Shannon information

        Ac = np.zeros((7, 8, 7))
        Ac[0, :, 0] = v_fingerID
        Ac[1, :, 1] = v_cue
        Ac[2, :, 2] = v_cert
        Ac[3, :, 3] = v_surprise
        Ac[4, :, 0] = v_cue
        Ac[5, :, 0] = v_cert
        Ac[6, :, 0] = v_surprise

        G_fingerID = np.outer(v_fingerID, v_fingerID)
        G_cue = np.outer(v_cue, v_cue)
        G_cert = np.outer(v_cert, v_cert)
        G_surprise = np.outer(v_surprise, v_surprise)

        M = []
        M.append(pcm.FixedModel('null', np.eye(8)))
        M.append(pcm.FixedModel('stimFinger', G_fingerID))
        M.append(pcm.FixedModel('cue', G_cue))
        M.append(pcm.FixedModel('cert', G_cert))
        M.append(pcm.FixedModel('surprise', G_surprise))
        M.append(pcm.ComponentModel('stimFinger+cue+cert+surprise (component)',
                                    np.array([G_fingerID, G_cue, G_cert, G_surprise])))
        M.append(pcm.FeatureModel('stimFinger+cue+cert+surprise (feature)', Ac))
        M.append(pcm.FreeModel('ceil', 8))  # Noise ceiling model

        snS = [102, 103, 104, 106, 107]

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(snS)

                G_hat_betas = np.zeros((N, 8, 8))
                Y = list()

                for s, sn in enumerate(snS):
                    reginfo = pd.read_csv(
                        os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                     f'subj{sn}_reginfo.tsv'), sep='\t')

                    betas = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                                 f'subj{sn}', f'ROI.{H}.{roi}.beta.npy'))
                    res = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                               f'subj{sn}', f'ROI.{H}.{roi}.res.npy'))

                    betas_prewhitened = betas / np.sqrt(res)

                    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
                    part_vec = reginfo.run

                    idx = cond_vec.isin([5, 6, 7, 8, 9, 10, 11, 12])

                    obs_des = {'cond_vec': cond_vec[idx],
                               'part_vec': part_vec[idx]}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

                    G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                           Y[s].obs_descriptors['part_vec'],
                                                           X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_in.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_cv.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_gr.exec.glm{args.glm}.{H}.{roi}.pkl'))

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_in.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_cv.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_gr.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_gr, f)

    if args.what == 'save_rois_planning':

        C = pcm.centering(5)
        v_cue = C @ np.array([-2, -1, 0, 1, 2])
        v_cert = C @ np.array([0, 1, 2, 1, 0])
        G_cue_plan = np.outer(v_cue, v_cue)
        G_cert_plan = np.outer(v_cert, v_cert)

        M = []
        M.append(pcm.FixedModel('null', np.eye(5)))
        M.append(pcm.FixedModel('cue', G_cue_plan))
        M.append(pcm.FixedModel('cert', G_cert_plan))
        M.append(pcm.ComponentModel('cue+cert', np.array([G_cert_plan, G_cue_plan])))
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

                    betas_prewhitened = betas / np.sqrt(res)

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

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_in.plan.glm{args.glm}.{H}.{roi}.pkl'))
                T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_cv.plan.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                         f'T_gr.plan.glm{args.glm}.{H}.{roi}.pkl'))
                            
                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_in.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_cv.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                       f'theta_gr.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                        pickle.dump(theta_gr, f)
