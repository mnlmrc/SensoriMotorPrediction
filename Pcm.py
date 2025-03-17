import warnings

import time

warnings.filterwarnings("ignore")

import argparse
import pickle

import PcmPy as pcm
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

import globals as gl
import pandas as pd
import numpy as np
import nibabel as nb
import os

import nitools as nt

import sys

import Functional_Fusion.atlas_map as am

def find_model(path, name):
    f = open(path, 'rb')
    M = pickle.load(f)
    for m in M:
        if m.name == name:
            return m, M.index(m)
    if m == M[-1]:
        raise Exception(f'Model name not found')

def make_execution_models():
    C = pcm.centering(8)

    v_fingerID = C @ np.array([1, 1, 1, 1, -1, -1, -1, -1])
    v_cue = C @ np.array([-1, 0, 1, 2, -2, -1, 0, 1])
    # v_cert = C @ np.array([0.1875, .25, 0.1875, 0, 0, 0.1875, .25, 0.1875])  # variance of a Bernoulli distribution
    # v_surprise = C @ -np.log2(np.array([.25, .5, .75, 1, 1, .75, .5, .25]))  # with Shannon information

    Ac = np.zeros((3, 8, 3))
    Ac[0, :, 0] = v_fingerID
    Ac[1, :, 1] = v_cue
    Ac[2, :, 2] = v_cue
    Ac[0, :, 2] = v_fingerID

    G_fingerID = np.outer(v_fingerID, v_fingerID)
    G_cue = np.outer(v_cue, v_cue)
    # G_cert = np.outer(v_cert, v_cert)
    # G_surprise = np.outer(v_surprise, v_surprise)

    G_component = np.array([G_fingerID, G_cue])

    M = []
    M.append(pcm.FixedModel('null', np.zeros((8, 8))))  # 0
    M.append(pcm.FixedModel('stimFinger', G_fingerID))  # 1
    M.append(pcm.FixedModel('cue', G_cue))  # 2
    # M.append(pcm.FixedModel('cert', G_cert))  # 3
    M.append(pcm.FixedModel('eye', np.eye(8)))  # 4
    # M.append(pcm.FixedModel('surprise', G_surprise))  # 5
    M.append(pcm.ComponentModel('stimFinger+cue (component)', G_component))  # 6
    M.append(pcm.FeatureModel('stimFinger+cue+stimFinger*cue (feature)', Ac))  # 7
    M.append(pcm.FreeModel('ceil', 8))  # 8

    return M


def make_execution_models_emg():
    C = pcm.centering(8)

    v_fingerID = C @ np.array([1, 1, 1, 1, -1, -1, -1, -1])
    v_cue = C @ np.array([2, -1, 0, 1, -2, -1, 0, 1, ])

    Ac = np.zeros((3, 8, 3))
    Ac[0, :, 0] = v_fingerID
    Ac[1, :, 1] = v_cue
    Ac[2, :, 0] = v_cue

    G_fingerID = np.outer(v_fingerID, v_fingerID)
    G_cue = np.outer(v_cue, v_cue)

    M = []
    M.append(pcm.FixedModel('null', np.zeros((8, 8))))
    M.append(pcm.FixedModel('stimFinger', G_fingerID))
    M.append(pcm.FixedModel('cue', G_cue))
    M.append(pcm.FixedModel('ind', np.eye(8)))
    M.append(pcm.ComponentModel('stimFinger+cue (component)',
                                np.array([G_fingerID, G_cue, ])))
    M.append(pcm.FeatureModel('stimFinger+cue (feature)', Ac))
    M.append(pcm.FreeModel('ceil', 8))  # Noise ceiling model

    return M


def make_planning_models():
    C = pcm.centering(5)

    v_cue = C @ np.array([-2, -1, 0, 1, 2])
    v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0])
    G_cue_plan = np.outer(v_cue, v_cue)
    G_cert_plan = np.outer(v_cert, v_cert)

    M = []
    M.append(pcm.FixedModel('null', np.zeros((5, 5))))  # 0
    M.append(pcm.FixedModel('cue', G_cue_plan))  # 1
    M.append(pcm.FixedModel('cert', G_cert_plan))  # 2
    M.append(pcm.FixedModel('eye', np.eye(5)))  # 3
    M.append(pcm.ComponentModel('cue+cert', np.array([G_cue_plan, G_cert_plan])))  # 4
    M.append(pcm.FreeModel('ceil', 5))  # 5

    return M


def make_individ_dataset(subatlas=None, args=None, sn=None, Hem=None):
    subj_dir = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{sn}')
    white = os.path.join(subj_dir, f'subj{sn}.{Hem}.white.32k.surf.gii')
    pial = os.path.join(subj_dir, f'subj{sn}.{Hem}.pial.32k.surf.gii')
    mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}', 'mask.nii')

    # Build atlas mapping
    amap = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
    amap.build()

    # Load regressor information
    reginfo_path = os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}', f'subj{sn}_reginfo.tsv')
    reginfo = pd.read_csv(reginfo_path, sep='\t')

    # Construct paths for beta images
    dnames = [os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}', f'beta_{i + 1:04d}.nii')
              for i in range(reginfo.shape[0])]

    betas = amap.extract_data_native(dnames)
    res = amap.extract_data_native([os.path.join(gl.baseDir, args.experiment,
                                                 f'glm{args.glm}', f'subj{sn}', 'ResMS.nii')])

    # Prewhiten betas
    betas_prewhitened = betas / np.sqrt(res)
    betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

    # Process condition vector
    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
    part_vec = reginfo.run

    cond_map = {
        'save_rois_execution': [5, 6, 7, 8, 9, 10, 11, 12],
        'save_tessel_execution': [5, 6, 7, 8, 9, 10, 11, 12],
        'save_rois_planning': [0, 1, 2, 3, 4],
        'save_tessel_planning': [0, 1, 2, 3, 4],
    }
    idx = cond_vec.isin(cond_map[args.what])

    # Create dataset
    obs_des = {'cond_vec': cond_vec[idx].to_numpy(),
               'part_vec': part_vec[idx].to_numpy()}
    Dataset = pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des)

    return Dataset


def fit_model_in_tessel(subatlas=None, args=None, M=None, Hem=None):
    Y = list()
    n_voxels = list()
    for s, sn in enumerate(args.snS):
        Dataset = make_individ_dataset(subatlas=subatlas, args=args, sn=sn, Hem=Hem)
        n_voxels.append(Dataset.n_channel)
        Y.append(Dataset)

    # if Y > 0:
    T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
    T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

    likelihood = T_cv.likelihood
    baseline = likelihood.loc[:, 'null'].values
    likelihood = likelihood - baseline.reshape(-1, 1)

    noise_upper = (T_gr.likelihood['ceil'] - baseline)

    noise_lower = likelihood.ceil

    return likelihood, noise_upper, noise_lower, baseline, theta_cv, n_voxels


def process_1tessel_execution(args=None, atlas=None, h=None, ntessel=None, M=None):
    # ntessel = ntessel + 1  # skip the first tessel

    Hem = ['L', 'R']

    col_names = [m.name for m in M]

    print(f'Hemisphere: {Hem[h]}, tessel #{ntessel}')
    atlas_hem = atlas.get_hemisphere(h)
    subatlas = atlas_hem.get_subatlas_image(os.path.join(gl.atlas_dir,
                                                         f'Icosahedron{args.ntessels}.{Hem[h]}.label.gii'), ntessel)

    T = {
        'likelihood': [],
        'noise_upper': [],
        'noise_lower': [],
        'baseline': [],
        'n_voxels': [],
        'col_names': [],
        'sn': []
    }

    theta_component = {
        'theta': [],
        '#comp': [],
        'sn': []
    }

    theta_feature = {
        'theta': [],
        '#feat': [],
        'sn': []
    }

    try:
        likelihood, noise_upper, noise_lower, baseline, theta_cv, n_voxels = \
            fit_model_in_tessel(subatlas=subatlas, args=args, M=M, Hem=Hem[h])

        for s, sn in enumerate(args.snS):
            for c, col in enumerate(col_names):
                T['likelihood'].append(likelihood[col][s])
                T['noise_upper'].append(noise_upper[s])
                T['noise_lower'].append(noise_lower[s])
                T['baseline'].append(baseline[s])
                T['n_voxels'].append(n_voxels[s])
                T['col_names'].append(col)
                T['sn'].append(sn)
            for c in range(M[4].n_param):
                theta_component['theta'].append(theta_cv[4][c, s])
                theta_component['sn'].append(sn)
                theta_component['#comp'].append(c)
            for c in range(M[5].n_param):
                theta_feature['theta'].append(theta_cv[5][c, s])
                theta_feature['sn'].append(sn)
                theta_feature['#feat'].append(c)

    except Exception as e:
        print(f"Error in tessel #{ntessel}: {e}")
        for s, sn in enumerate(args.snS):
            for c, col in enumerate(col_names):
                T['likelihood'].append(np.nan)
                T['noise_upper'].append(np.nan)
                T['noise_lower'].append(np.nan)
                T['baseline'].append(np.nan)
                T['n_voxels'].append(np.nan)
                T['col_names'].append(col)
                T['sn'].append(sn)
            for c in range(M[4].n_param):
                theta_component['theta'].append(np.nan)
                theta_component['sn'].append(sn)
                theta_component['#comp'].append(c)
            for c in range(M[5].n_param):
                theta_feature['theta'].append(np.nan)
                theta_feature['sn'].append(sn)
                theta_feature['#feat'].append(c)

    T = pd.DataFrame(T)
    theta_component = pd.DataFrame(theta_component)
    theta_feature = pd.DataFrame(theta_feature)

    return T, theta_component, theta_feature, subatlas.vertex[0]


def process_tessel_planning(args=None, atlas=None, h=None, ntessel=None, M=None):
    ntessel = ntessel + 1  # skip the first tessel

    Hem = ['L', 'R']

    col_names = [m.name for m in M]

    print(f'Hemisphere: {Hem[h]}, tessel #{ntessel}')
    atlas_hem = atlas.get_hemisphere(h)
    subatlas = atlas_hem.get_subatlas_image(os.path.join(gl.atlas_dir,
                                                         f'Icosahedron{args.ntessels}.{Hem[h]}.label.gii'), ntessel)

    T = {
        'likelihood': [],
        'noise_upper': [],
        'noise_lower': [],
        'baseline': [],
        'col_names': [],
        'sn': []
    }

    theta_component = {
        'theta': [],
        '#comp': [],
        'sn': []
    }

    try:
        likelihood, noise_upper, noise_lower, baseline, theta_cv = \
            fit_model_in_tessel(subatlas=subatlas, args=args, M=M)

        for s, sn in enumerate(args.snS):
            for c, col in enumerate(col_names):
                T['likelihood'].append(likelihood[col][s])
                T['noise_upper'].append(noise_upper[s])
                T['noise_lower'].append(noise_lower[s])
                T['baseline'].append(baseline[s])
                T['col_names'].append(col)
                T['sn'].append(sn)
            for c in range(M[4].n_param):
                theta_component['theta'].append(theta_cv[4][c, s])
                theta_component['sn'].append(sn)
                theta_component['#comp'].append(c)

    except Exception as e:
        print(f"Error in tessel #{ntessel}: {e}")
        for sn in range(len(args.snS)):
            for c, col in enumerate(col_names):
                T['likelihood'].append(np.nan)
                T['noise_upper'].append(np.nan)
                T['noise_lower'].append(np.nan)
                T['baseline'].append(np.nan)
                T['col_names'].append(col)
                T['sn'].append(sn)
            for c in range(M[4].n_param):
                theta_component['theta'].append(np.nan)
                theta_component['sn'].append(sn)
                theta_component['#comp'].append(c)

    T = pd.DataFrame(T)
    theta_component = pd.DataFrame(theta_component)

    return T, theta_component, subatlas.vertex[0]


def main(args):

    if args.what == 'save_tessel_execution':

        M = make_execution_models()
        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.exec.glm{args.glm}.pkl'), "wb") as f:
            pickle.dump(M, f)
        col_names = [m.name for m in M]

        struct = ['CortexLeft', 'CortexRight']

        atlas, _ = am.get_atlas('fs32k')

        for h, H in enumerate(['L', 'R']):

            # Parallel processing of tessels
            with parallel_backend("loky"):  # use threading for debug in PyCharm
                results = Parallel(n_jobs=args.n_jobs)(
                    delayed(process_1tessel_execution)(args=args, atlas=atlas, h=h, ntessel=ntessel, M=M)
                    for ntessel in range(args.ntessels) #np.arange(340, 364, 1)
                )

            # Aggregate results from parallel processes
            T = np.full((len(args.snS), 32492, len(M)+4), np.nan)
            theta_component = np.full((len(args.snS), 32492, M[4].n_param), np.nan)
            theta_feature = np.full((len(args.snS), 32492, M[5].n_param), np.nan)

            for s, sn in enumerate(args.snS):
                for Tt, tc, tf, vertex_id in results:
                    for c, col in enumerate(col_names):
                        LL = Tt[(Tt['sn'] == sn) & (Tt['col_names'] == col)]['likelihood']
                        T[s, vertex_id, c] = LL
                    T[s, vertex_id, -4] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
                    T[s, vertex_id, -3] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
                    T[s, vertex_id, -2] = Tt[(Tt['sn'] == sn)]['baseline'].unique()
                    T[s, vertex_id, -1] = Tt[(Tt['sn'] == sn)]['n_voxels'].unique()
                    for c in range(M[4].n_param):
                        theta = tc[(tc['sn'] == sn) & (tc['#comp'] == c)]['theta']
                        theta_component[s, vertex_id, c] = theta
                    for c in range(M[5].n_param):
                        theta = tf[(tf['sn'] == sn) & (tf['#feat'] == c)]['theta']
                        theta_feature[s, vertex_id, c] = theta

            # save giftis
            for s, sn in enumerate(args.snS):
                gifti_img_T = nt.make_func_gifti(T[s], anatomical_struct=struct[h],
                                                 column_names=col_names+['noise_upper', 'noise_lower', 'baseline', 'n_voxels'])
                gifti_img_theta_component = nt.make_func_gifti(theta_component[s], anatomical_struct=struct[h],
                                                               column_names=['stimFinger', 'cue',])
                gifti_img_theta_feature = nt.make_func_gifti(theta_feature[s], anatomical_struct=struct[h],
                                                             column_names=['stimFinger', 'cue', 'stimFinger*cue'])
                nb.save(gifti_img_T, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                                  f'ML.Icosahedron{args.ntessels}.glm{args.glm}.pcm.exec.{H}.func.gii'))
                nb.save(gifti_img_theta_component, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                                                f'theta.Icosahedron{args.ntessels}.component.glm{args.glm}.pcm.exec.{H}.func.gii'))
                nb.save(gifti_img_theta_feature, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                                              f'theta.Icosahedron{args.ntessels}.feature.glm{args.glm}.pcm.exec.{H}.func.gii'))

    if args.what == 'save_tessel_planning':

        M = make_planning_models()
        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.exec.glm{args.glm}.pkl'), "wb") as f:
            pickle.dump(M, f)
        col_names = [m.name for m in M]

        struct = ['CortexLeft', 'CortexRight']

        atlas, _ = am.get_atlas('fs32k')

        for h, H in enumerate(['L', 'R']):

            # Parallel processing of tessels
            with parallel_backend("loky"):
                results = Parallel(n_jobs=args.n_jobs)(
                    delayed(process_tessel_planning)(args=args, atlas=atlas, h=h, ntessel=ntessel, M=M)
                    for ntessel in range(args.ntessels) #np.arange(340, 364, 1) #
                )

            # Serial rpocessing of tessels
            for ntessel in range(args.ntessels):
                results = process_tessel_planning(args=args, atlas=atlas, h=h, ntessel=ntessel, M=M)

            # Aggregate results from parallel processes
            T = np.full((len(args.snS), 32492, len(M)), np.nan)
            theta_component = np.full((len(args.snS), 32492, M[4].n_param), np.nan)

            for s, sn in enumerate(args.snS):
                for Tt, tc, vertex_id in results:
                    for c, col in enumerate(col_names):
                        T[s, vertex_id, c] = Tt[(Tt['sn'] == sn) & (Tt['col_names'] == col)]['likelihood']
                    for c in range(M[4].n_param):
                        theta_component[s, vertex_id, c] = tc[(tc['sn'] == sn) & (tc['#comp'] == c)]['theta']

            # save giftis
            for s, sn in enumerate(args.snS):
                gifti_img_T = nt.make_func_gifti(T[s], anatomical_struct=struct[h], column_names=col_names)
                gifti_img_theta_component = nt.make_func_gifti(theta_component[s], anatomical_struct=struct[h],
                                                               column_names=['cue', 'cert'])
                nb.save(gifti_img_T, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                                  f'ML.Icosahedron{args.ntessels}.glm{args.glm}.pcm.plan.{H}.func.gii'))
                nb.save(gifti_img_theta_component, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                                                f'theta.Icosahedron{args.ntessels}.component.glm{args.glm}.pcm.plan.{H}.func.gii'))

    if args.what == 'save_emg_execution':

        M = make_execution_models_emg()
        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.emg.pkl'), "wb") as f:
            pickle.dump(M, f)

        snS = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110]

        N = len(snS)

        G_obs = np.zeros((N, 8, 8))
        Y = list()
        for s, sn in enumerate(snS):
            npz = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}',
                                       f'{args.experiment}_{sn}_binned.npz'), allow_pickle=True)

            emg = npz['data_array'][-1]
            descr = npz['descriptor'].item()
            timepoints = list(descr['time windows'].keys())

            dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                           f'{args.experiment}_{sn}.dat'), sep='\t')
            dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
            dat['cue'] = dat['cue'].map(gl.cue_mapping)
            dat['BN'] = dat['BN'].astype(str)

            cov = emg.T @ emg

            emg = emg / np.sqrt(np.diag(cov))

            dat[['ch_' + str(x) for x in range(emg.shape[-1])]] = emg

            dat = dat.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = dat['stimFinger'] + ',' + dat['cue']
            part_vec = dat['BN']

            obs_des = {'cond_vec': cond_vec,
                       'part_vec': part_vec}

            Y.append(pcm.dataset.Dataset(dat[['ch_' + str(x) for x in range(emg.shape[-1])]].to_numpy(),
                                         obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

        path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, f'G_obs.emg.Vol.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.emg.Vol.pkl'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.emg.Vol.pkl'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.emg.Vol.pkl'))

        with open(os.path.join(path, f'theta_in.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'save_force_execution':

        M = make_execution_models()
        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.force.pkl'), "wb") as f:
            pickle.dump(M, f)

        N = len(args.snS)

        G_obs = np.zeros((N, 8, 8))
        Y = list()
        for s, sn in enumerate(args.snS):
            force = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                             f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
            force = force[force['GoNogo'] == 'go'] if 'GoNogo' in force else force # select only go trial
            force['cue'] = force['cue'].map(gl.cue_mapping)
            force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
            force = force.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = force['cue'] + ',' + force['stimFinger']
            part_vec = force['BN']

            force = force[['thumb', 'index', 'middle', 'ring', 'pinkie']].to_numpy()

            cov = force.T @ force

            force = force / np.sqrt(np.diag(cov)) # prewhitening using variance of each channel

            obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
                       'part_vec': part_vec}

            Y.append(pcm.dataset.Dataset(force, obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

        path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, f'G_obs.force.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.force.pkl'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.force.pkl'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.force.pkl'))

        with open(os.path.join(path, f'theta_in.force.pkl'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.force.pkl'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.force.pkl'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'save_rois_planning':

        M = make_planning_models()

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)

                G_obs = np.zeros((N, 5, 5))
                Y = list()
                for s, sn in enumerate(args.snS):
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

                    G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
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

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

                os.makedirs(path, exist_ok=True)

                np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.{H}.{roi}.npy'), G_obs)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_in.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_cv.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_gr.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_gr, f)

    if args.what == 'save_rois_execution':

        M = make_execution_models()
        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.exec.glm{args.glm}.pkl'), "wb") as f:
            pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)

                G_obs = np.zeros((N, 8, 8))
                Y = list()
                for s, sn in enumerate(args.snS):
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

                    G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                     Y[s].obs_descriptors['part_vec'],
                                                     X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True,
                                                              fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_in.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_cv.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_gr.exec.glm{args.glm}.{H}.{roi}.pkl'))

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

                os.makedirs(path, exist_ok=True)

                np.save(os.path.join(path, f'G_obs.exec.glm{args.glm}.{H}.{roi}.npy'), G_obs)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_in.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_cv.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_gr.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_gr, f)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--ntessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')