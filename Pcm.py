import warnings

import time

warnings.filterwarnings("ignore")

import argparse
import pickle

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def find_model(M, name):
    if type(M) == str:
        f = open(M, 'rb')
        M = pickle.load(f)
    if type(M) == list:
        for m in M:
            if m.name == name:
                return m, M.index(m)
        if m == M[-1]:
            raise Exception(f'Model name not found')

def normalize_G(G):
    return (G - G.mean()) / G.std()

def normalize_Ac(Ac):
    for a in range(Ac.shape[0]):
        tr = np.trace(Ac[a] @ Ac[a].T)
        Ac[a] = Ac[a] / np.sqrt(tr)
    return Ac

def calc_normalized_likelihood_in_parcel(T_cv, T_gr, parcel_field='roi', parcel_name=None):
    likelihood = T_cv.likelihood
    baseline = likelihood.loc[:, 'null'].values
    likelihood = likelihood - baseline.reshape(-1, 1)

    noise_upper = (T_gr.likelihood['ceil'] - baseline).mean()

    noise_lower_abs = likelihood.ceil.mean()

    assert noise_upper > noise_lower_abs

    likelihood = likelihood / noise_lower_abs
    noise_upper = noise_upper / noise_lower_abs

    noise_lower = likelihood.ceil.mean()

    LL = pd.melt(likelihood)
    LL[parcel_field] = parcel_name
    LL['noise_lower'] = noise_lower
    LL['noise_upper'] = noise_upper

    return LL

def get_likelihood_in_parcel(T_cv, T_gr, parcel_field='roi', parcel_name=None):
    likelihood = T_cv.likelihood
    baseline = likelihood.loc[:, 'null'].values
    likelihood = likelihood - baseline.reshape(-1, 1)

    noise_upper = (T_gr.likelihood['ceil'] - baseline).mean()

    noise_lower = likelihood.ceil.mean()

    assert noise_upper > noise_lower

    LL = pd.melt(likelihood)
    LL[parcel_field] = parcel_name
    LL['noise_lower'] = noise_lower
    LL['noise_upper'] = noise_upper

    return LL

def make_execution_models(centering=False):

    C = pcm.centering(8)

    if centering:
        v_fingerID = C @ np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = C @ np.array([1, 2, 3, 4, 2, 3, 4, 5])
        v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = C @ -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information
    else:
        v_fingerID = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = np.array([1, 2, 3, 4, 2, 3, 4, 5])
        v_cert = np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information

    Ac = np.zeros((6, 8, 6))
    Ac[0, :, 0] = v_fingerID
    Ac[0, :, 4] = v_fingerID
    Ac[1, :, 1] = v_cue
    Ac[1, :, 5] = v_cue
    Ac[2, :, 2] = v_cert
    Ac[3, :, 3] = v_surprise
    Ac[4, :, 4] = v_cue
    Ac[5, :, 5] = v_fingerID

    Ac = normalize_Ac(Ac)

    G_fingerID = np.outer(v_fingerID, v_fingerID)
    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    G_surprise = np.outer(v_surprise, v_surprise)
    G_component = np.array([G_fingerID / np.trace(G_fingerID),
                            G_cue / np.trace(G_cue),
                            G_cert / np.trace(G_cert),
                            G_surprise / np.trace(G_surprise)
                            ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))  # 0
    M.append(pcm.FixedModel('finger', G_fingerID))  # 1
    M.append(pcm.FixedModel('cue', G_cue))  # 2
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 3
    M.append(pcm.FixedModel('surprise', G_surprise))  # 5
    M.append(pcm.ComponentModel('component', G_component))  # 6
    M.append(pcm.FeatureModel('feature', Ac))  # 7
    M.append(pcm.FreeModel('ceil', 8))  # 8

    return M


def make_planning_models(experiment, test_planning_force=True):
    # C = pcm.centering(5)

    v_cue = np.array([1, 2, 3, 4, 5])
    v_cert = np.array([0, 0.1875, .25, 0.1875, 0])

    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)

    path_G_force = os.path.join(gl.baseDir, experiment, gl.pcmDir, 'G_obs.force.plan.npy')
    if test_planning_force:
        G_force = np.load(os.path.join(gl.baseDir, experiment, gl.pcmDir, 'G_obs.force.plan.npy'))

    M = []
    M.append(pcm.FixedModel('null', np.zeros((5, 5))))  # 0
    M.append(pcm.FixedModel('cue', G_cue))  # 1
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 2
    M.append(pcm.FixedModel('equal distance', np.eye(5)))
    if test_planning_force:
        M.append(pcm.FixedModel('planning force', G_force.mean(axis=0)))
    M.append(pcm.ComponentModel('component', np.array([G_cue / np.trace(G_cue),
                                                       G_cert / np.trace(G_cert),])))  # 4
    M.append(pcm.FreeModel('ceil', 5))  # 5

    return M


class Tessellation():
    def __init__(self, experiment=None, participants_id=None, glm=None, M=None, reg_interest=None, reg_mapping=None,
                 n_tessels=None, n_jobs=None):
        self.experiment = experiment
        self.participants_id = participants_id
        self.glm = glm
        self.M = M
        self.col_names = [m.name for m in self.M]
        self.reg_interest = reg_interest
        self.reg_mapping = reg_mapping
        self.n_tessels = n_tessels
        self.n_jobs = n_jobs

        # define atlas
        self.atlas, _ = am.get_atlas('fs32k')
        self.path_tessel_atlas = {
            'L': os.path.join(gl.atlas_dir, f'Icosahedron{self.n_tessels}.L.label.gii'),
            'R': os.path.join(gl.atlas_dir, f'Icosahedron{self.n_tessels}.R.label.gii')
        }

        # define structures
        self.struct = ['CortexLeft', 'CortexRight']

        # define hemispheres
        self.Hem = ['L', 'R']

        # init results
        self.results = {
            'L': None,
            'R': None,
        }

    def _make_individ_dataset(self, H, subatlas, sn):

        # define path to glm
        glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{self.glm}', f'subj{sn}')

        # define path to surfaces
        surf_path = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{sn}')

        # retrieve surfaces
        white = os.path.join(surf_path, f'subj{sn}.{H}.white.32k.surf.gii')
        pial = os.path.join(surf_path, f'subj{sn}.{H}.pial.32k.surf.gii')

        # define glm mask
        mask = os.path.join(glm_path, 'mask.nii')

        # Build atlas mapping
        amap = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
        amap.build()

        # load betas from cifti
        cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))

        # extract betas
        beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

        # extract reginfo from cifti. When building the cifti, scalar axis must contain "condition_label.part_label"
        reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
        part_vec = np.array([int(r[1]) for r in reginfo])
        cond_vec = np.array([r[0] for r in reginfo])

        # Optional: use different regressor name, e.g., a number for ordering purposes
        if self.reg_mapping is not None:
            cond_vec = np.vectorize(self.reg_mapping.get)(cond_vec)

        # Optional: restrict to some regressors, use the new mapped names
        if self.reg_interest is not None:
            idx = np.isin(cond_vec, self.reg_interest)

        # Define obs_des to include in dataset descriptors
        obs_des = {'cond_vec': cond_vec[idx],
                   'part_vec': part_vec[idx]}

        # load residuals
        res = nb.load(os.path.join(glm_path, 'ResMS.nii'))

        betas = amap.extract_data_native([beta_img])
        res = amap.extract_data_native([res])

        # Prewhiten betas
        betas_prewhitened = betas / np.sqrt(res)

        # remove nans
        betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

        return pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des)


    def _fit_model_in_tessel(self, H, subatlas):
        Y = list()
        n_voxels = list()
        for s, sn in enumerate(self.participants_id):
            Dataset = self._make_individ_dataset(H, subatlas, sn)
            n_voxels.append(Dataset.n_channel)
            Y.append(Dataset)

        try:
            T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
            T_gr, _ = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')

            # for i in range(len(theta_cv)):
            #     n_param = self.M[i].n_param
            #     theta_cv[i] = theta_cv[i][:n_param] / np.linalg.norm(theta_cv[i][:n_param])

            likelihood = T_cv.likelihood
            baseline = likelihood.loc[:, 'null'].values
            likelihood = likelihood - baseline.reshape(-1, 1)
            noise_upper = (T_gr.likelihood['ceil'] - baseline)
            noise_lower = likelihood.ceil

        except Exception as e:
            print(f"Error in tessel: {e}")
            n_cols = len(self.col_names)
            n_subj = len(self.participants_id)
            likelihood = {col: np.full(n_subj, np.nan) for col in self.col_names}
            noise_upper = np.full(n_subj, np.nan)
            noise_lower = np.full(n_subj, np.nan)
            baseline = np.full(n_subj, np.nan)
            theta_cv = [np.full((m.n_param, n_subj), np.nan) for m in self.M]

        return likelihood, noise_upper, noise_lower, baseline, theta_cv, n_voxels


    def make_subatlas_tessel(self, H, ntessel):
        print(f'Hemisphere: {H}, tessel #{ntessel}\t')
        atlas_hem = self.atlas.get_hemisphere(self.Hem.index(H))
        subatlas = atlas_hem.get_subatlas_image(self.path_tessel_atlas[H], ntessel)
        return subatlas

    def _store_T_and_theta_from_tessel(self, H, ntessel):

        subatlas = self.make_subatlas_tessel(H, ntessel)

        T = {
            'likelihood': [],
            'noise_upper': [],
            'noise_lower': [],
            'baseline': [],
            'n_voxels': [],
            'col_names': [],
            'sn': []
        }

        theta = {}
        for md in self.M:
            if md.n_param > 0: # skip models with 0 params i.e. Fixed Models
                theta[md.name] = {
                    'theta': [],
                    '#param': [],
                    'sn': []
                }

        likelihood, noise_upper, noise_lower, baseline, theta_cv, n_voxels = self._fit_model_in_tessel(H, subatlas)

        for s, sn in enumerate(self.participants_id):
            for c, col in enumerate(self.col_names):
                T['likelihood'].append(likelihood[col][s])
                T['noise_upper'].append(noise_upper[s])
                T['noise_lower'].append(noise_lower[s])
                T['baseline'].append(baseline[s])
                T['n_voxels'].append(n_voxels[s])
                T['col_names'].append(col)
                T['sn'].append(sn)
            for m, md in enumerate(self.M):
                if md.n_param > 0:
                    for c in range(md.n_param):
                        theta[md.name]['theta'].append(theta_cv[m][c, s])
                        theta[md.name]['sn'].append(sn)
                        theta[md.name]['#param'].append(c)

        T = pd.DataFrame(T)
        for md in self.M:
            if md.n_param > 0:
                theta[md.name] = pd.DataFrame(theta[md.name])

        return T, theta, subatlas.vertex[0]


    def do_parallel_pcm_in_tessels(self):
        for H in self.Hem:

            # Parallel processing of tessels
            with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
                self.results[H] = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._store_T_and_theta_from_tessel)(H, ntessel)
                    for ntessel in range(self.n_tessels)
                )

            # # Serial processing of tessels
            # for ntessel in range(5):
            #     self.results[H] = self._store_T_and_theta_from_tessel(H, ntessel)


    def _extract_results_from_parallel_process(self, H, sn):
        results = self.results[H]

        # Aggregate results from parallel processes
        T = np.full((32492, len(self.M) + 4), np.nan)
        theta = {}
        for md in self.M:
            theta[md.name] = np.full((32492, md.n_param), np.nan)

        for Tt, th, vertex_id in results:
            if len(vertex_id)>0 :
                for c, col in enumerate(self.col_names):
                    LL = Tt[(Tt['sn'] == sn) & (Tt['col_names'] == col)]['likelihood']
                    T[vertex_id, c] = LL
                T[vertex_id, -4] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
                T[vertex_id, -3] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
                T[vertex_id, -2] = Tt[(Tt['sn'] == sn)]['baseline'].unique()
                T[vertex_id, -1] = Tt[(Tt['sn'] == sn)]['n_voxels'].unique()
                for md in self.M:
                    for c in range(md.n_param):
                        theta_tmp = th[md.name]
                        theta[md.name][vertex_id, c] = theta_tmp['theta'][
                            (theta_tmp['sn'] == sn) & (theta_tmp['#param'] == c)]

        return T, theta

    def make_group_giftis_likelihood(self, H):
        T = []
        column_names = self.col_names + ['noise_upper', 'noise_lower', 'baseline', 'n_voxels']
        for sn in self.participants_id:
            Tt, _ = self._extract_results_from_parallel_process(H, sn)
            T.append(Tt)
        T = np.array(T).mean(axis=0)
        gifti_img_T = nt.make_func_gifti(T,
                                         anatomical_struct=self.struct[self.Hem.index(H)],
                                         column_names=column_names, )

        return gifti_img_T

    def make_group_giftis_theta(self, H, model):
        theta = []
        for sn in self.participants_id:
            _, th = self._extract_results_from_parallel_process(H, sn)
            theta_tmp = th[model]
            theta.append(theta_tmp)

        theta = np.array(theta).mean(axis=0)
        column_names = [f'param #{n+1}' for n in range(theta.shape[1])]
        gifti_img_theta = nt.make_func_gifti(theta,
                                             anatomical_struct=self.struct[self.Hem.index(H)],
                                             column_names=column_names)

        return gifti_img_theta

    def make_group_cifti_likelihood(self):
        giftis = []
        for H in self.Hem:
            giftis.append(self.make_group_giftis_likelihood(H))

        return nt.join_giftis_to_cifti(giftis)

    def make_group_cifti_theta(self, model):
        giftis = []
        for H in self.Hem:
            giftis.append(self.make_group_giftis_theta(H, model))

        return nt.join_giftis_to_cifti(giftis)


def main(args):

    if args.what == 'save_tessel_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,  f'M.exec.glm{args.glm}.pkl'), "wb")
        pickle.dump(M, f)
        Tess = Tessellation(args.experiment,
                            args.snS,
                            args.glm,
                            M,
                            [5, 6, 7, 8, 9, 10, 11, 12],
                            gl.regressor_mapping,
                            args.n_tessels,
                            args.n_jobs)
        Tess.do_parallel_pcm_in_tessels()
        cifti_T = Tess.make_group_cifti_likelihood()
        nb.save(cifti_T, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'ML.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))
        cifti_theta_component = Tess.make_group_cifti_theta('component')
        nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))
        cifti_theta_feature = Tess.make_group_cifti_theta('feature')
        nb.save(cifti_theta_feature, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_feature.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))

    if args.what == 'save_tessel_planning':
        M = make_planning_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Tess = Tessellation(args.experiment,
                            args.snS,
                            args.glm,
                            M,
                            [0, 1, 2, 3, 4],
                            gl.regressor_mapping,
                            args.n_tessels,
                            args.n_jobs)
        Tess.do_parallel_pcm_in_tessels()
        cifti_T = Tess.make_group_cifti_likelihood()
        nb.save(cifti_T, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'ML.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.plan.dscalar.nii'))
        cifti_theta_component = Tess.make_group_cifti_theta('component')
        nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                    f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.plan.dscalar.nii'))

    if args.what == 'save_emg_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.pkl'), "wb")
        pickle.dump(M, f)

        # snS = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110]

        N = len(args.snS)

        for epoch in args.epochs:
            G_obs = np.zeros((N, 8, 8))
            Y = list()
            for s, sn in enumerate(args.snS):
                npz = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}',
                                           f'{args.experiment}_{sn}_binned.npz'), allow_pickle=True)

                descr = npz['descriptor'].item()
                timepoints = list(descr['time windows'].keys())

                emg = npz['data_array'][timepoints.index(epoch)]

                dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                               f'{args.experiment}_{sn}.dat'), sep='\t')
                dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
                dat['cue'] = dat['cue'].map(gl.cue_mapping)
                dat['BN'] = dat['BN'].astype(str)

                cov = emg.T @ emg

                emg = emg / np.sqrt(np.diag(cov))

                dat[['ch_' + str(x) for x in range(emg.shape[-1])]] = emg

                dat = dat.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
                cond_vec = dat['cue'] + ',' + dat['stimFinger']
                part_vec = dat['BN']

                obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
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

            np.save(os.path.join(path, f'G_obs.emg.{epoch}.npy'), G_obs)

            T_in.to_pickle(os.path.join(path, f'T_in.emg.{epoch}.pkl'))
            T_cv.to_pickle(os.path.join(path, f'T_cv.emg.{epoch}.pkl'))
            T_gr.to_pickle(os.path.join(path, f'T_gr.emg.{epoch}.pkl'))

            with open(os.path.join(path, f'theta_in.emg.{epoch}.pkl'), 'wb') as f:
                pickle.dump(theta_in, f)
            with open(os.path.join(path, f'theta_cv.emg.{epoch}.pkl'), 'wb') as f:
                pickle.dump(theta_cv, f)
            with open(os.path.join(path, f'theta_gr.emg.{epoch}.pkl'), 'wb') as f:
                pickle.dump(theta_gr, f)

    if args.what == 'save_force_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.force.exec.p'), "wb")
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

            force = force[['thumb1', 'index1', 'middle1', 'ring1', 'pinkie1']].to_numpy()

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

        np.save(os.path.join(path, f'G_obs.force.exec.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.force.exec.p'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.force.exec.p'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.force.exec.p'))

        with open(os.path.join(path, f'theta_in.force.exec.p'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.force.exec.p'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.force.exec.p'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'save_force_planning':

        M = make_planning_models(test_planning_force=False)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.force.plan.p'), "wb")
        pickle.dump(M, f)

        N = len(args.snS)

        G_obs = np.zeros((N, 5, 5))
        Y = list()
        for s, sn in enumerate(args.snS):
            force = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                             f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
            #force = force[force['GoNogo'] == 'nogo'] if 'GoNogo' in force else force # select only go trial
            force['cue'] = force['cue'].map(gl.cue_mapping)
            # force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
            force = force.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = force['cue'] #+ ',' + force['stimFinger']
            part_vec = force['BN']

            force = force[['thumb0', 'index0', 'middle0', 'ring0', 'pinkie0']].to_numpy()

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

        np.save(os.path.join(path, f'G_obs.force.plan.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.force.plan.p'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.force.plan.p'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.force.plan.p'))

        with open(os.path.join(path, f'theta_in.force.plan.p'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.force.plan.p'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.force.plan.p'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'save_rois_planning':

        M = make_planning_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.pkl'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)

                G_obs = np.zeros((N, 5, 5))
                Y = list()
                for s, sn in enumerate(args.snS):
                    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}')
                    cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))
                    beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

                    mask = nb.load(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                    coords = nt.get_mask_coords(mask)

                    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

                    res_img = nb.load(os.path.join(glm_path, 'ResMS.nii'))
                    res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

                    betas_prewhitened = betas / np.sqrt(res)
                    betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])

                    idx = np.isin(cond_vec, [0, 1, 2, 3, 4])

                    obs_des = {'cond_vec': cond_vec[idx],
                               'part_vec': part_vec[idx]}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

                    G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                     Y[s].obs_descriptors['part_vec'],
                                                     X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
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

                f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_in.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                pickle.dump(theta_in, f)

                f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_cv.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                pickle.dump(theta_cv, f)

                f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_gr.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                pickle.dump(theta_gr, f)

    if args.what == 'save_rois_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.exec.glm{args.glm}.pkl'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)

                G_obs = np.zeros((N, 8, 8))
                Y = list()
                for s, sn in enumerate(args.snS):
                    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}')
                    cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))
                    beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

                    mask = nb.load(
                        os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{sn}', f'ROI.{H}.{roi}.nii'))
                    coords = nt.get_mask_coords(mask)

                    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

                    res_img = nb.load(os.path.join(glm_path, 'ResMS.nii'))
                    res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

                    betas_prewhitened = betas / np.sqrt(res)
                    betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])

                    idx = np.isin(cond_vec, [5, 6, 7, 8, 9, 10, 11, 12])

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
    parser.add_argument('--epochs', nargs='+', type=str, default=['SLR', 'LLR', 'Vol'])
    parser.add_argument('--n_tessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')