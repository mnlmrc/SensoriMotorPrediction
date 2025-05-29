import warnings

import time

from globals import regressor_mapping

warnings.filterwarnings("ignore")

import argparse
import pickle
from itertools import combinations
import rsatoolbox as rsa
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rdms import D_to_rdm

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

def make_execution_models(centering=True, normalize=True):

    C = pcm.centering(8)

    if centering:
        v_fingerID = C @ np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = C @ np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
        v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = C @ -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information
    else:
        v_fingerID = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
        v_cert = np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information

    Ac = np.zeros((6, 8, 5))
    Ac[0, :, 0] = v_fingerID
    Ac[1, :, 1] = v_cue
    Ac[2, :, 1] = v_fingerID
    Ac[3, :, 2] = v_cue
    Ac[4, :, 3] = v_cert
    Ac[5, :, 4] = v_surprise

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
    M.append(pcm.FixedModel('surprise', G_surprise))  # 4
    M.append(pcm.ComponentModel('component', G_component))  # 5
    M.append(pcm.FeatureModel('feature', Ac))  # 6
    M.append(pcm.FreeModel('ceil', 8))  # 7

    return M

def make_planning_models(centering=True):
    C = pcm.centering(5)

    if centering:
        v_cue = C @ np.array([-1, -.5, 0, .5, 1])
        v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0])
        # v_shift = C @ np.array([0, 1, 0, -1, 0])
    else:
        v_cue = np.array([-1, -.5, 0, .5, 1])
        v_cert = np.array([0, 0.1875, .25, 0.1875, 0])

    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    # G_shift = np.outer(v_shift, v_shift)

    M = []
    M.append(pcm.FixedModel('null', np.zeros((5, 5))))  # 0
    M.append(pcm.FixedModel('cue', G_cue))  # 1
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 2
    # M.append(pcm.FixedModel('shift probability', G_shift))
    # M.append(pcm.FixedModel('equal distance', np.eye(5)))
    M.append(pcm.ComponentModel('component', np.array([G_cue / np.trace(G_cue),
                                                       G_cert / np.trace(G_cert),
                                                       # G_shift / np.trace(G_shift),
                                                       # np.eye(5) / 5
                                                      ])))  # 4
    M.append(pcm.FreeModel('ceil', 5))  # 5

    return M

def G_obs_in_timepoint(Data, dat, timepoint):
    Y = list()
    N = len(Data)
    G_obs = np.zeros((N, 8, 8))
    for s, (D, dd) in enumerate(zip(Data, dat)):
        emg = D[..., timepoint]

        cov = emg.T @ emg

        # emg = emg / np.sqrt(np.diag(cov))

        dd[['ch_' + str(x) for x in range(emg.shape[-1])]] = emg

        dd = dd.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()

        cond_vec = dd['cue'] + ',' + dd['stimFinger']
        part_vec = dd['BN']

        obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
                   'part_vec': part_vec}

        Y.append(pcm.dataset.Dataset(dd[['ch_' + str(x) for x in range(emg.shape[-1])]].to_numpy(),
                                     obs_descriptors=obs_des))

        G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                         Y[s].obs_descriptors['part_vec'],
                                         X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))
        # tr = np.trace(G)
        # Y[s].measurements /= np.sqrt(np.abs(tr))

    return G_obs, Y

# need to become a class here:
def fit_model_in_timepoint(M, Data, dat, timepoint):
    G_obs, Y = G_obs_in_timepoint(Data, dat, timepoint)

    T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=False, verbose=True, fixed_effect='block')
    # T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
    # T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

    # likelihood = T_cv.likelihood
    # baseline = likelihood.loc[:, 'null'].values
    # likelihood = likelihood - baseline.reshape(-1, 1)
    # noise_upper = (T_gr.likelihood['ceil'] - baseline)
    # noise_lower = likelihood.ceil

    return theta_in

def store_T_and_theta_from_timepoint(M, Data, dat, timepoint):
    col_names = [m.name for m in M]

    # T = {
    #     'likelihood': [],
    #     'noise_upper': [],
    #     'noise_lower': [],
    #     'baseline': [],
    #     'col_names': [],
    #     'sn': []
    # }

    theta = {}
    for md in M:
        if md.n_param > 0: # skip models with 0 params i.e. Fixed Models
            theta[md.name] = {
                'theta': [],
                '#param': [],
                'sn': []
            }

    theta_in = fit_model_in_timepoint(M, Data, dat, timepoint)

    for s, sn in enumerate(args.snS):
        # for c, col in enumerate(col_names):
            # T['likelihood'].append(likelihood[col][s])
            # # T['noise_upper'].append(noise_upper[s])
            # # T['noise_lower'].append(noise_lower[s])
            # T['baseline'].append(baseline[s])
            # T['col_names'].append(col)
            # T['sn'].append(sn)
        for m, md in enumerate(M):
            if md.n_param > 0:
                for c in range(md.n_param):
                    theta[md.name]['theta'].append(theta_in[m][c, s])
                    theta[md.name]['sn'].append(sn)
                    theta[md.name]['#param'].append(c)

    # T = pd.DataFrame(T)
    for md in M:
        if md.n_param > 0:
            theta[md.name] = pd.DataFrame(theta[md.name])

    return theta, timepoint

def extract_results_from_parallel_process(M, results, sn):
    col_names = [m.name for m in M]

    # Aggregate results from parallel processes
    # T = np.full((6444, len(M) + 3), np.nan)
    theta = {}
    for md in M:
        theta[md.name] = np.full((6444, md.n_param), np.nan)

    for th, tp in results:
        # for c, col in enumerate(col_names):
        #     LL = Tt[(Tt['sn'] == sn) & (Tt['col_names'] == col)]['likelihood']
        #     T[tp, c] = LL
        # T[tp, -3] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
        # T[tp, -2] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
        # T[tp, -1] = Tt[(Tt['sn'] == sn)]['baseline'].unique()
        for md in M:
            for c in range(md.n_param):
                theta_tmp = th[md.name]
                theta[md.name][tp, c] = theta_tmp['theta'][
                    (theta_tmp['sn'] == sn) & (theta_tmp['#param'] == c)]

    return theta

def calc_crossvalidated_fits(G_obs, model, theta, model_name):
    """

    Args:
        G_obs: path to G_obs.npy
        model: path to model.pkl
        theta: path to theta.pkl
        model_name (str): model name among those listed in model

    Returns:

    """
    Mf, idxf = find_model(model, model_name)

    f = open(theta, "rb")
    theta = pickle.load(f)
    theta_f = theta[idxf][:Mf.n_param]

    G_obs = np.load(G_obs)

    # calculate predicted
    G_hat = []
    for i in range(theta_f.shape[1]):
        G_hat_tmp, _ = Mf.predict(theta_f[:, i])
        G_hat.append(G_hat_tmp)
    G_hat = np.array(G_hat)

    # calculate fit
    corr = np.zeros(G_hat.shape[0])
    for i in range(G_hat.shape[0]):
        # Average G_hat across all folds except i
        G_hat_mean = np.mean(np.delete(G_hat, i, axis=0), axis=0)

        D_hat_mean = pcm.G_to_dist(G_hat_mean)
        D_obs = pcm.G_to_dist(G_obs[i])

        rdm_hat = D_to_rdm(D_hat_mean, pattern_descriptors={'conds':[j for j in range(D_hat_mean.shape[0])]})
        rdm_obs = D_to_rdm(D_obs, pattern_descriptors={'conds': [j for j in range(D_obs.shape[0])]})

        corr[i] = rsa.rdm.compare(rdm_hat, rdm_obs, method='cosine')


    # calculate noise lower ceiling
    noise_lower = np.zeros(G_obs.shape[0])
    for i in range(G_obs.shape[0]):
        # Average G_hat across all folds except i
        G_obs_mean = np.mean(np.delete(G_obs, i, axis=0), axis=0)

        D_obs_mean = pcm.G_to_dist(G_obs_mean)
        D_obs = pcm.G_to_dist(G_obs[i])

        rdm_obs_mean = D_to_rdm(D_obs_mean, pattern_descriptors={'conds': [j for j in range(D_obs_mean.shape[0])]})
        rdm_obs = D_to_rdm(D_obs, pattern_descriptors={'conds': [j for j in range(D_obs.shape[0])]})

        noise_lower[i] = rsa.rdm.compare(rdm_obs, rdm_obs_mean, method='cosine')

    # calculate noise upper ceiling
    noise_upper = np.zeros(G_obs.shape[0])
    for i in range(G_obs.shape[0]):
        # Average G_hat across all folds except i
        G_obs_mean = np.mean(G_obs, axis=0)

        D_obs_mean = pcm.G_to_dist(G_obs_mean)
        D_obs = pcm.G_to_dist(G_obs[i])

        rdm_obs_mean = D_to_rdm(D_obs_mean, pattern_descriptors={'conds': [j for j in range(D_obs_mean.shape[0])]})
        rdm_obs = D_to_rdm(D_obs, pattern_descriptors={'conds': [j for j in range(D_obs.shape[0])]})

        noise_upper[i] = rsa.rdm.compare(rdm_obs, rdm_obs_mean, method='cosine')

    return corr, noise_upper, noise_lower


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
            'L': os.path.join(gl.atlasDir, f'Icosahedron{self.n_tessels}.L.label.gii'),
            'R': os.path.join(gl.atlasDir, f'Icosahedron{self.n_tessels}.R.label.gii')
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
        res = amap.extract_data_native([res]).squeeze()

        # Replace near-zero values with np.nan
        tol = 1e-6
        betas[:, np.isclose(res, 0, atol=tol)] = np.nan
        res[np.isclose(res, 0, atol=tol)] = np.nan

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
            T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')

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
            # theta_cv = [np.full((m.n_param, n_subj), np.nan) for m in self.M]
            theta_gr = [np.full((m.n_param + n_subj), np.nan) for m in self.M]

        return likelihood, noise_upper, noise_lower, baseline, theta_gr, n_voxels


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
                    # 'sn': []
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
                    theta[md.name]['theta'].append(theta_cv[m][c])
                    # theta[md.name]['sn'].append(sn)
                    theta[md.name]['#param'].append(c)

        T = pd.DataFrame(T)
        for md in self.M:
            if md.n_param > 0:
                theta[md.name] = pd.DataFrame(theta[md.name])

        return T, theta, subatlas.vertex[0]


    def run_parallel_pcm_across_tessels(self):
        for H in self.Hem:

            # Parallel processing of tessels
            with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
                self.results[H] = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._store_T_and_theta_from_tessel)(H, ntessel)
                    for ntessel in range(self.n_tessels)
                )

            # # Serial processing of tessels
            # for ntessel in range(1, 2):
            #     self.results[H] = self._store_T_and_theta_from_tessel(H, ntessel)


    def _extract_results_from_parallel_process(self, H, sn=None):
        results = self.results[H]

        # Aggregate results from parallel processes
        T = np.full((32492, len(self.M) + 4), np.nan)
        theta = {}
        for md in self.M:
            theta[md.name] = np.full((32492, md.n_param), np.nan)

        if sn is not None:
            for Tt, _, vertex_id in results:
                if len(vertex_id)>0 :
                    for c, col in enumerate(self.col_names):
                        LL = Tt[(Tt['sn'] == sn) & (Tt['col_names'] == col)]['likelihood']
                        T[vertex_id, c] = LL
                    T[vertex_id, -4] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
                    T[vertex_id, -3] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
                    T[vertex_id, -2] = Tt[(Tt['sn'] == sn)]['baseline'].unique()
                    T[vertex_id, -1] = Tt[(Tt['sn'] == sn)]['n_voxels'].unique()
        else:
            for _, th, vertex_id in results:
                for md in self.M:
                    for c in range(md.n_param):
                        theta_tmp = th[md.name]
                        theta[md.name][vertex_id, c] = theta_tmp['theta'][theta_tmp['#param'] == c]

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
        # for sn in self.participants_id:
        _, th = self._extract_results_from_parallel_process(H)
        theta_tmp = th[model]
        # theta.append(theta_tmp)
        theta = th[model]

        # theta = np.array(theta).mean(axis=0)
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

class Rois():
    def __init__(self, snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=None, regr_of_interest=None,
                 n_jobs=16):
        self.snS = snS  # participants numbers
        self.M = M  # pcm models to fit
        self.glm_path = glm_path  # path to cifti_img
        self.cifti_img = cifti_img  # name of cifti_img
        self.roi_path = roi_path  # path to individual roi masks, which must be named <atlas_name>.<H>.<roi>.nii
        self.roi_imgs = roi_imgs  # name of roi files to use as masks, e.g. ROI.L.M1.nii or cerebellum.L.nii
        self.regressor_mapping = regressor_mapping  # dict, maps name of regressors to numbers to control in which order conditions appear in the G matrix
        self.regr_of_interest = regr_of_interest  # indexes from regressor mapping of the regressors we want to include in the analysis
        self.n_jobs = n_jobs

    def _make_roi_dataset(self, roi_img):
        N = len(self.snS)

        G_obs = np.zeros((N, len(self.regr_of_interest), len(self.regr_of_interest)))
        Y = list()
        for s, sn in enumerate(self.snS):
            cifti_img = nb.load(os.path.join(self.glm_path, f'subj{sn}',self.cifti_img))
            beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

            mask = nb.load(os.path.join(self.roi_path, f'subj{sn}', roi_img))
            coords = nt.get_mask_coords(mask)

            betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

            res_img = nb.load(os.path.join(self.glm_path, f'subj{sn}','ResMS.nii'))
            res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

            # Replace near-zero values with np.nan
            tol = 1e-6
            betas[:, np.isclose(res, 0, atol=tol)] = np.nan
            res[np.isclose(res, 0, atol=tol)] = np.nan

            betas_prewhitened = betas / np.sqrt(res)
            betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

            reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
            cond_vec = np.array([self.regressor_mapping[r[0]] for r in reginfo])
            part_vec = np.array([int(r[1]) for r in reginfo])

            idx = np.isin(cond_vec, self.regr_of_interest)

            obs_des = {'cond_vec': cond_vec[idx],
                       'part_vec': part_vec[idx]}

            Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                             Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

            # tr = np.trace(pcm.make_pd(G_obs[s]))
            # Y[s].measurements = Y[s].measurements / np.sqrt(tr)

        return Y, G_obs

    def _fit_model_to_dataset(self, Y):
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')

        return T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_pcm_in_roi(self, roi_img):
        Y, G_obs = self._make_roi_dataset(roi_img)
        T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = self._fit_model_to_dataset(Y)

        return G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_parallel_pcm_across_rois(self):
        ##Parallel processing of rois
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_pcm_in_roi)(roi)
                for roi in self.roi_imgs
            )

        # for roi in self.roi_imgs:
        #     self.run_pcm_in_roi(roi)

        results = self._extract_results_from_parallel_process(results,
                                      field_names=['G_obs', 'T_in', 'theta_in', 'T_cv', 'theta_cv', 'T_gr', 'theta_gr'])
        return results

    def fit_model_family_across_rois(self, model, basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_roi)(roi, model, basecomp, comp_names)
                for roi in self.roi_imgs
            )
        results = self._extract_results_from_parallel_process(results, ['T', 'theta'])
        return results

    def fit_model_family_in_roi(self, roi_img, model, basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
        MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        Y, _ = self._make_roi_dataset(roi_img)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)

        return T, theta

    def _extract_results_from_parallel_process(self, results, field_names):
        res_dict = {key: [] for key in ['roi_img'] + field_names}
        for r, result in enumerate(results):
            if len(result) != len(field_names):
                raise ValueError(f"Expected {len(field_names)} values, got {len(result)} at index {r}")
            res_dict['roi_img'].append(self.roi_imgs[r])
            for key, value in zip(field_names, result):
                res_dict[key].append(value)
        return res_dict

def main(args):

    if args.what == 'tessel_execution':

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
        Tess.run_parallel_pcm_across_tessels()
        cifti_T = Tess.make_group_cifti_likelihood()
        nb.save(cifti_T, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'ML.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))
        cifti_theta_component = Tess.make_group_cifti_theta('component')
        nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))
        cifti_theta_feature = Tess.make_group_cifti_theta('feature')
        nb.save(cifti_theta_feature, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_feature.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.exec.dscalar.nii'))

    if args.what == 'tessel_planning':
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
        # subatlas = Tess.make_subatlas_tessel('L', 1)
        # Tess._make_individ_dataset('L', subatlas, 102)
        Tess.run_parallel_pcm_across_tessels()
        cifti_T = Tess.make_group_cifti_likelihood()
        nb.save(cifti_T, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'ML.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.plan.dscalar.nii'))
        cifti_theta_component = Tess.make_group_cifti_theta('component')
        nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                    f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.plan.dscalar.nii'))

    if args.what == 'emg_G_obs_continuous':
        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :2148].mean(axis=-1, keepdims=True)
            # emg_norm = emg_rect / bs
            emg.append(emg_rect)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                                f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        G_obs = []
        for tp in range(6444):
            print(f'tp: {tp}/{6444}')
            G_obs_tmp, _ = G_obs_in_timepoint(emg, dat, tp)
            G_obs.append(G_obs_tmp)

        np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.emg.continuous.npy'), np.array(G_obs))

    if args.what == 'emg_continuous':

        M = make_execution_models()
        col_names = [m.name for m in M]

        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[...,:2148].mean(axis=-1, keepdims=True)
            # emg_norm = emg_rect / bs
            emg.append(emg_rect)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                               f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        # fit_model_in_timepoint(M, emg, dat, 0)
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(store_T_and_theta_from_timepoint)(M, emg, dat, tp)
                for tp in range(6444)
            )

        # for tp in range(6444):
        #     store_T_and_theta_from_timepoint(M, emg, dat, tp)

        theta = []
        for sn in args.snS:
            th = extract_results_from_parallel_process(M, results, sn)
            # T.append(Tt)
            theta.append(th['feature'])

        # T = np.array(T)
        theta = np.array(theta)

        # np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_cv.emg.continuous.npy'), T)
        np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_in.emg.continuous.npy'), theta)

    if args.what == 'emg_binned':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.pkl'), "wb")
        pickle.dump(M, f)

        wins = [(-1.0, 0.0), (.025, .05), (.05, .1), (.1, .5)]
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']

        fs = 2148
        latency = .05 * fs
        onset = int(1 * fs + latency)

        # snS = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110]

        N = len(args.snS)

        for epoch, win in zip(epochs, wins):
            G_obs = np.zeros((N, 8, 8))
            Y = list()
            for s, sn in enumerate(args.snS):
                emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
                emg_rect = np.abs(emg_raw)
                bs = emg_rect[..., :2148].mean(axis=-1, keepdims=True)
                emg_norm = emg_rect / bs
                emg = emg_norm[..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=-1)

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
            T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=False, verbose=True, fixed_effect='block')
            T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=False, verbose=True, fixed_effect='block')

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

    if args.what == 'emg_cv_fits':

        corr = {
            'corr': [],
            'noise_lower': [],
            'noise_upper': [],
            'sn': [],
            'epoch': [],
        }

        name = 'feature'
        for epoch in args.epochs:
            G_obs = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.emg.{epoch}.npy')
            model = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.pkl')
            theta = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_in.emg.{epoch}.pkl')
            corr_tmp, noise_upper_tmp,noise_lower_tmp = calc_crossvalidated_fits(G_obs, model, theta, name)

            corr['corr'].extend(corr_tmp)
            corr['noise_upper'].extend(noise_upper_tmp)
            corr['noise_lower'].extend(noise_lower_tmp)
            corr['sn'].extend(args.snS)
            corr['epoch'].extend([epoch] * len(corr_tmp))

        corr = pd.DataFrame(corr)
        corr.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'emg.cv_fits.tsv'), index=False, sep='\t')

    if args.what == 'roi_cv_fits':

        corr = {
            'corr': [],
            'noise_lower': [],
            'noise_upper': [],
            'sn': [],
            'roi': [],
            'Hem': [],
            'epoch': []
        }

        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1' ]
        Hem = ['L', 'R']
        epochs = ['plan', 'exec']

        for epoch in epochs:
            if epoch=='plan':
                name = 'component'
            elif epoch=='exec':
                name = 'feature'
            for H in Hem:
                for roi in rois:
                    G_obs = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.{epoch}.glm{args.glm}.{H}.{roi}.npy')
                    model = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{epoch}.glm{args.glm}.pkl')
                    theta = os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_in.{epoch}.glm{args.glm}.{H}.{roi}.pkl')
                    corr_tmp, noise_upper_tmp,noise_lower_tmp = calc_crossvalidated_fits(G_obs, model, theta, name)

                    corr['corr'].extend(corr_tmp)
                    corr['noise_upper'].extend(noise_upper_tmp)
                    corr['noise_lower'].extend(noise_lower_tmp)
                    corr['sn'].extend(args.snS)
                    corr['roi'].extend([roi] * len(corr_tmp))
                    corr['Hem'].extend([H] * len(corr_tmp))
                    corr['epoch'].extend([epoch] * len(corr_tmp))

        corr = pd.DataFrame(corr)
        corr.to_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'cv_fits.glm{args.glm}.tsv'), index=False, sep='\t')

    if args.what == 'force_execution':

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

    if args.what == 'force_planning':

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

    if args.what == 'rois_planning':

        M = make_planning_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)

        R = Rois(args.snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=gl.regressor_mapping,
                 regr_of_interest=[0, 1, 2, 3, 4])
        # res = R.run_pcm_in_roi(roi_imgs[0])
        res = R.run_parallel_pcm_across_rois()

        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T_in'][r].to_pickle(os.path.join(path, f'T_in.plan.glm{args.glm}.{H}.{roi}.p'))
                res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.plan.glm{args.glm}.{H}.{roi}.p'))
                res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.plan.glm{args.glm}.{H}.{roi}.p'))

                np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])

                f = open(os.path.join(path, f'theta_in.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_in'][r], f)
                f = open(os.path.join(path, f'theta_cv.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_cv'][r], f)
                f = open(os.path.join(path, f'theta_gr.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_gr'][r], f)

    if args.what == 'model_family_rois_planning':
        M = make_planning_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)

        R = Rois(args.snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=gl.regressor_mapping,
                 regr_of_interest=[0, 1, 2, 3, 4])
        res = R.fit_model_family_across_rois('component', comp_names=['cue', 'uncertainty'])

        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T'][r].to_pickle(os.path.join(path, f'T.model_family.plan.glm{args.glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta'][r], f)

    if args.what == 'model_family_rois_execution':
        M = make_execution_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)

        R = Rois(args.snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=gl.regressor_mapping,
                 regr_of_interest=[5, 6, 7, 8, 9, 10, 11, 12])
        # R.fit_model_family_in_roi(roi_imgs[0], 'component',
        #                                      basecomp=np.eye(8)[None, :, :], #
        #                                      comp_names=['finger', 'cue', 'uncertainty', 'surprise'])
        res = R.fit_model_family_across_rois('component',
                                             basecomp=np.eye(8)[None, :, :], # basecomp needs to be num_basecompxNxN
                                             comp_names=['finger', 'cue', 'uncertainty', 'surprise'])

        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T'][r].to_pickle(os.path.join(path, f'T.model_family.exec.glm{args.glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta'][r], f)

    if args.what == 'rois_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)

        R = Rois(args.snS, M, glm_path, cifti_img, roi_path, roi_imgs, regressor_mapping=gl.regressor_mapping,
                 regr_of_interest=[5, 6, 7, 8, 9, 10, 11, 12])
        res = R.run_parallel_pcm_across_rois()

        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T_in'][r].to_pickle(os.path.join(path, f'T_in.exec.glm{args.glm}.{H}.{roi}.p'))
                res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.exec.glm{args.glm}.{H}.{roi}.p'))
                res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.exec.glm{args.glm}.{H}.{roi}.p'))

                np.save(os.path.join(path, f'G_obs.exec.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])

                f = open(os.path.join(path, f'theta_in.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_in'][r], f)
                f = open(os.path.join(path, f'theta_cv.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_cv'][r], f)
                f = open(os.path.join(path, f'theta_gr.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta_gr'][r], f)

    if args.what == 'cerebellum_planning':

        M = make_planning_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.pcmDir, f'M.plan.glm{args.glm}.p'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        for H in Hem:

            N = len(args.snS)

            G_obs = np.zeros((N, 5, 5))
            Y = list()
            for s, sn in enumerate(args.snS):
                glm_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}', f'subj{sn}')
                cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))
                beta_img = nt.volume_from_cifti(cifti_img, struct_names=['Cerebellum'])

                mask = nb.load(os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir, f'subj{sn}', f'cerebellum.{H}.nii'))
                coords = nt.get_mask_coords(mask)

                betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

                res_img = nb.load(os.path.join(glm_path, 'wdResMS.nii'))
                res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

                # Replace near-zero values with np.nan
                tol = 1e-6
                betas[:, np.isclose(res, 0, atol=tol)] = np.nan
                res[np.isclose(res, 0, atol=tol)] = np.nan

                betas_prewhitened = betas / np.sqrt(res)
                betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                cond_vec = np.array([gl.regressor_mapping[r[0].replace(' ', '')] for r in reginfo])
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

            path = os.path.join(gl.baseDir, args.experiment,'SUIT',  gl.pcmDir)

            os.makedirs(path, exist_ok=True)

            T_in.to_pickle(os.path.join(path, f'T_in.plan.glm{args.glm}.cerebellum.{H}.p'))
            T_cv.to_pickle(os.path.join(path, f'T_cv.plan.glm{args.glm}.cerebellum.{H}.p'))
            T_gr.to_pickle(os.path.join(path, f'T_gr.plan.glm{args.glm}.cerebellum.{H}.p'))

            np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.cerebellum.{H}.npy'), G_obs)

            f = open(os.path.join(path, f'theta_in.plan.glm{args.glm}.cerebellum.{H}.p'), 'wb')
            pickle.dump(theta_in, f)

            f = open(os.path.join(path, f'theta_cv.plan.glm{args.glm}.cerebellum.{H}.p'), 'wb')
            pickle.dump(theta_cv, f)

            f = open(os.path.join(path, f'theta_gr.plan.glm{args.glm}.cerebellum.{H}.p'), 'wb')
            pickle.dump(theta_gr, f)

    if args.what == 'cerebellum_execution':

        M = make_planning_models(args.experiment)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.pkl'), "wb")
        pickle.dump(M, f)

        Hem = ['L', 'R']
        for H in Hem:

            N = len(args.snS)

            G_obs = np.zeros((N, 5, 5))
            Y = list()
            for s, sn in enumerate(args.snS):
                glm_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}', f'subj{sn}')
                cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))
                beta_img = nt.volume_from_cifti(cifti_img, struct_names=['Cerebellum'])

                mask = nb.load(os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir, f'subj{sn}', f'cerebellum.{H}.nii'))
                coords = nt.get_mask_coords(mask)

                betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

                res_img = nb.load(os.path.join(glm_path, 'wdResMS.nii'))
                res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

                # Replace near-zero values with np.nan
                tol = 1e-6
                betas[np.isclose(betas, 0, atol=tol)] = np.nan
                res[np.isclose(res, 0, atol=tol)] = np.nan

                betas_prewhitened = betas / np.sqrt(res)
                betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                cond_vec = np.array([gl.regressor_mapping[r[0].replace(' ', '')] for r in reginfo])
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

            path = os.path.join(gl.baseDir, args.experiment,'SUIT',  gl.pcmDir)

            os.makedirs(path, exist_ok=True)

            T_in.to_pickle(os.path.join(path, f'T_in.plan.glm{args.glm}.{H}.p'))
            T_cv.to_pickle(os.path.join(path, f'T_cv.plan.glm{args.glm}.{H}.p'))
            T_gr.to_pickle(os.path.join(path, f'T_gr.plan.glm{args.glm}.{H}.p'))

            np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.cerebellum.{H}.npy'), G_obs)

            f = open(os.path.join(gl.baseDir, args.experiment, 'SUIT',  gl.pcmDir,
                                  f'theta_in.plan.glm{args.glm}.cerebellum.{H}.pkl'), 'wb')
            pickle.dump(theta_in, f)

            f = open(os.path.join(gl.baseDir, args.experiment, 'SUIT',  gl.pcmDir,
                                   f'theta_cv.plan.glm{args.glm}.cerebellum.{H}.pkl'), 'wb')
            pickle.dump(theta_cv, f)

            f = open(os.path.join(gl.baseDir, args.experiment, 'SUIT',  gl.pcmDir,
                                   f'theta_gr.plan.glm{args.glm}.cerebellum.{H}.pkl'), 'wb')
            pickle.dump(theta_gr, f)

    if args.what == 'G_obs_rois_plan-exec':

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)

                G_obs = np.zeros((N, 13, 13))
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

                    # Replace near-zero values with np.nan
                    tol = 1e-6
                    betas[:, np.isclose(res, 0, atol=tol)] = np.nan
                    res[np.isclose(res, 0, atol=tol)] = np.nan

                    betas_prewhitened = betas / np.sqrt(res)
                    betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])

                    obs_des = {'cond_vec': cond_vec,
                               'part_vec': part_vec}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))

                    G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                     Y[s].obs_descriptors['part_vec'],
                                                     X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

                os.makedirs(path, exist_ok=True)

                np.save(os.path.join(path, f'G_obs.plan-exec.glm{args.glm}.{H}.{roi}.npy'), G_obs)

    if args.what == 'correlation_plan-exec':

        Mflex = pcm.CorrelationModel("flex", num_items=1, corr=None, cond_effect=False)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(args.snS)
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

                    # Replace near-zero values with np.nan
                    tol = 1e-6
                    betas[:, np.isclose(res, 0, atol=tol)] = np.nan
                    res[np.isclose(res, 0, atol=tol)] = np.nan

                    betas_prewhitened = betas / np.sqrt(res)
                    betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                    cond_vec = np.array([r[0] for r in reginfo])
                    part_vec = np.array([int(r[1]) for r in reginfo])

                    betas_reduced = []
                    cond_vec_reduced = []
                    part_vec_reduced = []
                    for part in np.unique(part_vec):
                        cond_vec_tmp = cond_vec[part_vec == part]
                        beta_tmp = betas_prewhitened[part_vec == part, :]

                        plan = (beta_tmp[cond_vec_tmp == '100%'] - beta_tmp[cond_vec_tmp == '0%']).squeeze()
                        exec = (beta_tmp[np.char.find(cond_vec_tmp, 'ring') >= 0].mean(axis=0) -
                                     beta_tmp[np.char.find(cond_vec_tmp, 'index') >= 0].mean(axis=0)).squeeze()

                        betas_reduced.append(plan)
                        betas_reduced.append(exec)
                        cond_vec_reduced.append('plan')
                        cond_vec_reduced.append('exec')
                        part_vec_reduced.append(part)
                        part_vec_reduced.append(part)

                    obs_des = {'cond_vec': np.array(cond_vec_reduced),
                               'part_vec': np.array(part_vec_reduced)}

                    Y.append(pcm.dataset.Dataset(np.array(betas_reduced), obs_descriptors=obs_des))

                T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False)
                T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True)

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_in.corr.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_gr.corr.glm{args.glm}.{H}.{roi}.pkl'))

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

                os.makedirs(path, exist_ok=True)

                f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_in.corr.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                pickle.dump(theta_in, f)

                f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                      f'theta_gr.corr.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                pickle.dump(theta_gr, f)

    if args.what == 'correlation_plan-exec_within_finger':

        Mflex = pcm.CorrelationModel("flex", num_items=1, corr=None, cond_effect=False)

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for finger in ['index', 'ring']:
            for H in Hem:
                for roi in rois:

                    N = len(args.snS)
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

                        # Replace near-zero values with np.nan
                        tol = 1e-6
                        betas[:, np.isclose(res, 0, atol=tol)] = np.nan
                        res[np.isclose(res, 0, atol=tol)] = np.nan

                        betas_prewhitened = betas / np.sqrt(res)
                        betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

                        reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
                        cond_vec = np.array([r[0] for r in reginfo])
                        part_vec = np.array([int(r[1]) for r in reginfo])

                        betas_reduced = []
                        cond_vec_reduced = []
                        part_vec_reduced = []
                        for part in np.unique(part_vec):
                            cond_vec_tmp = cond_vec[part_vec == part]
                            beta_tmp = betas_prewhitened[part_vec == part, :]

                            if finger=='index':
                                plan = (beta_tmp[cond_vec_tmp == '0%'] - beta_tmp[cond_vec_tmp == '75%']).squeeze()
                                exec = (beta_tmp[cond_vec_tmp == '0%,index'] - beta_tmp[cond_vec_tmp == '75%,index']).squeeze()
                            elif finger=='ring':
                                plan = (beta_tmp[cond_vec_tmp == '100%'] - beta_tmp[cond_vec_tmp == '25%']).squeeze()
                                exec = (beta_tmp[cond_vec_tmp == '100%,ring'] - beta_tmp[cond_vec_tmp == '25%,ring']).squeeze()

                            betas_reduced.append(plan)
                            betas_reduced.append(exec)
                            cond_vec_reduced.append('plan')
                            cond_vec_reduced.append('exec')
                            part_vec_reduced.append(part)
                            part_vec_reduced.append(part)

                        obs_des = {'cond_vec': np.array(cond_vec_reduced),
                                   'part_vec': np.array(part_vec_reduced)}

                        Y.append(pcm.dataset.Dataset(np.array(betas_reduced), obs_descriptors=obs_des))

                        # if finger == 'index':
                        #     probs = ['0%', '25%', '50%', '75%']
                        #     regressor_mapping = {
                        #         '0%': 0,
                        #         '25%': 1,
                        #         '50%': 2,
                        #         '75%': 3,
                        #         '0%,index': 4,
                        #         '25%,index': 5,
                        #         '50%,index': 6,
                        #         '75%,index': 7,
                        #     }
                        # elif finger == 'ring':
                        #     probs = ['25%', '50%', '75%', '100%']
                        #     regressor_mapping = {
                        #         '25%': 0,
                        #         '50%': 1,
                        #         '75%': 2,
                        #         '100%': 3,
                        #         '25%,ring': 4,
                        #         '50%,ring': 5,
                        #         '75%,ring': 6,
                        #         '100%,ring': 7,
                        #     }
                        #
                        # mask = np.array([r[0] in probs or finger in r[0] for r in reginfo])
                        # betas_prewhitened = betas_prewhitened[mask, :]
                        # cond_vec = np.array(cond_vec)[mask]
                        # part_vec = np.array(part_vec)[mask]
                        #
                        # obs_des = {'cond_vec': np.vectorize(regressor_mapping.get)(cond_vec),
                        #            'part_vec': np.array(part_vec)}
                        #
                        # Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))

                    T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False)
                    T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True)

                    T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                f'T_in.corr.{finger}.glm{args.glm}.{H}.{roi}.pkl'))
                    T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                f'T_gr.corr.{finger}.glm{args.glm}.{H}.{roi}.pkl'))

                    path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

                    os.makedirs(path, exist_ok=True)

                    f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'theta_in.corr.{finger}.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                    pickle.dump(theta_in, f)

                    f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                          f'theta_gr.corr.{finger}.glm{args.glm}.{H}.{roi}.pkl'), 'wb')
                    pickle.dump(theta_gr, f)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,115 ])
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epochs', nargs='+', type=str, default=['SLR', 'LLR', 'Vol'])
    parser.add_argument('--n_tessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')