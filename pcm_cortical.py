import warnings

import time

from globals import regressor_mapping

warnings.filterwarnings("ignore")

import argparse
import pickle
from itertools import combinations
import rsatoolbox as rsa

from rdms import D_to_rdm

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

def prewhiten(betas, res, lam=0.1, eps=1e-8):
    """
    betas: (n_cond, V)
    res:   (V,) ResMS  OR  residuals as (T, V) or (V, T)
    Returns: betas_wh, keep_mask
    """
    n_cond, V = betas.shape
    keep = np.ones(V, dtype=bool)

    if res.ndim == 1:
        r = res.astype(float)
        bad = ~np.isfinite(r) | np.isclose(r, 0.0, atol=1e-6) | np.isnan(betas).all(axis=0)
        keep &= ~bad
        scale = np.sqrt(np.clip(r[keep], eps, None))
        return betas[:, keep] / scale, keep

    # 2-D residuals
    R = res
    if R.shape == (V, R.shape[1]):     # (V, T)
        R = R.T                        # -> (T, V)
    if R.shape[1] != V:
        raise ValueError("Residuals do not match number of voxels in betas.")

    # drop bad voxels
    bad = ~np.isfinite(R).all(axis=0) | np.isclose(R.var(axis=0), 0.0, atol=1e-10) | np.isnan(betas).all(axis=0)
    keep &= ~bad
    R = R[:, keep]
    B = betas[:, keep]

    T = R.shape[0] - 1
    Sigma = (R.T @ R) / T

    # regularisation
    if lam and lam > 0:
        mu = np.mean(np.diag(Sigma))
        Sigma = (1 - lam) * Sigma + lam * mu * np.eye(Sigma.shape[0])

    w, U = np.linalg.eigh(Sigma)
    w = np.clip(w, eps, None)
    W = (U * (1.0 / np.sqrt(w))) @ U.T   # Σ^{-1/2}

    return B @ W


def calc_prewhitened_betas(glm_path=None, cifti_img='beta.dscalar.nii', res_img='ResMS.nii', roi_path=None, roi_img=None,
                           struct_names=['CortexLeft', 'CortexRight'], reg_mapping=None, reg_interest=None):
    """
    Get pre-whitened betas from ROI to submit to RSA/PCM
    Args:
        glm_path:
        cifti_img:
        res_img:
        roi_path:
        roi_img:
        struct_names:
        reg_mapping:
        reg_interest:

    Returns:

    """
    cifti_img = nb.load(os.path.join(glm_path, cifti_img))
    beta_img = nt.volume_from_cifti(cifti_img, struct_names=struct_names)

    mask = nb.load(os.path.join(roi_path, roi_img))
    coords = nt.get_mask_coords(mask)

    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

    res_img = nb.load(os.path.join(glm_path, res_img))
    if isinstance(res_img, nb.Cifti2Image):
        res_img = nt.volume_from_cifti(res_img, struct_names=struct_names)
    if isinstance(res_img, nb.nifti1.Nifti1Image):
        res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

    betas_prewhitened = prewhiten(betas, res, lam=0.1, eps=1e-8)

    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
    cond_vec = np.array([r[0] for r in reginfo])
    part_vec = np.array([int(r[1]) for r in reginfo])

    obs_des = {'cond_vec': cond_vec,
               'part_vec': part_vec}

    # Optional: use different regressor name, e.g., a number for ordering purposes
    if reg_mapping is not None:
        cond_vec = np.vectorize(reg_mapping.get)(cond_vec)
        obs_des['cond_vec'] = cond_vec

    # Optional: restrict to some regressors, use the new mapped names
    if reg_interest is not None:
        idx = np.isin(cond_vec, reg_interest)
        betas_prewhitened = betas_prewhitened[idx]
        obs_des = {'cond_vec': cond_vec[idx],
                   'part_vec': part_vec[idx]}

    return betas_prewhitened, obs_des


def bootstrap_correlation(idx, Y, Mflex, sigma_floor=1e-4):
    """

    Args:
        Y:
        Mflex:
        sigma_floor:

    Returns:

    """

    S = len(Y)
    y = [Y[i] for i in idx]

    _, theta_gr = pcm.fit_model_group(y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)

    theta_gr, _ = pcm.group_to_individ_param(theta_gr[0], Mflex, S)

    sigma2_1 = np.exp(theta_gr[0, 0])
    sigma2_2 = np.exp(theta_gr[1, 0])
    sigma2_e = np.exp(theta_gr[-1])
    r = Mflex.get_correlation(theta_gr)

    sd = np.sqrt(sigma2_1 * sigma2_2)
    if sd < sigma_floor * np.sqrt(sigma2_e).max():
        print(f'No reliable signal, discarding bootstrap resample')
        return None
    else:
        return r[0]


def bootstrap_summary(r_bootstrap, alpha=0.05):
    """
    Given the retained bootstrap correlations, return:
      - central (1-2*alpha) CI (so for alpha=.05 -> 90% CI)
      - functions for one-sided tests: r < x and r > x
    """
    r_bootstrap = np.asarray(r_bootstrap)
    if r_bootstrap.size == 0:
        raise ValueError("No valid bootstrap replicates retained.")

    lo = np.quantile(r_bootstrap, alpha)        # lower bound of central CI
    hi = np.quantile(r_bootstrap, 1 - alpha)    # upper bound of central CI

    def pval_r_less_than(x):
        # p ≈ proportion of bootstrap >= x  (upper tail)
        return float(np.mean(r_bootstrap >= x))

    def pval_r_greater_than(x):
        # p ≈ proportion of bootstrap <= x  (lower tail)
        # NOTE: tends to be liberal in the paper (lower bound too high)
        return float(np.mean(r_bootstrap <= x))

    return (lo, hi), pval_r_less_than, pval_r_greater_than


class Tessellation():
    def __init__(self, snS=None, surf_path=None, glm_path=None, M=None, reg_interest=None, reg_mapping=None,
                 n_tessels=None, n_jobs=None):
        self.snS = snS
        self.surf_path = surf_path
        self.glm_path = glm_path
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

        # define path to surfaces
        surf_path = os.path.join(self.surf_path, f'subj{sn}')

        # retrieve surfaces
        white = os.path.join(surf_path, f'subj{sn}.{H}.white.32k.surf.gii')
        pial = os.path.join(surf_path, f'subj{sn}.{H}.pial.32k.surf.gii')

        # define glm mask
        mask = os.path.join(self.glm_path, f'subj{sn}', 'mask.nii')

        # Build atlas mapping
        amap = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
        amap.build()

        # load betas from cifti
        cifti_img = nb.load(os.path.join(self.glm_path, f'subj{sn}', f'beta.dscalar.nii'))

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
        res = nb.load(os.path.join(self.glm_path, f'subj{sn}', 'ResMS.nii'))

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
        for s, sn in enumerate(self.snS):
            Dataset = self._make_individ_dataset(H, subatlas, sn)
            n_voxels.append(Dataset.n_channel)
            Y.append(Dataset)

        try:
            # T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
            T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')

            # for i in range(len(theta_cv)):
            #     n_param = self.M[i].n_param
            #     theta_cv[i] = theta_cv[i][:n_param] / np.linalg.norm(theta_cv[i][:n_param])

            likelihood = T_gr.likelihood
            baseline = likelihood.loc[:, 'null'].values
            likelihood = likelihood - baseline.reshape(-1, 1)
            # noise_upper = (T_gr.likelihood['ceil'] - baseline)
            # noise_lower = likelihood.ceil

        except Exception as e:
            print(f"Error in tessel: {e}")
            n_cols = len(self.col_names)
            n_subj = len(self.snS)
            likelihood = {col: np.full(n_subj, np.nan) for col in self.col_names}
            # noise_upper = np.full(n_subj, np.nan)
            # noise_lower = np.full(n_subj, np.nan)
            baseline = np.full(n_subj, np.nan)
            # theta_cv = [np.full((m.n_param, n_subj), np.nan) for m in self.M]
            theta_gr = [np.full((m.n_param + n_subj), np.nan) for m in self.M]

        return likelihood, baseline, theta_gr, n_voxels


    def make_subatlas_tessel(self, H, ntessel):
        print(f'Hemisphere: {H}, tessel #{ntessel}\t')
        atlas_hem = self.atlas.get_hemisphere(self.Hem.index(H))
        subatlas = atlas_hem.get_subatlas_image(self.path_tessel_atlas[H], ntessel)
        return subatlas

    def _store_T_and_theta_from_tessel(self, H, ntessel):

        subatlas = self.make_subatlas_tessel(H, ntessel)

        T = {
            'likelihood': [],
            # 'noise_upper': [],
            # 'noise_lower': [],
            'baseline': [],
            'n_voxels': [],
            'col_names': [],
            # 'sn': []
        }

        theta = {}
        for md in self.M:
            if md.n_param > 0: # skip models with 0 params i.e. Fixed Models
                theta[md.name] = {
                    'theta': [],
                    '#param': [],
                    # 'sn': []
                }

        likelihood, baseline, theta_gr, n_voxels = self._fit_model_in_tessel(H, subatlas)

        # for s, sn in enumerate(self.participants_id):
        for c, col in enumerate(self.col_names):
            T['likelihood'].append(likelihood[col])
            # T['noise_upper'].append(noise_upper[s])
            # T['noise_lower'].append(noise_lower[s])
            T['baseline'].append(baseline)
            T['n_voxels'].append(n_voxels)
            T['col_names'].append(col)
            # T['sn'].append(sn)
        for m, md in enumerate(self.M):
            if md.n_param > 0:
                for c in range(md.n_param):
                    theta[md.name]['theta'].append(theta_gr[m][c])
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


    def _extract_results_from_parallel_process(self, H):
        results = self.results[H]

        # Aggregate results from parallel processes
        T = np.full((32492, len(self.M) + 2), np.nan)
        theta = {}
        for md in self.M:
            theta[md.name] = np.full((32492, md.n_param), np.nan)

        # # if sn is not None:
        # for Tt, _, vertex_id in results:
        #     if len(vertex_id)>0 :
        #         for c, col in enumerate(self.col_names):
        #             LL = Tt[Tt['col_names'] == col]['likelihood']
        #             T[vertex_id, c] = LL
        #         # T[vertex_id, -4] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
        #         # T[vertex_id, -3] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
        #         T[vertex_id, -2] = Tt['baseline'].unique()
        #         T[vertex_id, -1] = Tt['n_voxels'].unique()
        # # else:
        for _, th, vertex_id in results:
            for md in self.M:
                for c in range(md.n_param):
                    theta_tmp = th[md.name]
                    theta[md.name][vertex_id, c] = theta_tmp['theta'][theta_tmp['#param'] == c]

        return T, theta

    def make_group_giftis_likelihood(self, H):
        T = []
        column_names = self.col_names + ['noise_upper', 'noise_lower', 'baseline', 'n_voxels']
        # for sn in self.participants_id:
        Tt, _ = self._extract_results_from_parallel_process(H)
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
    def __init__(self, snS=None, M=None, glm_path=None, cifti_img=None, res_img='ResMS.nii', roi_path=None,
                 roi_imgs=None, regressor_mapping=None, struct_names=['CortexLeft', 'CortexRight'],
                 regr_interest=None, n_jobs=16):
        self.snS = snS  # participants ids
        self.M = M  # pcm models to fit
        self.glm_path = glm_path  # path to cifti_img (should be the folder containinting the betas...)
        self.cifti_img = cifti_img  # name of cifti_img (e.g., beta.dscalar.nii)
        self.res_img = res_img # name of res image for univariate prewhitening
        self.roi_path = roi_path  # path to individual roi masks, which must be named <atlas_name>.<H>.<roi>.nii
        self.roi_imgs = roi_imgs  # name of roi files to use as masks, e.g. ROI.L.M1.nii or cerebellum.L.nii
        self.struct_names = struct_names
        self.regressor_mapping = regressor_mapping  # dict, maps name of regressors to numbers to control in which order conditions appear in the G matrix
        self.regr_interest = regr_interest  # indexes from regressor mapping of the regressors we want to include in the analysis
        self.n_jobs = n_jobs

    def _make_roi_dataset(self, roi_img):
        N = len(self.snS)

        G_obs = np.zeros((N, len(self.regr_interest), len(self.regr_interest)))
        Y = list()
        for s, sn in enumerate(self.snS):
            print(f'making dataset...subj{sn} - {roi_img}')
            betas_prewhitened, obs_des = calc_prewhitened_betas(glm_path=self.glm_path + '/' + f'subj{sn}',
                                                                cifti_img='beta.dscalar.nii',
                                                                res_img=self.res_img,
                                                                roi_path=self.roi_path,
                                                                roi_img=f'subj{sn}' + '/' + roi_img,
                                                                struct_names=self.struct_names,
                                                                reg_mapping=self.regressor_mapping,
                                                                reg_interest=self.regr_interest,)
            Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))
            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                             Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

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
        elif isinstance(M, pcm.FeatureModel):
            MF = pcm.model.ModelFamily(M, comp_names=comp_names, basecomponents=basecomp)
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


def pcm_tessel(M, epoch, args):
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    surf_path = os.path.join(gl.baseDir, args.experiment, gl.wbDir)
    Tess = Tessellation(args.snS,
                        surf_path,
                        glm_path,
                        M,
                        gl.reg_interest,
                        gl.regressor_mapping,
                        args.n_tessels,
                        args.n_jobs)
    Tess.run_parallel_pcm_across_tessels()
    cifti_theta_component = Tess.make_group_cifti_theta('component')
    nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.{epoch}.dscalar.nii'))
    cifti_theta_feature = Tess.make_group_cifti_theta('feature')
    nb.save(cifti_theta_feature, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                              f'theta_feature.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.{epoch}.dscalar.nii'))

def pcm_rois(M, epoch, args):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    roi_imgs = [f'ROI.{H}.{roi}.nii' for H in Hem for roi in rois]
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    cifti_img = 'beta.dscalar.nii'
    roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    R = Rois(args.sns, M, glm_path, cifti_img,
             roi_path=roi_path,
             roi_imgs=roi_imgs,
             regressor_mapping=gl.regressor_mapping,
             regr_interest=[0, 1, 2, 3, 4] if epoch == 'plan' else [5, 6, 7, 8, 9, 10, 11, 12,],
             res_img='residual.dtseries.nii',
             n_jobs=args.n_jobs
             )
    res = R.run_parallel_pcm_across_rois()

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


def main(args):
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    roi_path = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    pcm_path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
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
        R = Rois(args.sns, M, glm_path, cifti_img, roi_path=roi_path, roi_imgs=roi_imgs, n_jobs=args.n_jobs,
                 regressor_mapping=gl.regressor_mapping, regr_interest=[0, 1, 2, 3, 4], res_img='residual.dtseries.nii')
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
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        R = Rois(args.sns, M, glm_path, cifti_img, roi_path=roi_path, roi_imgs=roi_imgs, n_jobs=args.n_jobs,
                 regressor_mapping=gl.regressor_mapping, regr_interest=[5, 6, 7, 8, 9, 10, 11, 12], res_img='residual.dtseries.nii')
        res = R.fit_model_family_across_rois('component', comp_names=['finger', 'cue', 'surprise'],
                                             basecomp=np.eye(8)[None, :, :])
        for H in Hem:
            for roi in rois:
                r = res['roi_img'].index(f'ROI.{H}.{roi}.nii')

                path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
                os.makedirs(path, exist_ok=True)

                res['T'][r].to_pickle(os.path.join(path, f'T.model_family.exec.glm{args.glm}.{H}.{roi}.p'))
                f = open(os.path.join(path, f'theta.model_family.exec.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(res['theta'][r], f)
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
        rng = np.random.default_rng(0) # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for H in Hem:
            for roi in rois:
                N = len(args.sns)
                Y = list()
                r = roi_imgs.index(f'ROI.{H}.{roi}.nii')
                print(f'doing...ROI.{H}.{roi}')
                for s, sn in enumerate(args.sns):
                    betas_prewhitened, obs_des = calc_prewhitened_betas(glm_path=os.path.join(glm_path, f'subj{sn}'),
                                                               cifti_img='beta.dscalar.nii',
                                                               res_img='residual.dtseries.nii',
                                                               roi_path=os.path.join(roi_path, f'subj{sn}'),
                                                               roi_img=roi_imgs[r])
                    cond_vec, part_vec = obs_des['cond_vec'], obs_des['part_vec']
                    betas_reduced, cond_vec_reduced, part_vec_reduced = [], [], []
                    for part in np.unique(part_vec):
                        mask = part_vec == part
                        beta_tmp = betas_prewhitened[mask]
                        cond_tmp = cond_vec[mask]

                        plan = (beta_tmp[(cond_tmp == '0-100%') | (cond_tmp == '25-75%')].mean(axis=0) -
                                beta_tmp[(cond_tmp == '100-0%') | (cond_tmp == '75-25%')].mean(axis=0)).squeeze()
                        exec = (beta_tmp[np.char.find(cond_tmp, 'ring') >= 0].mean(axis=0) -
                                beta_tmp[np.char.find(cond_tmp, 'index') >= 0].mean(axis=0)).squeeze()

                        betas_reduced.extend([plan, exec])
                        cond_vec_reduced.extend(['plan', 'exec'])
                        part_vec_reduced.extend([part] * 2)

                    obs_des_reduced = {'cond_vec': np.array(cond_vec_reduced),
                               'part_vec': np.array(part_vec_reduced)}

                    Y.append(pcm.dataset.Dataset(np.array(betas_reduced), obs_descriptors=obs_des_reduced))

                T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=False)
                T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
                T_in.to_pickle(os.path.join(pcm_path, f'T_in.corr.glm{args.glm}.{H}.{roi}.p'))
                T_gr.to_pickle(os.path.join(pcm_path, f'T_gr.corr.glm{args.glm}.{H}.{roi}.p'))

                f = open(os.path.join(pcm_path, f'theta_in.corr.glm{args.glm}.{H}.{roi}.p'), 'wb')
                pickle.dump(theta_in, f)
                f = open(os.path.join(pcm_path, f'theta_gr.corr.glm{args.glm}.{H}.{roi}.p'), 'wb')
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

                np.save(os.path.join(pcm_path, f'r_bootstrap.{H}.{roi}.npy'), r_bootstrap)





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