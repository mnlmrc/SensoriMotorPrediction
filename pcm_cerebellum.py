import os
import pickle
import globals as gl
import argparse
from pcm_cortical import Rois, make_planning_models
import numpy as np
import nibabel as nb
import nitools as nt
import PcmPy as pcm

import time

def main(args):

    if args.what == 'cerebellum_planning':
        M = make_planning_models()
        # f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.glm{args.glm}.p'), "rb")
        # M = pickle.load(f)

        Hem = ['L', 'R']
        rois = ['M2', 'M3', 'D3']
        roi_imgs = [f'cerebellum.{H}.{roi}{H}.nii' for H in Hem for roi in rois]
        glm_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}')
        cifti_img = 'beta.dscalar.nii'
        roi_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir)

        R = Rois(args.snS, M, glm_path, cifti_img, 'wdResMS.nii', roi_path, roi_imgs,
                 struct_names=['CIFTI_STRUCTURE_CEREBELLUM'],
                 regressor_mapping=gl.regressor_mapping, regr_of_interest=[0, 1, 2, 3, 4])
        res = R.run_pcm_in_roi(roi_imgs[4])
        # res = R.run_parallel_pcm_across_rois()

        # for H in Hem:
        #     for roi in rois:
        #         r = res['roi_img'].index(f'cerebellum.{H}.{roi}{H}.nii')
        #
        #         path = os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.pcmDir)
        #         os.makedirs(path, exist_ok=True)
        #
        #         res['T_in'][r].to_pickle(os.path.join(path, f'T_in.plan.glm{args.glm}.{H}.{roi}.p'))
        #         res['T_cv'][r].to_pickle(os.path.join(path, f'T_cv.plan.glm{args.glm}.{H}.{roi}.p'))
        #         res['T_gr'][r].to_pickle(os.path.join(path, f'T_gr.plan.glm{args.glm}.{H}.{roi}.p'))
        #
        #         np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.{H}.{roi}.npy'), res['G_obs'][r])
        #
        #         f = open(os.path.join(path, f'theta_in.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
        #         pickle.dump(res['theta_in'][r], f)
        #         f = open(os.path.join(path, f'theta_cv.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
        #         pickle.dump(res['theta_cv'][r], f)
        #         f = open(os.path.join(path, f'theta_gr.plan.glm{args.glm}.{H}.{roi}.p'), 'wb')
        #         pickle.dump(res['theta_gr'][r], f)

        # M = make_planning_models(args.experiment)
        # f = open(os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.pcmDir, f'M.plan.glm{args.glm}.p'), "wb")
        # pickle.dump(M, f)
        #
        # Hem = ['L', 'R']
        # for H in Hem:
        #     for roi in rois:
        #
        #         N = len(args.snS)
        #
        #         G_obs = np.zeros((N, 5, 5))
        #         Y = list()
        #         for s, sn in enumerate(args.snS):
        #             glm_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}', f'subj{sn}')
        #             cifti_img = nb.load(os.path.join(glm_path, f'beta.dscalar.nii'))
        #             beta_img = nt.volume_from_cifti(cifti_img, struct_names=['Cerebellum'])
        #
        #             mask = nb.load(os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir, f'subj{sn}',
        #                                         f'cerebellum.{H}.{roi}{H}.nii'))
        #             coords = nt.get_mask_coords(mask)
        #
        #             betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T
        #
        #             res_img = nb.load(os.path.join(glm_path, 'wdResMS.nii'))
        #             res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)
        #
        #             # Replace near-zero values with np.nan
        #             tol = 1e-6
        #             betas[:, np.isclose(res, 0, atol=tol)] = np.nan
        #             res[np.isclose(res, 0, atol=tol)] = np.nan
        #
        #             betas_prewhitened = betas / np.sqrt(res)
        #             betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]
        #
        #             reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
        #             cond_vec = np.array([gl.regressor_mapping[r[0].replace(' ', '')] for r in reginfo])
        #             part_vec = np.array([int(r[1]) for r in reginfo])
        #
        #             idx = np.isin(cond_vec, [0, 1, 2, 3, 4])
        #
        #             obs_des = {'cond_vec': cond_vec[idx],
        #                        'part_vec': part_vec[idx]}
        #
        #             Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))
        #
        #             G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
        #                                              Y[s].obs_descriptors['part_vec'],
        #                                              X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))
        #
        #         T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        #         T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        #         T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        #
        #         path = os.path.join(gl.baseDir, args.experiment,'SUIT',  gl.pcmDir)
        #
        #         os.makedirs(path, exist_ok=True)
        #
        #         T_in.to_pickle(os.path.join(path, f'T_in.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'))
        #         T_cv.to_pickle(os.path.join(path, f'T_cv.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'))
        #         T_gr.to_pickle(os.path.join(path, f'T_gr.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'))
        #
        #         np.save(os.path.join(path, f'G_obs.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.npy'), G_obs)
        #
        #         f = open(os.path.join(path, f'theta_in.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'), 'wb')
        #         pickle.dump(theta_in, f)
        #
        #         f = open(os.path.join(path, f'theta_cv.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'), 'wb')
        #         pickle.dump(theta_cv, f)
        #
        #         f = open(os.path.join(path, f'theta_gr.plan.glm{args.glm}.cerebellum.{H}.{roi}{H}.p'), 'wb')
        #         pickle.dump(theta_gr, f)

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


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[102])  #[ 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, ]
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--n_jobs', type=int, default=16)

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')