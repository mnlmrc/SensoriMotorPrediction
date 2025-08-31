import os
import pickle
import globals as gl
import argparse
from pcm_cortical import Rois
import numpy as np
import nibabel as nb
import nitools as nt
import PcmPy as pcm

import time

def pcm_cerebellum(M, epoch, args):
    glm_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}')
    roi_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir)
    Hem = ['L', 'R']
    rois = ['M2', 'M3', 'D3']
    roi_imgs = [f'cerebellum.{H}.{roi}{H}.nii' for H in Hem for roi in rois]
    cifti_img = 'beta.dscalar.nii'
    res_img = 'wdResMS.nii'
    R = Rois(args.sns, M, glm_path, cifti_img,
             roi_path=roi_path,
             roi_imgs=roi_imgs,
             regressor_mapping=gl.regressor_mapping,
             regr_interest=[0, 1, 2, 3, 4] if epoch == 'plan' else [5, 6, 7, 8, 9, 10, 11, 12, ],
             res_img=res_img,
             struct_names=['Cerebellum', 'Cerebellum'],)
    # Y, G = R._make_roi_dataset(roi_imgs[0])
    # R._fit_model_to_dataset(Y)
    res = R.run_parallel_pcm_across_rois()
    path = os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.pcmDir)
    for H in Hem:
        for roi in rois:
            r = res['roi_img'].index(f'cerebellum.{H}.{roi}{H}.nii')
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
    if args.what == 'rois_planning':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        pcm_cerebellum(M, 'plan', args)
    if args.what == 'rois_execution':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        pcm_cerebellum(M, 'exec', args)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[ 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, ])  #
    parser.add_argument('--atlas', type=str, default='ROI')
    # parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--n_jobs', type=int, default=16)

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')