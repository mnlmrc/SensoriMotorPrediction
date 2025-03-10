import nibabel as nb
import os

import pandas as pd

import globals as gl
import numpy as np
import pickle

import argparse

import nitools as nt
from nitools import spm

import time
import Functional_Fusion.atlas_map as am


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='test_cifti')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=102)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    Hem = ['L', 'R']

    pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
    numTR = pinfo[pinfo['sn'] == args.sn].numTR.reset_index(drop=True)[0]

    if args.what == 'save_rois_timeseries':
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']

        SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}'))  #
        SPM.get_info_from_spm_mat()

        for H in Hem:
            for roi in rois:
                print(f'Processing... Hem: {H}, {roi}')

                # get coordinates
                mask_img = nb.load(
                    os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                 f'{args.atlas}.{H}.{roi}.nii'))
                coords = nt.get_mask_coords(mask_img)

                # get raw time series in roi
                data = nt.sample_images(SPM.rawdata_files, coords)

                # rerun glm
                beta, info, y_filt, y_hat, y_adj, residuals = SPM.rerun_glm(data)

                runs = np.repeat(info['run_number'].unique(), numTR)

                timeseries = {
                    'run': runs,
                    'beta': beta,
                    'y_filt': y_filt,
                    'y_hat': y_hat,
                    'y_adj': y_adj,
                    'residuals': residuals,
                }

                path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                    f'hrf.{args.atlas}.{H}.{roi}.p')
                f = open(path, 'wb')
                pickle.dump(timeseries, f)

    if args.what == 'test_cifti':

        SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}'))  #
        SPM.get_info_from_spm_mat()

        # mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        # atlas = am.AtlasVolumetric('ROI', mask)

        maskL = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.L.nii')
        L = am.AtlasVolumetric('L', maskL, structure='CortexLeft')

        maskR = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.R.nii')
        R = am.AtlasVolumetric('R', maskR, structure='CortexRight')

        betaL, residualL, info = SPM.get_betas(maskL)
        betaR, residualR, _ = SPM.get_betas(maskR)

        row_axis = nb.cifti2.ScalarAxis(info['reg_name'])

        ciftiL = L.data_to_cifti(betaL, row_axis)
        ciftiR = R.data_to_cifti(betaR, row_axis)

        save_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        # nb.save(ciftiL, save_path + '/' + 'beta.L.dscalar.nii')
        # nb.save(ciftiR, save_path + '/' + 'beta.R.dscalar.nii')

        bmL = L.get_brain_model_axis()
        bmR = R.get_brain_model_axis()

        brain_axis = bmL + bmR
        uniquex, index, count, = np.unique(brain_axis.voxel, axis=0, return_counts=True, return_index=True)
        brain_axis = brain_axis[index[count == 1]]

        beta = np.hstack((betaL, betaR))
        beta = beta[:, index[count == 1]]
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))

        cifti = nb.Cifti2Image(
            dataobj=beta,  # Stack them along the rows (adjust as needed)
            header=header  # Use one of the headers (may need to modify)
        )

        # nb.save(ciftiL, save_path + '/' + 'beta.dscalar.nii')
        # path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        # img_raw, img_filt, img_hat, img_adj, img_res = get_ciftis(mask=mask, SPM=SPM, TR=1000)
        #
        # nb.save(img_raw, os.path.join(path, 'ROI.S1.y_raw.dtseries.nii'))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Finished in {end - start} seconds')


# # load residuals for prewhitening
# res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
# ResMS = nt.sample_image(res_img, coords[0], coords[1], coords[2], 0)
#
# if stats == 'mean':
#     y_raw = data.mean(axis=1)
# elif stats == 'whiten':
#     y_raw = (data / np.sqrt(ResMS)).mean(axis=1)
# elif stats == 'pca':
#     pass
#
# fdata = SPM.spm_filter(SPM.weight @ data)
# beta = SPM.pinvX @ fdata
# pdata = SPM.design_matrix @ beta
#
#
#
#
#
