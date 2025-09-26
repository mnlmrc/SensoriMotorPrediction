import argparse
import time

import pandas as pd
import numpy as np
import os
import scipy
import glob

from nitools import spm
import Functional_Fusion.atlas_map as am
import imaging_pipelines.betas as bt

import globals as gl

import nibabel as nb
import nitools as nt


def main(args=None):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    struct = ['CortexLeft', 'CortexRight']
    path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    if args.what == 'make_betas_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'Hem.{H}.nii') for H in Hem]
        cifti = bt.make_cifti_betas(path_glm, masks, struct)
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    if args.what == 'make_betas_cifti_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='make_betas_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'make_betas_cifti_cerebellum':
        path_rois = os.path.join(gl.baseDir, args.experiment, 'SUIT', gl.roiDir)
        path_glm = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        reginfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                           f'subj{args.sn}_reginfo.tsv'), sep="\t")
        betas = [f'{path_glm}' + '/' + f'wdbeta_{i+1:04d}.nii' for i in range(reginfo.shape[0])]
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'cerebellum.{H}.nii') for H in Hem]
        print(f'mask: {masks}')
        row_axis = nb.cifti2.ScalarAxis(reginfo['name'].str.strip() + '.' + reginfo['run'].astype(str).str.strip())
        cifti = bt.make_cifti_betas(path_glm, masks, struct=['Cerebellum', 'Cerebellum'], betas=betas, row_axis=row_axis)
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    if args.what == 'make_betas_cifti_cerebellum_all':
        for sn in args.sns:
            print(f'doing participant {sn}')
            args = argparse.Namespace(
                what='make_betas_cifti_cerebellum',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'make_contrasts_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'Hem.{H}.nii') for H in Hem]
        reginfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                           f'subj{args.sn}_reginfo.tsv'), sep="\t")
        regressors = reginfo['name'].str.replace(' ', '')
        cifti = bt.make_cifti_contrasts(path_glm, masks, struct, regressors=regressors)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    if args.what == 'make_contrasts_cifti_all':
        for sn in args.sns:
            args = argparse.Namespace(
                what='make_contrasts_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'roi_contrasts':
        con_dict = {
            'con': [],
            'condition': [],
            'sn': [],
            'roi': [],
            'Hem': [],
            'epoch': []
        }
        for sn in args.sns:
            print(f'Processing subj{sn}')
            path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}')
            cifti = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
            regr = cifti.header.get_axis(0).name[[0, 4, 7, 10, 2, 1, 5, 8, 11, 6, 9, 12, 3]]
            vol = nt.volume_from_cifti(cifti, struct_names=struct)
            for H in Hem:
                for roi in rois:
                    mask = os.path.join(path_rois, f'subj{sn}', f'ROI.{H}.{roi}.nii')
                    coords = nt.get_mask_coords(mask)
                    con = nt.sample_image(vol, coords[0], coords[1], coords[2],0)
                    con = np.nanmean(con, axis=0)[[0, 4, 7, 10, 2, 1, 5, 8, 11, 6, 9, 12, 3]]
                    for i, c in enumerate(con):
                        con_dict['con'].append(c)
                        con_dict['condition'].append(regr[i])
                        con_dict['sn'].append(str(sn))
                        con_dict['roi'].append(roi)
                        con_dict['Hem'].append(H)
                        epoch = 'exec' if ('index' in regr[i]) or ('ring' in regr[i]) else 'plan'
                        con_dict['epoch'].append(epoch)
        con = pd.DataFrame.from_dict(con_dict)
        con.to_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', 'ROI.con.avg.tsv'),
                   sep='\t', index=False)
    if args.what == 'make_residuals_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'Hem.{H}.nii') for H in Hem]
        residuals = bt.make_cifti_residuals(path_glm, masks, struct)
        nb.save(residuals, path_glm + '/' + 'residual.dtseries.nii')
    # if args.what == 'make_4D_residuals_nifti':
    #     path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
    #     cifti = nb.load(path_glm + '/' + 'residual.dtseries.nii')
    #     nifti = nt.volume_from_cifti(cifti, struct_names=struct)
    #     nb.save(nifti, path_glm + '/' + 'residual.nii')
    if args.what == 'make_residuals_cifti_all':
        for sn in args.sns:
            print(f'Processing subj{sn}...')
            arg = argparse.Namespace(
                what='make_residuals_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(arg)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)
    finish = time.time()

    print(f'Execution time: {finish - start} seconds')
