import argparse
import time

import pandas as pd
import numpy as np
import os
import scipy
import shutil

from nitools import spm
import Functional_Fusion.atlas_map as am
import imaging_pipelines.betas as bt

import SensoriMotorPrediction.globals as gl

import nibabel as nb
import nitools as nt
import subprocess


def save_spm_as_mat7(sn, glm):
    path_glm = os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'subj{sn}')
    spm_path = os.path.join(path_glm, 'SPM.mat')
    backup_path = spm_path + ".backup"

    if os.path.exists(backup_path):
        resp = input(
            f"Backup already exists for participant {sn}.\n"
            f"This will overwrite SPM.mat using the backup.\n"
            f"Continue? [y/N]: "
        ).strip().lower()

        if resp not in ["y", "yes"]:
            print("Skipping conversion.")
            return
        else:
            print("Proceeding with replacement.")

    else:
        # Step 1: Backup the original file
        shutil.copy(spm_path, backup_path)
        print(f"Backed up {spm_path} to {backup_path}")

    # Step 2: Run MATLAB command
    matlab_cmd = (
        f"matlab -nodesktop -nosplash -r "
        f"\"load('{spm_path}'); save('{spm_path}', '-struct', 'SPM', '-v7'); exit\""
    )

    subprocess.run(matlab_cmd, shell=True, check=True)
    print(f"Processed {spm_path} with MATLAB")


def make_cifti(sn, glm=None, type='beta', experiment='smp2'):
    print(f'doing participant {sn}, {type}...')
    path_glm = os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}')
    masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in gl.Hem]
    reginfo = pd.read_csv(os.path.join(path_glm, f'subj{sn}_reginfo.tsv'), sep='\t')
    row_axis = nb.cifti2.ScalarAxis(reginfo['name'] + '.' + reginfo['run'].astype(str))
    if type == 'beta':
        cifti = bt.make_cifti_betas(masks, gl.struct, path_glm=path_glm, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    elif type == 'repetition_suppression':
        cifti = nb.load(path_glm + '/' + 'beta.dscalar.nii')
        brain_axis = cifti.header.get_axis(1)
        data = cifti.get_fdata()
        rep1 = data[::2]
        rep2 = data[1::2]
        suppr = rep2 - rep1
        reginfo = reginfo[::2].reset_index()
        chord_sess_rep = reginfo.name.str.split(',', expand=True)
        run = reginfo.run
        row_axis = chord_sess_rep.astype(str)[0] + ',' + chord_sess_rep[1] + '.' + run.astype(str)
        row_axis = nb.cifti2.ScalarAxis(row_axis)
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_suppr = nb.Cifti2Image(dataobj=suppr,  header=header)
        nb.save(cifti_suppr, path_glm + '/' + 'beta.dscalar.nii')
    elif type == 'residual':
        residuals = bt.make_cifti_residuals(path_glm=path_glm, masks=masks, struct=gl.struct)
        nb.save(residuals, path_glm + '/' + 'residual.dtseries.nii')
    elif type == 'contrast':
        cifti = bt.make_cifti_contrasts(path_glm, masks, im.struct, reginfo.name)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    elif type =='psc':
        contrast = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
        intercept = nb.load(path_glm + '/' + 'intercept.dscalar.nii')
        SPM = spm.SpmGlm(path_glm)
        SPM.get_info_from_spm_mat()
        cifti = bt.make_cifti_psc(contrast=contrast, intercept=intercept, SPM=SPM, masks=masks, struct=im.struct)
        nb.save(cifti, path_glm + '/' + 'psc.dscalar.nii')
    elif type == 'intercept':
        session = reginfo.name.str.split(',', n=1, expand=True)[1]
        nRuns = [reginfo[session == sess].run.nunique() for sess in session.unique()]
        nRegressors = reginfo.shape[0]
        intercept = []
        for sess in range(dn.nSess):
            for run in range(nRuns[sess]):
                intercept.append(os.path.join(path_glm, f'beta_0{nRegressors + run + 1 + sess * nRuns[0]}.nii'))
        masks = [os.path.join(path_rois, f'Hem.{H}.nii') for H in im.Hem]
        cond_vec = np.sort(np.array([f'{sess},{run}' for run in range(nRuns[sess]) for sess in range(dn.nSess)]))
        row_axis = nb.cifti2.ScalarAxis(cond_vec)
        cifti = bt.make_cifti_betas(masks, im.struct, intercept, row_axis=row_axis, )
        nb.save(cifti, path_glm + '/' + 'intercept.dscalar.nii')
    else:
        raise Exception(f'Unknown type {type}. Must be beta, residual, contrast or intercept.')


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
