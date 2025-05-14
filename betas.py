import argparse
import time

import pandas as pd
import numpy as np
import os
import scipy
import glob

from nitools import spm
import Functional_Fusion.atlas_map as am

import globals as gl

import nibabel as nb
import nitools as nt


def make_cifti_betas(path_glm, masks, struct):
    SPM = spm.SpmGlm(path_glm)  #
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    betas, _, info = SPM.get_betas(coords)

    reg_name = np.array([n.split('*')[0] for n in info['reg_name']])

    row_axis = nb.cifti2.ScalarAxis(reg_name.astype(str) + '.' + info['run_number'].astype(str))

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=betas,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti


def make_cifti_contrasts(path_glm, masks, struct):
    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    files = glob.glob(os.path.join(path_glm, '*reginfo.tsv'))

    if files:
        reginfo = pd.read_csv(files[0], sep='\t')
    else:
        raise FileNotFoundError("No file ending with 'reginfo.tsv' found.")
    regressors = reginfo['name'].unique()

    contrasts = list()
    for regr, regressor in enumerate(regressors):
        vol = nb.load(os.path.join(path_glm, f'con_{regressor.replace(" ", "")}.nii'))
        con = nt.sample_image(vol, coords[0], coords[1], coords[2], 0)
        contrasts.append(con)

    contrasts = np.array(contrasts)

    regressor = [r.replace(" ", "") for r in regressors]
    row_axis = nb.cifti2.ScalarAxis(regressors)

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=contrasts,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti

def make_cifti_residuals(path_glm, masks, struct):
    SPM = spm.SpmGlm(path_glm)  #
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric(H, mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    res, _, info = SPM.get_residuals(coords)

    row_axis = nb.cifti2.SeriesAxis(1, 1, res.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=res,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti


def get_roi(experiment=None, sn=None, Hem=None, roi=None, atlas='ROI'):
    mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}',
                                        f'subj{sn}_{atlas}_region.mat'))
    R_cell = mat['R'][0]
    R = list()
    for r in R_cell:
        R.append({field: r[field].item() for field in r.dtype.names})

    # find roi
    R = R[[True if (r['name'].size > 0) and (r['name'] == roi) and (r['hem'] == Hem)
           else False for r in R].index(True)]

    return R


def get_roi_betas(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    R = get_roi(experiment, sn, Hem, roi)

    betas = list()
    for n_regr in np.arange(0, reginfo.shape[0]):
        print(f'ROI.{Hem}.{roi} - loading regressor #{n_regr + 1}')

        vol = nb.load(
            os.path.join(gl.baseDir, 'smp2', f'{gl.glmDir}{glm}', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
        beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        betas.append(beta)

    betas = np.array(betas)
    betas = betas[:, ~np.all(np.isnan(betas), axis=0)]

    assert betas.ndim == 2

    return betas


def get_roi_ResMS(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    R = get_roi(experiment, sn, Hem, roi)

    ResMS = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
    res = nt.sample_image(ResMS, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)

    res = res[~np.isnan(res)]

    return res


def get_roi_contrasts(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    regressors = reginfo['name'].unique()

    R = get_roi(experiment, sn, Hem, roi)

    contrasts = list()
    for regr, regressor in enumerate(regressors):
        vol = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                   f'con_{regressor.replace(" ", "")}.nii'))
        con = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        contrasts.append(con)

    contrasts = np.array(contrasts)
    contrasts = contrasts[:, ~np.all(np.isnan(contrasts), axis=0)]

    return contrasts


def main(args=None):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    struct = ['CortexLeft', 'CortexRight']
    path_rois = os.path.join(gl.baseDir, args.experiment, gl.roiDir)
    if args.what == 'save_betas_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'Hem.{H}.nii') for H in Hem]
        cifti = make_cifti_betas(path_glm, masks, struct)
        nb.save(cifti, path_glm + '/' + 'beta.dscalar.nii')
    if args.what == 'save_contrasts_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(path_rois, f'subj{args.sn}', f'Hem.{H}.nii') for H in Hem]
        cifti = make_cifti_contrasts(path_glm, masks, struct)
        nb.save(cifti, path_glm + '/' + 'contrast.dscalar.nii')
    if args.what == 'save_residuals_cifti':
        SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}'))  #
        SPM.get_info_from_spm_mat()

        for i, (s, H) in enumerate(zip(struct, Hem)):
            mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
            atlas = am.AtlasVolumetric(H, mask, structure=s)

            if i == 0:
                brain_axis = atlas.get_brain_model_axis()
                coords = nt.get_mask_coords(mask)
            else:
                brain_axis += atlas.get_brain_model_axis()
                coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

        res, _, info = SPM.get_residuals(coords)

        row_axis = nb.cifti2.SeriesAxis(1, 1, res.shape[0], 'second')

        save_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')

        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti = nb.Cifti2Image(
            dataobj=res,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )

        nb.save(cifti, save_path + '/' + 'residual.dtseries.nii')
    if args.what == 'save_betas_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_betas_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_contrasts_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_contrasts_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_contrasts_roi':
        con_dict = {
            'con': [],
            'condition': [],
            'sn': [],
            'roi': [],
            'Hem': []
        }
        for sn in args.snS:
            path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{sn}')
            cifti = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
            regr = cifti.header.get_axis(0).name[[0, 4, 7, 10, 2, 1, 5, 8, 11, 6, 9, 12, 3]]
            vol = nt.volume_from_cifti(cifti)
            for H in Hem:
                for roi in rois:
                    mask = os.path.join(path_rois, f'subj{sn}', f'ROI.{H}.{roi}.nii')
                    coords = nt.get_mask_coords(mask)
                    con = nt.sample_image(vol, coords[0], coords[1], coords[2], 0)
                    con = np.nanmean(con, axis=0)[[0, 4, 7, 10, 2, 1, 5, 8, 11, 6, 9, 12, 3]]
                    for i, c in enumerate(con):
                        con_dict['con'].append(c)
                        con_dict['condition'].append(regr[i])
                        con_dict['sn'].append(sn)
                        con_dict['roi'].append(roi)
                        con_dict['Hem'].append(H)

        con_df = pd.DataFrame(con_dict)
        con_df.to_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', 'ROI.con.avg.tsv'),
                      sep='\t',index=False)

    if args.what == 'save_residuals_cifti_all':
        for sn in args.snS:
            print(f'Processing subj{sn}...')
            arg = argparse.Namespace(
                what='save_residuals_cifti',
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
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)
    finish = time.time()

    print(f'Execution time: {finish - start} seconds')
