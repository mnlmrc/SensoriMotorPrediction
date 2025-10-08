import nibabel as nb
import os

import pandas as pd

import globals as gl
import numpy as np
import pickle

import argparse

import nitools as nt
from matplotlib import pyplot as plt
from nitools import spm

import time
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds

sn=104
experiment='smp2'
glm=12

# default [6, 16, 1, 1, 6, 0, 32]
mask_img = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj104', f'ROI.L.S1.nii')
SPM = spm.SpmGlm(os.path.join(gl.baseDir, experiment, f'glm12', f'subj104',))

print('Loading SPM...')
SPM.get_info_from_spm_mat()

X0 = SPM.design_matrix

SPM.convolve_glm(SPM.bf)

X1 = SPM.design_matrix

# coords = nt.get_mask_coords(mask_img)
# data = nt.sample_images(SPM.rawdata_files, coords, use_dataobj=True)
#
# beta, info, data_filt, data_hat, data_adj, residuals = SPM.rerun_glm(data.mean(axis=1)[:, None])
#
# print('Updating hrf params...')
# P = [6, 12, 1, 1, 6, 0, 32]
# _, y_hat, y_adj, residuals = SPM.update_hrf_params(P, mask_img)

# X1 = SPM.design_matrix

def inspect_hrf_params(experiment, glm, sn, GoNogo, atlas, roi, P):

    # default [6, 16, 1, 1, 6, 0, 32]
    mask_img = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}', f'{atlas}.L.{roi}.nii')
    SPM = spm.SpmGlm(os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}',))

    print('Loading SPM...')
    SPM.get_info_from_spm_mat()

    print('Updating hrf params...')
    _, y_hat, y_adj, _ = SPM.update_hrf_params(P, mask_img)

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}',
                                   f'{experiment}_{sn}.dat'), sep='\t')
    dat = dat[dat['GoNogo'] == GoNogo]
    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
    runs = pinfo[pinfo['sn'] == sn].FuncRuns.reset_index(drop=True)[0].split('.')
    nVols = pinfo[pinfo['sn'] == sn].numTR
    i = 0
    for BN in dat['BN'].unique():
        if str(BN) in runs:
            if i == 0:
                at = (dat[dat['BN']==BN].startTRReal).tolist()
            else:
                at.extend((dat[dat['BN']==BN].startTRReal + int(nVols * i)).tolist())
            i =+ 1
        else:
            print(f'excluding block {BN}')

    y_hat_cut = spm.cut(y_hat, 10, at, 20)
    y_adj_cut = spm.cut(y_adj, 10, at, 20)

    return y_hat_cut, y_adj_cut


def get_timeseries_in_voxels(path_glm, masks, struct):
    """

    Args:
        path_glm (str):
        masks (list): Must be non-overlapping voxels
        struct (list):

    Returns:

    """

    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        if isinstance(mask, str):
            atlas = am.AtlasVolumetric('region', mask, structure=s)
        else:
            raise Exception('mask must be the path to a .nii file')

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
        else:
            brain_axis += atlas.get_brain_model_axis()

    coords = nt.affine_transform_mat(brain_axis.voxel.T, brain_axis.affine)

    # get raw time series in roi
    y_raw = nt.sample_images(SPM.rawdata_files, coords)

    # rerun glm
    _, info, y_filt, y_hat, y_adj, _ = SPM.rerun_glm(y_raw)

    row_axis = nb.cifti2.SeriesAxis(1, 1, y_filt.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))

    cifti_yraw = nb.Cifti2Image(ataobj=y_raw,header=header,)
    cifti_yfilt = nb.Cifti2Image(dataobj=y_filt, header=header,)
    cifti_yhat = nb.Cifti2Image(dataobj=y_hat,header=header,)
    cifti_yadj = nb.Cifti2Image(dataobj=y_adj,header=header,)

    return cifti_yraw, cifti_yfilt, cifti_yhat, cifti_yadj

def get_timeseries_in_parcels(path_glm, masks, rois, struct, timeseries):

    for i, (s, mask, roi) in enumerate(zip(struct, masks, rois)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            label_vec, _ = atlas.get_parcel(roi)
            parcel_axis = atlas.get_parcel_axis()
        else:
            label_vec = np.concatenate((label_vec, atlas.get_parcel(roi)[0] + label_vec.max()), axis=0)
            parcel_axis += atlas.get_parcel_axis()

    data = timeseries.get_fdata()
    parcel_data, label = ds.agg_parcels(data, label_vec)

    row_axis = nb.cifti2.SeriesAxis(1, 1, parcel_data.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
    cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

    return cifti_parcel

def cut_timeseries_at_onsets(path_glm, masks, rois, struct, timeseries, at=None):
    for i, (s, mask, roi) in enumerate(zip(struct, masks, rois)):
        atlas = am.AtlasVolumetric(args.atlas, mask, structure=s)

        if i == 0:
            label_vec, _ = atlas.get_parcel(roi)
            parcel_axis = atlas.get_parcel_axis()
        else:
            label_vec = np.concatenate((label_vec, atlas.get_parcel(roi)[0] + label_vec.max()), axis=0)
            parcel_axis += atlas.get_parcel_axis()

    data = timeseries.get_fdata()

    y_cut = spm.cut(data, 10, at, 20).mean(axis=0)

    parcel_data, label = ds.agg_parcels(y_cut, label_vec)

    row_axis = nb.cifti2.SeriesAxis(-10, 1, parcel_data.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
    cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

    return cifti_parcel

def main(args):
    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']
    timeseries = ['y_raw', 'y_hat', 'y_adj', 'y_filt']
    if args.what == 'save_timeseries_cifti':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        print(f'participant {args.sn}, getting timeseries in voxels...')
        cifti_yraw, cifti_yfilt, cifti_yhat, cifti_yadj = get_timeseries_in_voxels(path_glm, masks, struct)
        nb.save(cifti_yraw, path_glm + '/' + 'y_raw.dtseries.nii')
        nb.save(cifti_yfilt, path_glm + '/' + 'y_filt.dtseries.nii')
        nb.save(cifti_yhat, path_glm + '/' + 'y_hat.dtseries.nii')
        nb.save(cifti_yadj, path_glm + '/' + 'y_adj.dtseries.nii')
    if args.what == 'save_timeseries_parcel':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        rois = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                for H in Hem]
        for ts in timeseries:
            print(f'participant {args.sn}, processing {ts} parcels...')
            cifti = nb.load(os.path.join(path_glm, f'{timeseries}.dtseries.nii'))
            cifti_parcel = get_timeseries_in_parcels(path_glm, masks, rois, struct, cifti)
            nb.save(cifti_parcel, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                        f'{args.atlas}.{ts}.ptseries.nii'))
    if args.what == 'save_timeseries_cut':
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        masks = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'Hem.{H}.nii')
                 for H in Hem]
        rois = [os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}', f'{args.atlas}.{H}.nii')
                for H in Hem]
        # define onsets (experiment-specific)
        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{args.sn}',
                                       f'{args.experiment}_{args.sn}.dat'), sep='\t')
        dat = dat[dat['GoNogo'] == args.GoNogo]
        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        runs = pinfo[pinfo['sn'] == args.sn].FuncRuns.reset_index(drop=True)[0].split('.')
        nVols = pinfo[pinfo['sn'] == args.sn].numTR.reset_index(drop=True)[0]
        i = 0
        for BN in dat['BN'].unique():
            if str(BN) in runs:
                if i == 0:
                    at = (dat[dat['BN']==BN].startTRReal).tolist()
                else:
                    at.extend((dat[dat['BN']==BN].startTRReal + int(nVols * i)).tolist())
                i += 1
            else:
                print(f'excluding block {BN}')
        for ts in timeseries:
            print(f'participant {args.sn}, processing {ts} cut parcels...')
            cifti = nb.load(os.path.join(path_glm, f'{ts}.dtseries.nii'))
            cifti_parcel_cut = cut_timeseries_at_onsets(path_glm, masks, rois, struct, cifti, at=at)
            nb.save(cifti_parcel_cut, os.path.join(gl.baseDir, args.experiment,
                                        f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                        f'{args.atlas}.{ts}.{args.GoNogo}.cut.ptseries.nii'))

    if args.what == 'save_timeseries_cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_timeseries_cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_timeseries_parcel_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_timeseries_parcel',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                atlas=args.atlas
            )
            main(args)
    if args.what == 'save_timeseries_cut_all':
        GoNogo = ['go', 'nogo']
        for sn in args.snS:
            for go in GoNogo:
                args = argparse.Namespace(
                    what='save_timeseries_cut',
                    experiment=args.experiment,
                    sn=sn,
                    glm=args.glm,
                    GoNogo=go,
                    atlas=args.atlas
                )
                main(args)
    if args.what == 'save_timeseries_all':
        commands = ['save_timeseries_cifti_all', 'save_timeseries_parcel_all', 'save_timeseries_cut_all']
        for cmd in commands:
            args = argparse.Namespace(
                what=cmd,
                experiment=args.experiment,
                glm=args.glm,
                atlas=args.atlas,
                snS=args.snS
            )
            main(args)

    if args.what == 'save_timeseries_cut_avg':
        dataGo, dataNogo = [], []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            y_adj_go = nb.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                            f'{args.atlas}.y_adj.go.cut.ptseries.nii'))
            y_adj_nogo = nb.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                            f'{args.atlas}.y_adj.nogo.cut.ptseries.nii'))

            dataGo.append(y_adj_go.dataobj)
            dataNogo.append(y_adj_nogo.dataobj)

            parcel_axis_tmp = y_adj_go.header.get_axis(1)
            parcel_axis_tmp.affine = None # remove affine to allow concatenation

            if args.snS.index(sn) == 0:
                parcel_axis = parcel_axis_tmp
                row_axis = y_adj_go.header.get_axis(0)
            else:
                parcel_axis += parcel_axis_tmp

        dataGo = np.hstack(dataGo)
        dataNogo = np.hstack(dataNogo)

        header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
        cifti_parcel_go = nb.Cifti2Image(dataGo, header=header)
        nb.save(cifti_parcel_go, os.path.join(gl.baseDir, args.experiment,
                                           f'{gl.glmDir}{args.glm}',
                                           f'{args.atlas}.y_adj.go.cut.ptseries.nii'))
        cifti_parcel_nogo = nb.Cifti2Image(dataNogo, header=header)
        nb.save(cifti_parcel_nogo, os.path.join(gl.baseDir, args.experiment,
                                           f'{gl.glmDir}{args.glm}',
                                           f'{args.atlas}.y_adj.nogo.cut.ptseries.nii'))

    if args.what == 'inspect_hrf_params':

        inspect_hrf_params(args.experiment,
                           args.glm,
                           args.sn,
                           args.GoNogo,
                           args.atlas,
                           'S1',
                           [6, 12, 1, 1, 6, 0, 32])


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112], type=int)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--roi', type=str, default='S1')
    parser.add_argument('--P', nargs='+', type=int, default=[6, 16, 1, 1, 6, 0, 32])
    parser.add_argument('--GoNogo', type=str, default=None)

    args = parser.parse_args()

    main(args)
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
