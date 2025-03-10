import argparse

import globals as gl

import os
import pandas as pd
import numpy as np
import nibabel as nb
import nitools as nt

import rsatoolbox as rsa

import PcmPy as pcm


def calc_G_cosine(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    betas = np.load(
        os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                     f'ROI.{Hem}.{roi}.beta.npy'))
    res = np.load(
        os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', f'ROI.{Hem}.{roi}.res.npy'))

    betas_prewhitened = betas / np.sqrt(res)
    betas_prewhitened = np.array(betas_prewhitened)

    condition = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)

    Z = pcm.matrix.indicator(condition)
    G, Sig = pcm.est_G_crossval(betas_prewhitened, Z, reginfo.run)

    cos = G_to_cosine(G)

    return cos


def G_to_cosine(G):
    """
    Converts a second moment matrix G into a cosine angle matrix.

    Parameters:
        G (numpy.ndarray)
            An n_cond x n_cond second-moment matrix.

    Returns:
        angle_matrix (numpy.ndarray)
            An n_cond x n_cond matrix where each entry (i, j) represents
            the cosine angle between condition i and condition j.
    """
    # Normalize each row to unit length
    norm_G = np.linalg.norm(G, axis=1, keepdims=True)
    G_norm = G / norm_G

    # Compute cosine similarity matrix
    cosine_similarity = np.dot(G_norm, G_norm.T)

    # Clip to prevent numerical issues outside the valid domain of arccos
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    # Compute cosine angles in radians
    cos = np.arccos(cosine_similarity)

    return cos


def calc_rdm_roi(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    # betas = np.load(
    #     os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
    #                  f'ROI.{Hem}.{roi}.beta.npy'))
    beta_img = nt.volume_from_cifti(nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                      f'beta.dscalar.nii')), struct_names = ['CortexLeft', 'CortexRight'])
    mask = nb.load(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}', f'ROI.{Hem}.{roi}.nii'))
    coords = nt.get_mask_coords(mask)

    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

    res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
    res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)
    # res = np.load(
    #     os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', f'ROI.{Hem}.{roi}.res.npy'))

    betas_prewhitened = betas / np.sqrt(res)

    betas_prewhitened = np.array(betas_prewhitened)
    betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': reginfo.name.str.replace(" ", ""),
                         'run': reginfo.run},
    )
    # remove_mean removes the mean ACROSS VOXELS for each condition
    rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
    rdm.rdm_descriptors = {'roi': [roi], 'hem': [Hem], 'index': [0]}
    rdm.reorder(rdm_index[f'glm{glm}'])

    return rdm


def calc_rdm_emg(experiment=None, sn=None):
    npz = np.load(os.path.join(gl.baseDir, experiment, 'emg', f'subj{sn}', f'{experiment}_{sn}_binned.npz'),
                  allow_pickle=True)

    emg = npz['data_array']
    descr = npz['descriptor'].item()
    timepoints = list(descr['time windows'].keys())

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}', f'{experiment}_{sn}.dat'),
                      sep='\t')
    dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
    dat['cue'] = dat['cue'].map(gl.cue_mapping)

    rdms = list()
    for tp in range(1, emg.shape[0]):
        emg_tmp = emg[tp]

        cov = emg_tmp.T @ emg_tmp

        emg_tmp = emg_tmp / np.sqrt(np.diag(cov))

        dat_tmp = dat.copy()
        dat_tmp[['ch_' + str(x) for x in range(emg.shape[-1])]] = emg_tmp

        dat_tmp = dat_tmp.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
        conds = dat_tmp['stimFinger'] + ',' + dat_tmp['cue']

        dataset = rsa.data.Dataset(
            dat_tmp[['ch_' + str(x) for x in range(emg.shape[-1])]].to_numpy(),
            channel_descriptors={
                'channel': np.array(['ch_' + str(x) for x in range(emg.shape[-1])])},
            obs_descriptors={'conds': conds,
                             'run': dat_tmp['BN']},
            descriptors={'timepoint': timepoints[tp]},
        )

        rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
        rdm.reorder(np.array([1, 2, 3, 0, 4, 5, 6, 7]))
        rdms.append(rdm)

    rdms = rsa.rdm.concat(rdms)

    return rdms


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    if args.what == 'save_rois_rdms':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                rdm = calc_rdm_roi(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm
                )
                path = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}',
                                      f'glm{args.glm}.{H}.{roi}.hdf5')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                rdm.save(path, overwrite=True, file_type='hdf5')

    if args.what == 'save_rdm_emg':
        rdms = calc_rdm_emg(
            experiment=args.experiment,
            sn=args.sn,
        )
        path = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}',
                                 'emg.hdf5')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rdms.save(path, overwrite=True, file_type='hdf5')


if __name__ == '__main__':
    rdm_index = {
        'glm12': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
    }

    main()
