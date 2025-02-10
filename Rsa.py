import argparse

import globals as gl

import os
import pandas as pd
import numpy as np

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

    betas = np.load(
        os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                     f'ROI.{Hem}.{roi}.beta.npy'))
    res = np.load(
        os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', f'ROI.{Hem}.{roi}.res.npy'))

    betas_prewhitened = betas / np.sqrt(res)

    betas_prewhitened = np.array(betas_prewhitened)

    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': reginfo.name.str.replace(" ", ""),
                         'run': reginfo.run})
    # remove_mean removes the mean ACROSS VOXELS for each condition
    rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
    rdm.rdm_descriptors = {'roi': [roi], 'hem': [Hem], 'index': [0]}
    rdm.reorder(rdm_index[f'glm{glm}'])

    return rdm


def calc_rdm_force(experiment=None, sn=None, Hem=None, roi=None, glm=None):

    df_force = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}',
                                     f'{experiment}_{sn}_force_single_trial.tsv'), sep='\t')
    df_force = df_force.groupby(['cue', 'stimFinger', 'BN']).mean(numeric_only=True)
    force = df_force[gl.channels['mov']].to_numpy()

    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': reginfo.name.str.replace(" ", ""),
                         'run': reginfo.run})
    # remove_mean removes the mean ACROSS VOXELS for each condition
    rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
    rdm.rdm_descriptors = {'roi': [roi], 'hem': [Hem], 'index': [0]}
    rdm.reorder(rdm_index[f'glm{glm}'])

    return rdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--glm', type=int, default=None)

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
                rdm.save(os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}',
                                      f'glm{args.glm}.{H}.{roi}.hdf5'), overwrite=True, file_type='hdf5')

    if args.what == 'save_rois_cosine':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:
                print(f'Hemisphere: {H}, region:{roi}')
                cos = calc_G_cosine(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm
                )
                np.save(os.path.join(gl.baseDir, args.experiment, gl.cosDir, f'subj{args.sn}',
                                     f'glm{args.glm}.{H}.{roi}.npy'), cos)


if __name__ == '__main__':
    rdm_index = {
        'glm12': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
    }

    main()
