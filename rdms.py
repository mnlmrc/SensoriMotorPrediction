import argparse

import globals as gl

import os
import pandas as pd
import numpy as np
import nibabel as nb
import nitools as nt

import rsatoolbox as rsa

import PcmPy as pcm
from rsatoolbox.inference import noise_ceiling


def D_to_rdm(D, descriptors=None, rdm_descriptors={}, pattern_descriptors=None, dissimilarity_measure=None):

    if D.ndim == 2:
        triu_rows, triu_cols = np.triu_indices(D.shape[0], k=1)
        dissimilarities = D[triu_rows, triu_cols]
    if D.ndim == 3:
        triu_rows, triu_cols = np.triu_indices(D.shape[-1], k=1)
        dissimilarities = D[:, triu_rows, triu_cols]
    rdm_dict = {
        'dissimilarities': dissimilarities,
        'descriptors': descriptors,
        'rdm_descriptors': rdm_descriptors,
        'pattern_descriptors': pattern_descriptors,
        'dissimilarity_measure': dissimilarity_measure,
    }
    rdm = rsa.rdm.rdms_from_dict(rdm_dict)

    return rdm


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
    # reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
    #                                    f'subj{sn}_reginfo.tsv'), sep="\t")

    cifti_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                      f'beta.dscalar.nii'))
    beta_img = nt.volume_from_cifti(cifti_img, struct_names = ['CortexLeft', 'CortexRight'])
    mask = nb.load(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}', f'ROI.{Hem}.{roi}.nii'))
    coords = nt.get_mask_coords(mask)

    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

    res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
    res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

    betas_prewhitened = betas / np.sqrt(res)

    betas_prewhitened = np.array(betas_prewhitened)
    betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
    conds = [r[0] for r in reginfo]
    run = [r[1] for r in reginfo]
    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': conds,
                         'run': run},
        descriptors={'ROI': roi, 'Hem': Hem, 'sn': sn}
    )
    # remove_mean removes the mean ACROSS VOXELS for each condition
    rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)
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


def main(args):
    Hem = ['L', 'R']
    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    if args.what == 'save_rois_rdms':
        rdms = []
        for H in Hem:
            for roi in rois:
                print(f'Participant {args.sn}, Hemisphere: {H}, region:{roi}')
                rdm = calc_rdm_roi(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    roi=roi,
                    glm=args.glm
                )
                rdms.append(rdm)
        path = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}',
                              f'glm{args.glm}.ROI.hdf5')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for rdm in rdms:
            rdm.descriptors['noise'] = None # kill noise to allow concatenation
        rdms = rsa.rdm.concat(rdms)
        rdms.save(path, overwrite=True, file_type='hdf5')
    if args.what == 'save_rois_rdms_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='save_rois_rdms',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'save_rois_rdms_avg':
        rdms = []
        for sn in args.snS:
            rdms_subj = rsa.rdm.load_rdm(
                os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{sn}', f'glm{args.glm}.ROI.hdf5'))
            rdms.append(rdms_subj)
        rdms = rsa.rdm.concat(rdms)
        rdms.save(os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'glm{args.glm}.ROI.hdf5'),
                  overwrite=True, file_type='hdf5')
    if args.what == 'save_rdm_emg':
        rdms = calc_rdm_emg(
            experiment=args.experiment,
            sn=args.sn,
        )
        path = os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'subj{args.sn}',
                                 'emg.hdf5')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rdms.save(path, overwrite=True, file_type='hdf5')

    if args.what == 'xval_corr':

        G_force = np.load(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'G_obs.force.plan.npy'))
        D_force = np.zeros_like(G_force)
        for g in range(G_force.shape[0]):
            D_force[g] = pcm.G_to_dist(G_force[g])
        rdm_descriptors= {'sn': args.snS}
        pattern_descriptors= {'conds': list(gl.regressor_mapping.keys())[:5]}
        dissimilarity_measure='Crossnobis'
        rdms_force = D_to_rdm(D_force,
                             rdm_descriptors=rdm_descriptors,
                             pattern_descriptors=pattern_descriptors,
                             dissimilarity_measure=dissimilarity_measure)
        rdms_rois = rsa.rdm.load_rdm(os.path.join(gl.baseDir, args.experiment, gl.rdmDir, f'glm{args.glm}.{args.atlas}.hdf5',))
        rdms_rois = rdms_rois.subsample_pattern('conds', ['0%', '25%', '50%', '75%', '100%'])

        xval_corr_dict = {
            'Hem': [],
            'roi': [],
            'sn': [],
            'corr': [],
            'cosine': [],
            'noise_ceiling_corr': [],
            'noise_ceiling_cosine': []
        }
        for H in Hem:
            for roi in rois:
                for sn in args.snS:
                    print(f'Hem:{H}, roi:{roi}, sn:{sn}')
                    sn_b = [i for i in args.snS if i != sn]

                    rdm_roi_a = rdms_rois.subsample('Hem', H).subsample('ROI', roi).subsample('sn', sn)
                    rdm_roi_b = rdms_rois.subsample('Hem', H).subsample('ROI', roi).subsample('sn', sn_b)

                    rdm_force_a = rdms_force.subsample('sn', sn)
                    rdm_force_b = rdms_force.subsample('sn', sn_b)

                    noise_ceiling_roi_corr = rsa.rdm.compare(rdm_roi_a, rdm_roi_b, method='corr')
                    noise_ceiling_force_corr = rsa.rdm.compare(rdm_force_a, rdm_force_b, method='corr')
                    noise_ceiling_roi_corr = np.maximum(0, noise_ceiling_roi_corr)
                    noise_ceiling_force_corr = np.maximum(0, noise_ceiling_force_corr)
                    noise_ceiling_corr = np.sqrt(noise_ceiling_roi_corr * noise_ceiling_force_corr)
                    corr = rsa.rdm.compare(rdm_roi_a, rdm_force_b, method='corr')

                    noise_ceiling_cosine = rsa.rdm.compare(rdm_roi_a, rdm_roi_b, method='cosine')
                    cosine = rsa.rdm.compare(rdm_roi_a, rdm_force_b, method='cosine')

                    xval_corr_dict['Hem'].append(H)
                    xval_corr_dict['roi'].append(roi)
                    xval_corr_dict['sn'].append(sn)
                    xval_corr_dict['corr'].append(corr.mean())
                    xval_corr_dict['cosine'].append(cosine.mean())
                    xval_corr_dict['noise_ceiling_cosine'].append(noise_ceiling_cosine.mean())
                    xval_corr_dict['noise_ceiling_corr'].append(noise_ceiling_corr.mean())

        xval_corr_df = pd.DataFrame(xval_corr_dict)
        xval_corr_df.to_csv(os.path.join(gl.baseDir, args.experiment, gl.rdmDir,
                                         f'{args.atlas}.glm{args.glm}.xval_corr.tsv'), sep='\t', index=False)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--snS', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])

    args = parser.parse_args()

    rdm_index = {
        'glm12': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
        'glm14': [0, 2, 3, 4, 1, 5]
    }

    main(args)
