import argparse
import pickle
import warnings

import PcmPy as pcm

warnings.filterwarnings("ignore")

import globals as gl
import pandas as pd
import numpy as np
import os


def make_execution_models():
    C = pcm.centering(8)

    v_fingerID = C @ np.array([1, 1, 1, 1, -1, -1, -1, -1])
    v_cue = C @ np.array([-1, 0, 1, 2, -2, -1, 0, 1, ])
    v_cert = C @ np.array([0.1875, .25, 0.1875, 0, 0, 0.1875, .25, 0.1875])  # variance of a Bernoulli distribution
    v_surprise = C @ -np.log2(np.array([.25, .5, .75, 1, 1, .75, .5, .25]))  # with Shannon information

    Ac = np.zeros((7, 8, 7))
    Ac[0, :, 0] = v_fingerID
    Ac[1, :, 1] = v_cue
    Ac[2, :, 2] = v_cert
    Ac[3, :, 3] = v_surprise
    Ac[4, :, 0] = v_cue
    Ac[5, :, 0] = v_cert
    Ac[6, :, 0] = v_surprise

    G_fingerID = np.outer(v_fingerID, v_fingerID)
    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    G_surprise = np.outer(v_surprise, v_surprise)

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))
    M.append(pcm.FixedModel('stimFinger', G_fingerID))
    M.append(pcm.FixedModel('cue', G_cue))
    M.append(pcm.FixedModel('cert', G_cert))
    M.append(pcm.FixedModel('surprise', G_surprise))
    M.append(pcm.ComponentModel('stimFinger+cue+cert+surprise (component)',
                                np.array([G_fingerID, G_cue, G_cert, G_surprise])))
    M.append(pcm.FeatureModel('stimFinger+cue+cert+surprise (feature)', Ac))
    M.append(pcm.FreeModel('ceil', 8))  # Noise ceiling model

    return M


def make_planning_models():
    C = pcm.centering(5)

    v_cue = C @ np.array([-2, -1, 0, 1, 2])
    v_cert = C @ np.array([0, 1, 2, 1, 0])
    G_cue_plan = np.outer(v_cue, v_cue)
    G_cert_plan = np.outer(v_cert, v_cert)

    M = []
    M.append(pcm.FixedModel('null', np.eye(5)))
    M.append(pcm.FixedModel('cue', G_cue_plan))
    M.append(pcm.FixedModel('cert', G_cert_plan))
    M.append(pcm.ComponentModel('cue+cert', np.array([G_cert_plan, G_cue_plan])))
    M.append(pcm.FreeModel('ceil', 5))  # Noise ceiling model

    return M


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='Icosahedron-1002')
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'save_rois_execution':

        M = make_execution_models()

        snS = [102, 103, 104, 106, 107]

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(snS)

                G_hat_betas = np.zeros((N, 8, 8))
                Y = list()

                for s, sn in enumerate(snS):
                    reginfo = pd.read_csv(
                        os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                     f'subj{sn}_reginfo.tsv'), sep='\t')

                    betas = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                                 f'subj{sn}', f'ROI.{H}.{roi}.beta.npy'))
                    res = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                               f'subj{sn}', f'ROI.{H}.{roi}.res.npy'))

                    betas_prewhitened = betas / np.sqrt(res)

                    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
                    part_vec = reginfo.run

                    idx = cond_vec.isin([5, 6, 7, 8, 9, 10, 11, 12])

                    obs_des = {'cond_vec': cond_vec[idx],
                               'part_vec': part_vec[idx]}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

                    G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                           Y[s].obs_descriptors['part_vec'],
                                                           X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_in.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_cv.exec.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_gr.exec.glm{args.glm}.{H}.{roi}.pkl'))

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_in.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_cv.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_gr.exec.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_gr, f)
    if args.what == 'save_emg_execution':

        M = make_execution_models()

        snS = [102, 103, 104, 106, 107]

        N = len(snS)

        G_hat_betas = np.zeros((N, 8, 8))
        Y = list()

        for s, sn in enumerate(snS):
            npz = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}',
                                       f'{args.experiment}_{sn}_binned.npz'), allow_pickle=True)

            emg = npz['data_array'][-1]
            descr = npz['descriptor'].item()
            timepoints = list(descr['time windows'].keys())

            dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                           f'{args.experiment}_{sn}.dat'), sep='\t')
            dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
            dat['cue'] = dat['cue'].map(gl.cue_mapping)

            cov = emg.T @ emg

            emg = emg / np.sqrt(np.diag(cov))

            dat[['ch_' + str(x) for x in range(emg.shape[-1])]] = emg

            dat = dat.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = dat['stimFinger'] + ',' + dat['cue']

            obs_des = {'cond_vec': cond_vec,
                       'part_vec': dat['BN']}

            Y.append(pcm.dataset.Dataset(dat[['ch_' + str(x) for x in range(emg.shape[-1])]], obs_descriptors=obs_des))

            G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                   Y[s].obs_descriptors['part_vec'],
                                                   X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

        T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                    f'T_in.emg.Vol.pkl'))
        T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                    f'T_cv.emg.Vol.pkl'))
        T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                    f'T_gr.emg.Vol.pkl'))

        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'theta_in.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_in, f)

        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'theta_cv.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_cv, f)

        with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'theta_gr.emg.Vol.pkl'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'save_rois_planning':

        M = make_planning_models()

        snS = [102, 103, 104, 106, 107]

        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for roi in rois:

                N = len(snS)

                G_hat_betas = np.zeros((N, 5, 5))
                Y = list()

                for s, sn in enumerate(snS):
                    reginfo = pd.read_csv(
                        os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}',
                                     f'subj{sn}_reginfo.tsv'), sep='\t')

                    betas = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                                 f'subj{sn}', f'ROI.{H}.{roi}.beta.npy'))
                    res = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}',
                                               f'subj{sn}', f'ROI.{H}.{roi}.res.npy'))

                    betas_prewhitened = betas / np.sqrt(res)

                    cond_vec = reginfo.name.str.replace(" ", "").map(gl.regressor_mapping)
                    part_vec = reginfo.run

                    idx = cond_vec.isin([0, 1, 2, 3, 4])

                    obs_des = {'cond_vec': cond_vec[idx],
                               'part_vec': part_vec[idx]}

                    Y.append(pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des))

                    G_hat_betas[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                                           Y[s].obs_descriptors['part_vec'],
                                                           X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

                T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
                T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True,
                                                              fixed_effect='block')
                T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

                T_in.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_in.plan.glm{args.glm}.{H}.{roi}.pkl'))
                T_cv.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_cv.plan.glm{args.glm}.{H}.{roi}.pkl'))
                T_gr.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                            f'T_gr.plan.glm{args.glm}.{H}.{roi}.pkl'))

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_in.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_in, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_cv.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_cv, f)

                with open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                       f'theta_gr.plan.glm{args.glm}.{H}.{roi}.pkl'), 'wb') as f:
                    pickle.dump(theta_gr, f)
