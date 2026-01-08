import warnings
import time
import argparse
import pickle
from itertools import combinations
import rsatoolbox as rsa
import PcmPy as pcm
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
import globals as gl
import pandas as pd
import numpy as np
import os
import sys
from imaging_pipelines.util import bootstrap_correlation, bootstrap_summary

warnings.filterwarnings("ignore")


class EMG():
    def __init__(self, M, Data, dat, wins, onset, fsample, n_jobs=16):

        self.M = M
        self.Data = Data
        self.dat = dat
        self.wins = wins
        self.onset = onset
        self.fsample = fsample
        self.n_jobs = n_jobs


    def G_obs_in_timepoint(self, win):
        """

        Args:
            Data: list of ntrials x timepoints datasets from each participants
            dat: list of DataFrames from dat files of each participant
            win: (start, end) tuple for timewindow

        Returns:

        """

        Data = self.Data
        dat = self.dat
        onset = self.onset
        fs = self.fsample

        Y = list()
        N = len(Data)
        G_obs = np.zeros((N, 8, 8))
        for s, (D, dd) in enumerate(zip(Data, dat)):
            emg = D[..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=-1)

            # cov = emg.T @ emg
            #
            # emg = emg / np.sqrt(np.diag(cov))

            channels = ['ch_' + str(x) for x in range(emg.shape[-1])]

            dd[channels] = emg
            dd = dd.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            dd_avg = dd.groupby(['stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            emg_avg = dd_avg[channels]

            cond_vec = dd['cue'] + ',' + dd['stimFinger']
            part_vec = dd['BN']

            obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
                       'part_vec': part_vec}

            meas = dd[channels].to_numpy()
            err = np.zeros_like(meas)
            for b, BN in enumerate(dd['BN'].unique()):
                err[b:b+emg_avg.shape[0]] = meas[b:b+emg_avg.shape[0]] - emg_avg

            cov = err.T @ err
            meas_prewhitened = meas / np.sqrt(np.diag(cov))

            Y.append(pcm.dataset.Dataset(meas_prewhitened, obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        return G_obs, Y

    # need to become a class here:
    def fit_model_in_timepoint(self, win):
        """

        Args:
            M: list of models to fit
            Data: list of ntrials x timepoints datasets from each participants
            dat: list of DataFrames from dat files of each participant
            win: (start, end) tuple for timewindow

        Returns:

        """

        print(f'fitting model in window: {win}')

        G_obs, Y = self.G_obs_in_timepoint(win)

        _, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, _ = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, _ = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')

        return G_obs, T_cv, T_gr, theta_in, win

    def run_parallel_pcm_across_timepoints(self):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_in_timepoint)(win)
                for win in self.wins
            )

        # for roi in self.roi_imgs:
        #     self.run_pcm_in_roi(roi)

        results = self._extract_results_from_parallel_process(results,
                                      field_names=['G_obs', 'T_cv', 'T_gr', 'theta_in', 'win'])
        return results

    def _extract_results_from_parallel_process(self, results, field_names):
        res_dict = {key: [] for key in field_names}
        for r, result in enumerate(results):
            if len(result) != len(field_names):
                raise ValueError(f"Expected {len(field_names)} values, got {len(result)} at index {r}")
            # res_dict['roi_img'].append(self.roi_imgs[r])
            for key, value in zip(field_names, result):
                res_dict[key].append(value)
        return res_dict


    def fit_model_family_across_rois(self, model, basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_timepoint)(win, model, basecomp, comp_names)
                for win in self.wins
            )
        results = self._extract_results_from_parallel_process(results, ['T', 'theta'])
        return results

    def fit_model_family_in_timepoint(self, win, model, basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
        MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        _, Y = self.G_obs_in_timepoint(win)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)

        return T, theta


def main(args):

    if args.what == 'force_planning':

        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)

        N = len(args.snS)

        G_obs = np.zeros((N, 5, 5))
        Y = list()
        for s, sn in enumerate(args.snS):
            force = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                             f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
            #force = force[force['GoNogo'] == 'nogo'] if 'GoNogo' in force else force # select only go trial
            force['cue'] = force['cue'].map(gl.cue_mapping)
            # force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
            force = force.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = force['cue'] #+ ',' + force['stimFinger']
            part_vec = force['BN']

            force = force[['thumb0', 'index0', 'middle0', 'ring0', 'pinkie0']].to_numpy()

            cov = force.T @ force

            force = force / np.sqrt(np.diag(cov)) # prewhitening using variance of each channel

            obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
                       'part_vec': part_vec}

            Y.append(pcm.dataset.Dataset(force, obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

        path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, f'G_obs.force.plan.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.force.plan.p'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.force.plan.p'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.force.plan.p'))

        with open(os.path.join(path, f'theta_in.force.plan.p'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.force.plan.p'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.force.plan.p'), 'wb') as f:
            pickle.dump(theta_gr, f)
    if args.what == 'correlation_cue-finger':
        rng = np.random.default_rng(0)  # seed for reproducibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        path_emg = os.path.join(gl.baseDir, args.experiment, 'emg',)
        path_behav = os.path.join(gl.baseDir, args.experiment, gl.behavDir)
        path_pcm = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
        fs = 2148
        wins = [(-.1, 0.0), (.025, .05), (.05, .1), (.1, .5)]
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']
        onset = 1.
        Y = {epoch: [] for epoch in epochs}
        for sn in args.sns:
            print(f'loading participant {sn}')
            emg_raw = np.load(os.path.join(path_emg, f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :int(onset * fs)].mean(axis=-1, keepdims=True)
            emg_norm = emg_rect - bs
            dat = pd.read_csv(os.path.join(path_behav, f'subj{sn}', f'{args.experiment}_{sn}.dat'), sep='\t')
            dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
            dat['cue'] = dat['cue'].map(gl.cue_mapping)
            cond_name = dat['cue'] + ',' + dat['stimFinger']
            cond_vec = cond_name.map(gl.regressor_mapping)
            part_vec = dat['BN']
            n_part = dat['BN'].nunique()
            n_chan = emg_norm.shape[1]
            mask_plan = {'index': np.array([0, 1, 0, 0, 1, 0, 0, 0] * n_part, dtype=bool),
                         'ring': np.array([0, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool)}
            mask_exec = {'index': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                         'ring': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}
            obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                       'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
            for w, (win, epoch) in enumerate(zip(wins, epochs)):
                emg_w = emg_norm[..., int((win[0] + onset) * fs):int((win[1] + onset) * fs)].mean(axis=-1)
                emg_gr, cond_vec_gr, part_vec_gr = pcm.group_by_condition(emg_w, cond_vec, part_vec, axis=0)

                plani = emg_gr[mask_plan['index']].reshape(n_part, 2, n_chan).mean(axis=1)
                planr = emg_gr[mask_plan['ring']].reshape(n_part, 2, n_chan).mean(axis=1)
                plan = plani - planr
                plan = plan - plan.mean(axis=-1, keepdims=True)

                execi = emg_gr[mask_exec['index']].reshape(n_part, 4, n_chan).mean(axis=1)
                execr = emg_gr[mask_exec['ring']].reshape(n_part, 4, n_chan).mean(axis=1)
                exec = execi - execr
                exec = exec - exec.mean(axis=-1, keepdims=True)

                data = np.r_[plan, exec]
                X = pcm.indicator(obs_des['part_vec'])
                beta, *_ = np.linalg.lstsq(X, data)  # dimord part, channel
                err = data - X @ beta
                cov = (err.T @ err) / data.shape[0]
                data_prewhitened = data / np.sqrt(np.diag(cov))
                Y[epoch].append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))

        N, E = len(args.sns), len(epochs)
        r_indiv = np.zeros((N, E))
        SNR = np.zeros_like(r_indiv)
        B = 1000
        r_group = np.zeros(E)
        for e, epoch in enumerate(epochs):
            print(f'doing ML estimation for epoch: {epoch}...')
            _, theta_in = pcm.fit_model_individ(Y[epoch], Mflex, fixed_effect=None, fit_scale=False, verbose=False)
            _, theta_gr = pcm.fit_model_group(Y[epoch], Mflex, fixed_effect=None, fit_scale=True, verbose=False)

            indeces = rng.integers(0, N, size=(B, N))
            results = Parallel(n_jobs=16, backend='loky')(
                delayed(bootstrap_correlation)(idx, Y[epoch], Mflex) for idx in indeces)
            r_bootstrap = np.array([r for r in results if r is not None])
            n_disc = len(results) - len(r_bootstrap)
            print(f'{epoch}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')

            np.save(os.path.join(path_pcm, f'r_bootstrap.corr_cue-finger.emg.{epoch}.npy'), r_bootstrap)
            f = open(os.path.join(path_pcm, f'theta_in.corr_cue-finger.emg.{epoch}.p'), 'wb')
            pickle.dump(theta_in, f)
            f = open(os.path.join(path_pcm, f'theta_gr.corr_cue-finger.emg.{epoch}.p'), 'wb')
            pickle.dump(theta_gr, f)
    if args.what == 'G_obs_win':
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)
        path_emg = os.path.join(gl.baseDir, args.experiment, 'emg', )
        path_behav = os.path.join(gl.baseDir, args.experiment, gl.behavDir)
        path_pcm = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
        fs = 2148
        wins = [(-.1, 0.0), (.025, .05), (.05, .1), (.1, .5)]
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']
        onset = 1.
        Y = {epoch: [] for epoch in epochs}
        G = np.zeros((len(args.sns), len(epochs), 8, 8))
        for s, sn in enumerate(args.sns):
            print(f'loading participant {sn}')
            emg_raw = np.load(os.path.join(path_emg, f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :int(onset * fs)].mean(axis=-1, keepdims=True)
            emg_norm = emg_rect - bs
            dat = pd.read_csv(os.path.join(path_behav, f'subj{sn}', f'{args.experiment}_{sn}.dat'), sep='\t')
            dat['stimFinger'] = dat['stimFinger'].map(gl.stimFinger_mapping)
            dat['cue'] = dat['cue'].map(gl.cue_mapping)
            cond_name = dat['cue'] + ',' + dat['stimFinger']
            cond_vec = cond_name.map(gl.regressor_mapping).to_numpy()
            part_vec = dat['BN'].to_numpy()
            for w, (win, epoch) in enumerate(zip(wins, epochs)):
                emg_w = emg_norm[..., int((win[0] + onset) * fs):int((win[1] + onset) * fs)].mean(axis=-1)
                emg_gr, cond_vec_gr, part_vec_gr = pcm.group_by_condition(emg_w, cond_vec, part_vec, axis=0)
                obs_des = {'cond_vec': cond_vec_gr, 'part_vec': part_vec_gr}
                X = pcm.indicator(part_vec_gr)
                beta, *_ = np.linalg.lstsq(X, emg_gr) # dimord part, channel
                err = emg_gr - X @ beta
                cov = (err.T @ err) / emg_gr.shape[0]
                emg_prewhitened = emg_gr / np.sqrt(np.diag(cov))
                Y = pcm.dataset.Dataset(emg_prewhitened, obs_descriptors=obs_des)
                G[s, w], _ = pcm.est_G_crossval(Y.measurements,
                                             Y.obs_descriptors['cond_vec'],
                                             Y.obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']))

        np.save(os.path.join(path_pcm, f'G_obs.emg.npy'), G)
    if args.what == 'corr2tsv':
        if args.what == 'corr2tsv':
            corrs = ['cue-finger']
            epochs = ['Pre', 'SLR', 'LLR', 'Vol']
            corr_dict = {
                'r_indiv': [],
                'r_group': [],
                'SNR': [],
                'corr': [],
                'ci_lo': [],
                'ci_hi': [],
                'epoch': [],
                'participant_id': []
            }
            f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
            Mflex = pickle.load(f)
            for corr in corrs:
                for epoch in epochs:
                    f = open(os.path.join(gl.baseDir, args.experiment,gl.pcmDir, f'theta_in.corr_{corr}.emg.{epoch}.p'), 'rb')
                    theta = pickle.load(f)[0]
                    r_bootstrap = np.load(
                        os.path.join(gl.baseDir, args.experiment,gl.pcmDir, f'r_bootstrap.corr_{corr}.emg.{epoch}.npy'))
                    f = open(os.path.join(gl.baseDir, args.experiment,gl.pcmDir, f'theta_gr.corr_{corr}.emg.{epoch}.p'), 'rb')
                    theta_g = pickle.load(f)[0]

                    N = theta.shape[1]
                    sigma2_1 = np.exp(theta[0])
                    sigma2_2 = np.exp(theta[1])
                    r_indiv = Mflex.get_correlation(theta)
                    sigma2_e = np.exp(theta[3])
                    SNR = np.sqrt(sigma2_1 * sigma2_2) / sigma2_e

                    theta_g, _ = pcm.group_to_individ_param(theta_g, Mflex, N)
                    r_group = Mflex.get_correlation(theta_g)
                    (ci_lo, ci_hi), _, _ = bootstrap_summary(r_bootstrap, alpha=0.025)

                    corr_dict['r_indiv'].extend(r_indiv)
                    corr_dict['r_group'].extend(r_group)
                    corr_dict['ci_lo'].extend([ci_lo] * r_indiv.shape[0])
                    corr_dict['ci_hi'].extend([ci_hi] * r_indiv.shape[0])
                    corr_dict['SNR'].extend(SNR)
                    corr_dict['corr'].extend([corr] * r_indiv.shape[0])
                    corr_dict['participant_id'].extend(args.sns)
                    corr_dict['epoch'].extend([epoch] * r_indiv.shape[0])
            df_corr = pd.DataFrame(corr_dict)
            df_corr.to_csv(os.path.join(gl.baseDir, args.experiment,gl.pcmDir, 'correlations.EMG.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp0')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[100, 101, 102, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epochs', nargs='+', type=str, default=['Pre', 'SLR', 'LLR', 'Vol'])

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')