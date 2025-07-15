import warnings

import time

from globals import regressor_mapping

warnings.filterwarnings("ignore")

import argparse
import pickle
from itertools import combinations
import rsatoolbox as rsa

from rdms import D_to_rdm

import PcmPy as pcm
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

import globals as gl
import pandas as pd
import numpy as np
import os

from pcm_cortical import make_execution_models, make_planning_models, find_model

import sys


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

    if args.what == 'emg_G_obs_continuous':
        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :2148].mean(axis=-1, keepdims=True)
            # emg_norm = emg_rect / bs
            emg.append(emg_rect)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                                f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        G_obs = []
        for tp in range(6444):
            print(f'tp: {tp}/{6444}')
            G_obs_tmp, _ = G_obs_in_timepoint(emg, dat, tp)
            G_obs.append(G_obs_tmp)

        np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.emg.continuous.npy'), np.array(G_obs))

    if args.what == 'continuous':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.pkl'), "wb")
        pickle.dump(M, f)

        _, idx = find_model(M, 'feature')

        width = 0.025  # 20 ms
        n_wins = 100
        start = -0.1
        end = 0.5
        wins = [(t - width / 2, t + width / 2) for t in np.linspace(start, end, n_wins)]

        fs = 2148
        onset = int(1 * fs)

        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :onset].mean(axis=-1, keepdims=True)
            emg_norm = emg_rect - bs
            emg.append(emg_norm)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                               f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        Emg = EMG(M, emg, dat, wins, onset, 2148)
        # Emg.G_obs_in_timepoint(wins[0])
        res_dict = Emg.run_parallel_pcm_across_timepoints()

        theta_in = []
        for w, win in enumerate(wins):
            th = res_dict['theta_in'][w][idx]
            theta_in.append(th)

        theta_in = np.array(theta_in)

        descr = {
            'wins': wins,
            'width': width
        }

        np.savez(
            os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'theta_in.emg.continuous.npz'),
            theta=theta_in,
            descr=descr  # This dict will be saved as a single object
        )

    if args.what == 'binned':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.pkl'), "wb")
        pickle.dump(M, f)

        wins = [(-.1, 0.0), (.025, .05), (.05, .1), (.1, .5)]
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']

        fs = 2148
        # latency = .055 * fs
        onset = int(1 * fs)

        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :onset].mean(axis=-1, keepdims=True)
            emg_norm = emg_rect - bs
            emg.append(emg_norm)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                               f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        Emg = EMG(M, emg, dat, wins, onset, 2148)
        Emg.G_obs_in_timepoint(wins[0])
        res_dict = Emg.run_parallel_pcm_across_timepoints()

        # return res_dict

        for w, win in enumerate(wins):
            theta_in = res_dict['theta_in'][w]
            T_cv = res_dict['T_cv'][w]
            T_gr= res_dict['T_gr'][w]
            G_obs = res_dict['G_obs'][w]

            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_in.emg.{epochs[w]}.p'), "wb")
            pickle.dump(theta_in, f)

            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_cv.emg.{epochs[w]}.p'), "wb")
            pickle.dump(T_cv, f)

            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T_gr.emg.{epochs[w]}.p'), "wb")
            pickle.dump(T_gr, f)

            np.save(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.emg.{epochs[w]}.npy'), G_obs)

        return res_dict

    if args.what == 'model_family':
        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.emg.p'), "wb")
        pickle.dump(M, f)

        wins = [(-1.0, 0.0), (.025, .05), (.05, .1), (.1, .5)]
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']

        fs = 2148
        # latency = .05 * fs
        onset = int(1 * fs)

        emg = []
        dat = []
        for sn in args.snS:
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :onset].mean(axis=-1, keepdims=True)
            emg_norm = emg_rect - bs
            emg.append(emg_norm)
            dat_tmp = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                               f'{args.experiment}_{sn}.dat'), sep='\t')
            dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
            dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
            dat_tmp['BN'] = dat_tmp['BN'].astype(str)
            dat.append(dat_tmp)

        Emg = EMG(M, emg, dat, wins, onset, 2148)
        res_dict = Emg.fit_model_family_across_rois('component', basecomp=np.eye(8)[None, :, :], # basecomp needs to be num_basecompxNxN
                                             comp_names=['finger', 'cue', 'surprise'])

        for w, win in enumerate(wins):
            theta = res_dict['theta'][w]
            T = res_dict['T'][w]

            T.to_pickle(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'T.model_family.emg.{epochs[w]}.p'))

            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta.model_family.emg.{epochs[w]}.p'), 'wb')
            pickle.dump(theta, f)

    if args.what == 'force_execution':

        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.force.exec.p'), "wb")
        pickle.dump(M, f)

        N = len(args.snS)

        G_obs = np.zeros((N, 8, 8))
        Y = list()
        for s, sn in enumerate(args.snS):
            force = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                             f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
            force = force[force['GoNogo'] == 'go'] if 'GoNogo' in force else force # select only go trial
            force['cue'] = force['cue'].map(gl.cue_mapping)
            force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
            force = force.groupby(['BN', 'stimFinger', 'cue']).mean(numeric_only=True).reset_index()
            cond_vec = force['cue'] + ',' + force['stimFinger']
            part_vec = force['BN']

            force = force[['thumb1', 'index1', 'middle1', 'ring1', 'pinkie1']].to_numpy()

            cov = force.T @ force

            force = force / np.sqrt(np.diag(cov)) # prewhitening using variance of each channel

            obs_des = {'cond_vec': cond_vec.map(gl.regressor_mapping),
                       'part_vec': part_vec}

            Y.append(pcm.dataset.Dataset(force, obs_descriptors=obs_des))

            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements, Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, M, fit_scale=True, verbose=True, fixed_effect='block')

        path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)

        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, f'G_obs.force.exec.npy'), G_obs)

        T_in.to_pickle(os.path.join(path, f'T_in.force.exec.p'))
        T_cv.to_pickle(os.path.join(path, f'T_cv.force.exec.p'))
        T_gr.to_pickle(os.path.join(path, f'T_gr.force.exec.p'))

        with open(os.path.join(path, f'theta_in.force.exec.p'), 'wb') as f:
            pickle.dump(theta_in, f)
        with open(os.path.join(path, f'theta_cv.force.exec.p'), 'wb') as f:
            pickle.dump(theta_cv, f)
        with open(os.path.join(path, f'theta_gr.force.exec.p'), 'wb') as f:
            pickle.dump(theta_gr, f)

    if args.what == 'force_planning':

        M = make_planning_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                               f'M.force.plan.p'), "wb")
        pickle.dump(M, f)

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

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp0')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[100, 101, 102, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epochs', nargs='+', type=str, default=['Pre', 'SLR', 'LLR', 'Vol'])

    args = parser.parse_args()

    res_dict = main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')