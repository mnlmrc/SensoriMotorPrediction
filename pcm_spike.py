import PcmPy as pcm
import mat73
import scipy.io as sio
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle
from pcm_models import find_model, normalize_Ac
from pcm_lfp import make_execution_models
import glob
from joblib import Parallel, delayed, parallel_backend


def load_spike(file_path):
    mat = sio.loadmat(file_path)
    return mat['spikes_s']


def align_spike(spike, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    cueTime = trial_info.probTime.to_numpy()
    pertTime = trial_info.pertTime.to_numpy()
    spike = spike[:, 0]
    n_unit = spike[0].shape[1]
    spike_aligned = np.zeros((preProb + postProb + prePert + postPert, n_unit, len(spike))) # time_unit_trial
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange])
        spike_aligned[..., t] = spike[t][fullRange]
    return spike_aligned


def save_spike_aligned(monkey='Malfoy', rec=1):
    print('loading spikes...')
    path = os.path.join(gl.baseDir, args.experiment, 'spikes', monkey)
    trial_info = pd.read_csv(path + f'/trial_info-{rec}.tsv', sep='\t')
    spike = load_spike(path + f'/spike-{rec}.mat')
    spike = spike[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    spike_aligned = align_spike(spike, trial_info)
    np.save(os.path.join(path, f'spike_aligned-{rec}.npy'), spike_aligned)


class Spikes:
    def __init__(self, M, spike, err=None, cond_vec=None, part_vec=None, t_axis=1, n_jobs=16):
        self.M = M
        self.spike = spike
        self.err = err # for prewhitening
        self.cond_vec = cond_vec
        self.part_vec = part_vec
        self.t_axis = t_axis
        self.timepoints = spike.shape[t_axis]
        self.n_jobs = n_jobs

    def G_obs_in_timepoint(self, t,):
        """

        Args:
            t (int): timepoint

        Returns:

        """

        spike = self.spike[:, t,]

        # do prewhitening
        if self.err is not None:
            err = self.err[:, t,]
            cov = err.T @ err
            spike = spike / np.sqrt(np.diag(cov))
            spike = spike[:, ~np.all(np.isnan(spike), axis=0)]

        obs_des = {'cond_vec': self.cond_vec,
                   'part_vec': self.part_vec}

        Y = pcm.dataset.Dataset(spike, obs_descriptors=obs_des)

        G_obs, _ = pcm.est_G_crossval(Y.measurements, Y.obs_descriptors['cond_vec'],
                                         Y.obs_descriptors['part_vec'],
                                         X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']))

        return G_obs, Y

    def fit_model_in_timepoint(self, t):
        """

        Args:
            M: list of models to fit
            Data: list of ntrials x timepoints datasets from each participants
            dat: list of DataFrames from dat files of each participant
            win: (start, end) tuple for timewindow

        Returns:

        """

        print(f'fitting model in window: {t}')

        G_obs, Y = self.G_obs_in_timepoint(t, )
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')

        return G_obs, T_in, theta_in, t


    def run_parallel_pcm_across_timepoints(self):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_in_timepoint)(t)
                for t in range(self.timepoints)
            )
        results = self._extract_results_from_parallel_process(results, field_names=['G_obs', 'T_in', 'theta_in', 't'])
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


def run_pcm(epoch='plan', monkey='Malfoy', M=None, model='component', datatype='aligned', rec=1):

    _, idx = find_model(M, model)

    print('loading spikes...')
    path = os.path.join(gl.baseDir, args.experiment, 'spikes', monkey)

    spike_rec = np.load(os.path.join(path, f'spike_{datatype}-{rec}.npy'))

    trial_info = pd.read_csv(os.path.join(path, f'trial_info-{rec}.tsv'), sep='\t')
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
    trial_info.cond = trial_info.cond.map(mapping)

    roi = pd.read_csv(os.path.join(path, f'roi-{rec}.txt'), sep='\t')['brainID'].to_numpy()
    roi_unique = np.unique(roi)

    for r in roi_unique:
        spike = spike_rec[:, roi==r, :]
        print('grouping by condition...')
        if epoch=='plan':
            spike_grouped, cond_vec, part_vec = pcm.group_by_condition(spike, trial_info.prob, trial_info.block, axis=-1)
        elif epoch=='exec':
            spike_grouped, cond_vec, part_vec = pcm.group_by_condition(spike, trial_info.cond, trial_info.block, axis=-1)

        print('preparing for prewhitening...')
        if epoch == 'plan':
            n_cond = trial_info.prob.unique().size
        if epoch == 'exec':
            n_cond = trial_info.cond.unique().size
        spike_grouped_reshaped = spike_grouped.reshape(n_cond, int(spike_grouped.shape[0] / n_cond),
                                                   spike_grouped.shape[1], spike_grouped.shape[2],)
        spike_grouped_avg = spike_grouped_reshaped.mean(axis=0, keepdims=True)
        spike_err = spike_grouped_reshaped - spike_grouped_avg
        spike_err = spike_err.reshape(spike_grouped.shape)

        print('doing pcm...')
        Spk = Spikes(M, spike_grouped, cond_vec=cond_vec, part_vec=part_vec)

        # Spk.G_obs_in_timepoint(13)
        # Spk.fit_model_in_timepoint(55)
        res_dict = Spk.run_parallel_pcm_across_timepoints()

        theta_in = []
        for t in range(spike.shape[0]):
            th = res_dict['theta_in'][t][idx].squeeze()
            theta_in.append(th)

        theta_in = np.array(theta_in)
        G_obs = np.array(res_dict['G_obs'])

        pass

        np.save(os.path.join(gl.baseDir, args.experiment, 'spikes', gl.pcmDir,
                             f'theta_in.spike.{model}.{monkey}.{r}.{datatype}.{epoch}-{rec}.npy'), theta_in, )
        np.save(os.path.join(gl.baseDir, args.experiment, 'spikes', gl.pcmDir,
                             f'G_obs.spike.{monkey}.{r}.{datatype}.{epoch}-{rec}.npy'), G_obs, )

def main(args):
    recordings = {
        'Malfoy': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'Pert': [3, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19, ],
    }
    rois = ['PFC', 'preSMA', 'SMA', 'PMd', 'M1', 'S1', 'VPL']
    if args.what == 'continuous_plan':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        for rec in recordings[args.monkey]:
            run_pcm('plan', args.monkey, M=M, model=args.model, rec=rec)
    if args.what=='continuous_exec':
        M = make_execution_models()
        for rec in recordings[args.monkey]:
            run_pcm('exec', args.monkey, M=M, model=args.model, rec=rec)
    if args.what == 'binned_plan':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        run_pcm('plan', args.monkey, M=M, model='component', datatype='binned')
    if args.what=='binned_exec':
        M = make_execution_models()
        run_pcm('exec', args.monkey, M=M, model='component', datatype='binned')
    if args.what=='align_spikes':
        for r in recordings[args.monkey]:
            save_spike_aligned(args.monkey, rec=r)
    # if args.what=='assemble_rois':
    #     path = os.path.join(gl.baseDir, args.experiment, 'spikes', gl.pcmDir)
    #     for roi in rois:
    #         pattern = os.path.join(path, f'G_obs.spike.{args.monkey}.{roi}.aligned.{args.epoch}-*.npy')
    #         files = glob.glob(pattern)
    #         for file in files:
    #             G_obs = np.load(file,)
    #             pass
    if args.what == 'save_binned':
        save_lfp_binned(args.monkey)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--model', type=str, default='component')
    parser.add_argument('--monkey', type=str, default='Pert')

    args = parser.parse_args()

    res_dict = main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')