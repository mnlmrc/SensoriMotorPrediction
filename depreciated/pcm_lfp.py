import PcmPy as pcm
import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle
from pcm_models import find_model, normalize_Ac

from joblib import Parallel, delayed, parallel_backend


def make_execution_models(centering=True):

    C = pcm.centering(8)

    if centering:
        v_finger = C @ np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = C @ np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
        v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = C @ -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information
    else:
        v_finger = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        v_cue = np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
        v_cert = np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
        v_surprise = -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information

    Ac = np.zeros((3, 8, 3))
    Ac[0, :, 0] = v_finger
    Ac[1, :, 1] = v_finger
    Ac[2, :, 1] = v_cue

    Ac = normalize_Ac(Ac)

    G_finger = np.outer(v_finger, v_finger)
    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    G_surprise = np.outer(v_surprise, v_surprise)
    G_component = np.array([G_finger / np.trace(G_finger),
                            G_cue / np.trace(G_cue),
                            G_cert / np.trace(G_cert),
                            G_surprise / np.trace(G_surprise)
                            ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))  # 0
    M.append(pcm.FixedModel('finger', G_finger))  # 1
    M.append(pcm.FixedModel('cue', G_cue))  # 2
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 3
    M.append(pcm.FixedModel('surprise', G_surprise))  # 4
    M.append(pcm.ComponentModel('component', G_component))  # 5
    M.append(pcm.FeatureModel('feature', Ac))  # 6
    M.append(pcm.FreeModel('ceil', 8))  # 7

    return M


def load_lfp(file_path):
    mat = mat73.loadmat(file_path)
    return mat['lfp']

def align_lfp(lfp, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    cueTime = trial_info.probTime.to_numpy()
    pertTime = trial_info.pertTime.to_numpy()
    lfp_aligned = np.zeros((preProb + postProb + prePert + postPert, lfp.shape[1], lfp.shape[2], lfp.shape[3]))
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange])
        lfp_aligned[...,t] = lfp[fullRange,...,t]
    return lfp_aligned

def make_freq_masks(cfg):
    foi = cfg['foi']
    delta = (foi >= 1) & (foi < 3)
    theta = (foi >= 3) & (foi < 8)
    alpha_beta = (foi >= 8) & (foi < 25)
    alpha = (foi >= 8) & (foi < 12)
    beta = (foi >= 12) & (foi < 24)
    gamma = (foi >= 24) & (foi < 100)

    freq_masks = {
        'delta': delta,
        'theta': theta,
        'alpha-beta': alpha_beta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
    }

    return freq_masks


class LFPs:
    def __init__(self, M, lfp, err, freq_masks=None, cond_vec=None, part_vec=None, t_axis=1, n_jobs=16):
        self.M = M
        self.lfp = lfp
        self.freq_masks = freq_masks
        self.cond_vec = cond_vec
        self.part_vec = part_vec
        self.t_axis = t_axis
        self.timepoints = lfp.shape[t_axis]
        self.n_jobs = n_jobs

    def G_obs_in_timepoint(self, t, freq='delta'):
        """

        Args:
            t (int): timepoint
            freq (str): frequency band

        Returns:

        """
        if isinstance(freq, str):
            freq_mask = self.freq_masks[freq]
            lfp = self.lfp[:, t, freq_mask].mean(axis=1)
            err = self.err[:, t, freq_mask].mean(axis=1)
        elif isinstance(freq, int):
            lfp = self.lfp[:, t, freq]
            err = self.err[:, t, freq]

        cov = err.T @ err

        lfp_prewhitened = lfp / np.sqrt(np.diag(cov))

        obs_des = {'cond_vec': self.cond_vec,
                   'part_vec': self.part_vec}

        Y = pcm.dataset.Dataset(lfp_prewhitened, obs_descriptors=obs_des)

        G_obs, _ = pcm.est_G_crossval(Y.measurements, Y.obs_descriptors['cond_vec'],
                                         Y.obs_descriptors['part_vec'],
                                         X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']))
        return G_obs, Y

    def fit_model_in_timepoint(self, t, freq='delta'):
        """

        Args:
            M: list of models to fit
            Data: list of ntrials x timepoints datasets from each participants
            dat: list of DataFrames from dat files of each participant
            win: (start, end) tuple for timewindow

        Returns:

        """

        print(f'fitting model in window: {t}, {freq}')

        G_obs, Y = self.G_obs_in_timepoint(t, freq)
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')

        return G_obs, T_in, theta_in, t, freq


    def run_parallel_pcm_across_timepoints(self, freq='delta'):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_in_timepoint)(t, freq)
                for t in range(self.timepoints)
            )

        results = self._extract_results_from_parallel_process(results,
                                      field_names=['G_obs', 'T_in', 'theta_in', 't', 'freq'])
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


def save_lfp_aligned(monkey='Malfoy'):
    print('loading lfps...')
    path = os.path.join(gl.baseDir, args.experiment, 'LFPs', monkey)
    cfg = mat73.loadmat(path + '/cfg.mat')['cfg']
    trial_info = pd.read_csv(path + '/trial.tsv', sep='\t')
    lfp = load_lfp(path + '/lfp.PMd.mat')

    lfp = lfp[..., (trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]

    lfp_aligned = align_lfp(lfp, trial_info)

    np.save(os.path.join(path, f'lfp_aligned.PMd.npy'), lfp_aligned)


def save_lfp_binned(monkey='Malfoy', centres=np.array([25, 75, 125, 175]), width=50):
    path = os.path.join(gl.baseDir, args.experiment, 'LFPs', monkey)
    cfg = mat73.loadmat(path + '/cfg.mat')['cfg']
    freq_masks = make_freq_masks(cfg)
    lfp_aligned = np.load(os.path.join(path, 'lfp_aligned.PMd.npy'))

    lfp_binned = np.zeros((len(centres), lfp_aligned.shape[1], lfp_aligned.shape[2], lfp_aligned.shape[3]))

    for c, centre in enumerate(centres):
        lfp_binned[c] = lfp_aligned[int(centre - width / 2):int(centre + width / 2)].mean(axis=0)

    np.save(os.path.join(path, f'lfp_binned.PMd.npy'), lfp_binned)


def save_trial_info(monkey='Malfoy'):
    path = os.path.join(gl.baseDir, args.experiment, 'LFPs', monkey)
    trial_info = pd.read_csv(path + '/trial.tsv', sep='\t')
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]

    mapping = {
        1: 1,
        2: 8,
        3: 3,
        4: 6,
        5: 2,
        6: 5,
        7: 4,
        8: 7
    }

    trial_info.cond = trial_info.cond.map(mapping)
    trial_info.to_csv(os.path.join(path, 'trial_info.tsv'), sep='\t')


def run_pcm_in_freq_bands(epoch='plan', monkey='Malfoy', M=None, model='component', datatype='aligned'):

    _, idx = find_model(M, model)

    print('loading lfps...')
    path = os.path.join(gl.baseDir, args.experiment, 'LFPs', monkey)
    cfg = mat73.loadmat(path + '/cfg.mat')['cfg']

    freq_masks = make_freq_masks(cfg)

    lfp = np.load(os.path.join(path, f'lfp_{datatype}.PMd.npy'))
    trial_info = pd.read_csv(os.path.join(path, 'trial_info.tsv'), sep='\t')

    print('grouping by condition...')
    if epoch=='plan':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.prob, trial_info.block, axis=-1)
    elif epoch=='exec':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)

    print('doing prewhitening...')
    lfp_grouped_shape = lfp_grouped.shape
    if epoch == 'plan':
        n_cond = trial_info.prob.unique().size
    if epoch == 'exec':
        n_cond = trial_info.cond.unique().size
    lfp_grouped_reshaped = lfp_grouped.reshape(n_cond, int(lfp_grouped.shape[0] / n_cond),
                                               lfp_grouped.shape[1], lfp_grouped.shape[2], lfp_grouped.shape[3])
    lfp_grouped_avg = lfp_grouped_reshaped.mean(axis=0, keepdims=True)
    lfp_err = lfp_grouped_reshaped - lfp_grouped_avg
    lfp_err = lfp_err.reshape(lfp_grouped_shape)

    print('doing pcm...')
    Lfp = LFPs(M, lfp_grouped, lfp_err, freq_masks, cond_vec, part_vec)
    Lfp.G_obs_in_timepoint(0, 'delta')
    for freq in freq_masks.keys():

        res_dict = Lfp.run_parallel_pcm_across_timepoints(freq)

        theta_in = []
        for t in range(lfp.shape[0]):
            th = res_dict['theta_in'][t][idx].squeeze()
            theta_in.append(th)

        theta_in = np.array(theta_in)
        G_obs = np.array(res_dict['G_obs'])

        np.save(os.path.join(gl.baseDir, args.experiment, 'LFPs', gl.pcmDir,
                             f'theta_in.lfp.{monkey}.PMd.{freq}.{datatype}.{epoch}.npy'), theta_in, )
        np.save(os.path.join(gl.baseDir, args.experiment, 'LFPs', gl.pcmDir,
                             f'G_obs.lfp.{monkey}.PMd.{freq}.{datatype}.{epoch}.npy'), G_obs, )

def run_pcm_across_freqs(epoch='plan', monkey='Malfoy', M=None, model='component', datatype='aligned'):

    _, idx = find_model(M, model)
    n_param = M[idx].n_param

    print('loading lfps...')
    path = os.path.join(gl.baseDir, args.experiment, 'LFPs', monkey)

    lfp = np.load(os.path.join(path, f'lfp_{datatype}.PMd.npy'))
    trial_info = pd.read_csv(os.path.join(path, 'trial_info.tsv'), sep='\t')

    print('grouping by condition...')
    if epoch=='plan':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.prob, trial_info.block, axis=-1)
    elif epoch=='exec':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)

    print('doing prewhitening...')
    lfp_grouped_shape = lfp_grouped.shape
    if epoch == 'plan':
        n_cond = trial_info.prob.unique().size
    if epoch == 'exec':
        n_cond = trial_info.cond.unique().size
    lfp_grouped_reshaped = lfp_grouped.reshape(n_cond, int(lfp_grouped.shape[0] / n_cond),
                                               lfp_grouped.shape[1], lfp_grouped.shape[2], lfp_grouped.shape[3])
    lfp_grouped_avg = lfp_grouped_reshaped.mean(axis=0, keepdims=True)
    lfp_err = lfp_grouped_reshaped - lfp_grouped_avg
    lfp_err = lfp_err.reshape(lfp_grouped_shape)

    print('doing pcm...')
    Lfp = LFPs(M, lfp_grouped, lfp_err, cond_vec=cond_vec, part_vec=part_vec)
    theta_in = np.zeros((lfp.shape[1], lfp.shape[0], n_param + 1,))
    G_obs = np.zeros((lfp.shape[1], lfp.shape[0], n_cond, n_cond))
    for freq in range(lfp.shape[1]):
        res_dict = Lfp.run_parallel_pcm_across_timepoints(freq)
        theta_in_tmp = []
        for t in range(lfp.shape[0]):
            th = res_dict['theta_in'][t][idx].squeeze()
            theta_in_tmp.append(th)
        theta_in[freq] = np.array(theta_in_tmp)
        G_obs[freq] = np.array(res_dict['G_obs'])

    np.save(os.path.join(gl.baseDir, args.experiment, 'LFPs', gl.pcmDir,
                         f'theta_in.lfp.{monkey}.PMd.{datatype}.{epoch}.npy'), theta_in, )
    np.save(os.path.join(gl.baseDir, args.experiment, 'LFPs', gl.pcmDir,
                         f'G_obs.lfp.{monkey}.PMd.{datatype}.{epoch}.npy'), G_obs, )


def main(args):
    if args.what == 'continuous_plan_freq_bands':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        run_pcm_in_freq_bands('plan', args.monkey, M=M, model='component')
    if args.what=='continuous_exec_freq_bands':
        M = make_execution_models()
        run_pcm_in_freq_bands('exec', args.monkey, M=M, model='component')
    if args.what == 'continuous_plan_spectrum':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        run_pcm_across_freqs('plan', args.monkey, M=M, model='component')
    if args.what=='continuous_exec_spectrum':
        M = make_execution_models()
        run_pcm_across_freqs('exec', args.monkey, M=M, model='component')
    if args.what == 'binned_plan':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)
        run_pcm('plan', args.monkey, M=M, model='component', datatype='binned')
    if args.what=='binned_exec':
        M = make_execution_models()
        run_pcm('exec', args.monkey, M=M, model='component', datatype='binned')
    if args.what=='save_aligned':
        save_lfp_aligned(args.monkey)
    if args.what == 'save_binned':
        save_lfp_binned(args.monkey)
    if args.what=='save_trial_info':
        save_trial_info(args.monkey)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--monkey', type=str, default='Malfoy')

    args = parser.parse_args()

    res_dict = main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')


