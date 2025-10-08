import PcmPy as pcm
import scipy.io as sio
import mat73
import numpy as np
import pandas as pd
import time
import argparse
import glob
import os
import globals as gl
from sklearn.preprocessing import MinMaxScaler
import pickle
from pcm_models import find_model
from imaging_pipelines.util import bootstrap_correlation
from lfp import make_freq_masks
from joblib import Parallel, delayed, parallel_backend

class LFPs:
    def __init__(self, M: list,
                 lfp: np.ndarray,
                 freq_masks: dict=None,
                 cond_vec=None, part_vec=None, n_jobs=16):
        self.M = M
        self.lfp = lfp
        self.freq_masks = freq_masks
        self.cond_vec = cond_vec
        self.part_vec = part_vec
        self.timepoints = lfp.shape[1]
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
            lfp = self.lfp[:, t, :, freq_mask].mean(axis=0)
        elif isinstance(freq, int):
            lfp = self.lfp[:, t, :, freq]

        # lfp = lfp.copy()
        # for part in np.unique(self.part_vec):
        #     lfp_ = lfp[self.part_vec==part]
        #     G_ = lfp_ @ lfp_.T
        #     tr = np.trace(G_)
        #     lfp_ = lfp_ / np.sqrt(tr)
        #     lfp[self.part_vec==part] = lfp_

        obs_des = {'cond_vec': self.cond_vec,
                   'part_vec': self.part_vec}

        Y = pcm.dataset.Dataset(lfp, obs_descriptors=obs_des)
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


def run_pcm(epoch='plan', monkey='Malfoy', roi='PMd', M=None, model='component', rec=1):

    _, idx = find_model(M, model)
    n_param = M[idx].n_param

    print('loading lfps...')
    lfp = np.load(os.path.join(baseDir, lfpDir, monkey, f'lfp_aligned.{roi}-{rec}.npy'))
    trial_info = pd.read_csv(os.path.join(baseDir, recDir, monkey , f'trial_info-{rec}.tsv'), sep='\t')
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
    trial_info.cond = trial_info.cond.map(mapping)

    print('grouping by condition...')
    if epoch=='plan':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.prob, trial_info.block, axis=-1)
    elif epoch=='exec':
        lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)

    if epoch == 'plan':
        n_cond = trial_info.prob.unique().size
    if epoch == 'exec':
        n_cond = trial_info.cond.unique().size

    LFP = LFPs(M, lfp_grouped, cond_vec=cond_vec, part_vec=part_vec)
    theta_in = np.zeros((lfp.shape[2], lfp.shape[0], n_param + 1,))
    G_obs = np.zeros((lfp.shape[2], lfp.shape[0], n_cond, n_cond))
    for freq in range(lfp.shape[2]):
        LFP.fit_model_in_timepoint(0, freq)
        res_dict = LFP.run_parallel_pcm_across_timepoints(freq)
        theta_in_tmp = []
        for t in range(lfp.shape[0]):
            th = res_dict['theta_in'][t][idx].squeeze()
            theta_in_tmp.append(th)
        theta_in[freq] = np.array(theta_in_tmp)
        G_obs[freq] = np.array(res_dict['G_obs'])

    np.save(os.path.join(baseDir, pcmDir, monkey, f'theta_in.lfp.{model}.{roi}.{epoch}-{rec}.npy'), theta_in, )
    np.save(os.path.join(baseDir, pcmDir, monkey, f'G_obs.lfp.{roi}.{epoch}-{rec}.npy'), G_obs, )


def main(args):
    cuePre = 0
    cueIdx = 20
    cuePost = 84
    pertPre = cuePost
    pertIdx = pertPre + 30
    pertPost = pertPre + 70

    monkey = ['Malfoy', 'Pert']

    recordings = {
        'Malfoy': {
            'PMd': [19, 20, 21, 22, 23, 24],
            'S1': [26, 27, 28],
            'M1': [12, 13, 25, 27, 28]
        },
        'Pert': {
            'PMd': [4, 6, 7, 10, 20],
            'S1': [15],
            'M1': [2, 3, 14, 20]
        }
    }

    cfg = mat73.loadmat(os.path.join(baseDir, lfpDir, 'Malfoy', f'cfg.PMd-19.mat'))['cfg']
    freq_masks = make_freq_masks(cfg)

    if args.what=='continuous_plan':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)[:-1]
        for rec in args.recording:
            run_pcm('plan', args.monkey, M=M, roi=args.region, model=args.model, rec=rec)
    if args.what=='continuous_exec':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "rb")
        M = pickle.load(f)[:-1]
        for rec in args.recording:
            run_pcm('exec', args.monkey, M=M, roi=args.region, model=args.model, rec=rec)
    if args.what=='binned_plan':
        rois = ['PMd', 'S1']
        model = ['Cue', 'Uncertainty']
        epochs = {
            'Pre': (cuePre, cueIdx),
            'Cue': (cueIdx, cuePost),
            'Pert': (pertIdx, pertPost),
        }
        out_dict = {
            'monkey': [],
            'recording': [],
            'region': [],
            'band': [],
            'model': [],
            'epoch': [],
            'variance': [],
            'datatype': []
        }
        freqs = ['delta', 'theta', 'alpha', 'beta', 'alpha-beta', 'gamma']
        for roi in rois:
            for mon in monkey:
                for rec in recordings[mon][roi]:
                    theta_lfp = np.load(os.path.join(baseDir, pcmDir, mon, f'theta_in.lfp.component.{roi}.plan-{rec}.npy'))
                    var_expl_lfp = np.exp(theta_lfp[..., :-1])
                    for m, md in enumerate(model):
                        for f in freqs:
                            freq = freq_masks[f]
                            for epoch, interval in epochs.items():
                                out_dict['monkey'].append(mon)
                                out_dict['recording'].append(rec)
                                out_dict['region'].append(roi)
                                out_dict['band'].append(f)
                                out_dict['model'].append(md)
                                out_dict['variance'].append(var_expl_lfp[freq, interval[0]:interval[1], m].mean())
                                out_dict['epoch'].append(epoch)
                                out_dict['datatype'].append('lfp')
        out = pd.DataFrame(out_dict)
        out.to_csv(os.path.join(baseDir, pcmDir, 'var_expl.plan.lfp.tsv'), sep='\t', index=False)
    if args.what=='correlation_plan-exec':
            rng = np.random.default_rng(0)  # seed for reprodocibility
            f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
            Mflex = pickle.load(f)
            freq = freq_masks['beta']
            G, Y, i = [], [], 0
            for mon in monkey:
                for r, rec in enumerate(recordings[mon][args.region]):
                    print(f'doing {mon}, recording {rec}')

                    lfp = np.load(os.path.join(baseDir, lfpDir, mon, f'lfp_aligned.{args.region}-{rec}.npy'))
                    trial_info = pd.read_csv(os.path.join(baseDir, recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                    trial_info.cond = trial_info.cond.map(mapping)

                    lfp_grouped_plan, _, part_vec = pcm.group_by_condition(lfp, trial_info.prob, trial_info.block,
                                                                           axis=-1)
                    lfp_grouped_exec, _, _ = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)
                    lfp_grouped_plan = lfp_grouped_plan[..., freq].mean(axis=-1)
                    lfp_grouped_exec = lfp_grouped_exec[..., freq].mean(axis=-1)
                    n_part = len(np.unique(part_vec))
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}

                    mask_plan = {'ext': np.array([1, 1, 0, 0, 0] * n_part, dtype=bool),
                                 'flx': np.array([0, 0, 0, 1, 1] * n_part, dtype=bool)}
                    mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                                 'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                    plan_ext = lfp_grouped_plan[mask_plan['ext']].reshape(n_part, 2, 154, 32).mean(axis=1)
                    plan_flx = lfp_grouped_plan[mask_plan['flx']].reshape(n_part, 2, 154, 32).mean(axis=1)
                    plan = plan_ext - plan_flx
                    plan = plan[:, cuePost - 20:cuePost].mean(axis=1)
                    plan = plan - plan.mean(axis=-1, keepdims=True)

                    exec_ext = lfp_grouped_exec[mask_exec['ext']].reshape(n_part, 4, 154, 32).mean(axis=1)
                    exec_flx = lfp_grouped_exec[mask_exec['flx']].reshape(n_part, 4, 154, 32).mean(axis=1)
                    exec = exec_ext - exec_flx
                    exec = exec[:, pertIdx + 4:pertIdx + 24].mean(axis=1)
                    exec = exec - exec.mean(axis=-1, keepdims=True)

                    data = np.r_[plan, exec]
                    T, C = data.shape
                    X = pcm.indicator(obs_des['part_vec'])
                    beta, *_ = np.linalg.lstsq(X, data)
                    err = data - X @ beta
                    cov = (err.T @ err) / T
                    data_prewhitened = data / np.sqrt(np.diag(cov))

                    Y.append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
                    G.append(pcm.est_G_crossval(
                        Y[i].measurements,
                        Y[i].obs_descriptors['cond_vec'],
                        Y[i].obs_descriptors['part_vec'])[0])
                    i += 1

            np.save(os.path.join(baseDir, pcmDir, f'G_obs.lfp.corr_plan-exec.{args.region}.npy'), np.array(G))
            T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=True)
            T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
            T_in.to_pickle(os.path.join(baseDir, pcmDir, f'T_in.lfp.corr_plan-exec.{args.region}.p'))
            T_gr.to_pickle(os.path.join(baseDir, pcmDir, f'T_gr.lfp.corr_plan-exec.{args.region}.p'))

            f = open(os.path.join(baseDir, pcmDir, f'theta_in.lfp.corr_plan-exec.{args.region}.p'), 'wb')
            pickle.dump(theta_in, f)
            f = open(os.path.join(baseDir, pcmDir, f'theta_gr.lfp.corr_plan-exec.{args.region}.p'), 'wb')
            pickle.dump(theta_gr, f)

            # do bootstrap
            B = 1000
            S = len(Y)
            indeces = rng.integers(0, S, size=(B, S))
            results = Parallel(n_jobs=16, backend='loky')(
                delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces
            )
            r_bootstrap = np.array([r for r in results if r is not None])
            n_disc = len(results) - len(r_bootstrap)
            print(f'{args.region}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')
            np.save(os.path.join(baseDir, pcmDir, f'r_bootstrap.lfp.corr_plan-exec.{args.region}.npy'), r_bootstrap)
    if args.what=='correlation_cue-direction':
        rng = np.random.default_rng(0)  # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        freq = freq_masks['beta']
        G, Y, i = [], [], 0
        for mon in monkey:
            for roi in args.regions:
                for r, rec in enumerate(recordings[mon][roi]):
                    print(f'doing {mon}, recording {rec}')

                    lfp = np.load(os.path.join(baseDir, lfpDir, mon, f'lfp_aligned.{roi}-{rec}.npy'))
                    trial_info = pd.read_csv(os.path.join(baseDir, recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                    trial_info.cond = trial_info.cond.map(mapping)

                    lfp_grouped, _, part_vec = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)
                    lfp_grouped = lfp_grouped[..., freq].mean(axis=-1)
                    n_part = len(np.unique(part_vec))
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}

                    mask_plan = {'ext': np.array([0, 1, 0, 0, 1, 0, 0, 0] * n_part, dtype=bool),
                                 'flx': np.array([0, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool)}
                    mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                                 'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                    plan_ext = lfp_grouped[mask_plan['ext']].reshape(n_part, 2, 154, 32).mean(axis=1)
                    plan_flx = lfp_grouped[mask_plan['flx']].reshape(n_part, 2, 154, 32).mean(axis=1)
                    plan = plan_ext - plan_flx
                    plan = plan[:, pertIdx+4:pertIdx+24].mean(axis=1)
                    plan = plan - plan.mean(axis=-1, keepdims=True)

                    exec_ext = lfp_grouped[mask_exec['ext']].reshape(n_part, 4, 154, 32).mean(axis=1)
                    exec_flx = lfp_grouped[mask_exec['flx']].reshape(n_part, 4, 154, 32).mean(axis=1)
                    exec = exec_ext - exec_flx
                    exec = exec[:, pertIdx+4:pertIdx+24].mean(axis=1)
                    exec = exec - exec.mean(axis=-1, keepdims=True)

                    data = np.r_[plan, exec]
                    T, C = data.shape
                    X = pcm.indicator(obs_des['part_vec'])
                    beta, *_ = np.linalg.lstsq(X, data)
                    err = data - X @ beta
                    cov = (err.T @ err) / err.shape[0]
                    data_prewhitened = data / np.sqrt(np.diag(cov))
                    Y.append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
                    G.append(pcm.est_G_crossval(
                        Y[-1].measurements,
                        Y[-1].obs_descriptors['cond_vec'],
                        Y[-1].obs_descriptors['part_vec'])[0])

        np.save(os.path.join(baseDir, pcmDir, f'G_obs.lfp.corr_cue-dir.{"-".join(args.regions)}.npy'), np.array(G))
        T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=True)
        T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
        T_in.to_pickle(os.path.join(baseDir, pcmDir, f'T_in.lfp.corr_cue-dir.{"-".join(args.regions)}.p'))
        T_gr.to_pickle(os.path.join(baseDir, pcmDir, f'T_gr.lfp.corr_cue-dir.{"-".join(args.regions)}.p'))

        f = open(os.path.join(baseDir, pcmDir, f'theta_in.lfp.corr_cue-dir.{"-".join(args.regions)}.p'), 'wb')
        pickle.dump(theta_in, f)
        f = open(os.path.join(baseDir, pcmDir, f'theta_gr.lfp.corr_cue-dir.{"-".join(args.regions)}.p'), 'wb')
        pickle.dump(theta_gr, f)

        # do bootstrap
        B = 1000
        S = len(Y)
        indeces = rng.integers(0, S, size=(B, S))
        results = Parallel(n_jobs=16, backend='loky')(
            delayed(bootstrap_correlation)(idx, Y, Mflex) for idx in indeces
        )
        r_bootstrap = np.array([r for r in results if r is not None])
        n_disc = len(results) - len(r_bootstrap)
        print(f'{"-".join(args.regions)}: kept {len(r_bootstrap)}/{B} (discarded {n_disc})')
        np.save(os.path.join(baseDir, pcmDir, f'r_bootstrap.lfp.corr_cue-dir.{"-".join(args.regions)}.npy'), r_bootstrap)
    if args.what=='correlation_continuous':
        rng = np.random.default_rng(0)  # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        G = []
        T, F = 154, 40
        Y = {(t, f): [] for t in range(T) for f in range(F)}
        for mon in monkey:
            for r, rec in enumerate(recordings[mon][args.region]):
                print(f'doing {mon}, recording {rec}')
                lfp = np.load(os.path.join(baseDir, lfpDir, mon, f'lfp_aligned.{args.region}-{rec}.npy'))
                trial_info = pd.read_csv(os.path.join(baseDir, recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                trial_info.cond = trial_info.cond.map(mapping)
                lfp_grouped, _, part_vec = pcm.group_by_condition(lfp, trial_info.cond, trial_info.block, axis=-1)
                n_part = len(np.unique(part_vec))
                _, _, C, _ = lfp_grouped.shape
                mask_plan = {'ext': np.array([0, 1, 0, 0, 1, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool)}
                mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                plan_ext = lfp_grouped[mask_plan['ext']].reshape(n_part, 2, T, C, F).mean(axis=1)
                plan_flx = lfp_grouped[mask_plan['flx']].reshape(n_part, 2, T, C, F).mean(axis=1)
                plan_tf = plan_ext - plan_flx

                exec_ext = lfp_grouped[mask_exec['ext']].reshape(n_part, 4, T, C, F).mean(axis=1)
                exec_flx = lfp_grouped[mask_exec['flx']].reshape(n_part, 4, T, C, F).mean(axis=1)
                exec_tf = exec_ext - exec_flx
                # exec_t = exec_tf[:, pertIdx+4:pertIdx+12].mean(axis=1)

                obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                           'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}

                Gg = np.zeros((T, F, 2, 2))
                for t in range(T):
                    for f in range(F):
                        plan = plan_tf[:, t, :, f]
                        plan = plan - plan.mean(axis=-1, keepdims=True)
                        exec = exec_tf[:, t, :, f]
                        exec = exec - exec.mean(axis=-1, keepdims=True)
                        data = np.r_[plan, exec]
                        X = pcm.indicator(obs_des['part_vec'])
                        beta, *_ = np.linalg.lstsq(X, data)
                        err = data - X @ beta
                        cov = (err.T @ err) / data.shape[0]
                        data_prewhitened = data / np.sqrt(np.diag(cov))
                        Y[(t, f)].append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
                        Gg[t, f] = pcm.est_G_crossval(Y[(t, f)][-1].measurements,
                                                    Y[(t, f)][-1].obs_descriptors['cond_vec'],
                                                    Y[(t, f)][-1].obs_descriptors['part_vec'])[0]
                G.append(Gg)

        N = len(Y[(0, 0)])
        r_indiv = np.zeros((N, F, T))
        SNR = np.zeros_like(r_indiv)
        r_group = np.zeros((F, T))
        for t in range(T):
            for f in range(F):
                print(f'doing ML estimation for t={t}, f={f}...')
                _, theta = pcm.fit_model_individ(Y[(t, f)], Mflex, fixed_effect=None, fit_scale=False, verbose=False)
                _, theta_gr = pcm.fit_model_group(Y[(t, f)], Mflex, fixed_effect=None, fit_scale=True, verbose=False)
                sigma2_1 = np.exp(theta[0][0])
                sigma2_2 = np.exp(theta[0][1])
                r_indiv[:, f, t] = Mflex.get_correlation(theta[0])
                sigma2_e = np.exp(theta[0][3])
                SNR[:, f, t] = np.sqrt(sigma2_1 * sigma2_2) / sigma2_e
                theta_gr, _ = pcm.group_to_individ_param(theta_gr[0], Mflex, N)
                r_group[f, t] = Mflex.get_correlation(theta_gr)[0]

        np.save(os.path.join(baseDir, pcmDir, f'G_obs.lfp.corr_tf.{args.region}.npy'), np.array(G))
        np.save(os.path.join(baseDir, pcmDir, f'r_indiv.lfp.corr_tf.{args.region}.npy'), r_indiv)
        np.save(os.path.join(baseDir, pcmDir, f'SNR.lfp.corr_tf.{args.region}.npy'), SNR)
        np.save(os.path.join(baseDir, pcmDir, f'r_group.lfp.corr_tf.{args.region}.npy'), r_group)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--model', type=str, default='component')
    parser.add_argument('--recording', nargs='+', type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument( '--region', type=str, default='PMd')
    parser.add_argument('--regions', type=list, default=['M1', 'S1'])
    parser.add_argument('--monkey', type=str, default='Malfoy')

    args = parser.parse_args()

    baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction/'
    lfpDir = 'LFPs'
    recDir = 'Recordings'
    pcmDir = 'pcm'

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')