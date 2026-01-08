import PcmPy as pcm
import scipy.io as sio
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle
from pcm_models import find_model
from depreciated.pcm_lfp import make_execution_models
from joblib import Parallel, delayed, parallel_backend
from imaging_pipelines.util import bootstrap_correlation, bootstrap_summary
from sigproc.statistics import permutation_t_test_1samp_tf

def load_spike(file_path):
    mat = sio.loadmat(file_path)
    return mat['spikes_s']

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
    def __init__(self, M, spike, cond_vec=None, part_vec=None, n_jobs=16):
        self.M = M
        self.spike = spike
        self.cond_vec = cond_vec
        self.part_vec = part_vec
        self.timepoints = spike.shape[1]
        self.n_jobs = n_jobs

    def G_obs_in_timepoint(self, t,):
        """

        Args:
            t (int): timepoint

        Returns:

        """

        spike = self.spike[:, t,]

        # spike = spike.copy()
        # for part in np.unique(self.part_vec):
        #     spike_ = spike[self.part_vec==part]
        #     G_ = spike_ @ spike_.T
        #     tr = np.trace(G_)
        #     spike_ = spike_ / np.sqrt(tr)
        #     spike[self.part_vec==part] = spike_

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

    def fit_model_family_in_timepoint(self, t, model,  basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
            MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        _, Y = self.G_obs_in_timepoint(t)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)
        c_bf = MF.component_bayesfactor(T.likelihood, method='AIC', format='DataFrame')
        return c_bf.to_numpy()


    def fit_model_family_across_timepoints(self, model, basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            c_bf = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_timepoint)(t, model, basecomp, comp_names)
                for t in range(self.timepoints)
            )
        c_bf = np.vstack(c_bf)
        return c_bf

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

    print('loading spikes...')
    spike = np.load(os.path.join(baseDir, spkDir, monkey, f'spk_aligned.{roi}-{rec}.npy'))
    spike = np.sqrt(spike)

    trial_info = pd.read_csv(os.path.join(baseDir, recDir, monkey, f'trial_info-{rec}.tsv'), sep='\t')
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
    trial_info.cond = trial_info.cond.map(mapping)

    print('grouping by condition...')
    if epoch=='plan':
        spike_grouped, cond_vec, part_vec = pcm.group_by_condition(spike, trial_info.prob, trial_info.block, axis=-1)
    elif epoch=='exec':
        spike_grouped, cond_vec, part_vec = pcm.group_by_condition(spike, trial_info.cond, trial_info.block, axis=-1)

    print('doing pcm...')
    Spk = Spikes(M, spike_grouped, cond_vec=cond_vec, part_vec=part_vec)

    res_dict = Spk.run_parallel_pcm_across_timepoints()

    theta_in = []
    for t in range(spike.shape[0]):
        th = res_dict['theta_in'][t][idx].squeeze()
        theta_in.append(th)

    theta_in = np.array(theta_in)
    G_obs = np.array(res_dict['G_obs'])

    c_bf = Spk.fit_model_family_across_timepoints(model=model)

    np.save(os.path.join(baseDir, pcmDir ,monkey, f'theta_in.spk.{model}.{roi}.{epoch}-{rec}.npy'), theta_in, )
    np.save(os.path.join(baseDir, pcmDir ,monkey, f'G_obs.spk.{roi}.{epoch}-{rec}.npy'), G_obs, )
    np.save(os.path.join(gl.nhpDir, gl.pcmDir, monkey, f'c_bf.spk.{model}.{roi}.{epoch}-{rec}.npy'), c_bf, )

def main(args):

    monkey = ['Malfoy', 'Pert']

    if args.what == 'continuous':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{args.epoch}.p'), "rb")
        M = pickle.load(f)[:-1]
        for mon in monkey:
            for rec in gl.recordings[mon][args.region]:
                run_pcm(args.epoch, mon, M=M, roi=args.region, model=args.model, rec=rec)
    if args.what=='tot_variance':
        for mon in monkey:
            for r, rec in enumerate(gl.recordings[mon][args.region]):
                print(f'doing {mon}, recording {rec}')
                spk = np.load(os.path.join(gl.nhpDir, gl.spkDir, mon, f'spk_aligned.{args.region}-{rec}.npy'))
                spk = np.sqrt(spk)
                trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.spkDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                trial_info.cond = trial_info.cond.map(mapping)
                spk_grouped, cond_vec, part_vec = pcm.group_by_condition(spk,
                                                                         trial_info.prob if args.epoch=='plan' else trial_info.cond,
                                                                         trial_info.block, axis=-1)
                n_timep = spk_grouped.shape[1]
                Var = np.zeros(n_timep)
                for t in range(n_timep):
                    Y = spk_grouped[:, t]
                    G_obs, _ = pcm.est_G(Y, cond_vec, part_vec, X=pcm.indicator(part_vec))
                    Var[t] = np.trace(G_obs)
                np.save(os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.spk.{args.region}.{args.epoch}-{rec}.npy'), Var)
    if args.what=='cluster-based_perm':
        for seg in ['Cue', 'Pert']:
            c_bf = []
            print('loading component bayes factor...')
            for mon in monkey:
                for r, rec in enumerate(gl.recordings[mon][args.region]):
                    print(f'loading component bayes factor for {mon}, recording{rec})')
                    c_bf_tmp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, mon,
                                         f'c_bf.spk.{args.model}.{args.region}.{args.epoch}-{rec}.npy'))
                    if seg=='Cue':
                        c_bf_tmp = c_bf_tmp[:gl.cuePost, :]
                    elif seg=='Pert':
                        c_bf_tmp = c_bf_tmp[gl.pertPre:, :]
                    c_bf.append(c_bf_tmp)
            c_bf = np.array(c_bf)
            n_sess, n_timep, n_comp = c_bf.shape
            significant_bf = np.zeros((n_timep, n_comp))
            n_comp = c_bf.shape[-1]
            print('doing permutations...')
            for i in range(n_comp):
                _, _, significant_bf[:,  i] = permutation_t_test_1samp_tf(c_bf[:, :, i])

            np.save(os.path.join(gl.nhpDir, gl.pcmDir,
                                 f'significant_bf.spk.{seg}.{args.region}.{args.epoch}.npy'), significant_bf)
    if args.what=='correlation_plan-exec':
        rng = np.random.default_rng(0)  # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        G, Y, i = [], [], 0
        for mon in monkey:
            for r, rec in enumerate(recordings[mon][args.region]):
                print(f'doing {mon}, recording {rec}')

                spk = np.load(os.path.join(baseDir, spkDir, mon, f'spk_aligned.{args.region}-{rec}.npy'))
                trial_info = pd.read_csv(os.path.join(baseDir, recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                trial_info.cond = trial_info.cond.map(mapping)

                spk_grouped_plan, _, part_vec = pcm.group_by_condition(spk, trial_info.prob, trial_info.block, axis=-1)
                spk_grouped_exec, _, _ = pcm.group_by_condition(spk, trial_info.cond, trial_info.block, axis=-1)
                n_part = len(np.unique(part_vec))
                obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                           'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                n_unit = spk.shape[1]
                mask_plan = {'ext': np.array([1, 1, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 1, 1] * n_part, dtype=bool)}
                mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                plan_ext = spk_grouped_plan[mask_plan['ext']].reshape(n_part, 2, 154, n_unit).mean(axis=1)
                plan_flx = spk_grouped_plan[mask_plan['flx']].reshape(n_part, 2, 154, n_unit).mean(axis=1)
                plan = plan_ext - plan_flx
                plan = plan[:, cuePost - 20:cuePost].mean(axis=1)
                plan = plan - plan.mean(axis=-1, keepdims=True)

                exec_ext = spk_grouped_exec[mask_exec['ext']].reshape(n_part, 4, 154, n_unit).mean(axis=1)
                exec_flx = spk_grouped_exec[mask_exec['flx']].reshape(n_part, 4, 154, n_unit).mean(axis=1)
                exec = exec_ext - exec_flx
                exec = exec[:, pertIdx+4:pertIdx+24].mean(axis=1)
                exec = exec - exec.mean(axis=-1, keepdims=True)

                data = np.r_[plan, exec]
                X = pcm.indicator(obs_des['part_vec'])
                beta, *_ = np.linalg.lstsq(X, data)
                err = data - X @ beta
                cov = (err.T @ err) / err.shape[0]
                data_prewhitened = data / np.sqrt(np.diag(cov))

                Y.append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
                G.append(pcm.est_G_crossval(
                    Y[i].measurements,
                    Y[i].obs_descriptors['cond_vec'],
                    Y[i].obs_descriptors['part_vec'])[0])
                i += 1

        np.save(os.path.join(baseDir, pcmDir, f'G_obs.spk.corr_plan-exec.{args.region}.npy'), np.array(G))
        T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=True)
        T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
        T_in.to_pickle(os.path.join(baseDir, pcmDir, f'T_in.spk.corr_plan-exec.{args.region}.p'))
        T_gr.to_pickle(os.path.join(baseDir, pcmDir, f'T_gr.spk.corr_plan-exec.{args.region}.p'))

        f = open(os.path.join(baseDir, pcmDir, f'theta_in.spk.corr_plan-exec.{args.region}.p'), 'wb')
        pickle.dump(theta_in, f)
        f = open(os.path.join(baseDir, pcmDir, f'theta_gr.spk.corr_plan-exec.{args.region}.p'), 'wb')
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
        np.save(os.path.join(baseDir, pcmDir, f'r_bootstrap.spk.corr_plan-exec.{args.region}.npy'), r_bootstrap)
    if args.what=='correlation_cue-direction':
        rng = np.random.default_rng(0)  # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        time_interval = {'early':(gl.pertIdx + 4, gl.pertIdx + 24),
                         'late': (gl.pertPost - 20, gl.pertPost)}
        for tint in ['early', 'late']:
            G, Y, i = [], [], 0
            for mon in monkey:
                for roi in args.regions:
                    for r, rec in enumerate(gl.recordings[mon][roi]):
                        print(f'doing {mon}, recording {rec}')

                        spk = np.load(os.path.join(gl.nhpDir, gl.spkDir, mon, f'spk_aligned.{roi}-{rec}.npy'))
                        spk = np.sqrt(spk)
                        trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                        trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                        mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                        trial_info.cond = trial_info.cond.map(mapping)

                        spk_grouped, _, part_vec = pcm.group_by_condition(spk, trial_info.cond, trial_info.block, axis=-1)
                        n_part = len(np.unique(part_vec))
                        obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                                   'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                        n_unit = spk.shape[1]
                        mask_plan = {'ext': np.array([0, 1, 0, 0, 1, 0, 0, 0] * n_part, dtype=bool),
                                     'flx': np.array([0, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool)}
                        mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                                     'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                        plan_ext = spk_grouped[mask_plan['ext']].reshape(n_part, 2, 154, n_unit).mean(axis=1)
                        plan_flx = spk_grouped[mask_plan['flx']].reshape(n_part, 2, 154, n_unit).mean(axis=1)
                        plan = plan_ext - plan_flx
                        plan = plan[:, time_interval[tint][0]:time_interval[tint][1]].mean(axis=1)
                        plan = plan - plan.mean(axis=-1, keepdims=True)

                        exec_ext = spk_grouped[mask_exec['ext']].reshape(n_part, 4, 154, n_unit).mean(axis=1)
                        exec_flx = spk_grouped[mask_exec['flx']].reshape(n_part, 4, 154, n_unit).mean(axis=1)
                        exec = exec_ext - exec_flx
                        exec = exec[:, time_interval[tint][0]:time_interval[tint][1]].mean(axis=1)
                        exec = exec - exec.mean(axis=-1, keepdims=True)

                        data = np.r_[plan, exec]
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

            np.save(os.path.join(baseDir, pcmDir, f'G_obs.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.npy'), np.array(G))
            T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=True)
            T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
            T_in.to_pickle(os.path.join(baseDir, pcmDir, f'T_in.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.p'))
            T_gr.to_pickle(os.path.join(baseDir, pcmDir, f'T_gr.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.p'))

            f = open(os.path.join(baseDir, pcmDir, f'theta_in.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.p'), 'wb')
            pickle.dump(theta_in, f)
            f = open(os.path.join(baseDir, pcmDir, f'theta_gr.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.p'), 'wb')
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
            np.save(os.path.join(baseDir, pcmDir, f'r_bootstrap.spk.corr_cue-dir.{"-".join(args.regions)}.{tint}.npy'), r_bootstrap)
    if args.what=='correlation_continuous':
        rng = np.random.default_rng(0)  # seed for reprodocibility
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)

        G = []
        T = 154
        Y = {t: [] for t in range(T)}
        for mon in monkey:
            for r, rec in enumerate(recordings[mon][args.region]):
                print(f'doing {mon}, recording {rec}')
                spk = np.load(os.path.join(baseDir, spkDir, mon, f'spk_aligned.{args.region}-{rec}.npy'))
                trial_info = pd.read_csv(os.path.join(baseDir, recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                trial_info.cond = trial_info.cond.map(mapping)
                spk_grouped, _, part_vec = pcm.group_by_condition(spk, trial_info.cond, trial_info.block, axis=-1)
                spk_grouped = np.sqrt(spk_grouped)
                n_part = len(np.unique(part_vec))
                _, _, C = spk_grouped.shape
                mask_plan = {'ext': np.array([0, 1, 0, 0, 1, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 1, 0, 0, 1, 0] * n_part, dtype=bool)}
                mask_exec = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                             'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}

                plan_ext = spk_grouped[mask_plan['ext']].reshape(n_part, 2, T, C).mean(axis=1)
                plan_flx = spk_grouped[mask_plan['flx']].reshape(n_part, 2, T, C).mean(axis=1)
                plan_t = plan_ext - plan_flx

                exec_ext = spk_grouped[mask_exec['ext']].reshape(n_part, 4, T, C).mean(axis=1)
                exec_flx = spk_grouped[mask_exec['flx']].reshape(n_part, 4, T, C).mean(axis=1)
                exec_t = exec_ext - exec_flx
                # exec = exec[:, pertIdx+4:pertIdx+12].mean(axis=1)
                # exec = exec - exec.mean(axis=-1, keepdims=True)

                obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                           'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}

                Gg = np.zeros((T, 2, 2))
                for t in range(T):
                    plan = plan_t[:, t, :]
                    plan = plan - plan.mean(axis=-1, keepdims=True)
                    exec = exec_t[:, t, :]
                    exec = exec - exec.mean(axis=-1, keepdims=True)
                    data = np.r_[plan, exec]
                    err = data - data.mean(axis=0, keepdims=True)
                    cov = (err.T @ err) / data.shape[0]
                    data_prewhitened = data / np.sqrt(np.diag(cov))
                    Y[t].append(pcm.dataset.Dataset(data_prewhitened, obs_descriptors=obs_des))
                    Gg[t] = pcm.est_G_crossval(Y[t][-1].measurements,
                                                  Y[t][-1].obs_descriptors['cond_vec'],
                                                  Y[t][-1].obs_descriptors['part_vec'])[0]
                G.append(Gg)

        N = len(Y[0])
        r_indiv = np.zeros((N, T))
        SNR = np.zeros_like(r_indiv)
        r_group = np.zeros(T)
        for t in range(T):
            print(f'doing ML estimation for t={t}...')
            _, theta = pcm.fit_model_individ(Y[t], Mflex, fixed_effect=None, fit_scale=False, verbose=False)
            _, theta_gr = pcm.fit_model_group(Y[t], Mflex, fixed_effect=None, fit_scale=True, verbose=False)
            sigma2_1 = np.exp(theta[0][0])
            sigma2_2 = np.exp(theta[0][1])
            r_indiv[:, t] = Mflex.get_correlation(theta[0])
            sigma2_e = np.exp(theta[0][3])
            SNR[:, t] = np.sqrt(sigma2_1 * sigma2_2) / sigma2_e
            theta_gr, _ = pcm.group_to_individ_param(theta_gr[0], Mflex, N)
            r_group[t] = Mflex.get_correlation(theta_gr)[0]

        np.save(os.path.join(baseDir, pcmDir, f'G_obs.spk.corr_tf.{args.region}.npy'), np.array(G))
        np.save(os.path.join(baseDir, pcmDir, f'r_indiv.spk.corr_tf.{args.region}.npy'), r_indiv)
        np.save(os.path.join(baseDir, pcmDir, f'SNR.spk.corr_tf.{args.region}.npy'), SNR)
        np.save(os.path.join(baseDir, pcmDir, f'r_group.spk.corr_tf.{args.region}.npy'), r_group)
    if args.what=='spk2tsv':
        components = {
            'plan': ['expectation', 'uncertainty'],
            'exec': ['sensory input', 'expectation', 'surprise']
        }
        pcm_dict = {
            'epoch': [],
            'roi': [],
            'weight': [],
            'monkey': [],
            'component': [],
            'session': [],
            'noise': []
        }
        rois = ['PMd', 'M1', 'S1']
        for epoch in ['plan', 'exec']:
            for roi in rois:
                weight = []
                for mon in monkey:
                    for rec in gl.recordings[mon][roi]:
                        theta_spk_comp = np.load(
                            os.path.join(gl.nhpDir, gl.pcmDir, mon, f'theta_in.spk.component.{roi}.{epoch}-{rec}.npy'))
                        var_tot_spk = np.load(
                            os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.spk.{roi}.{epoch}-{rec}.npy'))
                        weight_norm = np.exp(theta_spk_comp[..., :-1]) / var_tot_spk.T[..., None]
                        noise = np.exp(theta_spk_comp[..., -1]) / var_tot_spk.T[..., None]
                        weight.append(weight_norm)
                        for md in range(weight_norm.shape[-1]):
                            weight_avg = weight_norm[gl.cueIdx:gl.cuePost, md].mean()
                            pcm_dict['epoch'].append(epoch)
                            pcm_dict['roi'].append(roi)
                            pcm_dict['weight'].append(weight_avg)
                            pcm_dict['noise'].append(noise[gl.cueIdx:gl.cuePost].mean())
                            pcm_dict['component'].append(components[epoch][md])
                            pcm_dict['session'].append(rec)
                            pcm_dict['monkey'].append(mon[0])
                weight = np.array(weight)
                df_weight = pd.DataFrame(pcm_dict)
                np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.spk.{roi}.{epoch}.npy'), weight)
        df_weight.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'weight.spk.10-20Hz.tsv'), sep='\t', index=False)
    if args.what == 'corr2tsv':
        corrs = ['cue-dir']
        corr_dict = {
            'r_indiv': [],
            'r_group': [],
            'SNR': [],
            'corr': [],
            'ci_lo': [],
            'ci_hi': [],
            'roi': [],
            'time_interval': [],
            'session_id': []
        }
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for tint in ['early', 'late']:
            for corr in corrs:
                f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_in.spk.corr_{corr}.M1-S1.{tint}.p'), 'rb')
                theta = pickle.load(f)[0]
                r_bootstrap = np.load(
                    os.path.join(gl.nhpDir, gl.pcmDir, f'r_bootstrap.spk.corr_{corr}.M1-S1.{tint}.npy'))
                f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_gr.spk.corr_{corr}.M1-S1.{tint}.p'), 'rb')
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
                corr_dict['session_id'].extend(np.linspace(1, r_indiv.shape[0], r_indiv.shape[0]))
                corr_dict['roi'].extend(['M1-S1'] * r_indiv.shape[0])
                corr_dict['time_interval'].extend([tint] * r_indiv.shape[0])
        df_corr = pd.DataFrame(corr_dict)
        df_corr.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'correlations.spk.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--model', type=str, default='component')
    parser.add_argument('--monkey', type=str, default='Pert')
    parser.add_argument('--region', type=str, default='PMd')
    parser.add_argument('--regions', type=list, default=['M1', 'S1'])
    parser.add_argument('--recording', nargs='+', type=int, default=[19, 20, 21, 22, 23])

    args = parser.parse_args()

    baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction/'
    spkDir = 'spikes'
    recDir = 'recordings'
    pcmDir = 'pcm'

    res_dict = main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')