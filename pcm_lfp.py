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
from imaging_pipelines.util import bootstrap_correlation, bootstrap_summary
from lfp import make_freq_masks
from joblib import Parallel, delayed, parallel_backend
from sigproc.statistics import permutation_t_test_1samp_tf

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

    def fit_model_family_in_timepoint(self, t, model, freq='delta',  basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
            MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        _, Y = self.G_obs_in_timepoint(t, freq)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)
        c_bf = MF.component_bayesfactor(T.likelihood, method='AIC', format='DataFrame')
        return c_bf.to_numpy()


    def fit_model_family_across_timepoints(self, model, freq='delta', basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            c_bf = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_timepoint)(t, model, freq, basecomp, comp_names)
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
    else:
        raise ValueError(f'Unknown epoch: {epoch}')

    if epoch == 'plan':
        n_cond = trial_info.prob.unique().size
    if epoch == 'exec':
        n_cond = trial_info.cond.unique().size

    LFP = LFPs(M, lfp_grouped, cond_vec=cond_vec, part_vec=part_vec)
    theta_in = np.zeros((lfp.shape[2], lfp.shape[0], n_param + 1,))
    G_obs = np.zeros((lfp.shape[2], lfp.shape[0], n_cond, n_cond))
    c_bf = np.zeros((lfp.shape[2], lfp.shape[0], n_param,))
    for freq in range(lfp.shape[2]):
        # fit model
        res_dict = LFP.run_parallel_pcm_across_timepoints(freq)
        theta_in_tmp = []
        for t in range(lfp.shape[0]):
            th = res_dict['theta_in'][t][idx].squeeze()
            theta_in_tmp.append(th)
        theta_in[freq] = np.array(theta_in_tmp)
        G_obs[freq] = np.array(res_dict['G_obs'])

        # model family
        c_bf[freq] = LFP.fit_model_family_across_timepoints(freq=freq, model=model)

    np.save(os.path.join(gl.nhpDir, gl.pcmDir, monkey, f'theta_in.lfp.{model}.{roi}.{epoch}-{rec}.npy'), theta_in, )
    np.save(os.path.join(gl.nhpDir, gl.pcmDir, monkey, f'G_obs.lfp.{roi}.{epoch}-{rec}.npy'), G_obs, )
    np.save(os.path.join(gl.nhpDir, gl.pcmDir, monkey, f'c_bf.lfp.{model}.{roi}.{epoch}-{rec}.npy'), c_bf, )


def main(args):

    monkey = ['Malfoy', 'Pert']

    cfg = mat73.loadmat(os.path.join(baseDir, lfpDir, 'Malfoy', f'cfg.PMd-19.mat'))['cfg']
    freq_masks = make_freq_masks(cfg)

    if args.what=='continuous':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{args.epoch}.p'), "rb")
        M = pickle.load(f)[:-1]
        for mon in monkey:
            for rec in gl.recordings[mon][args.region]:
                run_pcm(args.epoch, mon, M=M, roi=args.region, model=args.model, rec=rec)
    if args.what=='tot_variance':
        for mon in monkey:
            for r, rec in enumerate(gl.recordings[mon][args.region]):
                print(f'doing {mon}, recording {rec}, epoch {args.epoch}, region {args.region}')
                lfp = np.load(os.path.join(gl.nhpDir, gl.lfpDir, mon, f'lfp_aligned.{args.region}-{rec}.npy'))
                trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
                trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
                lfp_grouped, cond_vec, part_vec = pcm.group_by_condition(lfp,
                                                                         trial_info.prob if args.epoch=='plan' else trial_info.cond,
                                                                         trial_info.block, axis=-1)
                n_cond, n_timep, n_unit, n_freq = 5, lfp_grouped.shape[1], lfp_grouped.shape[2], lfp_grouped.shape[3]
                n_sample, n_feat = lfp_grouped.shape[0], lfp_grouped.shape[-1]
                Var = np.zeros((n_timep, n_freq))
                for f in range(n_freq):
                    for t in range(n_timep):
                        Y = lfp_grouped[:, t, :, f]
                        G_obs, _ = pcm.est_G(Y, cond_vec, part_vec, X=pcm.indicator(part_vec))
                        Var[t, f] = np.trace(G_obs)
                np.save(os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.lfp.{args.region}.{args.epoch}-{rec}.npy'), Var)
    if args.what=='cluster-based_perm':
        for seg in ['Cue', 'Pert']:
            c_bf = []
            freq_thresh = np.where(cfg['foi'] > 5)[0][0]
            print('loading component bayes factor...')
            for mon in monkey:
                for r, rec in enumerate(gl.recordings[mon][args.region]):
                    print(f'loading component bayes factor for {mon}, recording{rec})')
                    c_bf_tmp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, mon,
                                         f'c_bf.lfp.{args.model}.{args.region}.{args.epoch}-{rec}.npy'))
                    if seg=='Cue':
                        c_bf_tmp = c_bf_tmp[:, :gl.cuePost, :]
                    elif seg=='Pert':
                        c_bf_tmp = c_bf_tmp[:, gl.pertPre:, :]
                    c_bf.append(c_bf_tmp)
            c_bf = np.array(c_bf)
            n_sess, n_freq, n_timep, n_comp = c_bf.shape
            significant_bf = np.zeros((n_freq, n_timep, n_comp))
            n_comp = c_bf.shape[-1]
            print('doing permutations...')
            for i in range(n_comp):
                _, pval, significant_bf[freq_thresh:, :, i] = permutation_t_test_1samp_tf(c_bf[:, freq_thresh:, :, i])
            np.save(os.path.join(gl.nhpDir, gl.pcmDir,
                                 f'significant_bf.lfp.{seg}.{args.region}.{args.epoch}.npy'), significant_bf)
    if args.what=='correlation_plan-exec':
            rng = np.random.default_rng(0)  # seed for reprodocibility
            f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
            Mflex = pickle.load(f)
            freqs = ['alpha', 'beta', 'gamma']
            for f in freqs:
                freq = freq_masks[f]
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
        freqs = ['alpha', 'beta', 'gamma']
        for fband in freqs:
            freq = freq_masks[fband]
            G, Y, i = [], [], 0
            for mon in monkey:
                for roi in args.regions:
                    for r, rec in enumerate(gl.recordings[mon][roi]):
                        print(f'doing {mon}, recording {rec}, {fband} band')

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
                        plan = plan[:, gl.pertIdx+4:gl.pertIdx+24].mean(axis=1)
                        plan = plan - plan.mean(axis=-1, keepdims=True)

                        exec_ext = lfp_grouped[mask_exec['ext']].reshape(n_part, 4, 154, 32).mean(axis=1)
                        exec_flx = lfp_grouped[mask_exec['flx']].reshape(n_part, 4, 154, 32).mean(axis=1)
                        exec = exec_ext - exec_flx
                        exec = exec[:, gl.pertIdx+4:gl.pertIdx+24].mean(axis=1)
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

            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'G_obs.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.npy'), np.array(G))
            T_in, theta_in = pcm.fit_model_individ(Y, Mflex, fixed_effect=None, fit_scale=False, verbose=True)
            T_gr, theta_gr = pcm.fit_model_group(Y, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
            T_in.to_pickle(os.path.join(gl.nhpDir, gl.pcmDir, f'T_in.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.p'))
            T_gr.to_pickle(os.path.join(gl.nhpDir, gl.pcmDir, f'T_gr.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.p'))

            f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_in.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.p'), 'wb')
            pickle.dump(theta_in, f)
            f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_gr.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.p'), 'wb')
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
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'r_bootstrap.lfp.corr_cue-dir.{"-".join(args.regions)}.{fband}.npy'), r_bootstrap)
    if args.what=='lfp2tsv':
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
        freq1, freq2 = 10, 20
        freq_mask = (cfg['foi'] > freq1) & (cfg['foi'] < freq2)
        for epoch in ['plan', 'exec']:
            for roi in rois:
                weight = []
                for mon in monkey:
                    for rec in gl.recordings[mon][roi]:
                        theta_lfp_comp = np.load(
                            os.path.join(gl.nhpDir, gl.pcmDir, mon, f'theta_in.lfp.component.{roi}.{epoch}-{rec}.npy'))
                        var_tot_lfp = np.load(
                            os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.lfp.{roi}.{epoch}-{rec}.npy'))
                        weight_norm = np.exp(theta_lfp_comp[..., :-1]) / var_tot_lfp.T[..., None]
                        noise = np.exp(theta_lfp_comp[..., -1]) / var_tot_lfp.T #[..., None]
                        weight.append(weight_norm)
                        for md in range(weight_norm.shape[-1]):
                            weight_avg = weight_norm[freq_mask, gl.cueIdx:gl.cuePost, md].mean()
                            pcm_dict['epoch'].append(epoch)
                            pcm_dict['roi'].append(roi)
                            pcm_dict['weight'].append(weight_avg)
                            pcm_dict['noise'].append(noise[freq_mask, gl.cueIdx:gl.cuePost].mean())
                            pcm_dict['component'].append(components[epoch][md])
                            pcm_dict['session'].append(rec)
                            pcm_dict['monkey'].append(mon[0])
                weight = np.array(weight)
                df_weight = pd.DataFrame(pcm_dict)
                np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.lfp.{roi}.{epoch}.npy'), weight)
        df_weight.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'weight.lfp.10-20Hz.tsv'), sep='\t', index=False)
    if args.what=='corr2tsv':
        corrs = ['cue-dir']
        freqs = ['alpha', 'beta', 'gamma']
        corr_dict = {
            'r_indiv': [],
            'r_group': [],
            'SNR': [],
            'corr': [],
            'ci_lo': [],
            'ci_hi': [],
            'freq': [],
            'roi': [],
            'session_id': []
        }
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        for corr in corrs:
            for freq in freqs:
                f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_in.lfp.corr_{corr}.M1-S1.{freq}.p'), 'rb')
                theta = pickle.load(f)[0]
                r_bootstrap = np.load(
                    os.path.join(gl.nhpDir, gl.pcmDir, f'r_bootstrap.lfp.corr_{corr}.M1-S1.{freq}.npy'))
                f = open(os.path.join(gl.nhpDir, gl.pcmDir, f'theta_gr.lfp.corr_{corr}.M1-S1.{freq}.p'), 'rb')
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
                corr_dict['freq'].extend([freq] * r_indiv.shape[0])
                corr_dict['roi'].extend(['M1-S1'] * r_indiv.shape[0])
        df_corr = pd.DataFrame(corr_dict)
        df_corr.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'correlations.lfp.tsv'), sep='\t', index=False)
    if args.what=='corrective_drive':
        cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, 'Malfoy', f'cfg.PMd-19.mat'))['cfg']
        freq_masks = make_freq_masks(cfg)
        mask_freq = freq_masks['beta'] + freq_masks['gamma']
        rois = ['M1', 'S1']
        sig = []
        for roi in rois:
            for mon in monkey:
                for rec in gl.recordings[mon][roi]:
                    print(f'doing {mon}, recording {rec}-{roi}')
                    trial_info = pd.read_csv(
                        os.path.join(gl.nhpDir, gl.recDir, f'{mon}', f'trial_info-{rec}.tsv'), sep='\t')
                    idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
                    trial_info = trial_info.loc[idx].reset_index()
                    mapping = {1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7}
                    trial_info.cond = trial_info.cond.map(mapping)
                    lfp = np.load(os.path.join(gl.nhpDir, gl.lfpDir, f'{mon}', f'lfp_aligned.{roi}-{rec}.npy'))
                    lfp_win = lfp[gl.pertIdx + 4:gl.pertIdx + 24, :, mask_freq, :].mean(axis=(0, 2))
                    lfp_grouped, _, part_vec = pcm.group_by_condition(lfp_win, trial_info.cond, trial_info.block,
                                                                      axis=-1)
                    part = np.unique(part_vec)
                    n_part = len(part)
                    obs_des = {'cond_vec': np.r_[np.zeros(n_part), np.ones(n_part)],
                               'part_vec': np.r_[np.arange(0, n_part), np.arange(0, n_part)]}
                    mask_dir = {'ext': np.array([1, 1, 1, 1, 0, 0, 0, 0] * n_part, dtype=bool),
                                'flx': np.array([0, 0, 0, 0, 1, 1, 1, 1] * n_part, dtype=bool)}
                    ext = lfp_grouped[mask_dir['ext']].reshape(n_part, 4, 32).mean(axis=1)
                    flx = lfp_grouped[mask_dir['flx']].reshape(n_part, 4, 32).mean(axis=1)
                    v_pert = ext - flx
                    for r, row in trial_info.iterrows():
                        block = row.block
                        mask_part = part != block
                        mask_trial = (trial_info.block != block).to_numpy()
                        v_pert_tmp = v_pert[mask_part].mean(axis=0)
                        v_pert_tmp /= np.linalg.norm(v_pert_tmp)
                        train = (trial_info.block != block).to_numpy()
                        kin_tmp = kin[:, r]
                        lfp_tmp = lfp[:, :, mask_freq, r].mean(axis=-1)
                        mu = lfp[:, :, mask_freq][..., mask_trial].mean(axis=(2, 3))
                        lfp_tmp_centered = lfp_tmp - mu
                        sig_tmp = lfp_tmp_centered @ v_pert_tmp[None, :].T
                        sig.append(sig_tmp)
        np.save(os.path.join(gl.nhpDir, gl.behavDir, f'PE.npy'), np.array(sig))


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--n_jobs', type=int, default=16)
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--model', type=str, default='component')
    #parser.add_argument('--recording', nargs='+', type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument('--region', type=str, default='PMd')
    parser.add_argument('--regions', type=list, default=['M1', 'S1'])
    #parser.add_argument('--monkey', type=str, default='Malfoy')

    args = parser.parse_args()

    baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction/'
    lfpDir = 'LFPs'
    recDir = 'Recordings'
    pcmDir = 'pcm'

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')