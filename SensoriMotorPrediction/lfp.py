import time
import scipy.io as sio
import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import SensoriMotorPrediction.globals as gl
import pickle

def _load_lfp(file_path):
    mat = mat73.loadmat(file_path)
    return mat['lfp']


def _align_lfp(lfp, cfg, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    #cueTime = trial_info.probTime.to_numpy() - 1
    #pertTime = trial_info.pertTime.to_numpy() - trial_info.probTime.to_numpy()
    cueTime = trial_info.cueTime.to_numpy()
    pertTime = trial_info.goTime.to_numpy() - trial_info.cueTime.to_numpy()
    pertTime = cueTime + pertTime
    toi = cfg['cfg']['toi']
    n_freq = lfp.shape[2]
    n_elec = lfp.shape[1]
    n_trial = lfp.shape[-1]
    lfp_aligned = np.zeros((preProb + postProb + prePert + postPert, n_elec, n_freq, n_trial)) # time_unit_trial
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange]).astype(int)
        lfp_aligned[..., t] = lfp[fullRange, :, :, t]
    return lfp_aligned


def align_lfp(monkey='Malfoy', roi='M1', rec=1):
    print(f'loading lfps Recording-{rec}...')
    trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, f'{monkey}/trial_info-{rec}.tsv'), sep='\t')
    lfp = load_lfp(os.path.join(gl.nhpDir, gl.lfpDir, f'{monkey}/lfp.{roi}-{rec}.mat'))
    cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, f'{monkey}/cfg.{roi}-{rec}.mat'))
    lfp = lfp[..., (trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
    lfp_aligned = align_lfp(lfp, cfg, trial_info,)
    np.save(os.path.join(gl.nhpDir, gl.lfpDir, monkey, f'lfp_aligned.{roi}-{rec}.npy'), lfp_aligned)


def baseline_normalise(rois=('PMd', 'M1', 'S1')):
    """Pool trial-averaged LFP power across recordings and express it as change from baseline (dB).

    Loads `lfp_aligned.avg.{roi}-{rec}.npy` for every recording of every monkey in `rois` and
    divides it, frequency by frequency, by that recording's baseline power (mean power from the
    start of the aligned window up to 5 bins before cue onset), in decibels.

    Saves `lfp_aligned.avg.dB.npz` in the LFPs directory, holding the baseline-normalised power
    (recording x time x frequency), the ROI label of each recording, the time axis the
    recordings are aligned to and the frequencies of interest.
    """
    t_cue = np.linspace(0, gl.cuePost - 1, gl.cuePost)
    t_pert = np.linspace(gl.pertPre, gl.pertPost - 1, gl.pertPost - gl.pertPre) + 5
    t = np.concatenate((t_cue, t_pert))
    bs_mask = (t >= 0) & (t <= gl.cueIdx - 5)

    lfp_list, roi_list = [], []
    foi = None
    for roi in rois:
        for mon in gl.monkey:
            for rec in gl.recordings_roi[mon][roi]:
                print(f'loading lfps {mon}, {roi}, Recording-{rec}...')
                lfp_aligned = np.load(os.path.join(gl.nhpDir, gl.lfpDir, mon,
                                                   f'lfp_aligned.avg.{roi}-{rec}.npy'))
                if foi is None:
                    cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, mon, f'cfg.{roi}-{rec}.mat'))
                    foi = cfg['cfg']['foi']
                lfp_list.append(lfp_aligned)
                roi_list.append(roi)

    lfp = np.stack(lfp_list)                    # (recording, time, freq)
    roi_labels = np.array(roi_list)

    bs_lfp = lfp[:, bs_mask, :].mean(axis=1)    # (recording, freq)
    lfp_dB = 10 * np.log10(lfp / bs_lfp[:, None, :])

    np.savez(os.path.join(gl.nhpDir, gl.lfpDir, 'lfp_aligned.avg.dB.npz'),
             lfp_dB=lfp_dB, roi=roi_labels, t=t, foi=foi)

    return lfp_dB, roi_labels, t, foi


def make_freq_masks(cfg):
    foi = cfg['foi']
    delta = (foi >= 1) & (foi < 3)
    theta = (foi >= 3) & (foi < 8)
    alpha_beta = (foi >= 8) & (foi < 25)
    alpha = (foi >= 8) & (foi < 13)
    beta = (foi >= 13) & (foi < 25)
    gamma = (foi >= 25) & (foi < 100)

    freq_masks = {
        'delta': delta,
        'theta': theta,
        'alpha-beta': alpha_beta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
    }

    return freq_masks


def main(args):
    if args.what=='align':
        rec = args.recording[0] if isinstance(args.recording, list) else args.recording
        print(f'loading lfps Recording-{rec}...')
        trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, f'{args.monkey}/trial_info-{rec}.tsv'), sep='\t')
        lfp = load_lfp(os.path.join(gl.nhpDir, gl.lfpDir, f'{args.monkey}/lfp.{args.region}-{rec}.mat'))
        cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, f'{args.monkey}/cfg.{args.region}-{rec}.mat'))
        lfp = lfp[..., (trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        lfp_aligned = align_lfp(lfp, cfg, trial_info, postProb=30)
        np.save(os.path.join(gl.nhpDir, gl.lfpDir, f'{args.monkey}', f'lfp_aligned.{args.region}-{rec}.npy'), lfp_aligned)
    if args.what == 'align_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='align',
                region=args.region,
                recording=rec,
                monkey=args.monkey,)
            main(arg)
    if args.what=='average':
        rec = args.recording[0] if isinstance(args.recording, list) else args.recording
        print(f'doing lfps in {args.region} recording-{rec}...')
        lfp_aligned = np.load(os.path.join(gl.nhpDir, gl.lfpDir, args.monkey, f'lfp_aligned.{args.region}-{rec}.npy'))
        # trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, f'{args.monkey}/trial_info-{rec}.tsv'), sep='\t')
        # trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        # lfp_aligned_prob = np.array([lfp_aligned[..., trial_info.prob==prob].mean(axis=-1)
        #                              for prob in trial_info.prob.unique()])
        pass
        np.save(os.path.join(gl.nhpDir, gl.lfpDir, args.monkey, f'lfp_aligned.avg.{args.region}-{rec}.npy'),
                lfp_aligned.mean(axis=(1, 3)))
    if args.what == 'average_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='average',
                region=args.region,
                recording=rec,
                monkey=args.monkey,
            )
            main(arg)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='continuous')
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--recording', nargs='+', type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument( '--region', type=str, default='PMd')
    parser.add_argument('--monkey', type=str, default='Malfoy')

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')
