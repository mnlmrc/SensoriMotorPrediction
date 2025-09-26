import time
import scipy.io as sio
import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle

def load_lfp(file_path):
    mat = mat73.loadmat(file_path)
    return mat['lfp']


def align_lfp(lfp, cfg, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    cueTime = trial_info.probTime.to_numpy() - 1
    pertTime = trial_info.pertTime.to_numpy() - trial_info.probTime.to_numpy()
    pertTime = cueTime + pertTime
    toi = cfg['cfg']['toi']
    n_freq = lfp.shape[2]
    n_elec = lfp.shape[1]
    n_trial = lfp.shape[-1]
    lfp_aligned = np.zeros((preProb + postProb + prePert + postPert, n_elec, n_freq, n_trial)) # time_unit_trial
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange])
        lfp_aligned[..., t] = lfp[fullRange, :, :, t]
    return lfp_aligned


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
        print(f'loading lfps Recording-{args.recording}...')
        trial_info = pd.read_csv(os.path.join(baseDir, recDir, f'{args.monkey}/trial_info-{args.recording}.tsv'), sep='\t')
        lfp = load_lfp(os.path.join(baseDir, lfpDir, f'{args.monkey}/lfp.{args.region}-{args.recording}.mat'))
        cfg = mat73.loadmat(os.path.join(baseDir, lfpDir, f'{args.monkey}/cfg.{args.region}-{args.recording}.mat'))
        lfp = lfp[..., (trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        lfp_aligned = align_lfp(lfp, cfg, trial_info)
        np.save(os.path.join(baseDir, lfpDir, f'{args.monkey}', f'lfp_aligned.{args.region}-{args.recording}.npy'), lfp_aligned)
    if args.what=='align_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='align',
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

    baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction/'
    recDir = 'Recordings'
    lfpDir = 'LFPs'

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')
