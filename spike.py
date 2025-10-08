import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def align_spike(spike, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    cueTime = trial_info.probTime.to_numpy()
    pertTime = trial_info.pertTime.to_numpy()
    n_unit = spike[0].shape[1]
    spike_aligned = np.zeros((preProb + postProb + prePert + postPert, n_unit, len(spike))) # time_unit_trial
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange])
        spike_aligned[..., t] = spike[t][fullRange]
    return spike_aligned


def load_spike(file_path):
    mat = mat73.loadmat(file_path)
    spk = mat['spike_s']
    spk = [s[0] for s in spk]
    return spk


def main(args):
    if args.what=='align':
        print(f'loading spikes Recording-{args.recording}...')
        trial_info = pd.read_csv(os.path.join(baseDir, recDir, f'{args.monkey}', f'trial_info-{args.recording}.tsv'), sep='\t')
        spk = load_spike(os.path.join(baseDir, spkDir, f'{args.monkey}', f'spike_s.{args.region}-{args.recording}.mat'))
        idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
        spk = [spk[i] for i in idx]
        trial_info = trial_info.loc[idx].reset_index()
        spk_aligned = align_spike(spk, trial_info)
        np.save(os.path.join(baseDir, spkDir, f'{args.monkey}', f'spk_aligned.{args.region}-{args.recording}.npy'), spk_aligned)
    if args.what=='align_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='align',
                region=args.region,
                recording=rec,
                monkey=args.monkey,
            )
            main(arg)
    if args.what=='pca':
        pca = PCA(n_components=5)
        scaler = StandardScaler()
        for rec in args.recording:
            spk = np.load(os.path.join(baseDir, spkDir, f'{args.monkey}', f'spk_aligned.{args.region}-{rec}.npy'))
            Tp, N, Tr = spk.shape
            spk_stacked = np.transpose(spk, (0, 2, 1)).reshape(-1, spk.shape[1])
            spk_norm = scaler.fit_transform(spk_stacked)

            PCs = pca.fit_transform(spk_norm)
            PCs = PCs.reshape(Tp, Tr, -1)

            np.save(os.path.join(baseDir, spkDir, f'{args.monkey}', f'pcs.{args.region}-{rec}.npy'), PCs)
        pass


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--epoch', type=str, default='plan')
    parser.add_argument('--recording', nargs='+', type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument( '--region', type=str, default='PMd')
    parser.add_argument('--monkey', type=str, default='Malfoy')

    args = parser.parse_args()

    baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction/'
    recDir = 'Recordings'
    spkDir = 'spikes'

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')