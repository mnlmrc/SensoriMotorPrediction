import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import SensoriMotorPrediction.globals as gl
import pickle
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _load_spike(file_path):
    mat = mat73.loadmat(file_path)
    spk = mat['spike_s']
    spk = [s[0] for s in spk]
    return spk


def _align_spike(spike, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
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


def align_spike(monkey='Malfoy', roi='M1', rec=1):
    print(f'loading spikes Recording-{rec}...')
    trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, monkey, f'trial_info-{rec}.tsv'), sep='\t')
    spk = _load_spike(os.path.join(gl.nhpDir, gl.spkDir, monkey, f'spike_s.{roi}-{rec}.mat'))
    idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
    spk = [spk[i] for i in idx]
    trial_info = trial_info.loc[idx].reset_index()
    spk_aligned = _align_spike(spk, trial_info)
    np.save(os.path.join(gl.nhpDir, gl.spkDir, monkey, f'spk_aligned.{roi}-{rec}.npy'), spk_aligned)    


def baseline_subtract(rois=('PMd', 'M1', 'S1')):
    """Pool trial-averaged spiking activity across recordings and subtract the pre-cue baseline.

    Loads `spk_aligned.avg.{roi}-{rec}.npy` for every recording of every monkey in `rois`,
    averages it over units, and subtracts each recording's baseline (mean activity from the
    start of the aligned window up to 5 bins before cue onset).

    Saves `spk_aligned.avg.baseline_subtracted.npz` in the spikes directory, holding the
    baseline-subtracted activity (recording x time), the ROI label of each recording and the
    time axis the recordings are aligned to.
    """
    t_cue = np.linspace(0, gl.cuePost - 1, gl.cuePost)
    t_pert = np.linspace(gl.pertPre, gl.pertPost - 1, gl.pertPost - gl.pertPre) + 5
    t = np.concatenate((t_cue, t_pert))
    bs_mask = (t >= 0) & (t <= gl.cueIdx - 5)

    spk_list, roi_list = [], []
    for roi in rois:
        for mon in gl.monkey:
            for rec in gl.recordings_roi[mon][roi]:
                print(f'loading spikes {mon}, {roi}, Recording-{rec}...')
                spk_aligned = np.load(os.path.join(gl.nhpDir, gl.spkDir, mon,
                                                   f'spk_aligned.avg.{roi}-{rec}.npy'))
                spk_list.append(spk_aligned.mean(axis=-1))  # average over units
                roi_list.append(roi)

    spk = np.stack(spk_list)                # (recording, time)
    roi_labels = np.array(roi_list)

    bs_spk = spk[:, bs_mask].mean(axis=1)   # (recording,)
    spk = spk - bs_spk[:, None]

    np.savez(os.path.join(gl.nhpDir, gl.spkDir, 'spk_aligned.avg.baseline_subtracted.npz'),
             spk=spk, roi=roi_labels, t=t)

    return spk, roi_labels, t


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
    if args.what=='average':
        rec = args.recording[0] if isinstance(args.recording, list) else args.recording
        spk_aligned = np.load(os.path.join(baseDir, spkDir, args.monkey, f'spk_aligned.{args.region}-{rec}.npy'))
        # trial_info = pd.read_csv(os.path.join(baseDir, recDir, f'{args.monkey}/trial_info-{rec}.tsv'), sep='\t')
        # trial_info = trial_info[(trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0)]
        # spk_aligned_prob = np.array([spk_aligned[..., trial_info.prob==prob].mean(axis=-1)
        #                              for prob in trial_info.prob.unique()])
        np.save(os.path.join(baseDir, spkDir, args.monkey, f'spk_aligned.avg.{args.region}-{rec}.npy'),
                spk_aligned.mean(axis=-1))
    if args.what == 'average_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='average',
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