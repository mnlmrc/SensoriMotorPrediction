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

def align_kinematics(kin, trial_info, preProb=20, postProb=64, prePert=30, postPert=40,):
    cueTime = trial_info.probTime.to_numpy()
    pertTime = trial_info.pertTime.to_numpy()
    kin_aligned = np.zeros((preProb + postProb + prePert + postPert, len(kin))) # time_unit_trial
    for t, (cT, pT) in enumerate(zip(cueTime, pertTime)):
        probRange = np.arange(cT - preProb, cT + postProb)
        pertRange = np.arange(pT - prePert, pT + postPert)
        fullRange = np.concatenate([probRange, pertRange])
        kin_aligned[..., t] = kin[t][fullRange]
    return kin_aligned


def load_kinematics(file_path):
    mat = mat73.loadmat(file_path)
    kin = mat['elbKin']
    return kin


def main(args):
    if args.what=='align':
        print(f'loading kinematics Recording-{args.recording}...')
        trial_info = pd.read_csv(os.path.join(baseDir, recDir, f'{args.monkey}', f'trial_info-{args.recording}.tsv'), sep='\t')
        kin = load_kinematics(os.path.join(baseDir, kinDir, f'{args.monkey}', f'elbow_angle-{args.recording}.mat'))
        idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
        kin = [kin[i] for i in idx]
        trial_info = trial_info.loc[idx].reset_index()
        kin_aligned = align_kinematics(kin, trial_info)
        np.save(os.path.join(baseDir, kinDir, f'{args.monkey}', f'kin_aligned-{args.recording}.npy'), kin_aligned)
    if args.what=='align_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='align',
                recording=rec,
                monkey=args.monkey,
            )
            main(arg)


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
    kinDir = 'Behavioural'

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')