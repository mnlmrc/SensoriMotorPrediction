import mat73
import numpy as np
import pandas as pd
import time
import argparse
import os
import globals as gl
import pickle
import PcmPy as pcm
from lfp import make_freq_masks
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
    monkey = ['Pert', 'Malfoy']
    if args.what=='align':
        rec = args.recording[0] if isinstance(args.recording, list) else args.recording
        print(f'loading kinematics Recording-{rec}...')
        trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, f'{args.monkey}', f'trial_info-{rec}.tsv'), sep='\t')
        kin = load_kinematics(os.path.join(gl.nhpDir, gl.behavDir, f'{args.monkey}', f'elbow_angle-{rec}.mat'))
        idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
        kin = [kin[i] for i in idx]
        trial_info = trial_info.loc[idx].reset_index()
        kin_aligned = align_kinematics(kin, trial_info)
        np.save(os.path.join(gl.nhpDir, gl.behavDir, f'{args.monkey}', f'kin_aligned-{rec}.npy'), kin_aligned)
    if args.what=='align_all':
        for rec in args.recording:
            arg = argparse.Namespace(
                what='align',
                recording=rec,
                monkey=args.monkey,
            )
            main(arg)
    if args.what=='excursion':
        rois = ['M1', 'S1']
        excr_dict = {
            'roi': [],
            'excr_peak': [],
            'excr_auc': [],
            'monkey': [],
            'session': [],
            'cond': [],
            'pert': []
        }
        kin_group = []
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
                    kin = np.load(os.path.join(gl.nhpDir, gl.behavDir, f'{mon}', f'kin_aligned-{rec}.npy'))
                    for r, row in trial_info.iterrows():
                        kin_tmp = kin[:, r]
                        bs = kin_tmp[:gl.cueIdx].mean()
                        kin_tmp = kin_tmp - bs
                        if row.pertDirection==2:
                            kin_tmp = np.clip(kin_tmp, 0, None)
                        elif row.pertDirection==1:
                            kin_tmp = np.clip(kin_tmp * -1, 0, None)
                        kin_group.append(kin_tmp)
                        excr_peak = np.max(kin_tmp)
                        excr_auc = np.trapezoid(kin_tmp[gl.pertIdx:gl.pertIdx + 30])
                        excr_dict['roi'].append(roi)
                        excr_dict['excr_peak'].append(excr_peak)
                        excr_dict['excr_auc'].append(excr_auc)
                        excr_dict['session'].append(rec)
                        excr_dict['monkey'].append(mon[0])
                        excr_dict['pert'].append(row.pertDirection)
                        excr_dict['cond'].append(row.cond)
        excr_df = pd.DataFrame(excr_dict)
        excr_df.to_csv(os.path.join(gl.nhpDir, gl.behavDir, f'excursion.tsv'), sep='\t', index=False)
        np.save(os.path.join(gl.nhpDir, gl.behavDir, f'excursion.npy'), np.array(kin_group))
    if args.what=='corrective_drive':
        df_excr = pd.read_csv(os.path.join(gl.nhpDir, gl.behavDir, 'excursion.tsv'), sep='\t')
        excr = np.load(os.path.join(gl.nhpDir, gl.behavDir, 'excursion.npy'))
        pe = np.load(os.path.join(gl.nhpDir, gl.behavDir, 'PE.npy'))
        idx = ((df_excr['roi'] == 'M1')) & ((df_excr['cond'] == 4) | (df_excr['cond'] == 5))
        df_excr = df_excr[idx]
        excr = excr[idx]
        pe = np.abs(pe[idx])
        excr_peak = np.max(excr, axis=1)
        pe_peak = np.max(pe, axis=1)
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

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')