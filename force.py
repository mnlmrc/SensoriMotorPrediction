import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import globals as gl
import time
from util import corr_xval

def load_mov(filename):
    try:
        with open(filename, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            mov = [np.array(trial_data) for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filename}") from e

    return mov


def segment_mov(experiment=None, sn=None, session=None, blocks=None, prestim=gl.prestim, poststim=gl.poststim):
    ch_idx = [col in gl.channels['mov'] for col in gl.col_mov[experiment]]

    force = []
    for bl in blocks:

        filename = os.path.join(gl.baseDir, experiment, session, f'subj{sn}', f'{experiment}_{sn}_{int(bl):02d}.mov')

        mov = load_mov(filename)
        mov = np.concatenate(mov, axis=0)

        idx = mov[:, 1] > gl.planState[experiment]
        idxD = np.diff(idx.astype(int))
        stimOnset = np.where(idxD == 1)[0]

        print(f'Processing... subj{sn}, block {bl}, {len(stimOnset)} trials found...')

        for ons, onset in enumerate(stimOnset):
            # if self.dat.GoNogo.iloc[ons] == 'go':
            force.append(mov[onset - int(prestim * gl.fsample_mov):onset + int(poststim * gl.fsample_mov), ch_idx].T)

    descr = json.dumps({
        'experiment': experiment,
        'sn': sn,
        'fsample': gl.fsample_mov,
        'prestim': prestim,
        'poststim': poststim,
    })

    return np.array(force), descr


def calc_md(X):
    """

    Args:
        X: timepoints x channels data

    Returns:

    """
    N, m = X.shape
    F1 = X[0]
    FN = X[-1] - F1  # Shift the end point

    shifted_matrix = X - F1  # Shift all points

    d = list()

    for t in range(1, N - 1):
        Ft = shifted_matrix[t]

        # Project Ft onto the ideal straight line
        proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN

        # Calculate the Euclidean distance
        d.append(np.linalg.norm(Ft - proj))

    d = np.array(d)
    MD = d.mean()

    return MD, d

def calc_avg_force(experiment=None, sn=None, session=None, blocks=None, win=[(-1.5, 0), (.2, .4), ]):
    ch_idx = [col in gl.channels['mov'] for col in gl.col_mov[experiment]]

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, session, f'subj{sn}', f'{experiment}_{sn}.dat'),
                      sep='\t')

    force_dict = {
        'BN': [],
        'TN': [],
        'stimFinger': [],
        'cue': [],
        'MD': [],
    }
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    for i in range(len(win)):
        for f in fingers:
            force_dict[f'{f}{i}'] = []

    if 'GoNogo' in dat:
        force_dict['GoNogo'] = []

    for bl in blocks:

        dat_tmp = dat[dat['BN'] == int(bl)]

        filename = os.path.join(gl.baseDir, experiment, session, f'subj{sn}', f'{experiment}_{sn}_{int(bl):02d}.mov')

        mov = load_mov(filename)
        mov = np.concatenate(mov, axis=0)

        idx = mov[:, 1] > gl.planState[experiment]
        idxD = np.diff(idx.astype(int))
        stimOnset = np.where(idxD == 1)[0]

        print(f'Processing... subj{sn}, block {bl}, {len(stimOnset)} trials found...')

        assert(len(stimOnset) == len(dat_tmp))

        for ons, onset in enumerate(stimOnset):
            for i, w in enumerate(win):
                start = onset + int(w[0] * gl.fsample_mov)
                end = onset + int(w[1] * gl.fsample_mov)
                force_tmp = mov[start:end, ch_idx].mean(axis=0)


                for j, f in enumerate(fingers):
                    force_dict[f'{f}{i}'].append(force_tmp[j])

            # calc mean deviation
            X = mov[onset:onset + int(.5 * gl.fsample_mov), ch_idx]
            md, _ = calc_md(X)
            force_dict['MD'].append(md)
            force_dict['stimFinger'].append(dat_tmp.iloc[ons]['stimFinger'])
            force_dict['cue'].append(dat_tmp.iloc[ons]['cue'])
            force_dict['BN'].append(dat_tmp.iloc[ons]['BN'])
            force_dict['TN'].append(dat_tmp.iloc[ons]['TN'])
            force_dict['GoNogo'].append(dat_tmp.iloc[ons]['GoNogo'])

    force_df = pd.DataFrame(force_dict)

    return force_df

def main(args):

    pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')

    if args.what == 'mov2npz':
        if args.session == 'training':
            blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).runsTraining[0].split('.')
        elif args.session == 'behavioural':
            blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).FuncRuns[0].split('.')
        force_segmented, descr = segment_mov(experiment=args.experiment,
                                             sn=args.sn,
                                             session=args.session,
                                             blocks=blocks,
                                             prestim=gl.prestim,
                                             poststim=gl.poststim)
        print(f"Saving participant subj{args.sn}, session {args.session}...")
        np.savez(os.path.join(gl.baseDir, args.experiment, args.session, f'subj{args.sn}',
                              f'{args.experiment}_{args.sn}_force_segmented.npz'),
                 data_array=force_segmented, descriptor=descr, allow_pickle=False)
    if args.what == 'mov2npz_all':
        for sn in args.snS:
            args = argparse.Namespace(what='mov2npz',
                                        experiment=args.experiment,
                                      sn=sn,
                                      session=args.session,)
            main(args)
    if args.what == 'single_trial':

        if args.session == 'training':
            blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).runsTraining[0].split('.')
        elif args.session == 'behavioural':
            blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).FuncRuns[0].split('.')

        force_df = calc_avg_force(experiment=args.experiment,
                                  sn=args.sn,
                                  session=args.session,
                                  blocks=blocks, )
        force_df.to_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'subj{args.sn}',
                                     f'{args.experiment}_{args.sn}_force_single_trial.tsv'), sep='\t', index=False)
    if args.what == 'avg_continuous':

        force_avg = list()
        descr = {
            'cue': [],
            'stimFinger': [],
            'sn': [],
            'GoNogo': [],
            'finger': []
        }

        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')

        for sn in args.snS:
            runs = pinfo[pinfo['sn'] == sn].FuncRuns.reset_index(drop=True)[0].split('.')
            dat = pd.read_csv(
                os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}', f'{args.experiment}_{sn}.dat'),
                sep='\t')
            dat = dat[dat['BN'].astype(str).isin(runs)]
            npz = np.load(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                       f'{args.experiment}_{sn}_force_segmented.npz'))
            force = npz['data_array']
            for c, cue in enumerate(gl.cue_code):

                for f in range(force.shape[1]):
                    force_tmp = force[(dat.cue == cue) & (dat.stimFinger == 99999) & (dat.GoNogo == 'nogo'), f].mean(
                        axis=0, keepdims=True).squeeze()

                    force_avg.append(force_tmp)
                    descr['cue'].append(gl.cue_mapping[cue])
                    descr['stimFinger'].append(gl.stimFinger_mapping[99999])
                    descr['sn'].append(str(sn))
                    descr['finger'].append(gl.channels['mov'][f])
                    descr['GoNogo'].append('nogo')

                for sf, stimF in enumerate(gl.stimFinger_code):

                    print(f'subj{sn}, {cue}, {stimF}')

                    for f in range(force.shape[1]):
                        force_tmp = force[(dat.cue == cue) & (dat.stimFinger == stimF) & (dat.GoNogo == 'go'), f].mean(
                            axis=0, keepdims=True).squeeze()

                        force_avg.append(force_tmp)
                        descr['cue'].append(gl.cue_mapping[cue])
                        descr['stimFinger'].append(gl.stimFinger_mapping[stimF])
                        descr['sn'].append(str(sn))
                        descr['finger'].append(gl.channels['mov'][f])
                        descr['GoNogo'].append('go')

        np.savez(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'force.segmented.avg.npz'),
                 data_array=np.stack(force_avg, axis=0), descriptor=descr, allow_pickle=True)
    if args.what == 'single_trial_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='single_trial',
                experiment=args.experiment,
                sn=sn,
                session=args.session,
            )
            main(args)
    if args.what == 'corr_xval':
        within_block, between_block = [], []
        for sn in args.snS:
            behav_path = os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}')
            force = pd.read_csv(os.path.join(behav_path, f'{args.experiment}_{sn}_force_single_trial.tsv'), sep='\t')
            force = force[force['GoNogo'] == 'go']
            force = force.groupby(['BN', 'cue', 'stimFinger', ]).mean(numeric_only=True).reset_index()
            force['cue'] = force['cue'].map(gl.cue_mapping)
            force['stimFinger'] = force['stimFinger'].map(gl.stimFinger_mapping)
            force['cue,stimFinger'] = force['cue'] + ',' + force['stimFinger']
            force['cond_vec'] = force['cue,stimFinger'].map(gl.regressor_mapping)
            X = force[['thumb0', 'index0', 'middle0', 'ring0', 'pinkie0']].to_numpy()
            Y = force[['thumb1', 'index1', 'middle1', 'ring1', 'pinkie1']].to_numpy()
            part_vec = force['BN'].to_numpy()
            cond_vec = force['cond_vec'].to_numpy()

            wb, bb = corr_xval(X, Y, cond_vec, part_vec)
            within_block.append(wb)
            between_block.append(bb)

        within_block = np.array(within_block)
        between_block = np.array(between_block)

        np.save(os.path.join(gl.baseDir, args.experiment, gl.behavDir, 'corr_xval.plan_vs_exec.within_block.npy'), within_block)
        np.save(os.path.join(gl.baseDir, args.experiment, gl.behavDir, 'corr_xval.plan_vs_exec.between_block.npy'), between_block)

if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115], type=int)
    parser.add_argument('--session', type=str, default='behavioural')

    args = parser.parse_args()

    main(args)

    end = time.time()
    print(f'Time elapsed: {end - start} seconds')
