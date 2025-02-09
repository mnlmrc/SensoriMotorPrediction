import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import globals as gl


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


def calc_avg_force(experiment=None, sn=None, session=None, blocks=None, win=(.3, .5)):
    ch_idx = [col in gl.channels['mov'] for col in gl.col_mov[experiment]]

    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, session, f'subj{sn}', f'{experiment}_{sn}.dat'),
                      sep='\t')

    force_dict = {
        'BN': [],
        'TN': [],
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinkie': [],
        'stimFinger': [],
        'cue': [],
        'GoNogo': [],
        'RT': []
    }
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

            force_tmp = mov[onset + int(win[0] * gl.fsample_mov):
                                           onset + int(win[1] * gl.fsample_mov), ch_idx].mean(axis=0)

            force_dict['thumb'].append(force_tmp[0])
            force_dict['index'].append(force_tmp[1])
            force_dict['middle'].append(force_tmp[2])
            force_dict['ring'].append(force_tmp[3])
            force_dict['pinkie'].append(force_tmp[4])
            force_dict['stimFinger'].append(dat_tmp.iloc[ons]['stimFinger'])
            force_dict['RT'].append(dat_tmp.iloc[ons]['RT'])
            force_dict['cue'].append(dat_tmp.iloc[ons]['cue'])
            force_dict['GoNogo'].append(dat_tmp.iloc[ons]['GoNogo'])
            force_dict['BN'].append(dat_tmp.iloc[ons]['BN'])
            force_dict['TN'].append(dat_tmp.iloc[ons]['TN'])

    force_df = pd.DataFrame(force_dict)

    return force_df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--session', type=str, default='behavioural')

    args = parser.parse_args()

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
        snS = [102, 103, 104, 106, 107]

        force_avg = list()
        descr = {
            'cue': [],
            'stimFinger': [],
            'sn': [],
            'GoNogo': [],
            'finger': []
        }

        for sn in snS:
            dat = pd.read_csv(
                os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}', f'{args.experiment}_{sn}.dat'),
                sep='\t')
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

        np.savez(os.path.join(gl.baseDir, args.experiment, f'force.segmented.avg.npz'),
                 data_array=np.stack(force_avg, axis=0), descriptor=descr, allow_pickle=True)


if __name__ == '__main__':
    main()
