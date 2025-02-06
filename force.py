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
        'GoNogo': []
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
            force_dict['cue'].append(dat_tmp.iloc[ons]['cue'])
            force_dict['GoNogo'].append(dat_tmp.iloc[ons]['GoNogo'])
            force_dict['BN'].append(dat_tmp.iloc[ons]['BN'])
            force_dict['TN'].append(dat_tmp.iloc[ons]['TN'])

    force_df = pd.DataFrame(force_dict)

    return force_df


# class Force:
#     def __init__(self, experiment, session, participant_id=None):
#
#         self.experiment = experiment
#         self.session = session
#
#         self.path = self.get_path()
#
#         self.pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
#
#         self.prestim = int(gl.prestim * gl.fsample_mov)
#         self.poststim = int(gl.poststim * gl.fsample_mov)
#
#         # time windows in seconds
#         self.win = {
#             'Pre': (-.5, 0),
#             'LLR': (.1, .4),
#             'VOl': (.4, 1)
#         }
#
#         if participant_id is not None:
#             self.participant_id = participant_id
#             self.sn = int(''.join(filter(str.isdigit, participant_id)))
#             self.dat = pd.read_csv(os.path.join(self.path, participant_id, f'{experiment}_{self.sn}.dat'), sep='\t')
#
#     def get_path(self):
#
#         session = self.session
#         experiment = self.experiment
#         # participant_id = self.participant_id
#
#         if session == 'scanning':
#             path = os.path.join(gl.baseDir, experiment, gl.behavDir)
#         elif session == 'training':
#             path = os.path.join(gl.baseDir, experiment, gl.trainDir)
#         elif session == 'behavioural':
#             path = os.path.join(gl.baseDir, experiment, gl.behavDir)
#         elif session == 'pilot':
#             path = os.path.join(gl.baseDir, experiment, gl.pilotDir)
#         else:
#             raise ValueError('Session name not recognized.')
#
#         return path
#
#     def get_block(self):
#
#         pinfo = self.pinfo
#         sn = self.sn
#
#         if self.session == 'scanning':
#             blocks = pinfo[pinfo.sn == sn].runsSess1.iloc[0].split('.')
#         elif self.session == 'training':
#             blocks = pinfo[pinfo.sn == sn].runsTraining.iloc[0].split('.')
#         elif self.session == 'behavioural':
#             blocks = pinfo[pinfo.sn == sn].blocks_mov.iloc[0].split('.')
#         elif self.session == 'pilot':
#             blocks = pinfo[pinfo.sn == sn].blocks_mov.iloc[0].split('.')
#         else:
#             raise ValueError('Session name not recognized.')
#
#         return blocks
#
#     def sec2sample(self, sec):
#         samples = self.prestim + int(sec * gl.fsample_mov)
#
#         return samples
#
#     def load_mov(self, filename):
#         try:
#             with open(filename, 'rt') as fid:
#                 trial = 0
#                 A = []
#                 for line in fid:
#                     if line.startswith('Trial'):
#                         trial_number = int(line.split(' ')[1])
#                         trial += 1
#                         if trial_number != trial:
#                             warnings.warn('Trials out of sequence')
#                             trial = trial_number
#                         A.append([])
#                     else:
#                         data = np.fromstring(line, sep=' ')
#                         if A:
#                             A[-1].append(data)
#                         else:
#                             warnings.warn('Data without trial heading detected')
#                             A.append([data])
#
#                 mov = [np.array(trial_data) for trial_data in A]
#
#         except IOError as e:
#             raise IOError(f"Could not open {filename}") from e
#
#         return mov
#
#     def segment_mov(self):
#
#         sn = self.sn
#         experiment = self.experiment
#         participant_id = self.participant_id
#         path = self.get_path()
#         blocks = self.get_block()
#         ch_idx = [col in gl.channels['mov'] for col in gl.col_mov[experiment]]
#         prestim = self.prestim
#         poststim = self.poststim
#
#         force = []
#         for bl in blocks:
#             block = f'{int(bl):02d}'
#             filename = os.path.join(path, participant_id, f'{experiment}_{sn}_{block}.mov')
#
#             mov = self.load_mov(filename)
#             mov = np.concatenate(mov, axis=0)
#
#             idx = mov[:, 1] > gl.planState[experiment]
#             idxD = np.diff(idx.astype(int))
#             stimOnset = np.where(idxD == 1)[0]
#
#             print(f'Processing... {self.participant_id}, block {bl}, {len(stimOnset)} trials found...')
#
#             for ons, onset in enumerate(stimOnset):
#                 # if self.dat.GoNogo.iloc[ons] == 'go':
#                 force.append(mov[onset - prestim:onset + poststim, ch_idx].T)
#
#         descr = json.dumps({
#             'experiment': self.experiment,
#             'participant': self.participant_id,
#             'fsample': gl.fsample_mov,
#             'prestim': prestim,
#             'poststim': poststim,
#         })
#
#         return np.array(force), descr
#
#     def calc_avg_timec(self, GoNogo='go'):
#         """
#         Calculate the average force data across trials for each cue and stimulation finger.
#
#         Returns:
#             force_avg (numpy.ndarray): A 4D array with dimensions (cue, stimFinger, channel, time).
#         """
#
#         force = self.load_npz()
#         blocks = self.get_block()
#
#         # take only rows in dat that belong to good blocks based on participants.tsv
#         dat = self.dat[(self.dat.BN.isin(blocks) |
#                         self.dat.BN.isin(np.array(list(map(int, blocks)))))]
#
#         keep_trials = (dat.GoNogo == GoNogo)
#         force = force[keep_trials]
#         dat = dat[keep_trials]
#
#         if GoNogo == 'go':
#             force_avg = np.zeros((len(gl.cue_code), len(gl.stimFinger_code), force.shape[-2], force.shape[-1]))
#             for c, cue in enumerate(gl.cue_code):
#                 for sf, stimF in enumerate(gl.stimFinger_code):
#                     force_avg[c, sf] = force[(dat.cue == cue) & (dat.stimFinger == stimF)].mean(axis=0, keepdims=True)
#         elif GoNogo == 'nogo':
#             force_avg = np.zeros((len(gl.cue_code), force.shape[-2], force.shape[-1]))
#             for c, cue in enumerate(gl.cue_code):
#                 force_avg[c] = force[(dat.cue == cue)].mean(axis=0, keepdims=True)
#         else:
#             force_avg = None
#
#         return force_avg
#
#     def calc_bins(self):
#
#         force = self.load_npz()
#
#         df = pd.DataFrame()
#         for w in self.win.items():
#             for c, ch in enumerate(gl.channels['mov']):
#                 df[f'{w}/{ch}'] = force[:, c, w[0]:w[1]].mean(axis=-1)
#
#         df = pd.concat([self.dat, df], axis=1)
#
#         return df
#
#     def load_npz(self):
#
#         path = self.get_path()
#         experiment = self.experiment
#         participant_id = self.participant_id
#
#         sn = int(''.join([c for c in participant_id if c.isdigit()]))
#
#         npz = np.load(os.path.join(path, participant_id, f'{experiment}_{sn}.npz'))
#         force = npz['data_array']
#
#         return force
#
#     def calc_rdm(self, timew, GoNogo='go'):
#
#         force = self.load_npz()
#         blocks = self.get_block()
#
#         # take only rows in dat that belong to good blocks based on participants.tsv
#         dat = self.dat[(self.dat.BN.isin(blocks) |
#                         self.dat.BN.isin(np.array(list(map(int, blocks)))))]
#
#         keep_trials = (dat.GoNogo == GoNogo)
#         force = force[keep_trials]
#         dat = dat[keep_trials]
#
#         run = dat.BN
#         cue = dat.cue
#         stimFinger = dat.stimFinger
#
#         cue = cue.map(gl.cue_mapping)
#         stimFinger = stimFinger.map(gl.stimFinger_mapping)
#
#         cond_vec = [f'{sf},{c}' for c, sf in zip(cue, stimFinger)]
#
#         timew = (self.sec2sample(timew[0]), self.sec2sample(timew[1]))
#         timew = np.arange(timew[0], timew[1])
#
#         rdm = calc_rdm_unbalanced(force[..., timew].mean(axis=-1), gl.channels['mov'], cond_vec, run,
#                                   method='crossnobis')
#
#         if GoNogo == 'go':
#             rdm.reorder(np.array([1, 2, 3, 0, 4, 5, 6, 7]))
#         elif GoNogo == 'nogo':
#             rdm.reorder(np.array([0, 2, 3, 4, 1]))
#
#         return rdm
#
#     def calc_dist_timec(self, method='euclidean', GoNogo='go'):
#
#         force = self.load_npz()
#         blocks = self.get_block()
#
#         # take only rows in dat that belong to good blocks based on participants.tsv
#         dat = self.dat[(self.dat.BN.isin(blocks) |
#                         self.dat.BN.isin(np.array(list(map(int, blocks)))))]
#
#         keep_trials = (dat.GoNogo == GoNogo)
#         force = force[keep_trials]
#         dat = dat[keep_trials]
#
#         run = dat.BN
#         cue = dat.cue
#         stimFinger = dat.stimFinger
#
#         cue = cue.map(gl.cue_mapping)
#         stimFinger = stimFinger.map(gl.stimFinger_mapping)
#
#         cond_vec = [f'{sf},{c}' for c, sf in zip(cue, stimFinger)]
#
#         dist_stimFinger = np.zeros(force.shape[-1])
#         dist_cue = np.zeros(force.shape[-1])
#         dist_stimFinger_cue = np.zeros(force.shape[-1])
#         for t in range(force.shape[-1]):
#             print('participant %s' % self.participant_id + ', time point %f' % t)
#
#             # calculate rdm for timepoint t
#             force_tmp = force[:, :, t]
#
#             rdm = calc_rdm(force_tmp, gl.channels['mov'], cond_vec, run, method=method)
#
#             if GoNogo == 'go':
#                 rdm.reorder(np.array([0, 3, 1, 2, 7, 4, 6, 5]))
#             elif GoNogo == 'nogo':
#                 rdm.reorder(np.array([1, 4, 2, 0, 3]))
#
#             if GoNogo == 'go':
#                 dist_stimFinger[t] = rdm.dissimilarities[:, gl.mask_stimFinger].mean()
#                 dist_cue[t] = rdm.dissimilarities[:, gl.mask_cue].mean()
#                 dist_stimFinger_cue[t] = rdm.dissimilarities[:, gl.mask_stimFinger_cue].mean()
#
#             elif GoNogo == 'nogo':
#                 dist_cue[t] = rdm.dissimilarities.mean()
#
#         descr = json.dumps({
#             'participant': self.participant_id,
#             'mask_stimFinger': list(gl.mask_stimFinger.astype(str)) if GoNogo == 'go' else None,
#             'mask_cue': list(gl.mask_cue.astype(str)) if GoNogo == 'go' else None,
#             'mask_stimFinger_by_cue': list(gl.mask_stimFinger_cue.astype(str)) if GoNogo == 'go' else None,
#             'factor_order': ['stimFinger', 'cue', 'stimFinger_by_cue'],
#         })
#
#         dist = np.stack([dist_stimFinger, dist_cue, dist_stimFinger_cue])
#
#         return dist, descr

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--session', type=str, default='behavioural')

    args = parser.parse_args()

    pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')

    if args.session == 'training':
        blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).runsTraining[0].split('.')
    elif args.session == 'behavioural':
        blocks = pinfo[pinfo.sn == args.sn].reset_index(drop=True).FuncRuns[0].split('.')

    if args.what == 'mov2npz':
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
        force_df = calc_avg_force(experiment=args.experiment,
                                  sn=args.sn,
                                  session=args.session,
                                  blocks=blocks, )
        force_df.to_csv(os.path.join(gl.baseDir, args.experiment, args.session, f'subj{args.sn}',
                                     f'{args.experiment}_{args.sn}_force_single_trial.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
