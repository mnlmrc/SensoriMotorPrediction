import argparse

import pandas as pd
import numpy as np
import os
import globals as gl
from scipy.signal import resample
import pickle


def detect_trig(trig_sig, time_trig, amp_threshold=None, ntrials=None, debugging=False):
    """

    :param trig_sig:
    :param time_trig:
    :param amp_threshold:
    :param ntrials:
    :param debugging:
    :return:
    """

    ########## old trigger detection (subj 100-101)
    # trig_sig = trig_sig / np.max(trig_sig)
    # diff_trig = np.diff(trig_sig)
    # diff_trig[diff_trig < self.amp_threshold] = 0
    # locs, _ = find_peaks(diff_trig)
    ##############################################

    # trig_sig = pd.to_numeric(trig_sig).to_numpy()
    # time_trig = pd.to_numeric(time_trig).to_numpy()

    trig_sig[trig_sig < amp_threshold] = 0
    trig_sig[trig_sig > amp_threshold] = 1

    # Detecting the edges
    diff_trig = np.diff(trig_sig)

    locs = np.where(diff_trig == 1)[0]

    # Getting rise and fall times and indexes
    rise_idx = locs
    rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    # Filter out triggers that are less than 4 seconds apart
    filtered_rise_idx = [rise_idx[0]]  # Always keep the first one
    last_time = rise_times[0]

    for i in range(1, len(rise_times)):
        if rise_times[i] - last_time >= 4:
            filtered_rise_idx.append(rise_idx[i])
            last_time = rise_times[i]

    # Optionally, update rise_times too
    rise_idx = np.array(filtered_rise_idx)
    rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    # Sanity check
    if len(rise_idx) != ntrials:  # | (len(fall_idx) != Emg.ntrials):
        raise ValueError(f"Wrong number of trials: {len(rise_idx)}")

    return rise_times, rise_idx


def emg_segment(data, timestamp, prestim=None, poststim=None, fsample=None):
    """

    :param data:
    :param timestamp:
    :param prestim:
    :param poststim:
    :param fsample:
    :return:
    """
    muscle_names = data.columns
    n_muscles = len(muscle_names)
    ntrials = len(timestamp)
    timepoints = int(fsample * (prestim + poststim))

    emg_segmented = np.zeros((ntrials, n_muscles, timepoints))
    for tr, idx in enumerate(timestamp):
        for m, muscle in enumerate(muscle_names):
            emg_segmented[tr, m] = data[muscle][idx - int(prestim * fsample):
                                                idx + int(poststim * fsample)].to_numpy()

    return emg_segmented


def load_delsys(filepath, trigger_name=None, muscle_names=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param participant_id:
    :param experiment:
    :param block:
    :param muscle_names:
    :param trigger_name:
    :return:
    """
    # fname = f"{experiment}_{participant_id}_{block}.csv"
    # filepath = os.path.join(gl.make_dirs(experiment, "emg", participant_id), fname)

    # read data from .csv file (Delsys output)
    with open(filepath, 'rt') as fid:
        A = []
        for line in fid:
            # Strip whitespace and newline characters, then split
            split_line = [elem.strip() for elem in line.strip().split(',')]
            A.append(split_line)

    # identify columns with data from each muscle
    muscle_columns = {}
    for muscle in muscle_names:
        for c, col in enumerate(A[3]):
            if muscle in col:
                muscle_columns[muscle] = c + 1  # EMG is on the right of Timeseries data (that's why + 1)
                break
        for c, col in enumerate(A[5]):
            if muscle in col:
                muscle_columns[muscle] = c + 1
                break

    df_raw = pd.DataFrame(A[7:])  # get rid of header
    df_out = pd.DataFrame()  # init final dataframe

    for muscle in muscle_columns:
        df_out[muscle] = pd.to_numeric(df_raw[muscle_columns[muscle]],
                                       errors='coerce').replace('', np.nan).dropna()  # add EMG to dataframe

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    try:
        trigger = df_raw[trigger_column]
        trigger = resample(trigger.values, len(df_out))
    except IOError as e:
        raise IOError("Trigger not found") from e

    df_out[trigger_name] = trigger

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out


def main(args):
    if args.what=='segment_tms':

        muscle_names = [f'flx_D{i+1}' for i in range(5)] + [f'ext_D{i+1}' for i in range(5)] + ['FDI']

        filepath = os.path.join(gl.baseDir, args.experiment, 'emg',  f'Trial_1.csv')

        df = load_delsys(filepath, 'Trigger', muscle_names)
        trig_sig = df['Trigger'].to_numpy()
        trig_time = df['time'].to_numpy()

        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'{args.experiment}_{args.sn}.dat'), sep='\t')

        ntrials = dat.shape[0]

        idx, _ = detect_trig(trig_sig, trig_time, amp_threshold=2, ntrials=ntrials)
        df = df.drop(columns=['time', 'trigger'])

        df_segment = emg_segment(df, idx, .5, 1, 2148)

    if args.what=='make_bins':

        for sn in args.snS:
            emg = np.load(gl.baseDir, experiment, 'emg', f'subj{sn}', f'{experiment}_{sn}.npy')
            for win in args.wins:
                pass

    if args.what=='segment_emg':
        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        blocks = pinfo[pinfo.sn==args.sn].reset_index().blocks_emg[0].split('.')
        channels_emg = pinfo[pinfo.sn==args.sn].reset_index().channels_emg[0].split(',')
        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'subj{args.sn}',
                                       f'{args.experiment}_{args.sn}.dat'), sep='\t')

        emg = []
        for block in blocks:
            print(f'subj{args.sn} - block {block}')
            filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}',
                                                 f'{args.experiment}_{args.sn}_{block}.csv')
            dat_tmp = dat[dat.BN==int(block)]
            df_out = load_delsys(filepath, trigger_name='trigger', muscle_names=channels_emg)
            trig_sig = df_out.trigger.to_numpy()
            trig_time = df_out.time.to_numpy()
            ntrials = dat_tmp.shape[0]
            _, timestamp = detect_trig(trig_sig, trig_time, ntrials=ntrials, amp_threshold=2)

            emg.append(emg_segment(df_out.iloc[:, :-2], timestamp, prestim=1, poststim=2, fsample=2148))

        emg = np.vstack(emg)

        np.save(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}', 'emg_raw.npy'), emg)

    if args.what=='segment_emg_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='segment_emg',
                experiment=args.experiment,
                sn=sn,
            )
            main(args)


    if args.what=='save_participants_dict':

        channels = ['thumb_flex',
                    'index_flex',
                    'middle_flex',
                    'ring_flex',
                    'pinkie_flex',
                    'thumb_ext',
                    'index_ext',
                    'middle_ext',
                    'ring_ext',
                    'pinkie_ext',
                    'fdi']

        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')

        Dict = {ch: [] for ch in channels}
        for sn in args.snS:

            print(f'loading subj{sn}...')
            emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', f'emg_raw.npy'))
            emg_rect = np.abs(emg_raw)
            bs = emg_rect[..., :2148].mean(axis=-1, keepdims=True)
            emg = emg_rect / bs
            dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}',
                                           f'{args.experiment}_{sn}.dat'), sep='\t')
            blocks = [int(b) for b in pinfo[pinfo['sn'] == sn].blocks_emg.iloc[0].split('.')]
            dat = dat[dat.BN.isin(blocks)]
            ch_p = pinfo[pinfo['sn'] == sn].channels_emg.iloc[0].split(',')

            for ch in Dict.keys():
                if ch in ch_p:
                    idx = ch_p.index(ch)
                    emg_av = np.zeros((len(gl.cue_mapping.keys()), len(gl.stimFinger_mapping.keys()), emg.shape[-1]))
                    for sf, stimF in enumerate(list(gl.stimFinger_mapping.keys())):
                        for c, cue in enumerate(list(gl.cue_mapping.keys())):
                            emg_av[c, sf] = emg[(dat.cue == cue) & (dat.stimFinger == stimF), idx].mean(axis=0)

                    Dict[ch].append(emg_av)

        f = open(os.path.join(gl.baseDir, args.experiment, 'emg', 'emg.p'), 'wb')
        pickle.dump(Dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[100, 101, 102, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument('--experiment', type=str, default='smp0')

    args = parser.parse_args()

    main(args)

# def make_participants_dict(snS):
#
#     emg_dict = {
#         'thumb_flex': [],
#         'fdi': [],
#         'index_flex':[],
#         'middle_flex':[],
#         'ring_flex':[],
#         'pinkie_flex':[],
#         'thumb_ext':[],
#         'index_ext':[],
#         'middle_ext':[],
#         'ring_ext':[],
#         'pinkie_ext':[],
#         'sn': [],
#         'BN': [],
#         'TN': [],
#         'cue': [],
#         'stimFinger': []
#     }
#
#     channels = 'thumb_flex',
#         'fdi',
#         'index_flex',
#         'middle_flex',
#         'ring_flex',
#         'pinkie_flex':[],
#         'thumb_ext':[],
#         'index_ext':[],
#         'middle_ext':[],
#         'ring_ext':[],
#         'pinkie_ext':[],
#
#     emg_path = os.path.join(gl.baseDir, 'smp0', 'emg')
#     dat_path = os.path.join(gl.baseDir, 'smp0', 'behavioural')
#     for sn in snS:
#         emg = np.load(os.path.join(emg_path, f'subj{sn}', f'smp0_{sn}.npy'))
#         bins = pd.read_csv(os.path.join(emg_path, f'subj{sn}', f'smp0_{sn}_binned.tsv'), sep='\t')
#         dat = pd.read_csv(os.path.join(dat_path, f'subj{sn}', f'smp0_{sn}.dat'), sep='\t')
#         recorded_channels = bins.columns[1:-4]
#         for TN in range(dat.shape[0]):
#             emg_dict['TN'].append(dat.iloc[TN]['TN'])
#             emg_dict['cue'].append(dat.iloc[TN]['cue'])
#             emg_dict['stimFinger'].append(dat.iloc[TN]['stimFinger'])
#             emg_dict['sn'].append(sn)
#             emg_dict['BN'].append(dat.iloc[TN]['BN'])
#             for ch, channel in enumerate(channels):
#                 emg_tmp = emg[TN, ch]
#                 emg_dict[channel].append(emg_tmp)
#
#
#     pass
# def main(args):
#     if args.what=='make_participants_dict':
#         make_participants_dict(args.snS)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('what', nargs='?', default=None)
#     parser.add_argument('--snS', type=int, default=[100, 101, 102, 104, 105, 106])
#
#     args = parser.parse_args()
#
#     main(args)
