import argparse

import pandas as pd
import numpy as np
import os
import globals as gl
from scipy.signal import resample


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

    # Debugging plots
    if debugging:
        # Printing the number of triggers detected and number of trials
        print("\nNum Trigs Detected = {}".format(len(locs)))
        print("Num Trials in Run = {}".format(ntrials))
        print("====NumTrial should be equal to NumTrigs====\n\n\n")

        # plotting block
        plt.figure()
        plt.plot(trig_sig, 'k', linewidth=1.5)
        plt.plot(diff_trig, '--r', linewidth=1)
        plt.scatter(locs, diff_trig[locs], color='red', marker='o', s=30)
        plt.xlabel("Time (index)")
        plt.ylabel("Trigger Signal (black), Diff Trigger (red dashed), Detected triggers (red/blue points)")
        plt.ylim([-1.5, 1.5])
        plt.show()

    # Getting rise and fall times and indexes
    rise_idx = locs
    rise_times = time_trig[rise_idx]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', type=int, default=None)
    parser.add_argument('--experiment', type=str, default='smp3')

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
