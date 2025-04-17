import argparse

import pandas as pd
import numpy as np
import os
import globals as gl

def load_delsys(experiment=None, participant_id=None, block=None, muscle_names=None, trigger_name=None):
    """returns a pandas DataFrame with the raw EMG data recorded using the Delsys system

    :param participant_id:
    :param experiment:
    :param block:
    :param muscle_names:
    :param trigger_name:
    :return:
    """
    fname = f"{experiment}_{participant_id}_{block}.csv"
    filepath = os.path.join(gl.make_dirs(experiment, "emg", participant_id), fname)

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
