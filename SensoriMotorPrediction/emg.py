import argparse

import pandas as pd
import numpy as np
import os
import globals as gl
from scipy.signal import resample
import PcmPy as pcm
import pickle
from util import lp_filter
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def detect_trig(trig_sig, time_trig, amp_threshold=None, edge='rising', min_duration=.05):
    """
    Detect trigger onsets with optional minimum duration filtering.

    :param trig_sig: signal array
    :param time_trig: time vector (same length as trig_sig)
    :param amp_threshold: threshold to binarize signal
    :param debugging: if True, plots signal and detected triggers
    :param edge: 'rising' or 'falling'
    :param min_duration: minimum duration (in seconds) that the signal must remain above/below threshold
    :return: rise_times, rise_idx
    """

    if edge == 'rising':
        trig_bin = (trig_sig > amp_threshold).astype(int)
    elif edge == 'falling':
        trig_bin = (trig_sig < amp_threshold).astype(int)
    else:
        raise ValueError('edge must be either "rising" or "falling"')

    diff_trig = np.diff(trig_bin)
    rise_idx = np.where(diff_trig == 1)[0]
    fall_idx = np.where(diff_trig == -1)[0]

    # Make sure each rising edge has a corresponding falling edge
    if fall_idx.size == 0 or rise_idx.size == 0:
        return np.array([]), np.array([])

    # Ensure proper ordering
    if fall_idx[0] < rise_idx[0]:
        fall_idx = fall_idx[1:]
    if rise_idx[-1] > fall_idx[-1]:
        rise_idx = rise_idx[:-1]

    valid_rise_idx = []
    for r_idx, f_idx in zip(rise_idx, fall_idx):
        duration = time_trig[f_idx] - time_trig[r_idx]
        if duration >= min_duration:
            valid_rise_idx.append(r_idx)

    rise_idx = np.array(valid_rise_idx)
    rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

    # Remove triggers that are <4s apart
    if len(rise_idx) > 0:
        filtered_idx = [rise_idx[0]]
        last_time = rise_times[0]
        for i in range(1, len(rise_idx)):
            if rise_times[i] - last_time >= 4:
                filtered_idx.append(rise_idx[i])
                last_time = time_trig[rise_idx[i]]
        rise_idx = np.array(filtered_idx)
        rise_times = np.array([float(time_trig[idx]) for idx in rise_idx])

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

    df_raw = pd.DataFrame(A[8:])  # get rid of header
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


def align_emg(experiment='smp0', sn=sn):

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
    blocks = pinfo[pinfo.sn==args.sn].reset_index().blocks_emg_task[0]
    if type(blocks) is str:
        blocks = blocks.split(',')
    else:
        blocks = [blocks]
    channels_emg = pinfo[pinfo.sn==args.sn].reset_index().channels_emg[0].split(',')
    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, 'behavioural', f'subj{sn}', f'{experiment}_{sn}.dat'), sep='\t')

    emg = []
    for block in blocks:
        print(f'subj{args.sn} - block {block}')
        filepath = os.path.join(gl.baseDir, experiment, 'emg', f'subj{sn}', f'block_{block}.csv')
        dat_tmp = dat[dat.BN==int(block)]
        df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)
        trig_sig = df_out.Trigger.to_numpy()
        trig_time = df_out.time.astype(float).to_numpy()
        _, timestamp = detect_trig(trig_sig, trig_time, amp_threshold=args.thresh)
        emg.append(emg_segment(df_out.iloc[:, :-2], timestamp, prestim=0, poststim=4, fsample=2148))

    emg = np.vstack(emg)

    np.save(os.path.join(gl.baseDir, experiment, 'emg', f'subj{sn}', 'emg_raw.npy'), emg)


def main(args):

    if args.what=='segment_emg':
        pinfo = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'participants.tsv'), sep='\t')
        blocks = pinfo[pinfo.sn==args.sn].reset_index().blocks_emg_task[0]
        if type(blocks) is str:
            blocks = blocks.split(',')
        else:
            blocks = [blocks]
        channels_emg = pinfo[pinfo.sn==args.sn].reset_index().channels_emg[0].split(',')
        dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, 'behavioural', f'subj{args.sn}',
                                       f'{args.experiment}_{args.sn}.dat'), sep='\t')

        emg = []
        for block in blocks:
            print(f'subj{args.sn} - block {block}')
            filepath = os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}', f'block_{block}.csv')
            dat_tmp = dat[dat.BN==int(block)]
            df_out = load_delsys(filepath, trigger_name='Trigger', muscle_names=channels_emg)
            trig_sig = df_out.Trigger.to_numpy()
            trig_time = df_out.time.astype(float).to_numpy()
            # ntrials = dat_tmp.shape[0]
            _, timestamp = detect_trig(trig_sig, trig_time, amp_threshold=args.thresh)

            # for t, _ in enumerate(timestamp):
            #     stimFinger = dat_tmp.iloc[t]['stimFinger']
            #     if stimFinger==91999:
            #         timestamp[t] += int(.042 * 2148) # .046
            #     elif stimFinger==99919:
            #         timestamp[t] += int(.05 * 2148) # .06

            emg.append(emg_segment(df_out.iloc[:, :-2], timestamp, prestim=0, poststim=4, fsample=2148))

        emg = np.vstack(emg)

        np.save(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}', 'emg_raw.behav.task.npy'), emg)
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
            emg = emg_rect - bs
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
    if args.what == 'pca':
        print(f'loading subj{args.sn}...')
        emg_raw = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}', f'emg_raw.npy'))
        emg_rect = np.abs(emg_raw)
        bs = emg_rect[..., :2148].mean(axis=-1, keepdims=True)
        emg = emg_rect #/ bs
        shape = emg.shape
        emg_stacked = np.transpose(emg, (0, 2, 1)).reshape(-1, emg.shape[1])

        pca = PCA(n_components=3)
        # nmf = NMF(n_components=3, max_iter=1000)

        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        emg_stacked = scaler.fit_transform(emg_stacked)
        PCs = pca.fit_transform(emg_stacked)
        # PCs =  nmf.fit_transform(emg_stacked)
        PCs = PCs.reshape(emg.shape[0], emg.shape[-1], -1)  # (200, 6444, n_components)
        PCs = np.transpose(PCs, (0, 2, 1))
        np.save(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{args.sn}', 'pcs.npy'), PCs)
    if args.what=='pca_all':
        for sn in args.sns:
            args = argparse.Namespace(
                what='pca',
                experiment=args.experiment,
                sn=sn,
            )
            main(args)
    if args.what=='trajectories':
        pcs = []
        for sn in args.sns:
            print(f'doing participant {sn}...')
            dat = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{sn}', f'{args.experiment}_{sn}.dat'),
                              sep='\t')
            cue = dat.cue.map(gl.cue_mapping)
            stimFinger = dat.stimFinger.map(gl.stimFinger_mapping)
            cond_vec = cue + ',' + stimFinger
            cond_vec = cond_vec.map(gl.regressor_mapping).to_numpy()
            part_vec = np.ones_like(cond_vec)
            pcs_tmp = np.load(os.path.join(gl.baseDir, args.experiment, 'emg', f'subj{sn}', 'pcs.npy'))
            pcs_tmp = lp_filter(pcs_tmp, 10, gl.fsample_emg, axis=-1)
            pcs_tmp, _, _ = pcm.group_by_condition(pcs_tmp, cond_vec, part_vec, axis=0)
            pcs.append(pcs_tmp)
        #pcs = np.array(pcs).mean(axis=0)
        np.save(os.path.join(gl.baseDir, args.experiment, 'emg', 'trajectories.npy'), pcs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', type=int, default=[100, 101, 102, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument('--experiment', type=str, default='smp0')
    parser.add_argument('--datatype', type=str, default='task')
    parser.add_argument('--edge', type=str, default='rising')
    parser.add_argument('--thresh', type=float, default=2)

    args = parser.parse_args()

    main(args)

