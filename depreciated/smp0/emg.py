import os

import numpy as np
import pandas as pd
import smp0.globals as gl
from matplotlib import pyplot as plt
from scipy.signal import resample

from smp0.utils import hp_filter


def emg_hp_filter(data, n_ord=None, cutoff=None, fsample=None, muscle_names=None):
    """

    :param data:
    :param n_ord:
    :param cutoff:
    :param fsample:
    :param muscle_names:
    :return:
    """
    data_filtered = pd.DataFrame()
    for col in muscle_names:
        data_filtered[col] = hp_filter(data[col], n_ord=n_ord, cutoff=cutoff, fsample=fsample)

    return data_filtered


def emg_rectify(data, muscle_names=None):
    """

    :param data:
    :param muscle_names:
    :return:
    """
    data_rectified = pd.DataFrame()
    for col in muscle_names:
        data_rectified[col] = data[col].abs()  # Rectify

    return data_rectified


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

    trig_sig = pd.to_numeric(trig_sig).to_numpy()
    time_trig = pd.to_numeric(time_trig).to_numpy()

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
