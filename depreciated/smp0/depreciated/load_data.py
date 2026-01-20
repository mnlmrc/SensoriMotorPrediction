# Define a function to read the trial data from a file
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt


def load_participants(experiment):
    fname = path + experiment + '/' + 'participants.tsv'
    fid = open(fname, 'rt')
    participants = pd.read_csv(fid, delimiter='\t', engine='python')

    return participants


def count_blocks(experiment, participant_id, folder='mov', extension='.mov'):
    """Count the number of files with a given extension in a directory."""

    directory = path + experiment + '/subj' + participant_id + '/' + folder + '/'

    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            count += 1
    return count


def load_mov(experiment, participant_id, block):
    try:
        int(participant_id)
        fname = path + experiment + '/subj' + participant_id + '/mov/' + experiment + '_' + participant_id + '_' + block + '.mov'
    except:
        fname = path + experiment + '/' + participant_id + '/mov/' + experiment + '_' + participant_id + '_' + block + '.mov'

    try:
        with open(fname, 'rt') as fid:
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
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            rawForce = [np.array(trial_data)[:, 4:9] for trial_data in A]
            vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            time = [np.array(trial_data)[:, 1:4] for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {fname}") from e

    return rawForce, vizForce, time


def load_dat(experiment, participant_id):
    try:
        int(participant_id)
        fname = path + experiment + '/subj' + participant_id + '/' + experiment + '_' + participant_id + '.dat'
    except:
        fname = path + experiment + '/' + participant_id + '/' + experiment + '_' + participant_id + '.dat'

    try:
        fid = open(fname, 'rt')
        dat = pd.read_csv(fid, delimiter='\t', engine='python')
    except IOError as e:
        raise IOError(f"Could not open {fname}") from e

    # dat = pd.read_csv(path + experiment + '/subj' + participant_id + '/' + experiment + '_' + participant_id + '.dat', delimiter='\t')

    return dat


def hp_filter(data, fs, cutoff_freq=30, nOrd=4):

    numtaps = int(nOrd * fs / cutoff_freq)
    b = firwin(numtaps + 1, cutoff_freq, fs=fs, pass_zero='highpass')
    filtered_data = filtfilt(b, 1, data)

    return filtered_data


def load_emg(fname, muscle_names=None, fsample=None, trigger_name="trigger"):
    # fname = path + experiment + '/subj' + participant_id + '/emg/' + experiment + '_' + participant_id + '_' + str(
    #     block) + '.csv'

    # read data from .csv file (Delsys output)
    with open(fname, 'rt') as fid:
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
        df_out[muscle] = df_raw[muscle_columns[muscle]]  # add EMG to dataframe

    # High-pass filter and rectify EMG
    for col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')  # convert to floats
        df_out[col] = hp_filter(df_out[col], fsample)
        df_out[col] = df_out[col].abs()  # Rectify

    # add trigger column
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    df_out[trigger_name] = df_raw[trigger_column]

    # add time column
    df_out['time'] = df_raw.loc[:, 0]

    return df_out



# Replace 'your_data_file.txt' with the path to your data file
path = '/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/'  # replace with data path
# path = '/Volumes/Diedrichsen_data$/data/SensoriMotorPrediction/'
