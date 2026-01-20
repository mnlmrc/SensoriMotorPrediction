import os

import pandas as pd

from smp0.depreciated.depreciated import path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('MacOSX')

experiment = 'smp0'
participant_id = '102'
block = 1

muscle_names = ['thumb_flex', 'index_flex', 'middle_flex', 'ring_flex', 'pinkie_flex', 'thumb_ext',
                    'index_ext', 'middle_ext', 'ring_ext', 'pinkie_ext']  # approx recorded muscles

trigger_name = "trigger"

fname = f"{experiment}_{participant_id}_{block}.csv"
filepath = os.path.join(path, experiment, f"subj{participant_id}", 'emg', fname)

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
    df_out[muscle] = df_raw[muscle_columns[muscle]]  # add EMG to dataframe

    # High-pass filter and rectify EMG
for col in df_out.columns:
    df_out[col] = pd.to_numeric(df_out[col], errors='coerce')  # convert to floats
    df_out[col] = df_out[col].abs()  # Rectify

    # add trigger column
trigger_column = None
for c, col in enumerate(A[3]):
    if trigger_name in col:
        trigger_column = c + 1

trigger = pd.to_numeric(df_raw[trigger_column], errors='coerce').values

time = pd.to_numeric(df_raw.loc[:, 1], errors='coerce').dropna().values
timeTrigger = pd.to_numeric(df_raw[trigger_column - 1], errors='coerce').values

# plt.rcParams['path.simplify'] = True
#
# # Adjust the simplification threshold (the default is 1/9)
# # Lower values mean more simplification
# plt.rcParams['path.simplify_threshold'] = 0.05
#
#
# time = resample(time, 48000)
sig = pd.to_numeric(df_raw[2], errors='coerce').dropna().values
plt.plot(time, sig)
plt.plot(timeTrigger, trigger)
