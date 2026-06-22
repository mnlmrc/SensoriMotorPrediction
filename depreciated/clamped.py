import os.path

import numpy as np
import pandas as pd

from utils import detect_response_latency

import globals as gl

path = os.path.join(gl.baseDir, 'smp0', 'clamped')

clamped = np.load(os.path.join(path, 'mov', 'smp0_clamped.npy'))
dat = pd.read_csv(os.path.join(path, 'smp0_clamped.dat'), sep='\t')

stimFinger = dat.stimFinger
n_stimF = dat.stimFinger.unique()
idx_f = [1, 3]

fsample = 500
prestim = 1
threshold = .03

latency = list()
for sf, stimF in enumerate(dat.stimFinger.unique()):
    c_mean = clamped[stimFinger == stimF, idx_f[sf]].mean(axis=0)
    latency.append(detect_response_latency(c_mean, threshold=threshold,
                                           fsample=fsample) - prestim)

latency = pd.DataFrame({
    'index': latency[0],
    'ring': latency[1]
}, index=[0])
latency.to_csv(os.path.join(path, 'smp0_clamped_latency.tsv'), sep='\t')
