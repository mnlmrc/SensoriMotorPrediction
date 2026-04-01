import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

import numpy as np

baseDir = '/cifs/diedrichsen/data/SensoriMotorPrediction/'
if not os.path.exists(baseDir):
    baseDir="/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/"
if not os.path.exists(baseDir):
    baseDir="/Users/mnlmrc/Documents/data/SensoriMotorPrediction/"
if not os.path.exists(baseDir):
    print("No valid base directory found")
else:
    print("Base directory:", baseDir)

wbDir = "surfaceWB"
glmDir = "glm"
behavDir = "behavioural"
trainDir = "training"
imagingDir = "imaging_data"
rdmDir = "rdm"
surfDir = "surfaceWB"
roiDir = 'ROI'
cosDir = 'cosine'
pilotDir = 'pilot'
pcmDir = 'pcm'
nhpDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction'
spkDir = 'spikes'
lfpDir = 'LFPs'
recDir = 'Recordings'

nTR = 336
TR = 1


col_mov = {
    'smp0': ['trialNum', 'state', 'timeReal', 'time',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp2': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
}

channels = {'mov': ['thumb', 'index', 'middle', 'ring', 'pinkie']}

prestim = 1.5
poststim = 1

fsample_mov = 500
fsample_emg = 2148

planState = {
    'smp0': 2,
    'smp2': 3,
}

sns = [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]

cues = ['100-0%', '75-25%', '50-50%', '25-75%', '0-100%']
fingers = ['thumb', 'index', 'middle', 'ring', 'pinkie']
stimFinger = ['index', 'ring']

cue_code = [93, 12, 44, 21, 39]
stimFinger_code = [91999, 99919]

cue_mapping = {
                93: '0-100%',
                12: '25-75%',
                44: '50-50%',
                21: '75-25%',
                39: '100-0%'
            }
stimFinger_mapping = {91999: 'index',
                      99919: 'ring',
                      99999: 'nogo'}

regressor_mapping = {
    '100-0%': 0,
    '75-25%': 1,
    '50-50%': 2,
    '25-75%': 3,
    '0-100%': 4,
    '100-0%      ': 0,
    '75-25%      ': 1,
    '50-50%      ': 2,
    '25-75%      ': 3,
    '0-100%      ': 4,
    '100-0%,index': 5,
    '75-25%,index': 6,
    '50-50%,index': 7,
    '25-75%,index': 8,
    '75-25%,ring ': 9,
    '50-50%,ring ': 10,
    '25-75%,ring ': 11,
    '0-100%,ring ': 12,
    '100-0%,go   ': 13,
    '75-25%,go   ': 14,
    '50-50%,go   ': 15,
    '25-75%,go   ': 16,
    '0-100%,go   ': 17,
    '100-0%   ': 0,
    '75-25%   ': 1,
    '50-50%   ': 2,
    '25-75%   ': 3,
    '0-100%   ': 4,
    #'exec,index': 18,
    'exec,ring': 19
}

freqs = ['delta', 'theta', 'alpha-beta', 'gamma']
recordings = {
            'Malfoy': {
                'PFC': [17, 19, 20, 21, 22, 23, 24],
                'PMd': [10, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24],
                'M1': [12, 13, 25, 27, 28],
                'S1': [ 5, 9, 11, 15, 16, 26, 27, 28]
            },
            'Pert': {
                'PFC': [8, 9, 10, 11, 12],
                'PMd': [4, 6, 7, 10, 20],
                'M1': [2, 3, 14, 20],
                'S1': [15],

            }
        }

cuePre = 0
cueIdx = 20
cuePost = 84
pertPre = cuePost
pertIdx = pertPre + 30
pertPost = pertPre + 70

### colours ###
cmap_plan = plt.get_cmap('Greys')
col_plan = [cmap_plan(i) for i in np.linspace(.3, .9, 5)]

cmap_index = plt.get_cmap('Greens')
col_index = [cmap_index(i) for i in np.linspace(.3, .9, 5)][:4]

cmap_ring = plt.get_cmap('Oranges')
col_ring = [cmap_ring(i) for i in np.linspace(.3, .9, 5)][1:]

colour_mapping = {
    '100-0%': '#EC008C', #col_plan[0],
    '75-25%': '#BD60A5', #col_plan[1],
    '50-50%': '#8781BD', #col_plan[2],
    '25-75%': '#1D9AD6', #col_plan[3],
    '0-100%': '#00AEEF', #col_plan[4],
    '100-0%,index': '#EC008C', #col_index[0],
    '75-25%,index': '#F067A6', #col_index[1],
    '50-50%,index': '#F49AC1', #col_index[2],
    '25-75%,index': '#F9CBDF', #col_index[3],
    '75-25%,ring': '#B9E4FA', #col_ring[0],
    '50-50%,ring': '#6DCFF6', #col_ring[1],
    '25-75%,ring': '#00C2F3', #col_ring[2],
    '0-100%,ring': '#00AEEF', #col_ring[3],
}

###############

reg_interest = {
    'exec': [5, 6, 7, 8, 9, 10, 11, 12],
    'plan': [0, 1, 2, 3, 4]
}

# make rdm masks for cue vs stimFinger effect (plus interaction)
mask_stimFinger = np.zeros([28], dtype=bool)
mask_cue = np.zeros([28], dtype=bool)
mask_stimFinger_cue = np.zeros([28], dtype=bool)
mask_stimFinger[[4, 11, 17]] = True
mask_cue[[0, 1, 7, 25, 26, 27]] = True
mask_stimFinger_cue[[5, 6, 10, 12, 15, 16]] = True

borders = {'L': os.path.join(baseDir, 'smp2', surfDir, 'fs_LR.32k.L.border'),
           'R': os.path.join(baseDir, 'smp2', surfDir, 'fs_LR.32k.R.border')}

atlasDir = 'atlases' #next((Dir for Dir in atlasDirs if Path(Dir).exists()), None)
print(f'Atlases dir found: {atlasDir}')

struct = ['CortexLeft', 'CortexRight']
Hem = ['L', 'R']
rois = {
        'Desikan': [
            'rostralmiddlefrontal',
            'caudalmiddlefrontal',
            'precentral',
            'postcentral',
            'superiorparietal',
            'pericalcarine'
        ],
        'BA_handArea': [
            'ba4a', 'ba4p', 'ba3A', 'ba3B', 'ba1', 'ba2'
        ],
        'ROI': [
            'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp',
        ]
    }

