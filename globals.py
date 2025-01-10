import os
from pathlib import Path

import numpy as np

Dirs = ["/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/",
        "/cifs/diedrichsen/data/SensoriMotorPrediction/",
        "/Users/mnlmrc/Library/CloudStorage/GoogleDrive-mnlmrc@unife.it/My Drive/UWO/SensoriMotorPrediction/",
        "/Users/mnlmrc/Documents/data/SensoriMotorPrediction/"]

# Find the first existing directory
baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)

if baseDir:
    print(f"Base directory found: {baseDir}")
else:
    print("No valid base directory found.")

wbDir = "surfaceWB"
glmDir = "glm"
behavDir = "behavioural"
trainDir = "training"
imagingDir = "imaging_data"
rdmDir = "rdm"
roiDir = 'ROI'
pilotDir = 'pilot'

print("Base directory:", baseDir)

col_mov = {
    'smp0': ['trialNum', 'state', 'timeReal', 'time',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp1': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp2': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
}

participants = {
    'smp0': ['subj100',
             'subj101',
             'subj102',
             'subj103',
             'subj104',
             'subj105',
             'subj106',
             'subj107',
             'subj108',
             'subj109',
             'subj110'],
    'smp1': ['subj100',
             'subj101',
             'subj102',
             'subj103',
             'subj104',
             # 'subj105',
             # 'subj106'
             ],
    'smp2': ['subj100',
             'subj101',
             # 'subj102',
             # 'subj103',
             # 'subj104'
             ]
}

clabels = ['0%', '25%', '50%', '75%', '100%']

channels = {'mov': ['thumb', 'index', 'middle', 'ring', 'pinkie']}

prestim = 1
poststim = 1
fsample_mov = 500
TR = 1
N = {
    'smp0': len(participants['smp0']),
    'smp1': len(participants['smp1']),
    'smp2': len(participants['smp2']),
}
planState = {
    'smp0': 2,
    'smp1': 3,
    'smp2': 3
}

cue = ['0%', '25%', '50%', '75%', '100%']
stimFinger = ['index', 'ring']

cue_code = [93, 12, 44, 21, 39]
stimFinger_code = [91999, 99919]

cue_mapping = {
                93: '0%',
                12: '25%',
                44: '50%',
                21: '75%',
                39: '100%'
            }
stimFinger_mapping = {91999: 'index',
                      99919: 'ring',
                      99999: 'nogo'}

# make rdm masks for cue vs stimFinger effect (plus interaction)
mask_stimFinger = np.zeros([28], dtype=bool)
mask_cue = np.zeros([28], dtype=bool)
mask_stimFinger_cue = np.zeros([28], dtype=bool)
mask_stimFinger[[4, 11, 17]] = True
mask_cue[[0, 1, 7, 25, 26, 27]] = True
mask_stimFinger_cue[[5, 6, 10, 12, 15, 16]] = True

# flatmap stuff
borders = {'L': '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.border',
           'R': '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.border'}

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
            'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'
        ]
    }

rdm_index = {
    'glm10': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
    'glm11': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
    'glm12': [0, 4, 7, 10, 2, 5, 8, 11, 3, 1, 6, 9, 12],
}
