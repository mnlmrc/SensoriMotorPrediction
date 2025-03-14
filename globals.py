import os
from pathlib import Path
import matplotlib.pyplot as plt

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
surfDir = "surfaceWB"
roiDir = 'ROI'
cosDir = 'cosine'
pilotDir = 'pilot'
pcmDir = 'pcm'

print("Base directory:", baseDir)

col_mov = {
    'smp0': ['trialNum', 'state', 'timeReal', 'time',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp1': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp2': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
}

# participants = {
#     'smp0': ['subj100',
#              'subj101',
#              'subj102',
#              'subj103',
#              'subj104',
#              'subj105',
#              'subj106',
#              'subj107',
#              'subj108',
#              'subj109',
#              'subj110'],
#     'smp1': ['subj100',
#              'subj101',
#              'subj102',
#              'subj103',
#              'subj104',
#              # 'subj105',
#              # 'subj106'
#              ],
#     'smp2': ['subj100',
#              'subj101',
#              # 'subj102',
#              # 'subj103',
#              # 'subj104'
#              ]
# }

channels = {'mov': ['thumb', 'index', 'middle', 'ring', 'pinkie']}

prestim = 1
poststim = 1

fsample_mov = 500
# TR = 1
# N = {
#     'smp0': len(participants['smp0']),
#     'smp1': len(participants['smp1']),
#     'smp2': len(participants['smp2']),
# }
planState = {
    'smp0': 2,
    'smp1': 3,
    'smp2': 3
}

cues = ['0%', '25%', '50%', '75%', '100%']
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

regressor_mapping = {
    '0%': 0,
    '25%': 1,
    '50%': 2,
    '75%': 3,
    '100%': 4,
    '25%,index': 5,
    '50%,index': 6,
    '75%,index': 7,
    '100%,index': 8,
    '0%,ring': 9,
    '25%,ring': 10,
    '50%,ring': 11,
    '75%,ring': 12,
}

### colours ###
cmap = plt.get_cmap("coolwarm")
colors = [cmap(i) for i in np.linspace(0, 1, 5)]

colour_mapping = {
    '0%': colors[0],
    '25%': colors[1],
    '50%': colors[2],
    '75%': colors[3],
    '100%': colors[4],
    '25%,index': colors[1],
    '50%,index': colors[2],
    '75%,index': colors[3],
    '100%,index': colors[4],
    '0%,ring': colors[0],
    '25%,ring': colors[1],
    '50%,ring': colors[2],
    '75%,ring': colors[3],
}

###############

# make rdm masks for cue vs stimFinger effect (plus interaction)
mask_stimFinger = np.zeros([28], dtype=bool)
mask_cue = np.zeros([28], dtype=bool)
mask_stimFinger_cue = np.zeros([28], dtype=bool)
mask_stimFinger[[4, 11, 17]] = True
mask_cue[[0, 1, 7, 25, 26, 27]] = True
mask_stimFinger_cue[[5, 6, 10, 12, 15, 16]] = True

# flatmap stuff
# borderDirs = ["/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/",
#         "/home/ROBARTS/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/",]
#
# borderDir = next((Dir for Dir in borderDirs if Path(Dir).exists()), None)

borders = {'L': os.path.join(baseDir, 'smp2', surfDir, 'fs_LR.32k.L.border'),
           'R': os.path.join(baseDir, 'smp2', surfDir, 'fs_LR.32k.R.border')}

atlas_dir = ["/home/ROBARTS/memanue5/Documents/GitHub/Functional_Fusion/Functional_Fusion/Atlases/tpl-fs32k/",
             "/Users/mnlmrc/Documents/GitHub/Functional_Fusion/Functional_Fusion/Atlases/tpl-fs32k/"]

atlas_dir = next((Dir for Dir in atlas_dir if Path(Dir).exists()), None)

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
