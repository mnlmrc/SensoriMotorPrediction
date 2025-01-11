import os.path

import PcmPy as pcm
import scipy

import globals as gl
import pandas as pd
import numpy as np
import os
import nibabel as nb
import nitools as nt


import matplotlib.pyplot as plt


def make_Z_all(experiment='smp2', sn=None):

    participant_id = f'subj{sn}'

    # Load the .dat file
    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id, f'{experiment}_{sn}.dat'), sep='\t')

    # Define unique cues and fingers
    unique_cues = [93, 12, 44, 21, 39]
    unique_fingers = [91999, 99919, 99999]

    # Initialize the design matrix
    Z = np.zeros((len(dat), len(unique_cues) + len(unique_fingers)), dtype=int)

    # Fill in the design matrix
    for i, row in dat.iterrows():
        # Cue columns
        if row['cue'] in unique_cues:
            cue_index = unique_cues.index(row['cue'])
            Z[i, cue_index] = 1

        # Finger columns
        if row['stimFinger'] in unique_fingers:
            finger_index = unique_fingers.index(row['stimFinger']) + len(unique_cues)
            Z[i, finger_index] = 1

    return Z

def make_Z_cue(experiment='smp2', sn=None):

    participant_id = f'subj{sn}'

    # Load the .dat file
    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, participant_id, f'{experiment}_{sn}.dat'), sep='\t')

    # Define unique cues and fingers
    unique_cues = [93, 12, 44, 21, 39]

    # Initialize the design matrix
    Z = np.zeros((len(dat), len(unique_cues)), dtype=int)

    # Fill in the design matrix
    for i, row in dat.iterrows():
        # Cue columns
        if row['cue'] in unique_cues:
            cue_index = unique_cues.index(row['cue'])
            Z[i, cue_index] = 1

    return Z


def FixedModel(name, Z):

    G = np.matmul(Z.T, Z)
    M = pcm.model.FixedModel(name, G)

    return M, G

# def fitting(Y, M, experiment='smp2', sn=102):
#
#
#
#     return T, theta

# sn = 102
# experiment = 'smp2'
#
# Z_all = make_Z_all('smp2', sn)
# Z_cue = make_Z_cue('smp2', sn)
# M_all, G_all = FixedModel('all', Z_all)
# M_cue, G_cue = FixedModel('cue', Z_cue)
#
# mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}',
#                                                 f'subj{sn}_ROI_region.mat'))
# R_cell = mat['R'][0]
# R = list()
# for r in R_cell:
#     R.append({field: r[field].item() for field in r.dtype.names})
#
# # find roi where to calc RDM
# R = R[[True if (r['name'].size > 0) and (r['name'] == 'M1') and (r['hem'] == 'L')
#        else False for r in R].index(True)]
#
# reginfo = pd.read_csv(os.path.join(gl.baseDir, 'smp2', gl.glmDir + '12', f'subj{sn}', f'subj{sn}_reginfo.tsv'),
#                       sep='\t')
#
# betas = list()
# for n_regr in np.arange(0, reginfo.shape[0]):
#
#     print(f'loading regressor #{n_regr + 1}')
#
#     vol = nb.load(os.path.join(gl.baseDir, 'smp2', gl.glmDir + '12', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
#     beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
#     betas.append(beta)

