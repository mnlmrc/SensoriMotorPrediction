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


def make_Z_all(experiment='smp2', sn=None, glm=None):

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    # Extract percentage and finger information from the "name" column
    reginfo['percentage'] = reginfo['name'].str.extract(r'(\d+%)')[0]
    reginfo['finger'] = reginfo['name'].str.extract(r',(index|ring)')[0].fillna('nogo')

    # Define unique percentages and fingers for one-hot encoding
    unique_percentages = ['0%', '25%', '50%', '75%', '100%']
    unique_fingers = ['index', 'ring', 'nogo']

    # Initialize the design matrix
    Z = np.zeros((len(reginfo), len(unique_percentages) + len(unique_fingers)), dtype=int)

    # Fill in the design matrix
    for i, row in reginfo.iterrows():
        # Percentage columns
        percentage_idx = unique_percentages.index(row['percentage'])
        Z[i, percentage_idx] = 1

        # Finger columns
        finger_idx = unique_fingers.index(row['finger']) + len(unique_percentages)
        Z[i, finger_idx] = 1

    return Z


def make_Z_cue(experiment='smp2', sn=None, glm=None):

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    # Extract percentage and finger information from the "name" column
    reginfo['percentage'] = reginfo['name'].str.extract(r'(\d+%)')[0]
    reginfo['finger'] = reginfo['name'].str.extract(r',(index|ring)')[0].fillna('nogo')

    # Define unique percentages and fingers for one-hot encoding
    unique_percentages = ['0%', '25%', '50%', '75%', '100%']

    # Initialize the design matrix
    Z = np.zeros((len(reginfo), len(unique_percentages)), dtype=int)

    # Fill in the design matrix
    for i, row in reginfo.iterrows():
        # Percentage columns
        percentage_idx = unique_percentages.index(row['percentage'])
        Z[i, percentage_idx] = 1

    return Z


def FixedModel(name, Z):
    G = np.matmul(Z, Z.T)
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
