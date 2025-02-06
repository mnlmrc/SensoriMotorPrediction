import argparse
import os.path

import PcmPy as pcm
import scipy

import globals as gl
import pandas as pd
import numpy as np
import os
import subprocess
import nibabel as nb
import nitools as nt

import matplotlib.pyplot as plt

from betas import get_roi


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


def find_matlab_function(function_name):
    """Search for the MATLAB function file in known locations"""
    try:
        # Run the 'find' command and capture output
        result = subprocess.run(["find", "/", "-name", f"{function_name}.m"], stdout=subprocess.PIPE, text=True,
                                stderr=subprocess.DEVNULL)

        # Extract paths
        paths = result.stdout.strip().split("\n")

        # Check if paths exist and return the first valid directory
        for path in paths:
            if os.path.isfile(path):
                return os.path.dirname(path)

    except Exception as e:
        print(f"Error finding MATLAB function: {e}")

    return None


def get_tessel_betas(experiment=None, sn=None, atlas=None, Hem=None, idx=None, glm=None):
    R = get_roi(experiment, sn, Hem, f'label-{idx}', atlas=atlas)

    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    betas = list()
    for n_regr in np.arange(0, reginfo.shape[0]):
        vol = nb.load(
            os.path.join(gl.baseDir, 'smp2', f'{gl.glmDir}{glm}', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
        beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        betas.append(beta)

    betas = np.array(betas)
    betas = betas[:, ~np.all(np.isnan(betas), axis=0)]

    assert betas.ndim == 2

    return betas


# def tessellation(atlas='Icosahedron-1002'):
#
#     # matlab_cmd = (f"cd('~/Documents/GitHub/sensori-motor-prediction/smp2/'); "
#     #               f"smp2_anat('TESSELLATION:single_tessel', 'sn', 106, 'atlas', '{atlas}'); exit")
#     #
#     # subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_cmd])


if __name__ == '__main__':

    Z_stimFinger = np.zeros((13, 3))
    Z_stimFinger[0:5, 0] = 1
    Z_stimFinger[5:9, 1] = 1
    Z_stimFinger[9:13, 2] = 1

    Z_cue = np.zeros((13, 5))
    Z_cue[[0, 9], 0] = 1
    Z_cue[[1, 5, 10], 1] = 1
    Z_cue[[2, 6, 11], 2] = 1
    Z_cue[[3, 7, 12], 3] = 1
    Z_cue[[4, 8], 4] = 1

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='Icosahedron-1002')
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=None)

    args = parser.parse_args()

    if args.what == '_get_tessel_betas':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
        for H in Hem:
            for idx in range(924):
                print(f'Hemisphere: {H}, tessel:{idx + 1}')
                betas = get_tessel_betas(
                    experiment=args.experiment,
                    sn=args.sn,
                    Hem=H,
                    idx=idx + 1,
                    atlas=args.atlas,
                    glm=args.glm,
                )
