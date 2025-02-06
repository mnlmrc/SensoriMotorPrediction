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


def tessellation(atlas='Icosahedron-1002'):

    matlab_cmd = f"cd('~/Documents/GitHub/sensori-motor-prediction/smp2/'); smp2_anat('TESSELLATION:single_tessel', 'atlas', '{atlas}'); exit"

    subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_cmd])


if __name__ == '__main__':
    tessellation(atlas='Icosahedron-1002')
