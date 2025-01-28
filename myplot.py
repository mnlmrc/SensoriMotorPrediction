import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors as mcolors

import globals as gl


def make_colors(n_labels, ecol=('blue', 'red')):
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{ecol[0]}_to_{ecol[1]}",
                                                     [ecol[0], ecol[1]], N=100)
    norm = plt.Normalize(0, n_labels)
    colors = [cmap(norm(lab)) for lab in range(n_labels)]

    return colors


def get_clamp_lat():
    """
    Just get the latency of push initiation on the ring and index finger
    Returns:
        latency (tuple): latency_index, latency_ring

    """
    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    return latency


def make_tAx(data, latency=None):
    """
    Just make the time axis of any time plot aligned to the time of perturbation, taking into account the latency of
    the push initiation on the ring and index finger

    Args:
        data: a numpy array of the data that need to be plotted. Last dimension must be time

    Returns:
        numpy.ndarray (data.shape[-1)

    """
    if latency is None:
        latency = get_clamp_lat()

    tAx = (np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[0],
           np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[1])

    return tAx


def plot_bins(df):
    pass


def make_yref(axs, reference_length=5, pos='left'):
    midpoint_y = (axs.get_ylim()[0] + axs.get_ylim()[1]) / 5  # Calculate the one-third of the y-axis

    if pos == 'left':
        reference_x = axs.get_xlim()[0]
        axs.plot([reference_x, reference_x],
                 [midpoint_y - reference_length / 2, midpoint_y + reference_length / 2],
                 ls='-', color='k', lw=3, zorder=100)
        axs.text(reference_x, midpoint_y, f'{reference_length}N ', color='k',
                 ha='right', va='center', zorder=100)
    elif pos == 'right':
        reference_x = axs.get_xlim()[1]  # Position of the reference line
        axs.plot([reference_x, reference_x],
                 [midpoint_y - reference_length / 2, midpoint_y + reference_length / 2],
                 ls='-', color='k', lw=3, zorder=100)
        axs.text(reference_x, midpoint_y, f'{reference_length}N ', color='k',
                 ha='left', va='center', zorder=100)



# if __name__ == "__main__":
#     clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]
#
#     latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
#     latency = latency['index'][0], latency['ring'][0]
#
#     plot_timec('smp0',
#                'behav',
#                channels=gl.channels['mov'],
#                clamp=clamp,
#                xlim=[-.1, .5],
#                ylim=[0, 40],
#                filename='force.timec.behav.png')
#
#     plot_timec('smp1',
#                'training',
#                channels=gl.channels['mov'],
#                clamp=clamp,
#                xlim=[-.1, .5],
#                ylim=[0, 40],
#                filename='force.timec.training.png')
#
#     plot_timec('smp1',
#                'scanning',
#                channels=gl.channels['mov'],
#                clamp=clamp,
#                xlim=[-.1, .5],
#                ylim=[0, 40],
#                filename='force.timec.scanning.png')
#
#     plot_timec('smp2',
#                'pilot',
#                channels=gl.channels['mov'],
#                clamp=clamp,
#                xlim=[-.1, .5],
#                ylim=[0, 40],
#                filename='force.timec.pilot.png')
#
#     plt.show()
