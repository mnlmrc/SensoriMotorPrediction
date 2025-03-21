import numpy as np
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import surfAnalysisPy as surf
import os
import globals as gl

def plot_flatmap_cortical_activation(img, vmin=-20, vmax=20, xlim=None, ylim=None, figsize=(5, 6)):

    if xlim is None:
        xlim = {
            'L': [-80, 120],
            'R': [-120, 80],
        }
    if ylim is None:
        ylim = {
            'L': [-50, 150],
            'R': [-60, 140]
        }

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    cifti_img = nb.load(img)
    column_names = cifti_img.header.get_axis(0).name
    data = nt.split_cifti_to_giftis(cifti_img, type='func', column_names=column_names)

    for h, H in enumerate(['L', 'R']):

        darray = nt.get_gifti_data_matrix(data[h])

        col_names = nt.get_gifti_column_names(data[h])

        plan_col_names = [col for col in col_names if 'index' not in col and 'ring' not in col]
        exec_col_names = [col for col in col_names if 'index' in col or 'ring' in col]

        im = np.array([x in plan_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[0, h])
        surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='jet',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

        im = np.array([x in exec_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[1, h])
        surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='jet',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

    # make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='jet')
    cbar = fig.colorbar(sm, ax=[axs[1, 0], axs[1, 1]], orientation='horizontal', fraction=0.03)
    cbar.set_label('activation vs. baseline (a.u.)')

    # cosmetic
    axs[0, 0].set_title('Left hemisphere\nPlanning')
    axs[0, 1].set_title('Right hemisphere\nPlanning')
    axs[1, 0].set_title('Execution')
    axs[1, 1].set_title('Execution')

    return fig, axs


