import numpy as np
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import surfAnalysisPy as surf
import os
import globals as gl
from scipy.stats import ttest_1samp

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


def add_colorbar(fig, ax, cax, orientation='horizontal', fraction=.02, label='', anchor=None, pad=.1):
    cbar = fig.colorbar(cax, ax=ax, orientation=orientation, fraction=fraction, anchor=anchor)
    cbar.set_label(label)

def add_significant_patches_to_ml_rois(fig, axs, LL, box_width, patch_height):

    for m, md in enumerate(LL.model.unique()):
        for r, roi in enumerate(LL.roi.unique()):
            LL_tmp = LL[(LL['model'] == md) & (LL['roi'] == roi)]
            start = m - box_width / 2 + r * 0.1
            xInt = (start, start + 0.1)
            _, pval0 = ttest_1samp(LL_tmp['value'], popmean=0, alternative='greater')
            _, pval1 = ttest_1samp(LL_tmp['value'], popmean=1, alternative='less')
            if pval0 < .05:
                rect = Rectangle(
                    (xInt[0], 0),
                    xInt[1] - xInt[0],
                    patch_height,
                    fc='grey',
                    alpha=0.2,
                    lw=0
                )
                axs.add_patch(rect)

    return fig, axs

def add_lineplot_to_boxplot(fig, axs, data=None, x=None, y=None, hue=None, box_width=None, color='k', lw=1, ls='-', show_error=False):
    x_order = data[x].unique()
    hue_order = data[hue].unique()
    n_hues = len(hue_order)

    for i, X in enumerate(x_order):
        means = []
        err = []
        xAx = []
        for j, Hue in enumerate(hue_order):
            subset = data[(data[x] == X) & (data[hue] == Hue)]
            mean_val = subset[y].mean()
            err_val = subset[y].std() / np.sqrt(len(subset[y]))

            x_pos = i - box_width / 2 + (j + 0.5) * box_width / n_hues

            xAx.append(x_pos)
            means.append(mean_val)
            err.append(err_val)

        mean = np.array(means)
        err = np.array(err)
        axs.plot(xAx, means, color=color, lw=lw, ls=ls,zorder=1e6)
        if show_error:
            axs.fill_between(xAx, means-err, means+err, color=color, lw=0, alpha=.2,zorder=1e6)


    return fig, axs


def set_spines_and_ticks_width(ax,
                               spine_width=1.5,
                               spine_sides=('left', 'bottom', 'right', 'top'),
                               tick_width=1.5,
                               tick_length=3.5,
                               axes=('x', 'y'),
                               which='both'):
    """
    Adjust the width of spines and ticks on a matplotlib Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes to modify.
    - spine_width: float or dict
        Width for spines. If dict, use keys like {'left': 1.5, 'bottom': 2, ...}
    - spine_sides: tuple
        Which spines to modify. Default is all four.
    - tick_width: float
        Width of tick marks.
    - tick_length: float
        Length of tick marks.
    - axes: tuple
        Axes to apply tick changes to ('x', 'y', or both).
    - which: str
        'major', 'minor', or 'both' ticks.
    """
    # Set spine widths
    for side in spine_sides:
        if isinstance(spine_width, dict):
            if side in spine_width:
                ax.spines[side].set_linewidth(spine_width[side])
        else:
            ax.spines[side].set_linewidth(spine_width)

    # Set tick parameters
    for axis in axes:
        ax.tick_params(axis=axis, width=tick_width, length=tick_length, which=which)




