import argparse

import numpy as np
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import PcmPy as pcm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, FancyBboxPatch
import surfAnalysisPy as surf
from matplotlib.lines import Line2D
import os
import globals as gl
from scipy.stats import ttest_1samp
import matplotlib.transforms as mtransforms
import SUITPy.flatmap as flatmap
import pandas as pd
import mat73
from pcm_lfp import make_freq_masks
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_force_response(fig, axs, force, descr):
    tAx = np.linspace(-gl.prestim, gl.poststim, force.shape[-1])

    space = [[0, 2, 18, 22, 35], [0, 2, 18, 22, 35]]
    for f, finger in enumerate(descr.finger.unique()):
        for s, stimFinger in enumerate(descr.stimFinger.unique()):
            for c, cue in enumerate(descr.cue.unique()):
                force_tmp = force[(descr.cue == cue) & (descr.stimFinger == stimFinger) & (descr.finger == finger)]

                if stimFinger == 'index':
                    ax = axs[0]
                else:
                    ax = axs[1]

                y = force_tmp.mean(axis=0) + space[s][f]
                yerr = force_tmp.std(axis=0) / np.sqrt(force_tmp.shape[0])

                if ~np.isnan(y).any():
                    ax.plot(tAx, y, color=gl.colour_mapping[f'{cue},{stimFinger}'])
                    ax.fill_between(tAx, y - yerr, y + yerr, color=gl.colour_mapping[f'{cue},{stimFinger}'], lw=0,
                                    alpha=.2)

                if (s == 0) & (c == 1):
                    ax.text(-.1, y[0], finger, va='center', ha='right', )

    for ax in axs:
        ax.set_xlim([-.1, .5])
        ax.set_xticks([0, .2, .4])
        ax.spines[['bottom']].set_bounds(0, .4)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.axvline(0, ls='-', color='k', lw=.8)
        ax.set_yticks([])
        # ax.axvspan(.2, .4, color='grey', alpha=.2, lw=0)

    axs[0].text(0, ax.get_ylim()[1], 'index perturbation', va='bottom', ha='left')
    axs[1].text(0, ax.get_ylim()[1], 'ring perturbation', va='bottom', ha='left')

    make_yref(axs[1], reference_length=5, pos='right', color='k')

    fig.suptitle('Force response to finger perturbation')
    fig.supxlabel('time relative to perturbation (s)')

    # Create legend entries as colored lines (matching plotted lines)
    legend_elements = []
    for k, v in gl.colour_mapping.items():
        if 'ring' in k or 'index' in k:
            legend_elements.append(Line2D([0], [0], color=v, lw=2, label=k))

    # Add the legend to the figure, outside the right edge
    fig.legend(handles=legend_elements,
               loc='lower left',
               bbox_to_anchor=(.9, .5),
               fontsize=8,
               frameon=False, )

    fig.subplots_adjust(wspace=.3)

    return fig, axs



def plot_D_lfp(fig, axs, panel, G, ticklabels, vmin=None, vmax=None, sqrt=False, colorbar=False):
    if axs.ndim==2:
        ax = axs[panel[0], panel[1]]
    else:
        ax = axs[panel]
    D = pcm.G_to_dist(G)
    if sqrt:
        D = np.sign(D) * np.sqrt(np.abs(D))
    h = ax.imshow(D,vmin=vmin,vmax=vmax)
    ax.set_xticks(np.linspace(0, G.shape[1] - 1, G.shape[1]))
    ax.set_yticks(np.linspace(0, G.shape[1] - 1, G.shape[1]))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_yticklabels(ticklabels)

    return ax, h


def plot_mesh_lfp(axs, tAx, foi, var_expl, components, vmin=0, vmax=.1):
    for i in range(var_expl.shape[-1]):
        ax = axs[i]
        mesh = ax.pcolormesh(tAx, foi, var_expl[..., i], shading='auto', vmin=vmin, vmax=vmax)
        ax.set_yscale('log')
        ax.set_title(components[i])
        ax.set_xticks([])
        ax.axhline(3, color='w', lw=.8, ls='--')
        ax.axhline(8, color='w', lw=.8, ls='--')
        ax.axhline(13, color='w', lw=.8, ls='--')
        ax.axhline(25, color='w', lw=.8, ls='--')
        ax.axhline(100, color='w', lw=.8, ls='--')
    return axs, mesh


def plot_theta_lfp_mean(row, col, axs, var_expl_pre, var_expl_post, color):
    if axs.ndim == 2:
        ax = axs[row, col]
    else:
        ax = axs[col]
    ax.tick_params(which='both', left=False, bottom=False)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xlim([-.2, 1.2])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    var_expl = np.stack([var_expl_pre, var_expl_post])
    for i in range(var_expl_pre.shape[-1]):
        ax.plot(var_expl[:, i], color=color[i], marker='s', ms=5, mfc=color[i])
    return ax

def plot_theta_lfp(row, axs, var_expl, color):
    if axs.ndim==2:
        ax = axs[row, 0]
    else:
        ax = axs[0]
    for i in range(var_expl.shape[1]):
        ax.plot(var_expl[:, i], color=color[i])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(bottom=False)
    ax.axhline(0, color='k', lw=.8)

    return ax


def plot_force_aligned(force, descr, go_or_nogo, vsep, axs):
    tAx = np.linspace(-gl.prestim, gl.poststim, force.shape[-1])
    for cue in descr.cue.unique():
        for sf, stimF in enumerate(descr.stimFinger.unique()):
            for f, finger in enumerate(descr.finger.unique()):
                mask = (descr.cue == cue) & (descr.stimFinger == stimF) & (descr.GoNogo == go_or_nogo) & (descr.finger == finger)
                if not mask.any():
                    continue
                force_avg = force[mask].mean(axis=0).squeeze()
                force_err = force[mask].std(axis=0).squeeze() / np.sqrt(force[mask].shape[0])

                ax_idx = sf if go_or_nogo == 'go' else 0
                label = cue if (go_or_nogo == 'go' and sf == 2 and f == 0) else None

                # Determine color key
                color_key = f'{cue}' if stimF == 'nogo' else f'{cue},{stimF}'
                color = gl.colour_mapping.get(color_key, 'black')  # fallback color

                axs[ax_idx].plot(tAx, force_avg + f * vsep, color=color, label=label)
                axs[ax_idx].fill_between(tAx, force_avg + f * vsep - force_err, force_avg + f * vsep + force_err,
                                         color=color, lw=0, alpha=.2)

def annotate_finger_labels(force, descr, ax, vsep):
    for f, finger in enumerate(descr.finger.unique()):
        mask = (descr.GoNogo == 'nogo') & (descr.finger == finger)
        force_avg = force[mask].mean(axis=0).squeeze()
        force_err = force[mask].std(axis=0).squeeze() / np.sqrt(force[mask].shape[0])
        ax.text(.1, force_avg.mean() + f * vsep + force_err.mean() + .1, finger, va='bottom', ha='left')

def auto_margin(lines, margin_ratio=0.1, default_ylim=(-1, 1)):
    if not lines:
        return default_ylim
    all_y = np.concatenate([line.get_ydata() for line in lines if line.get_ydata().size > 0])
    if all_y.size == 0:
        return default_ylim
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return default_ylim
    margin = (y_max - y_min) * margin_ratio
    return [y_min - margin, y_max + margin]

def plot_flatmap_cortical_activation(img, vmin=-20, vmax=20, xlim=None, ylim=None, figsize=(5, 6),
                                     frame=(None, None, None, None), rounding=.2, cbar_orientation='vertical',cbar_fraction=.01):

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
        exec_col_names = [col for col in col_names if 'index' in col or 'ring' in col or 'exec' in col]

        im = np.array([x in plan_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[0, h])
        ax = surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='bwr',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

        # Add a patch with rounded corners on top
        bbox = FancyBboxPatch((frame[0], frame[1]), frame[2], frame[3],
                              boxstyle=f"round,pad=0,rounding_size={rounding}",
                              transform=ax.transAxes,
                              facecolor='none',
                              edgecolor='none',
                              zorder=10,
                              clip_on=False)
        for artist in ax.get_children():
            if hasattr(artist, 'set_clip_path'):
                artist.set_clip_path(bbox)

        im = np.array([x in exec_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[1, h])
        ax = surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='bwr',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

        # Add a patch with rounded corners on top
        bbox = FancyBboxPatch((frame[0], frame[1]), frame[2], frame[3],
                              boxstyle=f"round,pad=0,rounding_size={rounding}",
                              transform=ax.transAxes,
                              facecolor='none',
                              edgecolor='none',
                              zorder=10,
                              clip_on=False)
        for artist in ax.get_children():
            if hasattr(artist, 'set_clip_path'):
                artist.set_clip_path(bbox)

    # make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='bwr')
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.01)
    cbar.set_label('activation vs. baseline (a.u.)')

    # cosmetic
    axs[0, 0].set_title('Left hemisphere\nPlanning')
    axs[0, 1].set_title('Right hemisphere\nPlanning')
    axs[1, 0].set_title('Execution')
    axs[1, 1].set_title('Execution')

    return fig, axs


def plot_flatmap_cerebellar_activation(img, vmin=-20, vmax=20, xlim=None, ylim=None, figsize=(5, 6),
                                     frame=(None, None, None, None), rounding=.2, cbar_orientation='vertical',cbar_fraction=.01):

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

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    gifti = nb.load(img)

    darray = nt.get_gifti_data_matrix(gifti)

    col_names = nt.get_gifti_column_names(gifti)
    plan_col_names = [col for col in col_names if 'index' not in col and 'ring' not in col]
    exec_col_names = [col for col in col_names if 'index' in col or 'ring' in col or 'exec' in col]

    im = np.array([x in plan_col_names for x in col_names])
    darray_avg = np.array(darray[:, im]).mean(axis=1)
    plt.sca(axs[0])
    ax = flatmap.plot(data=darray_avg,
                      cmap='bwr',
                      cscale=[vmin, vmax],
                      new_figure=False,
                      colorbar=False,
                      render='matplotlib')

    im = np.array([x in exec_col_names for x in col_names])
    darray_avg = np.array(darray[:, im]).mean(axis=1)
    plt.sca(axs[1])
    ax = flatmap.plot(data=darray_avg,
                      cmap='bwr',
                      cscale=[vmin, vmax],
                      new_figure=False,
                      colorbar=False,
                      render='matplotlib')

    # make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='bwr')
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.01)
    cbar.set_label('activation vs. baseline (a.u.)')

    # cosmetic
    axs[0].set_title('Planning')
    axs[1].set_title('Execution')

    return fig, axs


def add_colorbar(fig, ax, cax, orientation='horizontal', fraction=.02, label='', anchor=None, pad=.1):
    cbar = fig.colorbar(cax, ax=ax, orientation=orientation, fraction=fraction, anchor=anchor)
    cbar.set_label(label)

def add_noise_ceiling(fig, ax, baseline, noise_ceiling, upper_ceiling=None, xlim=(None, None),
                      facecolor='lightgrey', alpha=0.3):
    noise_lower = np.nanmean(noise_ceiling)
    if upper_ceiling is not None:
        noise_upper = np.nanmean(upper_ceiling-baseline)
        noiserect = Rectangle((xlim[0], noise_lower), xlim[1]-xlim[0], noise_upper-noise_lower,
                              linewidth=0, facecolor=facecolor, zorder=1e6, alpha=alpha)
        ax.add_patch(noiserect)
    else:
        l = mlines.Line2D([xlim[0], xlim[1]], [noise_lower, noise_lower],color=[0,0,0], linestyle=':')
        ax.add_line(l)

def add_significant_patches_to_ml_rois(fig, axs, LL, box_width, patch_height=7, field='value', alternative0='greater',
                                       alternative1='less'):

    for m, md in enumerate(LL.model.unique()):
        for r, roi in enumerate(LL.roi.unique()):
            LL_tmp = LL[(LL['model'] == md) & (LL['roi'] == roi)]
            start = m - box_width / 2 + r * 0.1
            xInt = (start, start + 0.1)
            _, pval0 = ttest_1samp(LL_tmp[field], popmean=0, alternative=alternative0)
            _, pval1 = ttest_1samp(LL_tmp[field], popmean=1, alternative=alternative1)
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

def add_lineplot_to_boxplot(fig, axs, data=None, x=None, y=None, hue=None, box_width=.8, color='k', lw=1, ls='-', show_error=False):
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


def set_dark_background(fig):
    """Apply dark background styling to all axes in a Matplotlib Figure."""
    fig.patch.set_facecolor('black')  # Figure background

    for ax in fig.get_axes():
        # Axes background
        ax.set_facecolor('black')

        # Spines
        for spine in ax.spines.values():
            spine.set_color('white')

        # Tick marks and labels
        ax.tick_params(axis='both', labelcolor='white', color='white',)
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_color('white')
            tick.tick2line.set_color('white')
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_color('white')
            tick.tick2line.set_color('white')

        # Axis labels and title
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        # Optional: grid color
        ax.grid(color='gray')


def set_spines_and_ticks_width(ax,
                               spine_width=1.5,
                               spine_sides=('left', 'bottom', 'right', 'top'),
                               tick_width=1.5,
                               tick_length=3.5,
                               axes=('x', 'y'),
                               which='both',
                               colors='k'):
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
        ax.tick_params(axis=axis, width=tick_width, length=tick_length, which=which, colors=colors)



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


def make_yref(ax, reference_length=5, pos='left', unit='N', custom_text=None, color='k'):

    # Compute location in axes coordinates (0 to 1)
    midpoint_ax_y = 0.3  # halfway in the vertical direction
    length_ax = reference_length / (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Convert axes coordinates to display coordinates
    trans = ax.transAxes
    fig = ax.figure

    if pos == 'left':
        x_ref = -0.05  # just outside the axis on the left
        ha = 'right'
    elif pos == 'right':
        x_ref = 1.05  # just outside the axis on the right
        ha = 'left'
    else:
        raise ValueError("pos must be 'left' or 'right'")

    # Draw the line in axes coordinates
    ax.plot([x_ref, x_ref],
            [midpoint_ax_y - length_ax/2, midpoint_ax_y + length_ax/2],
            transform=trans,
            color=color, lw=2, zorder=100, clip_on=False)

    # Add text label
    text = custom_text if custom_text is not None else f' {reference_length}{unit}'
    ax.text(x_ref, midpoint_ax_y, text,
            transform=trans,
            color=color, ha=ha, va='center',
            zorder=100, clip_on=False)

# Define the updated function and test it on the provided file
def load_border_vertices_xml(filepath):
    vertices = []
    inside_vertices_block = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if "<Vertices>" in line:
                inside_vertices_block = True
                line = line.replace("<Vertices>", "")
            if inside_vertices_block:
                if "</Vertices>" in line:
                    line = line.replace("</Vertices>", "")
                    inside_vertices_block = False
                if line:
                    numbers = [int(x) for x in line.split()]
                    vertices.extend(numbers)
    return np.array(vertices)


def add_significance_bars(ax, tAx, sig, color='black', position='top', height=0.02, alpha=.5):
    """
    Adds a thin horizontal significance bar above or below the signal.

    Parameters:
    - ax: matplotlib axis
    - tAx: time axis (1D)
    - sig: boolean array (same shape as tAx) indicating significance
    - color: bar color
    - position: 'top' or 'bottom'
    - height: bar height as fraction of axis height (0.02 = 2%)
    """
    from itertools import groupby
    from operator import itemgetter
    import matplotlib.pyplot as plt

    # Use axis-relative coordinates for vertical placement
    transform = ax.get_xaxis_transform()  # x in data, y in axes coords

    y = 1 - height if position == 'top' else 0

    # Group significant regions
    sig_regions = [(tAx[g[0][0]], tAx[g[-1][0]]) for k, g in groupby(enumerate(sig), key=itemgetter(1))
                   if k for g in [list(g)]]

    for start, end in sig_regions:
        ax.add_patch(plt.Rectangle((start, y), end - start, height,
                                   transform=transform,
                                   color=color, alpha=alpha, linewidth=0, zorder=1e6))


# def save_figure_incremental(fig, base_name, ext='svg', overwrite=True):
#     """
#     Save a figure without overwriting by auto-incrementing the filename.
#
#     Parameters:
#         fig        : matplotlib figure object
#         base_name  : base name of the figure file (e.g., 'plot')
#         folder     : destination folder
#         ext        : file extension ('pdf', 'svg', etc.)
#     """
#     folder = os.path.join(gl.baseDir, 'figures')
#     os.makedirs(folder, exist_ok=True)
#     existing = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith('.' + ext)]
#
#     if overwrite is False:
#         # Extract number suffixes and find the next available one
#         suffixes = []
#         for f in existing:
#             parts = f.replace(f'.{ext}', '').split('_')
#             if parts[-1].isdigit():
#                 suffixes.append(int(parts[-1]))
#         next_suffix = max(suffixes, default=0) + 1
#         filename = f"{base_name}_{next_suffix}.{ext}"
#         filepath = os.path.join(folder, filename)
#     else:
#         filename = f"{base_name}.{ext}"
#         filepath = os.path.join(folder, filename)
#
#     fig.savefig(filepath, format=ext, dpi=600, bbox_inches='tight')
#     print(f"Figure saved to: {filepath}")


def make_axes_square(ax):
    pos = ax.get_position()  # Get [left, bottom, width, height] in figure fraction
    center_x = pos.x0 + pos.width / 2
    center_y = pos.y0 + pos.height / 2
    size = min(pos.width, pos.height)
    new_pos = [
        center_x - size / 2,  # new left
        center_y - size / 2,  # new bottom
        size,                 # new width
        size                  # new height
    ]
    ax.set_position(new_pos)


def pcm_spike(fig, axs, roi, epoch, monkey, rec):
    # temporal landmarks
    cuePre = 0
    cueIdx = 20
    cuePost = 84
    pertIdx = 114

    xtick = cueIdx if epoch == 'plan' else pertIdx
    xticklabel = 'Cue' if epoch == 'plan' else 'Pert'
    xlim = [cuePre, cuePost] if epoch == 'plan' else [cuePost, 154]
    rangePre = np.arange(cuePre, cueIdx) if epoch == 'plan' else np.arange(cuePost, pertIdx)
    rangePost = np.arange(cueIdx, cuePost) if epoch == 'plan' else np.arange(pertIdx, 145)

    # load data
    G_obs, var_expl, cov = [], [], []
    for r in rec:
        path = os.path.join(gl.baseDir, 'smp2', 'spikes', gl.pcmDir)
        G_obs.append(np.load(os.path.join(path, f'G_obs.spike.{monkey}.{roi}.aligned.{epoch}-{r}.npy')))

        # calc variance
        theta_c = np.load(os.path.join(path, f'theta_in.spike.component.{monkey}.{roi}.aligned.{epoch}-{r}.npy'))
        n_param_c = theta_c.shape[-1] - 1
        var_expl.append(np.sqrt(np.exp(theta_c[..., :n_param_c])))

        if epoch == 'exec':
            theta_f = np.load(os.path.join(path, f'theta_in.spike.feature.{monkey}.{roi}.aligned.{epoch}-{r}.npy'))
            cov.append(theta_f[:, 1] * theta_f[:, 2])

    var_expl = np.array(var_expl).mean(axis=0)
    G_obs = np.array(G_obs).mean(axis=0)
    cov = np.array(cov).mean(axis=0)

    color = ['red', 'blue'] if epoch == 'plan' else ['#FFCC33', 'red', 'blue', 'magenta', 'cyan', 'k']
    linestyle = ['-', '-', '-', '-', '-', '--']
    components = ['cue', 'uncertainty'] if epoch == 'plan' else ['direction', 'cue', 'uncertainty', 'surprise', 'direction*cue', 'noise ceiling']
    tr = np.sqrt(np.trace(G_obs, axis1=1, axis2=2))

    ax = plot_theta_lfp(0, axs, var_expl, color=color)

    extra_lines = []
    extra_labels = []

    # covariance
    if epoch == 'exec':  # only execution models contain feature model
        l_cov = ax.plot(cov, color='cyan')
        extra_lines.append(l_cov)
        extra_labels.append('covariance')

    l_tr = ax.plot(tr, ls='--', color='k')
    extra_lines.append(l_tr)
    extra_labels.append('noise ceiling')

    main_lines = [
        plt.Line2D([], [], color=c, linestyle=ls)
        for c, ls in zip(color, linestyle)
    ]
    labels = components + extra_labels
    lines = main_lines + extra_lines

    fig.legend(lines, labels, loc='center left', fontsize=9, ncol=1, bbox_to_anchor=(1, .5), frameon=False)

    # ax.spines['left'].set_bounds(0, .5)
    ax.axvline(xtick, color='k', lw=.8)
    ax.set_xticks([xtick])
    # ax.set_ylim([-.008, 3])
    ax.set_xticklabels([xticklabel])
    ax = plot_theta_lfp_mean(0, 1, axs, var_expl[rangePre].mean(axis=0),
                             var_expl[rangePost].mean(axis=0), color=color)
    axs[0].set_xlim(xlim)
    fig.supylabel('variance (a.u.)', fontsize='medium')
    fig.suptitle(f'Variance explained by component model (monkey {monkey[0]}, {roi})')
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1)

    return fig, axs


def main(args, **kwargs):
    path_fig = 'figures'
    if args.what=='force_response':
        experiment = 'smp2'
        npz = np.load(os.path.join(gl.baseDir, experiment, gl.behavDir, 'force.segmented.avg.npz'), allow_pickle=True)
        force = npz['data_array']
        descr = pd.DataFrame(npz['descriptor'].item())
        force = force[descr.GoNogo == 'go']
        descr = descr[descr.GoNogo == 'go']
        fig, axs = plt.subplots(1, 2, figsize=(4, 5), sharey=True, sharex=True, constrained_layout=True)
        fig, axs = plot_force_response(fig, axs, force, descr)
        fig.tight_layout()
        plt.savefig(os.path.join(path_fig, 'force_response.svg'))
        plt.show()
    if args.what=='lfp':
        pass
    if args.what=='pcm_lfp':
        cuePre = 0
        cueIdx = 20
        cuePost = 84
        pertIdx = 114
        monkey = kwargs.get('monkey', 'Pert')
        roi = kwargs.get('roi', 'PMd')
        epoch = kwargs.get('epoch', 'plan')
        xtick = cueIdx if epoch == 'plan' else pertIdx
        xticklabel = 'Cue' if epoch == 'plan' else 'Pert'
        xlim = [cuePre, cuePost] if epoch == 'plan' else [cuePost, 154]
        rangePre = np.arange(cuePre, cueIdx) if epoch == 'plan' else np.arange(cuePost, pertIdx)
        rangePost = np.arange(cueIdx, cuePost) if epoch == 'plan' else np.arange(pertIdx, 145)
        path = os.path.join(gl.baseDir, 'smp2', 'LFPs', gl.pcmDir)
        theta_in = np.load(os.path.join(path, f'theta_in.lfp.{monkey}.{roi}.aligned.{epoch}.npy'))
        cfg = mat73.loadmat(os.path.join(gl.baseDir, 'smp2', 'LFPs', monkey, 'cfg.mat'))['cfg']
        freq_mask = make_freq_masks(cfg)
        G_obs = np.load(os.path.join(path, f'G_obs.lfp.{monkey}.{roi}.aligned.{epoch}.npy'))
        n_param = theta_in.shape[-1] - 1
        var_expl = np.sqrt(np.exp(theta_in[..., :n_param]))
        color = ['red', 'blue'] if epoch == 'plan' else ['#FFCC33', 'red', 'blue', 'magenta']
        components = ['cue', 'uncertainty'] if epoch == 'plan' else ['finger', 'cue', 'uncertainty', 'surprise']
        freqs = ['delta', 'theta', 'alpha-beta', 'alpha', 'beta', 'gamma']
        fig, axs = plt.subplots(len(freqs), 2, sharex='col', sharey=True, figsize=(5, 8),
                                gridspec_kw={'width_ratios': [3, .5]})
        for f, freq in enumerate(freqs):
            tr = np.sqrt(np.trace(G_obs[freq_mask[freq]].mean(axis=0), axis1=1, axis2=2))
            var_expl_tmp = var_expl[freq_mask[freq]].mean(axis=0)
            ax = plot_theta_lfp(f, axs, var_expl_tmp, color=color)
            ax.plot(tr, ls='--', color='k')
            ax.set_title(freq)
            ax.spines['left'].set_bounds(0, .5)
            ax.axvline(xtick, color='k', lw=.8)
            ax.set_xticks([xtick])
            ax.set_ylim([-.008, .5])
            ax.set_xticklabels([xticklabel])
            ax = plot_theta_lfp_mean(f, 1, axs, var_expl_tmp[rangePre].mean(axis=0),
                                     var_expl_tmp[rangePost].mean(axis=0), color=color)
        axs[0, 0].set_xlim(xlim)
        fig.supylabel('variance (a.u.)', fontsize='medium')
        fig.legend(components, loc='lower right', fontsize=9, frameon=False, ncol=2)
        fig.suptitle(f'Variance explained by component model (monkey {monkey[0]}, {roi})')
        fig.tight_layout()
        fig.subplots_adjust(bottom=.1)
        plt.savefig(os.path.join(path_fig, f'pcm_lfp.{epoch}.{monkey}.{roi}.svg'))
        plt.show()
    if args.what == 'pcm_lfp_tf_plan':
        cuePre = 0
        cueIdx = 20
        cuePost = 84
        pertIdx = 114
        monkey = kwargs.get('monkey', 'Pert')
        roi = kwargs.get('roi', 'PMd')
        components = ['cue', 'uncertainty']
        epoch = 'plan'
        figsize = kwargs.get('figsize', (4, 4))
        path = os.path.join(gl.baseDir, 'smp2', 'LFPs', gl.pcmDir)
        theta_in = np.load(os.path.join(path, f'theta_in.lfp.{monkey}.{roi}.aligned.{epoch}.npy'))
        cfg = mat73.loadmat(os.path.join(gl.baseDir, 'smp2', 'LFPs', monkey, 'cfg.mat'))['cfg']
        n_params = theta_in.shape[-1] - 1
        var_expl = np.sqrt(np.exp(theta_in[..., :n_params]))
        tAx = np.linspace(0, var_expl.shape[1], var_expl.shape[1])
        fig, axs = plt.subplots(var_expl.shape[-1], sharex=True, sharey=True,
                                figsize=(int(figsize[0]), int(figsize[1])), constrained_layout=True)
        _, mesh = plot_mesh_lfp(axs, tAx, cfg['foi'], var_expl, components, vmin=0, vmax=.3)
        for ax in axs:
            ax.set_xlim([0, cuePost])
            ax.set_xticks([cueIdx])
            ax.set_xticklabels(['Cue'])
            ax.axvline(cueIdx, color='w', lw=.8)
        fig.supylabel('frequency (Hz)')
        fig.suptitle(f'Variance explained\nby component model (Monkey {monkey[0]}, {roi})')
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('variance (a.u.)')
        plt.savefig(os.path.join(path_fig, f'pcm_lfp_tf.{epoch}.{monkey}.{roi}.svg'))
        plt.show()
    if args.what == 'pcm_lfp_tf_exec':
        cuePre = 0
        cueIdx = 20
        cuePost = 84
        pertIdx = 114
        monkey = kwargs.get('monkey', 'Pert')
        roi = kwargs.get('roi', 'PMd')
        epoch = 'exec'
        components = kwargs.get('components', ['finger', 'cue', 'uncertainty', 'surprise'])
        figsize = kwargs.get('figsize', (4, 8))
        path = os.path.join(gl.baseDir, 'smp2', 'LFPs', gl.pcmDir)
        theta_in = np.load(os.path.join(path, f'theta_in.lfp.{monkey}.{roi}.aligned.{epoch}.npy'))
        cfg = mat73.loadmat(os.path.join(gl.baseDir, 'smp2', 'LFPs', monkey, 'cfg.mat'))['cfg']
        n_params = theta_in.shape[-1] - 1
        var_expl = np.sqrt(np.exp(theta_in[..., :n_params]))
        tAx = np.linspace(0, var_expl.shape[1], var_expl.shape[1])
        fig, axs = plt.subplots(var_expl.shape[-1], sharex=True, sharey=True,
                                figsize=(int(figsize[0]), int(figsize[1])), constrained_layout=True)
        _, mesh = plot_mesh_lfp(axs, tAx, cfg['foi'], var_expl, components, vmin=0, vmax=.3)
        for ax in axs:
            ax.set_xlim([cuePost, var_expl.shape[1]])
            ax.set_xticks([pertIdx])
            ax.set_xticklabels(['Pert'])
            ax.axvline(pertIdx, color='w', lw=.8)
        fig.supylabel('frequency (Hz)')
        fig.suptitle(f'Variance explained\nby component model (Monkey {monkey[0]}, {roi})')
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(mesh, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('variance (a.u.)')
        plt.savefig(os.path.join(path_fig, f'pcm_lfp_tf.{epoch}.{monkey}.{roi}.svg'))
        plt.show()
    if args.what == 'pcm_spike':
        monkey = kwargs.get('monkey', 'Pert')
        roi = kwargs.get('roi', 'PMd')
        epoch = kwargs.get('epoch', 'plan')
        fig, axs = plt.subplots(1, 2, sharex='col', sharey=True, figsize=(5, 3),
                                gridspec_kw={'width_ratios': [3, .5]})
        fig, axs = pcm_spike(roi, epoch, monkey)
        fig.savefig(os.path.join(path_fig, f'pcm_spike.{epoch}.{monkey}.{roi}.svg'))
        plt.show()


def parse_unknown_args(args):
    parsed = {}
    key = None
    for arg in args:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            parsed[key] = []  # Start a new list for this key
        elif key:
            parsed[key].append(arg)
        else:
            raise ValueError(f"Value {arg} has no associated flag.")

    # Flatten any single-value lists for convenience
    for k in parsed:
        if len(parsed[k]) == 1:
            parsed[k] = parsed[k][0]

    return parsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('what', nargs='?', default=None)
    args, unknown_args = parser.parse_known_args()

    kwargs = parse_unknown_args(unknown_args)

    main(args, **kwargs)