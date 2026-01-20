import argparse
import pickle
import pyvista as pv
import numpy as np
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import PcmPy as pcm
#from pcm_models import find_model
from sklearn.preprocessing import MinMaxScaler
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
import surfAnalysisPy as surf
from matplotlib.lines import Line2D
import os
import globals as gl
from scipy.stats import ttest_1samp, ttest_rel, linregress, t, permutation_test, binomtest
import SUITPy.flatmap as flatmap
import seaborn as sb
import pandas as pd
import mat73
from itertools import combinations
#from imaging_pipelines.util import bootstrap_summary

import warnings
import xarray as xr

warnings.filterwarnings('ignore')

def plot_aligned_force(fig, axs, force, descr):
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

def plot_aligned_emg(fig, axs, emg):

    # Time axis for each latency
    tAx = np.linspace(-1, 2, 6444)

    # Vertical lines and associated styles
    line_xs = [0, .025, .05, .1]
    line_styles = ['-', '--', '-.', ':']
    line_labels = ['SLR', 'LLR', 'Vol']

    for ch, channel in enumerate(emg.keys()):
        for c, cue in enumerate(gl.cue_mapping.values()):
            for s, stimFinger in enumerate(gl.stimFinger):

                ax = axs[s]

                data = np.array(emg[channel])
                y = np.nanmean(data[:, c, s], axis=0) + ch * .1
                yerr = np.nanstd(data[:, c, s], axis=0) / np.sqrt(10)

                if ~np.isnan(y).any():
                    ax.plot(tAx, y, color=gl.colour_mapping[f'{cue},{stimFinger}'])
                    ax.fill_between(tAx, y - yerr, y + yerr,
                                    color=gl.colour_mapping[f'{cue},{stimFinger}'], lw=0, alpha=.2)

                if (s == 0) & (c == 1):
                    label = (f'FDS$_{{{ch + 1}}}$' if ch < 5 else
                             f'EDC$_{{{ch - 4}}}$' if (ch >= 5) & (ch < 10) else
                             'FDI' if ch == 10 else '')
                    ax.text(-.025, y[0], label, va='center', ha='right', )

    # Draw vertical lines
    for ax in axs:
        ax.set_ylim((0, 1.15))
        for x, style in zip(line_xs, line_styles):
            ax.vlines(x, ymin=0, ymax=ax.get_ylim()[1], linestyles=style, color='k', lw=.8)

        ax.set_xlim([-.02, .2])

        #     ax.set_ylim([0, 1.15])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # Remove y-ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.spines['bottom'].set_bounds([0, .2])
        # ax.spines['bottom'].set_linewidth(2)
        # ax.tick_params(width=2)
        ax.set_xticks([0, .1, .2])
        # set_spines_and_ticks_width(ax, spine_width=1.5, spine_sides=('bottom',), tick_width=1.5, axes=('x',), which='both')

        # Place text labels between key vertical lines
        xlims = ax.get_xlim()
        text_positions = [
            (.025 + .05) / 2,  # "SLR"
            (.05 + .1) / 2,  # "LLR"
            (.1 + xlims[1]) / 2  # "Vol"
        ]
        for label, xpos in zip(line_labels, text_positions):
            ax.text(xpos, ax.get_ylim()[1], label, ha='center', va='top', rotation=90, )
    #
    # # Global labels and layout
    fig.supxlabel('time relative to perturbation (s)')
    make_yref(axs[1], reference_length=.1, pos='right', unit='mV', color='k')

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
               frameon=False)

    axs[0].text(0, ax.get_ylim()[1], 'index perturbation', va='bottom', ha='left')
    axs[1].text(0, ax.get_ylim()[1], 'ring perturbation', va='bottom', ha='left')

    fig.suptitle('EMG response to finger perturbation', )
    fig.subplots_adjust(wspace=.4)

    return fig, axs

def plot_aligned_deviation(fig, axs, force, descr):
    for s, stimFinger in enumerate(descr.stimFinger.unique()):
        for c, cue in enumerate(descr.cue.unique()):
            force1 = force[(descr.cue == cue) & (descr.stimFinger == stimFinger) & (descr.finger == 'index')]
            force2 = force[(descr.cue == cue) & (descr.stimFinger == stimFinger) & (descr.finger == 'ring')]

            if stimFinger == 'index':
                ax = axs[0]
            else:
                ax = axs[1]

            y1 = force1.mean(axis=0)
            y2 = force2.mean(axis=0)
            # yerr1 = force1.std(axis=0) / np.sqrt(force1.shape[0])
            yerr2 = force2.std(axis=0) / np.sqrt(force2.shape[0])

            if ~np.isnan(y1).any():
                ax.plot(y1, y2, color=gl.colour_mapping[f'{cue},{stimFinger}'])
                ax.fill_between(y1, y2 - yerr2, y2 + yerr2, color=gl.colour_mapping[f'{cue},{stimFinger}'], lw=0,
                                alpha=.2)

        ax.set_aspect('equal')

    axs[0].text(0, ax.get_ylim()[1], 'index perturbation', va='bottom', ha='left')
    axs[1].text(0, ax.get_ylim()[1], 'ring perturbation', va='bottom', ha='left')

    for a, ax in enumerate(axs):
        ax.set_yticks([0, 10])
        ax.set_xticks([0, 10])
        ax.spines[['left', 'bottom']].set_bounds(0, 10)
        if a == 0:
            ax.spines[['top', 'right', ]].set_visible(False)
        else:
            ax.spines[['top', 'right', 'left', ]].set_visible(False)
            ax.tick_params(which='both', left=False)

    # Create legend entries as colored lines (matching plotted lines)
    legend_elements = []
    for k, v in gl.colour_mapping.items():
        if 'ring' in k or 'index' in k:
            legend_elements.append(Line2D([0], [0], color=v, lw=2, label=k))

    # Add the legend to the figure, outside the right edge
    axs[1].legend(handles=legend_elements,
                  loc='upper right',
                  # bbox_to_anchor=(.9, .5),
                  fontsize=8,
                  frameon=False, )

    fig.supxlabel('index force (N)', fontsize=10)
    fig.supylabel('ring force (N)', fontsize=10)
    fig.suptitle('Force trajectories')

    return fig, axs

def plot_binned_behaviour(fig, axs, dat, y=('index0', 'ring0'), finger=('nogo', 'nogo'), markersize=2, jitter=.2):
    sb.barplot(dat[dat['stimFinger'] == finger[0]], x='cue', y=y[0], ax=axs[0], errorbar='se',
               width=1, palette=list(gl.colour_mapping.values())[5:9], order=list(gl.regressor_mapping)[0:4])
    sb.stripplot(
        dat[dat['stimFinger'] == finger[0]],
        x='cue', y=y[0], ax=axs[0], size=markersize,
        order=list(gl.regressor_mapping)[0:4],
        color='black', jitter=jitter, dodge=False,
    )
    sb.barplot(dat[dat['stimFinger'] == finger[1]], x='cue', y=y[1], ax=axs[1], errorbar='se',
                width=1, palette=list(gl.colour_mapping.values())[9:13], order=list(gl.regressor_mapping)[1:5])
    sb.stripplot(
        dat[dat['stimFinger'] == finger[1]],
        x='cue', y=y[1], ax=axs[1], size=markersize,
        order=list(gl.regressor_mapping)[1:5],
        color='black', jitter=jitter, dodge=False,
    )

    for ax in axs:
        ax.set_xticks([])
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
        ax.set_xlabel('')

    axs[1].spines[['left']].set_visible(False)
    axs[1].tick_params(width=0)

    return fig, axs

def plot_bold(fig, axs, T, H, rois):
    tAx = np.linspace(-10, 20, T['y_adj'].shape[-1]) + .5
    for r in range(len(rois)):

        ax = axs[r]

        hem = T['hem'] == H
        roi = T['name'] == rois[r]
        go = T['GoNogo'] == 'go'

        N = T['y_adj'][go & roi & hem]

        y_adj_go = np.nanmean(T['y_adj'][go & roi & hem], axis=0)
        y_adj_nogo = np.nanmean(T['y_adj'][~go & roi & hem], axis=0)

        y_adj_go_err = np.nanstd(T['y_adj'][go & roi & hem], axis=0) / np.sqrt(14)
        y_adj_nogo_err = np.nanstd(T['y_adj'][~go & roi & hem], axis=0) / np.sqrt(14)

        ax.plot(tAx, y_adj_go, color='#FF7F7F', label='go', ls='-')
        ax.plot(tAx, y_adj_nogo, color='#000080', label='nogo', ls='-')
        ax.fill_between(tAx, y_adj_go - y_adj_go_err, y_adj_go + y_adj_go_err,
                        color='#FF7F7F', alpha=.2, lw=0)
        ax.fill_between(tAx, y_adj_nogo - y_adj_nogo_err, y_adj_nogo + y_adj_nogo_err,
                        color='#000080', alpha=.2, lw=0)

        ax.axvline(0, color='k', ls='-', lw=.8)
        ax.axvline(2.5, color='k', ls='--', lw=.8)
        ax.axhline(0, color='k', ls='-', lw=.8)

        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_bounds(0, 12)
        ax.set_xticks([0, 6, 12])
        if r == 0:
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_bounds(-2, 2)
        else:
            ax.tick_params('y', width=0)

        ax.set_xlim((-1, 12))
        ax.set_ylim((-3, 3.6))

        ax.set_title(rois[r])

    ax.legend(frameon=False, bbox_to_anchor=(1, -.1), loc='upper right', )

    fig.supylabel('activation (a.u.)')
    fig.supxlabel('time relative to cue presentation onset (s)')
    fig.suptitle(f'Average BOLD timeseries')

    fig.subplots_adjust(bottom=.2, top=.8, left=.08)

    return fig, axs

def plot_surf(fig, ax, surf_data, H, vmin=-10, vmax=10, cmap='viridis', col=0, thresh=.01, title=None,
              overlay='overlay'):

    Hem = ['L', 'R']
    h = Hem.index(H)

    surf = nb.load(os.path.join(gl.atlasDir, f'fs_LR.32k.{H}.very_inflated.surf.gii'))
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.uint32)  # pyvista requires uint32
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()
    sulc = nt.get_gifti_data_matrix(nb.load(os.path.join(gl.atlasDir, 'fs_LR.32k.LR.sulc.dscalar.gii')))

    if isinstance(surf_data, nb.Cifti2Image):
        column_names = surf_data.header.get_axis(0).name
        giftis = nt.split_cifti_to_giftis(surf_data, type='func', column_names=column_names)
        data = nt.get_gifti_data_matrix(giftis[h])[:, col]
    elif isinstance(surf_data, nb.GiftiImage):
        data = nt.get_gifti_data_matrix(surf_data)[:, col]
    if isinstance(surf_data, np.ndarray):
        data = surf_data

    if H == 'L':
        sulc = sulc[:len(data)]
    else:
        sulc = sulc[len(data):]

    if thresh is not None:
        mask = (data > thresh) | (data < -thresh)
        data[~mask] = np.nan

    mesh = pv.PolyData(coords, faces)
    mesh.point_data["sulc"] = sulc
    mesh.point_data[overlay] = data

    border_verts = load_border_vertices_xml(os.path.join(gl.atlasDir, f'fs_LR.32k.{H}.border'))
    border = coords[border_verts]
    p = pv.Plotter(window_size=(600, 600), off_screen=True)
    p.add_mesh(mesh, scalars="sulc", cmap="Greys", clim=[-2, 2], lighting=True, show_scalar_bar=False)
    p.add_mesh(mesh,
               scalars=overlay,
               cmap=cmap if overlay=='overlay' else None,
               rgb=overlay=='rgb',
               clim=[vmin, vmax],
               lighting=True,
               show_scalar_bar=False)
    p.add_points(border[::3], color='w', point_size=6, render_points_as_spheres=True)
    p.set_background("white")
    if H == 'L':
        p.view_vector((-.8, 0, 1))
    elif H == 'R':
        p.view_vector((.8, 0, 1))
    p.show(screenshot='tmp.png', jupyter_backend='none')
    p.close()
    img = plt.imread('tmp.png')
    os.remove('tmp.png')
    h, w = img.shape[:2]
    pad = 220  # number of pixels to keep around the center
    cropped_img = img[h // 2 - 160:h // 2 + 125, w // 2 - 250:w // 2 + 250]
    ax.imshow(cropped_img)
    ax.axis('off')
    ax.set_title(title, fontsize=20, pad=18)

    return fig, ax

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

def plot_surf_label(fig, ax, gifti, H, cmap='viridis', view='lateral'):
    Hem = ['L', 'R']
    h = Hem.index(H)

    data = gifti.darrays[0].data.astype(float)

    surf = nb.load(f'/cifs/diedrichsen/data/Atlas_templates/fs_LR_32/fs_LR.32k.{H}.very_inflated.surf.gii')
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.uint32)  # pyvista requires uint32
    # data = nt.get_gifti_data_matrix(giftis)
    mask = data < 1
    data[mask] = np.nan
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()

    sulc = nt.get_gifti_data_matrix(
        nb.load(f'/cifs/diedrichsen/data/Atlas_templates/fs_LR_32/fs_LR.32k.LR.sulc.dscalar.gii'))
    if H == 'L':
        sulc = sulc[:len(data)]
    else:
        sulc = sulc[len(data):]

    mesh = pv.PolyData(coords, faces)
    mesh.point_data["overlay"] = data

    mesh.point_data["sulc"] = sulc

    border_verts = load_border_vertices_xml(
        f'/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_{H}/fs_LR.32k.{H}.border')
    border = coords[border_verts]

    p = pv.Plotter(window_size=(600, 600), off_screen=True)
    p.add_mesh(mesh, scalars="sulc", cmap="Greys", clim=[-2, 2], lighting=True, show_scalar_bar=False)
    p.add_mesh(mesh, scalars="overlay", cmap=cmap, clim=[1, np.nanmax(data)],  show_scalar_bar=False, categories=True)
    p.add_points(border[::3], color='w', point_size=6, render_points_as_spheres=True)
    p.set_background("white")
    if view == 'lateral':
        p.view_vector((-1, 0, 1))
    elif view == 'medial':
        p.view_vector((.8, 0, 0))
    p.show(screenshot='tmp.png')

    img = plt.imread('tmp.png')
    os.remove('tmp.png')
    h, w = img.shape[:2]
    pad = 220  # number of pixels to keep around the center
    cropped_img = img[h // 2 - 200:h // 2 + 180, w // 2 - 250:w // 2 + 250]
    ax.imshow(cropped_img)
    ax.axis('off')

    return fig, ax

def plot_avg_activation(fig, axs, con, H, rois):
    for r, roi in enumerate(rois):
        ax = axs[r]
        conditions = list(gl.regressor_mapping.keys())[:13]
        sb.barplot(con[(con['roi'] == roi) & (con['Hem'] == H)],
                   ax=ax,
                   y='con',
                   x='condition',
                   order=conditions,
                   palette=[gl.colour_mapping[cond] for cond in conditions],
                   # showfliers=False,
                   errorbar='se',
                   width=1,
                   legend=False
                   )
        ax.axhline(0, ls='-', color='k', lw=.8)
        ax.set_title(roi)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False, width=0)
        ax.spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        if r == 0:
            ax.spines[['left', ]].set_visible(True)
        else:
            ax.tick_params(axis='y', width=0)

    return fig, axs

def plot_dissimilarities(fig, axs, panel, D, ticklabels, vmin=None, vmax=None, sqrt=False, source=None):
    if panel is not None:
        ax = axs[panel]
    else:
        ax = axs

    mask = np.tri(D.shape[1], k=-1, dtype=bool)
    Dd = D[:, mask].mean(axis=-1)

    if sqrt:
        D = np.sign(D) * np.sqrt(np.abs(D))
    h = ax.imshow(D.mean(axis=0),vmin=vmin,vmax=vmax)
    ax.set_xticks(np.linspace(0, D.shape[1] - 1, D.shape[1]))
    ax.set_yticks(np.linspace(0, D.shape[1] - 1, D.shape[1]))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_yticklabels(ticklabels)

    tval, pval = ttest_1samp(Dd, 0, alternative='greater')
    print(f'{source}: tval={tval}, pval={pval}')

    return fig, axs

# def plot_likelihood(fig, axs, likelihood, x='roi', color='k', width=.8):
#     likelihood = pd.DataFrame(likelihood)
#     baseline = likelihood['baseline'].values
#     likelihood[['noise_upper', 'noise_lower', 'likelihood']] = likelihood[['noise_upper', 'noise_lower',
#                                                            'likelihood']] - baseline.reshape(-1, 1)
#
#     sb.barplot(data=likelihood, x=x, y='likelihood', ax=axs, errorbar='se', color=color, width=width)
#
#     xs = likelihood[x].unique()
#
#     for xi, Xx in enumerate(xs):
#         ll = likelihood[likelihood[x] == Xx]['likelihood']
#         noise_upper = likelihood[likelihood[x] == Xx]['noise_upper'].mean()
#         noise_lower = likelihood[likelihood[x] == Xx]['noise_lower'].mean()
#         noiserect = Rectangle(
#             (xi - width / 2, noise_lower),
#             width,
#             noise_upper - noise_lower,
#             linewidth=0,
#             facecolor=[0.5, 0.5, 0.5, 0.2],
#             zorder=1e6,
#             alpha=0.3
#         )
#         axs.add_patch(noiserect)
#         # tval, pval = ttest_1samp(ll, 0, alternative='greater')
#         print(f"{Xx}: {ll.mean() / noise_upper} upper noise")
#
#     fig, axs = add_sig_to_bars(fig, axs, likelihood, y='likelihood', x=x, alternative='greater')
#
#     axs.set_ylabel('log-Bayes Factor')
#     axs.set_xlabel('')
#     axs.spines[['top', 'right', 'bottom']].set_visible(False)
#     axs.spines['bottom'].set_bounds(0, len(xs) - 1)
#     axs.tick_params('x', bottom=False)
#
#     return fig, axs

# def plot_var_expl(fig, axs, panel, param_c, components=['finger', 'cue', 'surprise'],
#                   palette=['#D4AF37', 'red', 'magenta']):
#
#     var_expl = np.exp(param_c)
#
#     ax = axs[panel]
#
#     sb.barplot(data=var_expl.T, ax=ax, palette=palette, errorbar='se', width=1)
#     ax.set_xticks(ax.get_xticks())
#     ax.spines[['top', 'right', 'bottom']].set_visible(False)
#     if panel>0:
#         ax.spines['left'].set_visible(False)
#         ax.tick_params(axis='y', left=False)
#     else:
#         ax.spines['left'].set_position(('data', -1))
#
#     ax.set_ylabel('weight')
#     ax.set_xticks([])
#
#     return fig, axs


def plot_force_repr_corr(fig, axs, panel, param_c, diff):

    ax = axs[panel]

    var_expl = np.exp(param_c)

    # Regression for diff2
    x = diff
    y = var_expl[0]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept

    # Compute confidence intervals
    n = len(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = n - 2
    t_val = t.ppf(0.95, dof)

    se_line = np.sqrt(
        np.sum(residuals ** 2) / dof * (1 / n + (x_fit - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)))
    ci = t_val * se_line

    # Check confidence interval at x = 0
    ix_0 = np.argmin(np.abs(x_fit - 0))
    lower_bound = y_fit[ix_0] - ci[ix_0]
    upper_bound = y_fit[ix_0] + ci[ix_0]

    MSE = np.sum(residuals ** 2) / dof
    SE_intercept = np.sqrt(MSE * (1 / n + np.mean(x) ** 2 / np.sum((x - np.mean(x)) ** 2)))
    t_intercept = intercept / SE_intercept
    p = 1 - t.cdf(t_intercept, df=dof)

    if lower_bound > 0:
        print(f"Intercept is significantly > 0 (one-sided, p < 0.05), t={t_intercept}, p={p}")
    else:
        print(f'Intercept not significant, t={t_intercept}, p={p}')

    ax.plot(x_fit, y_fit, color='k', linestyle='--', label='Fit')
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color='k', alpha=.2, label='95% CI', lw=0)
    ax.scatter(x, y, color='k')

    ax.axvline(0, lw=.8, color='k')
    ax.axhline(0, lw=.8, color='k')

    # ax.set_title(roi)

    # Remove spines from 'left', 'top', and 'right'
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set ticks for x-axis and y-axis
    ax.set_xticks(np.linspace(0, .08, 2))

    return fig, axs

def plot_correlation(fig, axs, panel, x, y, alternative_slope='two-sided', alternative_intercept='two-sided'):
    if isinstance(axs, plt.Axes):
        ax = axs
    elif isinstance(axs, np.ndarray):
        ax = axs[panel]

    slope, intercept, r_value, p_slope, std_err = linregress(x, y, alternative=alternative_slope)

    R2 = r_value ** 2

    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + intercept

    # Compute confidence intervals
    n = len(x)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = n - 2
    t_val = t.ppf(0.975, dof)

    se_line = np.sqrt(
        np.sum(residuals ** 2) / dof * (1 / n + (x_fit - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ci = t_val * se_line

    # Check confidence interval at x = 0
    ix_0 = np.argmin(np.abs(x_fit - 0))
    lower_bound = y_fit[ix_0] - ci[ix_0]
    upper_bound = y_fit[ix_0] + ci[ix_0]

    MSE = np.sum(residuals ** 2) / dof
    SE_intercept = np.sqrt(MSE * (1 / n + np.mean(x) ** 2 / np.sum((x - np.mean(x)) ** 2)))
    t_intercept = intercept / SE_intercept
    if alternative_intercept=='two-sided':
        p_intercept = 2 * (1 - t.cdf(t_intercept, df=dof))
    elif alternative_intercept=='greater':
        p_intercept = 1 - t.cdf(t_intercept, df=dof)
    elif alternative_intercept=='less':
        p_intercept = t.cdf(t_intercept, df=dof)

    ax.plot(x_fit, y_fit, color='k', linestyle='--', label='Fit')
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color='k', alpha=.2, label='95% CI', lw=0)
    ax.scatter(x, y, color='k')

    ax.axvline(0, lw=.8, color='k', ls=':')
    ax.axhline(0, lw=.8, color='k', ls=':')

    ax.spines[['left', 'top', 'right']].set_visible(False) if panel>0 else ax.spines[['top', 'right']].set_visible(False)

    x_text = 0
    y_text = ax.get_ylim()[1]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    print(f'slope: {slope}, p = {p_slope:.3f}')
    print(f'intercept: {intercept}, p_intercept = {p_intercept:.3f}')
    print(f'R2 = {R2:.3f}')

    return fig, axs


def plot_interaction(fig, ax, interaction, x='roi', color='cyan', width=.8, alternative='two-sided'):
    interaction = pd.DataFrame(interaction)
    sb.barplot(data=interaction, x=x, y='interaction', ax=ax, errorbar='se', width=width, color=color)
    ax.set_xlabel('')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_ylabel('correlation')
    return fig, ax

# def plot_comp_bayes(fig, axs, panel, c_bf, components=['finger', 'cue', 'surprise'],
#                   palette=['#D4AF37', 'red', 'magenta']):
#
#     ax = axs[panel]
#     plt.sca(ax)
#     bars = pcm.vis.plot_component(c_bf, type='bf', palette=palette, errorbar='se', width=1)
#     ax.spines[['top', 'right', 'bottom']].set_visible(False)
#     if panel > 0:
#         ax.spines['left'].set_visible(False)
#         ax.tick_params(axis='y', left=False)
#     else:
#         ax.spines['left'].set_position(('data', -1))
#     ax.set_xlabel('')
#     ax.tick_params(axis='x', bottom=False, labelbottom=False)
#
#     return fig, axs

def plot_pcm_corr(fig, axs, panel, Mflex, theta, theta_g, r_bootstrap=None, alpha=0.025):
    ax = axs[panel]

    N = theta.shape[1]

    sigma2_1 = np.exp(theta[0])
    sigma2_2 = np.exp(theta[1])
    r_indiv = Mflex.get_correlation(theta)
    sigma2_e = np.exp(theta[3])
    SNR = np.sqrt(sigma2_1 * sigma2_2) / sigma2_e
    ax.scatter(SNR, r_indiv, color='k')

    theta_g, _ = pcm.group_to_individ_param(theta_g, Mflex, N)
    r_group = Mflex.get_correlation(theta_g)
    ax.axhline(r_group[0], color='r', linestyle='--')
    ax.axhline(0, color='k', linestyle='-', lw=.8)

    ax.set_ylim(-1.2, 1.2)

    ax.spines[['top', 'right', 'left']].set_visible(False)

    if panel == 0:
        ax.spines[['left']].set_visible(True)
    else:
        ax.tick_params(left=False)

    #if r_bootstrap is not None:
    #    (ci_lo, ci_hi), _, _ = bootstrap_summary(r_bootstrap, alpha=alpha)
    #    print(f"group estimate:{r_group[0]} central {(1 - 2 * alpha) * 100:.0f}% CI for r: [{ci_lo:.3f}, {ci_hi:.3f}]")
    #    ax.axhspan(ci_lo, ci_hi, lw=0, color='lightgrey', zorder=0)

    sigma_g_2_1 = np.exp(theta_g[0, 0])
    sigma_g_2_2 = np.exp(theta_g[1, 0])
    sigma_g_2_e = np.exp(theta_g[-1])

    sdts = np.sqrt(sigma_g_2_1 * sigma_g_2_2)
    if sdts < 1e-4 * np.sqrt(sigma2_e).max():
        print(f'Geom mean of variances: {sdts}, SD_err={np.sqrt(sigma2_e).max()}; '
              f'No reliable signal, discarding bootstrap resample')

    return fig, axs

def add_sig_1samp(fig, axs, panel, X, rotation=0, fontsize=10, type='ttest', alternative='greater'):
    ax = axs[panel]
    for i, col in enumerate(X.columns):
        vals = X[col].dropna().values
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n < 2:
            continue

        if type=='ttest':
            tval, p_val_1samp = ttest_1samp(vals, 0, alternative=alternative)
            print(f'{col}, tval({alternative})={tval}, pval={p_val_1samp}')
        elif type=='permutation':
            res = permutation_test(
                (vals,), np.mean,
                permutation_type='samples',
                vectorized=False,
                alternative=alternative,
                n_resamples=5000,
                random_state=0
            )
            p_val_1samp = res.pvalue
            print(f'{col}, permutation test ({alternative}), pval={p_val_1samp}')
        elif type=='sign':
            n_pos = np.sum(vals > 0)
            n = np.sum(vals != 0)
            res = binomtest(n_pos, n, 0.5, alternative=alternative)
            p_val_1samp = res.pvalue
            print(f'{col}, sign test ({alternative}), pval={p_val_1samp}')

        if p_val_1samp < 0.001:
            stars = '***'
        elif p_val_1samp < 0.01:
            stars = '**'
        elif p_val_1samp < 0.05:
            stars = '*'
        else:
            stars = None

        if stars:
            ax.text(i, 0, stars, ha='center', va='top', fontsize=fontsize, rotation=rotation)

    return fig, axs

def add_sig_to_bars(fig, ax, data, y=None, x=None, alternative='two-sided', fontsize=10):
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    offset = .05 * np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])

    xs = data[x].unique()
    for xi, Xx in enumerate(xs):
        datai = data[data[x] == Xx][y]
        tval, pval = ttest_1samp(datai, 0, alternative=alternative)
        print(f"{Xx}, tval={tval}, pval={pval}")
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = None
        if stars:
            mean = datai.mean()
            se = datai.std() / np.sqrt(datai.size)
            y_max = mean + se * np.sign(mean)
            ax.text(xi, y_max + offset * np.sign(mean), f'{stars}', ha='center', va='top', fontsize=fontsize)

    return fig, ax

def add_sig_var_expl(fig, axs, panel, param_c, components, fontsize=10):
    var_expl = np.exp(param_c)
    ax = axs[panel]

    offset = .1 * ax.get_ylim()[1]

    df = pd.DataFrame(var_expl.T, columns=components)

    y_max = None

    pairs = list(combinations(components, 2))
    for pair in pairs:
        a = df[pair[0]]
        b = df[pair[1]]

        i = components.index(pair[0])
        j = components.index(pair[1])

        tval, p_val_paired = ttest_rel(a, b)

        print(f'{pair[0]} vs. {pair[1]}, tval={tval}, pval={p_val_paired}')

        # Significance stars
        if p_val_paired < 0.001:
            stars = '***'
        elif p_val_paired < 0.01:
            stars = '**'
        elif p_val_paired < 0.05:
            stars = '*'
        else:
            stars = None

        if stars:
            if y_max is None:
                y_max = var_expl.mean(axis=1).max()
                y_argmax = var_expl.mean(axis=1).argmax()
                se = var_expl[y_argmax].std() / np.sqrt(var_expl.shape[1])
                y_max += se
                y_max0 = y_max
            else:
                y_max = y_max + offset #* y_max0

            # Compute x positions of the bars for the two models
            center = (i+j)/2
            x1 = center - .4 * np.abs(j-i)  # component
            x2 = center + .4  * np.abs(j-i)# feature

            # Draw bar and stars
            ax.plot([x1, x2], [y_max + offset , y_max + offset], lw=1.5, c='k')
            ax.text(center, y_max + .8 * offset, stars, ha='center', va='bottom', fontsize=fontsize)

    return fig, axs


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

# def pcm_spike(fig, axs, roi, epoch, monkey, rec):
#     # temporal landmarks
#     cuePre = 0
#     cueIdx = 20
#     cuePost = 84
#     pertIdx = 114
#
#     xtick = cueIdx if epoch == 'plan' else pertIdx
#     xticklabel = 'Cue' if epoch == 'plan' else 'Pert'
#     xlim = [cuePre, cuePost] if epoch == 'plan' else [cuePost, 154]
#     rangePre = np.arange(cuePre, cueIdx) if epoch == 'plan' else np.arange(cuePost, pertIdx)
#     rangePost = np.arange(cueIdx, cuePost) if epoch == 'plan' else np.arange(pertIdx, 145)
#
#     # load data
#     G_obs, var_expl, cov, stds = [], [], [], []
#     for r in rec:
#         path = os.path.join('/cifs/pruszynski/Marco/SensoriMotorPrediction', 'spikes', gl.pcmDir)
#         G_obs.append(np.load(os.path.join(path, f'G_obs.spike.{monkey}.{roi}.aligned.{epoch}-{r}.npy')))
#
#         # calc variance
#         theta_c = np.load(os.path.join(path, f'theta_in.spike.component.{monkey}.{roi}.aligned.{epoch}-{r}.npy'))
#         n_param_c = theta_c.shape[-1] - 1
#         var_expl.append(np.sqrt(np.exp(theta_c[..., :n_param_c])))
#
#         if epoch == 'exec':
#             theta_f = np.load(os.path.join(path, f'theta_in.spike.feature.{monkey}.{roi}.aligned.{epoch}-{r}.npy'))
#             cov.append(theta_f[:, 1] * theta_f[:, 2])
#             stds.append(np.sqrt((theta_f[:, 0]**2 + theta_f[:, 1]**2) * theta_f[:, 2]**2))
#
#     var_expl = np.array(var_expl).mean(axis=0)
#     G_obs = np.array(G_obs).mean(axis=0)
#     cov = np.array(cov)
#     stds = np.array(stds)
#     correlation = (cov / stds).mean(axis=0)
#
#     color = ['red', 'blue'] if epoch == 'plan' else ['#FFCC33', 'red', 'blue', 'magenta', 'k']
#     linestyle = ['-', '-', '-', '-', '--']
#     components = ['cue', 'uncertainty'] if epoch == 'plan' else ['direction', 'cue', 'uncertainty', 'surprise', 'noise ceiling']
#     tr = np.sqrt(np.trace(G_obs, axis1=1, axis2=2))
#
#     ax = plot_theta_lfp(0, axs, var_expl, color=color)
#
#     # covariance
#     if epoch == 'exec':  # only execution models contain feature model
#         inset = ax.inset_axes([0.12, 0.6, 0.3, 0.5], transform=ax.transAxes)
#         inset.plot(correlation, color='cyan')
#         inset.axvline(xtick, color='k', lw=.8)
#         inset.axhline(0, color='k', lw=.8)
#         inset.set_xticks([xtick])
#         inset.set_ylim((-1, 1))
#         # inset.spines[['bottom', 'right', 'top']].set_visible(False)
#         inset.spines['left'].set_bounds(-1, 1)
#         inset.set_xticklabels([xticklabel])
#         inset.set_xlim(xlim)
#         inset.set_title('correlation', fontsize=10)
#
#     ax.plot(tr, ls='--', color='k')
#
#     lines = [plt.Line2D([], [], color=c, linestyle=ls) for c, ls in zip(color, linestyle)]
#     fig.legend(lines, components, loc='center left', fontsize=9, ncol=1, bbox_to_anchor=(1, .5), frameon=False)
#
#     ax.axvline(xtick, color='k', lw=.8)
#     ax.set_xticks([xtick])
#     ax.set_xticklabels([xticklabel])
#     ax = plot_theta_lfp_mean(0, 1, axs, var_expl[rangePre].mean(axis=0),
#                              var_expl[rangePost].mean(axis=0), color=color)
#     axs[0].set_xlim(xlim)
#     fig.supylabel('variance (a.u.)', fontsize='medium')
#     fig.suptitle(f'Variance explained by component model (monkey {monkey[0]}, {roi})')
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=.1)
#
#     return fig, axs


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


def plot_flatmap_cerebellar_activation(img, vmin=-20, vmax=20, thresh=1, xlim=None, ylim=None, figsize=(5, 6), cmap='bwr',
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
    mask = (darray_avg > thresh) | (darray_avg < -thresh)
    darray_avg[~mask] = np.nan
    ax = flatmap.plot(data=darray_avg,
                      cmap=cmap,
                      cscale=[vmin, vmax],
                      new_figure=False,
                      colorbar=False,
                      render='matplotlib')

    im = np.array([x in exec_col_names for x in col_names])
    darray_avg = np.array(darray[:, im]).mean(axis=1)
    plt.sca(axs[1])
    mask = (darray_avg > thresh) | (darray_avg < -thresh)
    darray_avg[~mask] = np.nan
    ax = flatmap.plot(data=darray_avg,
                      cmap=cmap,
                      cscale=[vmin, vmax],
                      new_figure=False,
                      colorbar=False,
                      render='matplotlib')

    # # make colorbar
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # sm = ScalarMappable(norm=norm, cmap='bwr')
    # cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.01)
    # cbar.set_label('activation vs. baseline (a.u.)')

    # cosmetic
    axs[0].set_title('Planning')
    axs[1].set_title('Execution')

    return fig, axs


def add_colorbar(fig, ax, cax, orientation='horizontal', fraction=.02, label='', anchor=None, pad=.1):
    cbar = fig.colorbar(cax, ax=ax, orientation=orientation, fraction=fraction, anchor=anchor)
    cbar.set_label(label)


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


def main(args, **kwargs):
    path_fig = 'figures'
    if args.what=='force_aligned':
        experiment = 'smp2'
        npz = np.load(os.path.join('data', gl.behavDir, 'force.segmented.avg.npz'), allow_pickle=True)
        force = npz['data_array']
        descr = pd.DataFrame(npz['descriptor'].item())
        force = force[descr.GoNogo == 'go']
        descr = descr[descr.GoNogo == 'go']
        fig, axs = plt.subplots(1, 2, figsize=(4, 5), sharey=True, sharex=True, constrained_layout=True)
        fig, axs = plot_aligned_force(fig, axs, force, descr)
        fig.tight_layout()
        plt.savefig(os.path.join(path_fig, 'force_response.svg'))
        plt.show()
    if args.what=='force_binned':
        filepath = os.path.join('data', gl.behavDir, f'{args.experiment}_force_single_trial.tsv')
        dat = pd.read_csv(filepath, sep='\t', )
        dat = dat.groupby(['sn', 'cue', 'stimFinger', 'GoNogo', ]).mean(numeric_only=True).reset_index()
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(2, 2))
        fig, axs = plot_binned_behaviour(fig, axs, dat, y=['index1', 'ring1'], finger=('index', 'ring'))
        axs[0].spines[['left']].set_bounds(0, 10)
        axs[0].set_ylabel('Force (N)')
        fig.suptitle('Force response')
        fig.subplots_adjust(left=.25)
        plt.savefig(os.path.join(path_fig, 'force_binned.svg'))
        plt.show()
    if args.what=='dev_aligned':
        cut = float(kwargs.get('cut', .05))
        startSample = int(gl.prestim * gl.fsample_mov + cut * gl.fsample_mov)
        npz = np.load(os.path.join('data', gl.behavDir,  'force.segmented.avg.npz'), allow_pickle=True)
        force = npz['data_array']
        descr = pd.DataFrame(npz['descriptor'].item())
        force = force[descr.GoNogo == 'go', startSample:]
        force = force - force[:, 0][:, None]
        descr = descr[descr.GoNogo == 'go']
        fig, axs = plt.subplots(1, 2, figsize=(5, 2.75), sharey=True, sharex=True, constrained_layout=True)
        fig, axs = plot_aligned_deviation(fig, axs, force, descr)
        fig.tight_layout()
        plt.savefig(os.path.join(path_fig, 'dev_aligned.svg'))
        plt.show()
    if args.what=='dev_binned':
        filepath = os.path.join('data', gl.behavDir, f'{args.experiment}_force_single_trial.tsv')
        dat = pd.read_csv(filepath, sep='\t', )
        dat = dat.groupby(['sn', 'cue', 'stimFinger', 'GoNogo', ]).mean(numeric_only=True).reset_index()
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(2, 2))
        fig, axs = plot_binned_behaviour(fig, axs, dat, y=['MD', 'MD'], finger=('index', 'ring'))
        axs[0].spines[['left']].set_bounds(0, 1)
        axs[0].set_ylabel('Mean deviation (N)')
        fig.subplots_adjust(left=.25)
        plt.savefig(os.path.join(path_fig, 'dev_binned.svg'))
        plt.show()
    if args.what=='BOLD':
        fig, axs = plt.subplots(1, len(args.rois), sharey=True, sharex=True, figsize=(8, 3))

        tAx = np.arange(-1, 13)

        for r, roi in enumerate(args.rois):
            ax = axs[r]
            go = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'hrf_go.{args.H}.{roi}.npy'))
            nogo = np.load(os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'hrf_nogo.{args.H}.{roi}.npy'))
            y_adj_go = go.mean(axis=0)
            y_adj_go_err = go.std(axis=0) / np.sqrt(go.shape[0])
            y_adj_nogo = nogo.mean(axis=0)
            y_adj_nogo_err = nogo.std(axis=0) / np.sqrt(nogo.shape[0])

            ax.plot(tAx, y_adj_go, color='#FF7F7F', label='go', ls='-')
            ax.plot(tAx, y_adj_nogo, color='#000080', label='nogo', ls='-')
            ax.fill_between(tAx, y_adj_go - y_adj_go_err, y_adj_go + y_adj_go_err,
                            color='#FF7F7F', alpha=.2, lw=0)
            ax.fill_between(tAx, y_adj_nogo - y_adj_nogo_err, y_adj_nogo + y_adj_nogo_err,
                            color='#000080', alpha=.2, lw=0)

            ax.axvline(0, color='k', ls='-', lw=.8)
            ax.axvline(2.5, color='k', ls='--', lw=.8)
            ax.axhline(0, color='k', ls='-', lw=.8)

            ax.spines[['left', 'right', 'top']].set_visible(False)
            ax.spines['bottom'].set_bounds(0, 12)
            ax.set_xticks([0, 6, 12])
            if r == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_bounds(-2, 2)
            else:
                ax.tick_params('y', width=0)

            ax.set_xlim((-1, 12))
            ax.set_ylim((-3, 3.6))

            ax.set_title(roi)

        ax.legend(frameon=False)  # , bbox_to_anchor=(1, -.1), loc='upper right', )

        fig.supylabel('activation (a.u.)', fontsize=10)
        fig.supxlabel('time relative to cue presentation onset (s)', fontsize=10)
        fig.suptitle(f'Average BOLD timeseries')
        fig.tight_layout()

        plt.savefig(os.path.join(path_fig, 'BOLD.svg'))
        plt.show()
    if args.what=='surf_labels':
        view = kwargs.get('view', 'lateral')
        cmap = kwargs.get('cmap', 'Greys')
        rois = ['M1', 'S1', 'PMd', 'PMv', 'SMA', 'V1', 'SPLa', 'SPLp']
        gifti = nb.load(os.path.join(gl.atlasDir, f'ROI.32k.{args.H}.label.gii'))
        gifti = nt.make_label_gifti(gifti.darrays[0].data, anatomical_struct='CortexLeft', label_names=[''] + rois)
        labels = nt.get_gifti_labels(gifti)
        n_lab = gifti.darrays[0].data
        gifti.darrays[0].data = np.array([x if gifti.labeltable.labels_as_dict[x] in args.rois else 0 for x in n_lab])
        fig, ax = plt.subplots()
        fig, ax = plot_surf_label(fig, ax, gifti, args.H, cmap=cmap, view=view)
        fig.savefig(os.path.join(path_fig, f'surf_label.{view}.svg'))
        plt.show()
    if args.what=='surf_activation':
        thresh = 1
        vmin, vmax = tuple(map(float, kwargs.get('vlim', (-15, 15))))
        cold_colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]  # RGB tuples
        cold = LinearSegmentedColormap.from_list('cold_custom', cold_colors, N=128)
        hot = plt.cm.hot(np.linspace(0, 1, 128))
        cold_vals = cold(np.linspace(1, 0, 128))  # reversed
        combined = np.vstack((cold_vals, hot))
        cmap = LinearSegmentedColormap.from_list('coldhot', combined)
        cifti = nb.load(
            os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'glm{args.glm}.con.plan-exec.smooth.dscalar.nii'))
        fig, ax = plt.subplots()
        if args.epoch == 'plan':
            col=0
            title='response preparation'
        elif args.epoch == 'exec':
            col=1
            title='response execution'
        fig, ax = plot_surf(fig, ax, cifti, args.H, cmap=cmap, col=col, thresh=thresh, vmin=vmin, vmax=vmax)

        cold_half = LinearSegmentedColormap.from_list("cold_half", combined[:128])
        hot_half = LinearSegmentedColormap.from_list("hot_half", combined[128:])

        neg_norm = Normalize(vmin=vmin, vmax=-thresh)
        sm_neg = ScalarMappable(norm=neg_norm, cmap=cold_half)
        cax_neg = fig.add_axes([0.2, 0.1, 0.25, 0.025])
        cbar_neg = fig.colorbar(sm_neg, ax=ax, cax=cax_neg, fraction=0.03, pad=0.02, orientation='horizontal')
        cbar_neg.set_ticks([vmin, -thresh])
        fig.supxlabel('activation (a.u.)', y=.001)

        pos_norm = Normalize(vmin=thresh, vmax=vmax)
        sm_pos = ScalarMappable(norm=pos_norm, cmap=hot_half)
        cax_pos = fig.add_axes([0.55, 0.1, 0.25, 0.025])
        cbar_pos = fig.colorbar(sm_pos, ax=ax, cax=cax_pos, fraction=0.03, pad=0.08, orientation='horizontal')
        cbar_pos.set_ticks([thresh, vmax])
        plt.subplots_adjust(left=0, right=1, bottom=.2, top=.9)

        fig.savefig(os.path.join(path_fig, f'activation.{args.H}.{args.epoch}.svg'))

        plt.show()
    if args.what=='dissimilarities_cortical':
        figsize = tuple(map(float, kwargs.get('figsize', (8, 2))))
        vmin, vmax = tuple(map(float, kwargs.get('vlim', (-.01, .5))))
        if args.epoch == 'plan':
            ticklabels = list(gl.regressor_mapping.keys())[:5]
            suptitle = 'Crossnobis dissimilarities during response preparation'
        elif args.epoch == 'exec':
            ticklabels = list(gl.regressor_mapping.keys())[5:13]
            suptitle = 'Crossnobis dissimilarities during response execution'
        fig, axs = plt.subplots(1, len(args.rois), figsize=figsize, sharex=True, sharey=True,)
        for r, roi in enumerate(args.rois):
            G = np.load(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.{args.epoch}.glm{args.glm}.{args.H}.{roi}.npy'))
            D = pcm.G_to_dist(G)
            fig, axs = plot_dissimilarities(fig, axs, r, D, ticklabels, vmin=vmin, vmax=vmax, sqrt=True)
            axs[r].set_title(roi)
        cax = axs[-1].get_images()[0]
        cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=.008)
        cbar.set_label('dissimilarity (a.u.)')
        fig.suptitle(suptitle)
        plt.savefig(os.path.join(path_fig, f'dissimilarity.{args.epoch}.{args.H}.svg'))
        plt.show()
    # if args.what=='pcm_models':
    #     vmin, vmax = tuple(map(float, kwargs.get('vlim', (0, 1))))
    #     figsize = tuple(map(float, kwargs.get('figsize', (2.66, 2))))
    #     components = kwargs.get('components', ['sensory input', 'expectation', 'surprise'])
    #     Mc, _ = find_model(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{args.epoch}.p'), 'component')
    #     G_mod = Mc.Gc
    #     if args.epoch == 'plan':
    #         ticklabels = list(gl.regressor_mapping.keys())[:5]
    #         suptitle = 'Crossnobis dissimilarities during response preparation'
    #     elif args.epoch == 'exec':
    #         ticklabels = list(gl.regressor_mapping.keys())[5:13]
    #         suptitle = 'Crossnobis dissimilarities during response execution'
    #         Mf, _ = find_model(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.{args.epoch}.p'), 'feature')
    #         Ac_pos = (Mf.Ac * np.array([1, 1, 1, ])[:, None, None]).sum(axis=0)
    #         Gx_pos = Ac_pos @ Ac_pos.T
    #         Ac_neg = (Mf.Ac * np.array([1, 1, -1, ])[:, None, None]).sum(axis=0)
    #         Gx_neg = Ac_neg @ Ac_neg.T
    #         G_mod = np.r_[G_mod, Gx_pos[None, :, :], Gx_neg[None, :, :]]
    #     D_mod = pcm.G_to_dist(G_mod)
    #     fig, axs = plt.subplots(1, G_mod.shape[0], figsize=figsize, sharex=True, sharey=True, )
    #     for m, D in enumerate(D_mod):
    #         fig, axs = plot_dissimilarities(fig, axs, m, D[None, :, :], ticklabels, vmin=vmin, vmax=vmax, sqrt=True)
    #         axs[m].set_title(components[m])
    #     fig.suptitle('Representational models')
    #     fig.savefig(os.path.join(path_fig, f'models.{args.epoch}.svg'))
    #     plt.show()
    if args.what=='weight_cortical':
        figsize = tuple(map(float, kwargs.get('figsize', (4, 3))))
        df = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'component_model.BOLD.tsv'), sep='\t')
        df = df[(df['epoch'] == args.epoch) & (df['Hem'] == args.H)]
        fig, ax = plt.subplots(figsize=figsize)
        palette = ['red', 'blue'] if args.epoch=='plan' else ['#FFFF00', 'red', 'purple']
        df['norm_weight'] = np.sqrt(df['weight'] / df['noise'])
        sb.barplot(ax=ax, data=df, x='roi', y='norm_weight', hue='component', palette=palette, errorbar='se')
        sb.stripplot(ax=ax, data=df, x='roi', y='norm_weight', hue='component', size=2, color='black', jitter=.1,
                     dodge=True, legend=False)
        ax.set_title('Standardised representation weight')
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        if args.epoch=='plan':
            ax.set_ylim(0, .35)
        elif args.epoch=='exec':
            ax.set_ylim(0, .8)
        ax.set_xlabel('')
        ax.set_ylabel('weight')
        ax.tick_params(axis='x', bottom=False, )
        ax.legend(title=None, frameon=False)
        fig.savefig(os.path.join(path_fig, f'weight.cortical.{args.H}.{args.epoch}.svg'))
        plt.show()
    if args.what=='%weight_cortical':
        df = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'component_model.BOLD.tsv'), sep='\t')
        df = df[(df['epoch'] == args.epoch) & (df['Hem'] == args.H)]
        df_cluster = df.groupby(['cluster', 'participant_id', 'component']).mean(numeric_only=True).reset_index()
        if args.epoch=='plan':
            df_exp = df_cluster[df_cluster['component'] == 'expectation']
        elif args.epoch=='exec':
            df_cluster = df_cluster[(df_cluster['component'] == 'sensory input') | (df_cluster['component'] == 'surprise')]
            df_exp = df_cluster[df_cluster['component'] == 'sensory input']
        df_sum = df_cluster.groupby(['cluster', 'participant_id']).sum(numeric_only=True).reset_index()
        df_ratio = df_exp.copy()
        df_ratio['ratio'] = df_exp['weight'].to_numpy() / df_sum['weight'].to_numpy()
        df_ratio['cluster'] = pd.Categorical(df_ratio['cluster'], categories=['premotor-parietal', 'M1-S1'], ordered=True)

        fig, ax = plt.subplots(figsize=(.8, 2))

        sb.barplot(data=df_ratio, x='cluster', y='ratio', errorbar='se', color='grey')
        sb.stripplot(data=df_ratio, x='cluster', y='ratio', size=2, color='black', jitter=.1, dodge=True, legend=False,
                     order=['premotor-parietal', 'M1-S1'])
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
        ax.tick_params(axis='x', bottom=False, )
        ax.set_xlabel('')
        ax.set_yticks((0, 1))
        ax.set_xticklabels(['premotor-parietal', 'M1-S1'], rotation=90)
        if args.epoch == 'plan':
            ax.set_ylabel('% expectation')
        elif args.epoch == 'exec':
            ax.set_ylabel('% sensory input')
        fig.savefig(os.path.join(path_fig, f'%weight.expectation.{args.H}.{args.epoch}.svg'))
        plt.show()
    if args.what=='searchlight':
        comp = kwargs.get('comp', 'finger') if args.epoch=='exec' else None
        if args.H=='L':
            mclip = .2
            threshold = .05 / mclip if args.epoch=='plan' else .1 / mclip
        else:
            mclip = .1
            threshold = .02 / mclip if args.epoch == 'plan' else .04 / mclip
        scaler = MinMaxScaler()
        gifti = nb.load(os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'searchlight.var_expl.{args.epoch}.{args.H}.func.gii'))
        data = nt.get_gifti_data_matrix(gifti)
        data = data[:, [0, -1]] if args.epoch == 'exec' else data
        raw_min, raw_max = threshold * mclip * np.nanmax(data), np.nanmax(data)
        data = data / raw_max #scaler.fit_transform(data)
        data = np.clip(data / mclip, 0, 1)
        sulc = nt.get_gifti_data_matrix(
            nb.load(f'/cifs/diedrichsen/data/Atlas_templates/fs_LR_32/fs_LR.32k.LR.sulc.dscalar.gii'))
        sulc = sulc[:len(data)]
        sulc_norm = MinMaxScaler((0.3, 0.7)).fit_transform(sulc.reshape(-1, 1)).flatten()
        rgba = np.zeros((len(sulc_norm), 4))
        rgba[:, 0] = sulc_norm  # red = grey
        rgba[:, 1] = sulc_norm  # green = grey
        rgba[:, 2] = sulc_norm  # blue = grey
        rgba[:, 3] = 1 # opaque background
        mask = (data[:, 0] >= threshold) | (data[:, 1] >= threshold)
        mask1 = data[:, 0] >= threshold
        mask2 = data[:, 1] >= threshold
        if args.epoch == 'plan': # red/blue
            rgba[mask, 0] = data[mask, 0]  # red
            rgba[mask, 1] = 0  # green stays off for 2-color blend
            rgba[mask, 2] = data[mask, 1]  # blue
        elif args.epoch == 'exec': #yellow/cyan
            mask = mask1 if comp=='finger' else mask2
            rgba[mask, 0] = data[mask, 0] if comp=='finger' else data[mask, 1] #data[mask, 0] * (1 - data[mask, 1])
            rgba[mask, 1] = data[mask, 0] if comp=='finger' else 0 #data[mask, 1] #data[mask, 0] + data[mask, 1] - (data[mask, 1] * data[mask, 0])
            rgba[mask, 2] = data[mask, 1] if comp=='surprise' else 0 #data[mask, 1] * (1 - data[mask, 0])
            rgba[:, 1] = np.clip(rgba[:, 1], 0, 1) # clamp green if it exceeds 1
        fig, ax = plt.subplots()
        fig, ax = plot_surf(fig, ax, rgba, args.H, cmap=None, vmin=None, vmax=None, overlay='rgb')
        blue_half = LinearSegmentedColormap.from_list(
            "left_half", ["black", "blue" if args.epoch=='plan' else "yellow"])
        neg_norm = Normalize(vmin=0, vmax=raw_max* mclip)
        sm_neg = ScalarMappable(norm=neg_norm, cmap=blue_half)
        red_half = LinearSegmentedColormap.from_list(
            "right_half", ["black", "red" if args.epoch=='plan' else "purple"])
        pos_norm = Normalize(vmin=0, vmax=raw_max* mclip)
        sm_pos = ScalarMappable(norm=pos_norm, cmap=red_half)
        cax_neg = fig.add_axes([0.20, 0.10, 0.28, 0.025])  # left (blue)
        cax_pos = fig.add_axes([0.52, 0.10, 0.28, 0.025])  # right (red)
        cbar_neg = fig.colorbar(sm_neg, cax=cax_neg, orientation='horizontal')
        cbar_pos = fig.colorbar(sm_pos, cax=cax_pos, orientation='horizontal')
        #ticks = np.linspace(0, .05 if args.epoch=='plan' else .3, 3)
        # cbar_neg.set_ticks(ticks)
        # cbar_pos.set_ticks(ticks)
        cbar_neg.set_label('weight')
        if args.epoch=='plan':
            fig.savefig(os.path.join(path_fig, f'searchlight.var_expl.{args.epoch}.glm{args.glm}.{args.H}.svg'), dpi=300)
        elif args.epoch=='exec':
            fig.savefig(os.path.join(path_fig, f'searchlight.var_expl.{args.epoch}.{comp}.glm{args.glm}.{args.H}.svg'),  dpi=300)
        plt.show()
    if args.what=='dissimilarities_force':
        fig, ax = plt.subplots(figsize=(3, 2.5))
        vmin, vmax = -.00001, .0004
        G = np.load(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.force.plan.npy'))
        Df = pcm.G_to_dist(G)
        fig, ax = plot_dissimilarities(fig, ax, None, Df, list(gl.regressor_mapping.keys())[:5], vmin=vmin, vmax=vmax,
                                       sqrt=False, source='force')
        cax = ax.get_images()[0]
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=.04)
        cbar.set_label('dissimilarity (a.u.)')
        fig.suptitle('Force dissimilarities\nduring preparation')
        fig.tight_layout()
        fig.savefig(os.path.join(path_fig, 'finger_pre.svg'))
        plt.show()
    if args.what=='force_vs_expectation':
        rois = ['M1', 'S1']
        diff = np.zeros(len(args.sns))
        pcm_path = os.path.join(gl.baseDir, args.experiment, gl.pcmDir)
        G = np.load(os.path.join(pcm_path, f'G_obs.force.plan.npy'))
        Df = pcm.G_to_dist(G)

        fig, axs = plt.subplots(1, len(rois), sharex=True, sharey=True, figsize=(4, 3))

        for r, roi in enumerate(rois):
            G = np.load(os.path.join(pcm_path, f'G_obs.plan.glm{args.glm}.{args.H}.{roi}.npy'))
            D = pcm.G_to_dist(G)
            x = Df.mean(axis=(1, 2))
            y = D.mean(axis=(1, 2))
            fig, axs = plot_correlation(fig, axs, r, x, y, alternative_slope='two-sided', alternative_intercept='greater')
            axs[r].set_title(roi)

        for ax in axs:
            yticks = ax.get_yticks()
            xticks = ax.get_xticks()
            ax.spines['left'].set_bounds(yticks[1], yticks[-2])
            ax.spines['bottom'].set_bounds(xticks[1], xticks[-2])

        axs[0].set_ylabel('BOLD dissimilarity (a.u.)', fontsize=10)
        axs[1].tick_params('y', left=False)
        fig.supxlabel('force dissimilarity (a.u.)', fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(path_fig, 'force_vs_cue.svg'))
        plt.show()
    if args.what=='corr_cortical':
        corr = kwargs.get('corr', 'plan-exec')
        df_corr = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, 'correlations.BOLD.tsv'), sep='\t')

        fig, axs = plt.subplots(1, len(args.rois), sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)

        for r, roi in enumerate(args.rois):
            ax = axs[r]
            df_corr_tmp = df_corr[(df_corr['roi'] == roi) & (df_corr['Hem'] == args.H) & (df_corr['corr'] == corr)]
            r_indiv = df_corr_tmp.r_indiv.to_numpy()
            SNR = df_corr_tmp.SNR.to_numpy()
            r_group = df_corr_tmp.r_group.to_numpy()[0]
            ci_lo, ci_hi = df_corr_tmp.ci_lo.to_numpy()[0], df_corr_tmp.ci_hi.to_numpy()[0]
            ax.scatter(SNR, r_indiv, color='k')
            ax.axhline(r_group, color='r', linestyle='--')
            ax.axhline(0, color='k', linestyle='-', lw=.8)
            ax.axhspan(ci_lo, ci_hi, lw=0, color='lightgrey', zorder=0)
            ax.set_ylim(-1.2, 1.2)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            if r == 0:
                ax.spines[['left']].set_visible(True)
                ax.spines['left'].set_bounds(-1, 1)
            else:
                ax.tick_params(left=False)
            ax.set_title(roi)

        fig.supxlabel('SNR')
        axs[0].set_ylabel('correlation')

        if corr == 'plan-exec':
            fig.suptitle(f'Preparation-execution correlation')
        elif corr == 'cue-finger':
            fig.suptitle(f'Cue-finger correlation')

        plt.show()
    if args.what=='corr_emg':
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        figsize = tuple(map(float, kwargs.get('figsize', (4, 2))))
        corr = kwargs.get('corr', 'plan-exec')
        fig, axs = plt.subplots(1, len(args.epochs), sharex=True, sharey=True, figsize=figsize, constrained_layout=True)
        for r, epoch in enumerate(args.epochs):
            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_in.corr_{corr}.emg.{epoch}.p'), 'rb')
            theta = pickle.load(f)[0]
            r_bootstrap = np.load(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'r_bootstrap.corr_{corr}.emg.{epoch}.npy'))
            f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'theta_gr.corr_{corr}.emg.{epoch}.p'), 'rb')
            theta_g = pickle.load(f)[0]
            fig, axs = plot_pcm_corr(fig, axs, r, Mflex, theta, theta_g, r_bootstrap)
            # axs[r].set_xlim((-.025, .3))
            xticks = 0, 6#np.array(axs[r].get_xlim()).clip(0, np.inf)
            axs[r].set_xticks(xticks)
            axs[r].set_xticklabels([f'{xticks[0]:.02f}', f'{xticks[1]:.02f}'])
            axs[r].spines[['bottom']].set_bounds(xticks[0], xticks[1])
            axs[r].spines[['left']].set_bounds(-1, 1)
            axs[r].set_title(epoch)
        fig.supxlabel('SNR')
        axs[0].set_ylabel('correlation')
        axs[0].set_xlim(-.1, 6.1)
        if corr=='plan-exec':
            fig.suptitle(f'Preparation-execution\ncorrelation')
        elif corr=='cue-finger':
            fig.suptitle(f'Cue-finger correlation')
        fig.savefig(os.path.join(path_fig, f'corr_{corr}.emg.svg'))
        plt.show()
    if args.what=='corr_lfp':
        baseDir = '/cifs/pruszynski/Marco/SensoriMotorPrediction'
        f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'M.plan-exec.p'), "rb")
        Mflex = pickle.load(f)
        figsize = tuple(map(float, kwargs.get('figsize', (4, 2))))
        corr = kwargs.get('corr', 'plan-exec')
        dtype = ['lfp', 'spk']
        title = ['LFPs 8-13Hz', '13-25Hz', '25-100Hz','Spiking activity']
        freqs = ['alpha', 'beta', 'gamma']
        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=figsize)

        dt = 'lfp'
        for fr, freq in enumerate(freqs):
            print(fr)
            fname = f'{dt}.corr_{corr}.M1-S1.{freq}'
            f = open(os.path.join(baseDir, gl.pcmDir, f'theta_in.{fname}.p'), 'rb')
            theta = pickle.load(f)[0]
            r_bootstrap = np.load(os.path.join(baseDir, gl.pcmDir, f'r_bootstrap.{fname}.npy'))
            f = open(os.path.join(baseDir, gl.pcmDir, f'theta_gr.{fname}.p'), 'rb')
            theta_g = pickle.load(f)[0]
            fig, axs = plot_pcm_corr(fig, axs, fr, Mflex, theta, theta_g, r_bootstrap)
            axs[fr].set_title(title[fr])
            xticks = 0, 1.5#np.array(axs[fr].get_xlim()).clip(0, np.inf)
            axs[fr].set_xticks(xticks)
            axs[fr].set_xticklabels([f'{xticks[0]:.02f}', f'{xticks[1]:.02f}'])
            axs[fr].spines[['bottom']].set_bounds(xticks[0], xticks[1])
            axs[fr].spines[['left']].set_bounds(-1, 1)

        fname =  f'spk.corr_{corr}.M1-S1'
        f = open(os.path.join(baseDir, gl.pcmDir, f'theta_in.{fname}.p'), 'rb')
        theta = pickle.load(f)[0]
        r_bootstrap = np.load(os.path.join(baseDir, gl.pcmDir, f'r_bootstrap.{fname}.npy'))
        f = open(os.path.join(baseDir, gl.pcmDir, f'theta_gr.{fname}.p'), 'rb')
        theta_g = pickle.load(f)[0]
        fig, axs = plot_pcm_corr(fig, axs, 3, Mflex, theta, theta_g, r_bootstrap)
        axs[3].set_title(title[3])
        #xticks = np.array(axs[3].get_xlim()).clip(0, np.inf)
        axs[3].set_xticks(xticks)
        axs[3].set_xticklabels([f'{xticks[0]:.02f}', f'{xticks[1]:.02f}'])
        axs[3].spines[['bottom']].set_bounds(xticks[0], xticks[1])
        axs[3].spines[['left']].set_bounds(-1, 1)
        axs[3].set_xlim(-0.1, 1.6)

        fig.supxlabel('SNR')
        axs[0].set_ylabel('correlation')

        if corr=='plan-exec':
            fig.suptitle(f'Preparation-execution correlation')
        elif corr=='cue-dir':
            fig.suptitle(f'Cue-direction correlation (M1-S1)')
        fig.savefig(os.path.join(path_fig, f'corr_{corr}.lfp.svg'))

        plt.show()
    if args.what=='emg_aligned':
        f = open(os.path.join(gl.baseDir, args.experiment, 'emg', 'emg.p'), 'rb')
        emg = pickle.load(f)
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(4, 5))
        fig, axs = plot_emg_aligned(fig, axs, emg)
        fig.savefig(os.path.join(path_fig, f'emg_aligned.svg'))
        plt.show()
    if args.what=='dissimilarities_emg':
        vmin, vmax = -0.1, 1.8
        epochs = ['Pre', 'SLR', 'LLR', 'Vol']
        G = np.load(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'G_obs.emg.npy'))
        fig, axs = plt.subplots(1, 4, figsize=(6, 2.5), sharex=True, sharey=True)
        for e, epoch in enumerate(epochs):
            D = pcm.G_to_dist(G[:, e])
            fig, axs = plot_dissimilarities(fig, axs, e, D, list(gl.regressor_mapping.keys())[5:], vmin=vmin,
                                            vmax=vmax, sqrt=True, source=epoch)
            axs[e].set_title(epoch)
        cax = axs[-1].get_images()[0]
        cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=.008)
        cbar.set_label('dissimilarity (a.u.)')
        fig.suptitle('Crossnobis dissimilarities during response execution')
        fig.savefig(os.path.join(path_fig, f'dissimilarities_emg.svg'))
        plt.show()
    if args.what=='mds':
        pass
    if args.what=='weight_ephys':
        figsize = tuple(map(float, kwargs.get('figsize', (4, 3))))
        ylim = tuple(map(float, kwargs.get('ylim', (0, .3))))
        monkey = list(map(str, kwargs.get('monkey', ['Malfoy', 'Pert'])))
        models = list(map(str, kwargs.get('models', ['Expectation', 'Uncertainty'])))
        model = kwargs.get('model', 'Expectation')
        freq1, freq2 = tuple(map(float, kwargs.get('band', (10, 20))))
        epoch = args.epoch
        rois = args.rois

        fig, axs_ = plt.subplots(2, len(rois) + 1, sharex='col', figsize=figsize,
                                 gridspec_kw={"width_ratios": [80] * len(rois) + [2]}, constrained_layout=True)

        axs = axs_[:, :-1]
        ax_c = axs_[:, -1]

        freq1, freq2 = 10, 20
        cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, 'Malfoy/cfg.PMd-19.mat'))['cfg']
        foi = cfg['foi']
        freq_mask = (foi > freq1) & (foi < freq2)
        t_cue = np.linspace(0, gl.cuePost - 1, gl.cuePost)
        t_pert = np.linspace(gl.pertPre, gl.pertPost - 1, gl.pertPost - gl.pertPre) + 5
        t = np.concatenate((t_cue, t_pert))

        vmin, vmax = 0, .1
        color = [['darkred', 'navy'], ['lightcoral', 'lightblue']]
        md = models.index(model)
        for r, roi in enumerate(rois):
            # if len(rois) > 1:
            ax = axs[:, r]
            # else:
            #     ax = axs
            weight_lfp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.lfp.{roi}.{args.epoch}.npy'))[..., md].mean(axis=0)
            weight_lfp_band = weight_lfp[freq_mask].mean(axis=0)
            sig_lfp1 = np.load(os.path.join(gl.nhpDir, gl.pcmDir, f'significant_bf.lfp.Cue.{roi}.{args.epoch}.npy'))
            sig_lfp1 = sig_lfp1[..., md]
            weight_lfp1 = weight_lfp[:, :gl.cuePost]
            weight_lfp2 = weight_lfp[:, gl.pertPre:]
            weight_lfp_band1 = weight_lfp_band[:gl.cuePost]
            weight_lfp_band2 = weight_lfp_band[gl.pertPre:]
            h1 = ax[0].pcolormesh(t_cue, foi, weight_lfp1, vmin=vmin, vmax=vmax, cmap='viridis')
            h1.set_rasterized(True)
            ax[0].contour(t_cue, foi, sig_lfp1, levels=1, colors='lightcoral', linewidths=1)
            h2 = ax[0].pcolormesh(t_pert, foi, weight_lfp2, vmin=vmin, vmax=vmax, cmap='viridis')
            h2.set_rasterized(True)
            ax[0,].set_yscale('log')
            ax[0,].set_title(roi)
            ax[0,].axhline(freq1, color='k', lw='.8')
            ax[0,].axhline(freq2, color='k', lw='.8')
            ax[0,].set_ylabel('frequency (Hz)') if r == 0 else None
            weight_spk = np.load(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.spk.{roi}.{args.epoch}.npy'))[..., md].mean(axis=0)
            weight_spk1 = weight_spk[:gl.cuePost]
            weight_spk2 = weight_spk[gl.pertPre:]
            sig_spk1 = np.load(os.path.join(gl.nhpDir, gl.pcmDir, f'significant_bf.spk.Cue.{roi}.{args.epoch}.npy'))
            sig_spk1 = sig_spk1[..., 0]
            sig_spk1[sig_spk1 < 1] = np.nan
            sig_spk1[sig_spk1 == 1] = .29
            ax[1,].plot(t_cue, sig_spk1, color='brown', lw=2)
            ax[1,].plot(t_cue, weight_spk1, color='brown', label='spiking activity')
            ax[1,].plot(t_pert, weight_spk2, color='brown')
            ax[1,].plot(t_cue, weight_lfp_band1, color='lightcoral', label='LFPs (10-20Hz)')
            ax[1,].plot(t_pert, weight_lfp_band2, color='lightcoral')
            ax[1,].axvspan(gl.cueIdx, gl.cuePost, color='grey', alpha=.2, lw=0)
            ax[1,].set_ylim([-.01, .3])
            ax[1,].set_yticks((0, .3))
            ax[1,].spines['left'].set_bounds(0, .3)
            ax[1,].set_ylabel('weight') if r == 0 else None
            for i in range(2):
                ax[i,].axvline(gl.cueIdx, color='k', lw='.8')
                ax[i,].axvline(gl.pertIdx, color='k', lw='.8')
                ax[i,].set_xticks([gl.cueIdx, gl.pertIdx])
                ax[i,].set_xticklabels(['Cue', 'Pert'])
                ax[i,].set_yticks([]) if r > 0 else None
                ax[i,].spines[['bottom', 'right', 'top']].set_visible(False) if r == 0 else ax[i].spines[
                    ['bottom', 'right', 'top', 'left']].set_visible(False)

        # if len(rois) > 1:
        axs[1, -1].legend(frameon=False, ncol=1, fontsize=8, loc='upper left')
        yline = axs[1, 0].get_ylim()[1]
        # else:
        #     axs[1].legend(frameon=False, ncol=1, fontsize=8, loc='upper left')
        #     yline = axs[1].get_ylim()[1]

        fig.colorbar(h2, cax=ax_c[0], label='weight')
        for ax in ax_c[1:]:
            ax.remove()

        fig.suptitle(f'{model} weight in LFPs and spiking activity', va='center')


        axs_[1, 0].hlines(yline, gl.cueIdx + 20, gl.cueIdx + 40, color='k', )
        axs_[1, 0].text(gl.cueIdx + 30, yline, '200ms', va='top', ha='center')

        fig.savefig(os.path.join(path_fig, f'lfp.component.{model}.{args.epoch}.{".".join(rois)}.svg'))
        plt.show()
    if args.what=='%weight_ephys':
        df_exp = pd.DataFrame()
        for mod in ['lfp', 'spk']:
            df_ephys = pd.read_csv(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.{mod}.10-20Hz.tsv'), sep='\t')
            df_ephys = df_ephys[
                ((df_ephys['roi'] == 'PMd') | (df_ephys['roi'] == 'S1')) & (df_ephys['epoch'] == 'plan')]
            df_ephys['weight_norm'] = np.sqrt(df_ephys['weight'] / df_ephys['noise'])
            df_ephys['session'] = df_ephys['session'].astype(str) + df_ephys['monkey']
            df_exp_tmp = df_ephys[df_ephys['component'] == 'expectation'].reset_index(drop=True)
            df_exp_tmp['modality'] = mod
            df_exp = pd.concat([df_exp, df_exp_tmp])

        fig, ax = plt.subplots(figsize=(1.5, 2), constrained_layout=True)

        sb.barplot(data=df_exp, x='roi', y='weight_norm', hue='modality', errorbar='se', ax=ax,
                   palette=['lightcoral', 'brown'], color='grey')
        sb.stripplot(data=df_exp, x='roi', y='weight_norm', hue='modality', ax=ax, size=2, color='k', jitter=.1,
                     dodge=True, legend=False)
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
        ax.tick_params(axis='x', bottom=False, )
        ax.set_xlabel('')
        ax.set_xticks((0, 1))
        ax.set_xticklabels(['PMd', 'S1'], rotation=90)
        ax.set_ylabel('weight')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['LFPs (10–20Hz)', 'spiking activity'], frameon=False, title=False,
                  fontsize=8, bbox_to_anchor=(.5, 1), loc='lower center')
        fig.savefig(os.path.join(path_fig, f'%weight.expectation.ephys.plan.svg'))
        plt.show()
    if args.what=='desynchronisation':
        figsize = tuple(map(float, kwargs.get('figsize', (4, 3))))
        monkey = list(map(str, kwargs.get('monkey', ['Malfoy', 'Pert'])))
        rois = args.rois
        cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, 'Malfoy/cfg.PMd-19.mat'))['cfg']
        foi = cfg['foi']

        t_cue = np.linspace(0, gl.cuePost - 1, gl.cuePost)
        t_pert = np.linspace(gl.pertPre, gl.pertPost - 1, gl.pertPost - gl.pertPre) + 5
        t = np.concatenate((t_cue, t_pert))

        lfp = {'lfp': [], 'roi': []}
        spk = {'spk': [], 'roi': []}
        for roi in rois:
            for mon in monkey:
                for rec in gl.recordings[mon][roi]:
                    lfp_aligned = np.load(os.path.join(gl.nhpDir, gl.lfpDir, mon, f'lfp_aligned.avg.{roi}-{rec}.npy'))
                    spk_aligned = np.load(os.path.join(gl.nhpDir, gl.spkDir, mon, f'spk_aligned.avg.{roi}-{rec}.npy'))
                    lfp["lfp"].append(lfp_aligned)
                    lfp["roi"].append(roi)
                    spk["spk"].append(spk_aligned.mean(axis=-1))
                    spk["roi"].append(roi)
        lfp = xr.DataArray(data=np.stack(lfp["lfp"]),
                           dims=('roi', 'time', 'freq'),
                           coords={
                               'roi': lfp['roi'],
                               'freq': foi,
                               'time': t,})
        spk = xr.DataArray(data=np.stack(spk["spk"]),
                           dims=('roi', 'time',),
                           coords={
                               'roi': lfp['roi'],
                               'time': t,})
        fig, axs = plt.subplots(2, len(rois), sharex='col', sharey='row', figsize=(7, 3), constrained_layout=True)

        vmin, vmax = -3, 3

        bs_lfp = lfp.sel(time=slice(0, gl.cueIdx - 5)).mean(axis=1)
        bs_spk = spk.sel(time=slice(0, gl.cueIdx - 5)).mean(axis=1)
        lfp_dB = (10 * np.log10(lfp / bs_lfp))
        spk = spk - bs_spk

        for r, roi in enumerate(rois):
            ax = axs[:, r]
            ax[0].set_yscale('log')
            lfp_cue = lfp_dB.sel(roi=roi, time=slice(0, gl.cuePost - 1)).mean(axis=0)
            lfp_pert = lfp_dB.sel(roi=roi, time=slice(gl.pertPre, None)).mean(axis=0)
            h1 = ax[0].pcolormesh(t_cue, foi, lfp_cue.T, vmin=vmin, vmax=vmax, cmap='bwr', shading='gouraud')
            h2 = ax[0].pcolormesh(t_pert, foi, lfp_pert.T, vmin=vmin, vmax=vmax, cmap='bwr', shading='gouraud')
            h1.set_rasterized(True)
            h2.set_rasterized(True)
            ax[0].axhline(8, color='k', lw='.8')
            ax[0].axhline(13, color='k', lw='.8')
            ax[0].axhline(25, color='k', lw='.8')
            ax[0].axhline(100, color='k', lw='.8')
            ax[0].axvline(gl.cueIdx, color='k', lw='.8')
            ax[0].axvline(gl.pertIdx, color='k', lw='.8')
            ax[0].set_xticks([gl.cueIdx, gl.pertIdx])
            ax[0].set_xticklabels(['Cue', 'Pert'])
            ax[0].set_title(roi)
            spk_cue = spk.sel(roi=roi, time=slice(0, gl.cuePost - 1)).mean(axis=0)
            spk_pert = spk.sel(roi=roi, time=slice(gl.pertPre, None)).mean(axis=0)
            ax[1].plot(t_cue, spk_cue, color='k')
            ax[1].plot(t_pert, spk_pert, color='k')
            ax[1].spines[['right', 'top', 'bottom']].set_visible(False)
            ax[1].axvline(gl.cueIdx, color='k', lw='.8')
            ax[1].axvline(gl.pertIdx, color='k', lw='.8')
            ax[1].spines['left'].set_visible(False) if r > 0 else None
            ax[1].tick_params(left=False) if r > 0 else None

        yline = axs[1, 0].get_ylim()[1]
        axs[1, 0].hlines(yline, gl.cueIdx + 20, gl.cueIdx + 40, color='k', )
        axs[1, 0].text(gl.cueIdx + 30, yline, '200ms', va='top', ha='center')

        fig.colorbar(h1, ax=ax, label='power (dB)', fraction=0.02)
        axs[0, 0].set_ylabel('frequency (Hz)')
        axs[1, 0].set_ylabel('spiking rate (a.u.)')
        fig.suptitle('LFP power and spiking activity')

        fig.savefig(os.path.join(path_fig, f'lfp.desynchronisation.{".".join(rois)}.svg'))

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
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--H', type=str, default='L')
    parser.add_argument('--epoch', type=str, default='plan', choices=['plan', 'exec', 'plan-exec'])
    parser.add_argument('--epochs', type=list, default=['Pre', 'SLR', 'LLR', 'Vol'])
    parser.add_argument('--rois', nargs='+', type=str, default=['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp',])
    parser.add_argument('--sns', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    args, unknown_args = parser.parse_known_args()

    kwargs = parse_unknown_args(unknown_args)

    main(args, **kwargs)
