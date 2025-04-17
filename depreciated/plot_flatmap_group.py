import os
import sys

import nibabel as nb
import numpy as np

import globals as gl

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append('/Users/mnlmrc/Documents/GitHub')
import surfAnalysisPy as surf

surface = ['fs32k_L', 'fs32k_R']

if os.path.isdir('/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh'):
    meshDir = '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh'
elif os.path.isdir('/home/ROBARTS/memanue5/Documents/GitHub/surfAnalysis/standard_mesh'):
    meshDir = '/home/ROBARTS/memanue5/Documents/GitHub/surfAnalysis/standard_mesh'
else:
    meshDir = None

borders = [f'{meshDir}/fs_L/fs_LR.32k.L.border', f'{meshDir}/fs_R/fs_LR.32k.R.border']
underlay = [f'{meshDir}/fs_L/fs_LR.32k.LR.sulc.dscalar.nii', f'{meshDir}fs_R/fs_LR.32k.LR.sulc.dscalar.nii']

Hem = ['L', 'R']

# fig, axs = plt.subplots()

for h, H in enumerate(Hem):

    map_names = {
        f'con_0%,ring.{H}.func.gii, smooth 1': '0%, ring',
        f'con_25%,ring.{H}.func.gii, smooth 1': '25%, ring',
        f'con_50%,ring.{H}.func.gii, smooth 1': '50%, ring',
        f'con_75%,ring.{H}.func.gii, smooth 1': '75%, ring',
        f'con_25%,index.{H}.func.gii, smooth 1': '25%, ring',
        f'con_50%,index.{H}.func.gii, smooth 1': '50%, ring',
        f'con_75%,index.{H}.func.gii, smooth 1': '75%, ring',
        f'con_100%,index.{H}.func.gii, smooth 1': '100%, ring',
        f'con_0%.{H}.func.gii, smooth 1': '0%, ring',
        f'con_25%.{H}.func.gii, smooth 1': '25%, ring',
        f'con_50%.{H}.func.gii, smooth 1': '50%, ring',
        f'con_75%.{H}.func.gii, smooth 1': '75%, ring',
        f'con_100%.{H}.func.gii, smooth 1': '100%, ring',
        f'exec.{H}.func.gii, smooth 1': 'execution',
        f'plan.{H}.func.gii, smooth 1': 'planning',
    }

    map_hem = {
        'L': 'left hemisphere',
        'R': 'right hemisphere',
    }

    frame = {
        'L': [-60, 120, -40, 140],
        'R': [-100, 80, -60, 120]
    }

    mean = nb.load(os.path.join(gl.baseDir, 'smp1', gl.wbDir, 'group', f'mean.smooth.{H}.func.gii'))
    pval = nb.load(os.path.join(gl.baseDir, 'smp1', gl.wbDir, 'group', f'pval.{H}.func.gii'))
    label = nb.load(f'/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32/ROI.32k.{H}.label.gii')

    label_names = [''] + [lab.label for lab in label.labeltable.labels if hasattr(lab, 'label')]
    colors = [lab.rgba for lab in label.labeltable.labels]
    cmap = ListedColormap(colors)

    fig, axs = plt.subplots(figsize=(5.5, 4))
    plt.sca(axs)
    surf.plot.plotmap(label.darrays[0].data, surface[h],
                      underlay=None,
                      borders=borders[h],
                      # cscale=[-10, 10],
                      underscale=[-1.5, 1],
                      alpha=.5,
                      new_figure=False,
                      overlay_type='label',
                      label_names=label_names,
                      cmap=cmap,
                      colorbar=True)

    plt.ylim([1, 8])
    plt.savefig(os.path.join(gl.baseDir, 'smp1', 'figures', f'flatmap.ROI.{H}.rois.svg'), dpi=300)
    plt.savefig(os.path.join(gl.baseDir, 'smp1', 'figures', f'flatmap.ROI.{H}.rois.png'), dpi=300)

    for img in mean.darrays:
        # img.data[p.data > .05] = np.nan
        # if ('ring' in map_names[img.metadata["Name"]]) | ('index' in map_names[img.metadata["Name"]])

        fig, axs = plt.subplots(figsize=(5.5, 4))
        plt.sca(axs)
        surf.plot.plotmap(img.data, surface[h],
                          underlay=None,
                          borders=borders[h],
                          cscale=[-20, 20],
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          cmap='seismic',
                          colorbar=True,
                          frame=frame[H]
                          )

        plt.ylabel('activation (a.u.)')
        plt.suptitle(f'{map_hem[H]}, {map_names[img.metadata["Name"]]}')
        # plt.savefig(os.path.join(gl.baseDir, 'smp1', 'figures', f'flatmap.group.{H}.{img.metadata["Name"]}.svg'),
        #             dpi=300)
        # plt.savefig(os.path.join(gl.baseDir, 'smp1', 'figures', f'flatmap.group.{H}.{img.metadata["Name"]}.png'),
        #             dpi=300)
