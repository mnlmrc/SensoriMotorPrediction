import argparse
import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub')
import numpy as np
import nibabel as nb
# import nitools as nt
import os
import matplotlib.pyplot as plt
import globals as gl

import surfAnalysisPy as surf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument('--participant_id', default='subj101', help='Participant ID (e.g., subj100, subj101, ...)')
    parser.add_argument('--glm', default='4', help='GLM model (e.g., 1, 2, ...)')
    parser.add_argument('--measure', default='con', help='Measurement (e.g., psc, tval, ...)')
    parser.add_argument('--epoch', default='exec', help='Selected epoch')
    parser.add_argument('--stimFinger', nargs='+', default=[ 'index', 'ring'], help='Selected stimulated finger')
    parser.add_argument('--instr', nargs='+', default=['go'], help='Selected instruction')

    args = parser.parse_args()

    participant_id = args.participant_id
    glm = args.glm
    meas = args.measure
    sel_epoch = args.epoch
    sel_stimFinger = args.stimFinger
    sel_instr = args.instr

    experiment = 'smp1'

    data = [
        f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{participant_id}/glm{glm}.{meas}.L.func.gii',
        f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{participant_id}/glm{glm}.{meas}.R.func.gii']
    borders = ['/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.border',
               '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.border']
    underlay = ['/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.LR.sulc.dscalar.nii',
                '/Users/mnlmrc/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.LR.sulc.dscalar.nii']

    cscale = {
        'spmT': [-2.5, 2.5],
        'psc': [-1, 1],
        'con': [-20, 20]
    }

    D = [nb.load(data[0]), nb.load(data[1])]

    col = [d.metadata['Name'] for d in D[0].darrays]

    # build name
    cond = f'{meas}_{sel_epoch}'
    if (len(sel_instr) == 1) and sel_epoch is not 'exec':
        cond += f'_{sel_instr[0]}'
    cond += '-.nii'

    im = col.index(cond)

    title = ['Left hemisphere', 'Right hemisphere']

    surface = ['fs32k_L', 'fs32k_R']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for h in range(2):
        plt.sca(axs[h])
        surf.plot.plotmap(D[h].darrays[im].data, surface[h],
                          underlay=None,
                          borders=borders[h],
                          cscale=cscale[meas],
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False)
        plt.title(title[h])

    if sel_epoch is 'plan':
        plt.suptitle(f'{participant_id}\nepoch:{sel_epoch}, instr:{sel_instr}, stimFinger:{sel_stimFinger}')
    elif sel_epoch is 'exec':
        plt.suptitle(f"{participant_id}\nepoch:{sel_epoch}, instr:['go'], stimFinger:{sel_stimFinger}")

    plt.subplots_adjust(bottom=0, left=0, wspace=0, hspace=0, right=1, top=.8)

    # plt.savefig(os.path.join(gl.baseDir, experiment, 'figures', participant_id,
    #                          f"{cond}".rstrip('.nii') + '.png'))
