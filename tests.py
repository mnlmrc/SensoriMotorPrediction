import os
from nitools import spm
from hrf import update_hrf_params
import globals as gl
import pandas as pd
import matplotlib.pyplot as plt

sn=103
experiment='smp2'
glm=12

# default [6, 16, 1, 1, 6, 0, 32]
mask_img = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}', f'ROI.L.M1.nii')
SPM = spm.SpmGlm(os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}',))
SPM.get_info_from_spm_mat()

print('updating hrf params...')
y_filt, y_hat, y_adj = update_hrf_params(SPM, [10, 20, 1, 1, 6, 0, 32], mask_img)


dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}',
                               f'{experiment}_{sn}.dat'), sep='\t')
dat = dat[dat['GoNogo'] == 'go']
pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
runs = pinfo[pinfo['sn'] == sn].FuncRuns.reset_index(drop=True)[0].split('.')
nVols = pinfo[pinfo['sn'] == sn].numTR
i = 0
for BN in dat['BN'].unique():
    if str(BN) in runs:
        if i == 0:
            at = (dat[dat['BN']==BN].startTRReal).tolist()
        else:
            at.extend((dat[dat['BN']==BN].startTRReal + int(nVols * i)).tolist())
        i =+ 1
    else:
        print(f'excluding block {BN}')

y_adj = spm.avg_cut(y_adj, 10, at, 20)
y_hat = spm.avg_cut(y_hat, 10, at, 20)

fig, axs = plt.subplots()

axs.plot(y_adj.mean(axis=(0, 2)), color='green', ls='--')
axs.plot(y_hat.mean(axis=(0, 2)), color='green')

plt.show()