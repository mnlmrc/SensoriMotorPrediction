import nibabel as nb
import os
import globals as gl
import numpy as np
import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub/nitools')
from nitools import spm

experiment = 'smp2'
sn = 108
atlas = 'ROI'
roi = 'M1'
H = 'L'
glm = 12

runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

stats = 'whiten'

# retrieve SPM
SPM = spm.SpmGlm(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}'))
SPM.get_info_from_spm_mat()

# # get coordinates
# mask_img = nb.load(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}', f'{atlas}.{H}.{roi}.nii'))
# coords = nt.get_mask_coords(mask_img)
#
# # get raw time series in roi
# imaging_data = [os.path.join(gl.baseDir, experiment, gl.imagingDir, f'subj{sn}', f'subj{sn}_run_{run:02d}.nii')
#                 for run in runs]
# data = nt.sample_images(imaging_data, coords)
#
# # load residuals for prewhitening
# res_img = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
# ResMS = nt.sample_image(res_img, coords[0], coords[1], coords[2], 0)
#
# if stats == 'mean':
#     y_raw = data.mean(axis=1)
# elif stats == 'whiten':
#     y_raw = (data / np.sqrt(ResMS)).mean(axis=1)
# elif stats == 'pca':
#     pass
#
# fdata = SPM.spm_filter(SPM.weight @ data)
# beta = SPM.pinvX @ fdata
# pdata = SPM.design_matrix @ beta
#
#
#
#
#
