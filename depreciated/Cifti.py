import argparse
import time
from nitools import spm
import numpy as np
import nibabel as nb
import globals as gl
import os

def check_duplicate_voxels(cifti_img):
    # Load the volume NIFTI file
    vol_img = nb.load(cifti_img)
    vol_data = vol_img.get_fdata()

    # Identify duplicated voxels
    unique_voxels, counts = np.unique(vol_data[0], return_counts=True)
    duplicated_voxels = unique_voxels[counts > 1]

    if len(duplicated_voxels) > 0:
        print(f"Found {len(duplicated_voxels)} duplicated voxels")
    else:
        print("No duplicated voxels found")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='save_rois_betas')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=102)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']
    SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}'))  #
    SPM.get_info_from_spm_mat()

    if args.what == 'save_rois_timeseries':
        pass
    if args.what == 'save_rois_betas':
        path_mask = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}') + f'/{args.atlas}.'
        for roi in rois:
            print(f'Processing... {roi}')
            mask = (path_mask + f'L.{roi}.nii', path_mask + f'R.{roi}.nii')
            betaL, ResMS, info = SPM.get_betas(mask[0])
            betaR, ResMS, _ = SPM.get_betas(mask[1])

            beta = np.hstack((betaL, betaR))


            pass
            nb.save(img_beta, os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}',
                                           f'{args.atlas}.{roi}.beta.dscalar.nii'))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Time elapsed: {end - start}')
