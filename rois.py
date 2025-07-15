import sys
import os
import globals as gl
import nitools as nt
import nibabel as nb

import argparse
import time

import imaging_pipelines.rois as rois

import Functional_Fusion.atlas_map as am

import numpy as np


def exclude_overlapping_voxels(amap, exclude='all', exclude_thres=0.9):
    """
    Ensures that ROIs do not share voxels by excluding overlapping voxels based on their weights.

    Parameters:
        amap (list): A list of AtlasMapSurf objects, each containing:
                     - 'vox_list': (N, M) np.array of voxel indices (M = number of dimensions, e.g., 3 for [x, y, z])
                     - 'vox_weight': (N, M) np.array of weights corresponding to vox_list
        exclude (str or list of tuple): If 'all', compare all ROI pairs. Otherwise, provide a list of (i, j) tuples.
        exclude_thres (float): Threshold to determine which ROI retains a voxel.

    Returns:
        list: Updated amap with overlapping voxels removed.
    """

    # Initialize exclusion masks
    for roi in amap:
        roi.excl_mask = np.zeros(roi.vox_list.shape, dtype=bool).flatten()

    # Create list of ROI pairs to compare
    if exclude == 'all':
        exclude_pairs = [(i, j) for i in range(len(amap)) for j in range(i, len(amap))]
    else:
        exclude_pairs = exclude  # User-provided list of pairs

    # Process each pair of ROIs
    for j, k in exclude_pairs:
        vox_j, weight_j = amap[j].vox_list, amap[j].vox_weight
        vox_k, weight_k = amap[k].vox_list, amap[k].vox_weight

        # # Find common voxel indices
        # common_voxels, idx_j, idx_k = np.intersect1d(vox_j, vox_k, return_indices=True)

        EQ = vox_j.flatten()[:, np.newaxis] == vox_k.flatten()[np.newaxis, :]
        # EQ = np.all(vox_j[:, np.newaxis, :] == vox_k[np.newaxis, :, :], axis=2)

        idx_j, idx_k = np.where(EQ)

        for idx_j_v, idx_k_v in zip(idx_j, idx_k):
            wj, wk = weight_j.flatten()[idx_j_v], weight_k.flatten()[idx_k_v]
            total_weight = wj + wk

            if total_weight == 0:
                amap[j].excl_mask[idx_j_v] = True
                amap[k].excl_mask[idx_k_v] = True
            else:
                frac_j = wj / total_weight
                frac_k = wk / total_weight

                if frac_j > exclude_thres:  # Keep voxel in j, exclude from k
                    amap[k].excl_mask[idx_k_v] = True
                elif frac_k > exclude_thres:  # Keep voxel in k, exclude from j
                    amap[j].excl_mask[idx_j_v] = True
                else:  # Exclude from both
                    amap[j].excl_mask[idx_j_v] = True
                    amap[k].excl_mask[idx_k_v] = True

        # Apply exclusion mask to each ROI
    for roi in amap:
        mask = ~roi.excl_mask  # Keep only unexcluded voxels
        roi.vox_list = roi.vox_list.flatten()[mask]  # Reshape vox_list to keep valid entries
        roi.vox_weight = roi.vox_weight.flatten()[mask]
        roi.num_excl = np.sum(roi.excl_mask)  # Count excluded voxels
        del roi.excl_mask  # Remove temporary mask

    return amap


def main(args):
    exclude = {
        'ROI': [(1, 2), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 7), (3, 4), (3, 5), (7, 8)]
    }

    if args.what == 'make_cortical_rois':
        path_surf = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{args.sn}')
        white = [os.path.join(path_surf, f'subj{args.sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
        pial = [os.path.join(path_surf, f'subj{args.sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
        mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        atlas_name = 'ROI'
        atlas_dir = gl.atlasDir
        rois_dir = os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}')
        Rois = rois.SurfRois(atlas_name, white, pial, mask, atlas_dir, rois_dir)
        Rois.make_hemispheres()
        Rois.make_rois(exclude=exclude[atlas_name])

    if args.what=='make_cerebellar_rois':
        atlas_path = os.path.join(gl.baseDir, args.experiment, 'SUIT', 'atl-NettekovenSym32_space-SUIT_dseg.nii')
        space = 'SUIT1'
        _, _, labels = nt.read_lut(os.path.join(gl.baseDir, args.experiment, 'SUIT', 'atl-NettekovenSym32.lut'))
        crois = {'L': ['M2L', 'M3L', 'D3L'],
                'R': [ 'M2R', 'M3R', 'D3R']}
        deform = os.path.join(gl.baseDir, args.experiment, 'SUIT', 'anatomicals', f'subj{args.sn}',
                              f'y_subj{args.sn}_anatomical_suitdef.nii')
        mask = os.path.join(gl.baseDir, args.experiment, 'SUIT',f'glm{args.glm}',  f'subj{args.sn}', 'wdmask.nii')
        out_path = os.path.join(gl.baseDir, args.experiment, 'SUIT' ,gl.roiDir, f'subj{args.sn}', )
        os.makedirs(out_path, exist_ok=True)
        rois.make_cerebellum(atlas_path, space, labels, crois, None, mask, out_path)

    if args.what == 'make_cortical_rois_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='make_cortical_rois',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                atlas=args.atlas,

            )
            main(args)

    if args.what == 'make_cerebellar_rois_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='make_cerebellar_rois',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                atlas=args.atlas,

            )
            main(args)



if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,])
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)
    finish = time.time()

    print(f'Execution time:{finish-start} s')

