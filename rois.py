import sys
import os
import globals as gl
import nitools as nt
import nibabel as nb

import argparse
import time

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default='make_rois')
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    atlas, _ = am.get_atlas('fs32k')

    if args.what=='make_rois':

        Hem = ['L', 'R']

        roiMasks = []
        for h, H in enumerate(Hem):

            g_atlas = nb.load(os.path.join(gl.atlas_dir, f'{args.atlas}.32k.{Hem[h]}.label.gii'))

            labels = {
                ele.key: getattr(ele, 'label', '')
                for ele in g_atlas.labeltable.labels
            }

            amap = list()
            for nlabel, label in enumerate(labels.values()):
                print(f'making ROI: {label}, {H}')

                atlas_hem = atlas.get_hemisphere(h)
                subatlas = atlas_hem.get_subatlas_image(os.path.join(gl.atlas_dir,
                                                                     f'{args.atlas}.32k.{H}.label.gii'), nlabel)

                subj_dir = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{args.sn}')
                white = os.path.join(subj_dir, f'subj{args.sn}.{H}.white.32k.surf.gii')
                pial = os.path.join(subj_dir, f'subj{args.sn}.{H}.pial.32k.surf.gii')
                mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
                amap_tmp = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
                amap_tmp.build()

                # add roi name
                amap_tmp.name = label

                # add number of voxels
                amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

                amap.append(amap_tmp)

            print('excluding voxels...')
            amap = am.exclude_overlapping_voxels(amap, exclude=[(1, 2), (1, 6), (1, 7),
                                                             (2, 3), (2, 4), (2, 5), (2, 7),
                                                             (3, 4), (3, 5),
                                                             (7, 8)])
            for amap_tmp in amap:
                print(f'saving ROI {amap_tmp.name}, {H}')
                mask_out = amap_tmp.save_as_image(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                                               f'{args.atlas}.{H}.{amap_tmp.name}.nii'))
                if len(amap_tmp.name) > 0:
                    roiMasks.append(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                                               f'{args.atlas}.{H}.{amap_tmp.name}.nii'))

        am.parcel_combine(roiMasks,os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                                               f'{args.atlas}.nii'))

    if args.what=='make_hemispheres':

        Hem = ['L', 'R']

        amap = []
        for h, H in enumerate(Hem):
            atlas_hem = atlas.get_hemisphere(h)

            subj_dir = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{args.sn}')
            white = os.path.join(subj_dir, f'subj{args.sn}.{H}.white.32k.surf.gii')
            pial = os.path.join(subj_dir, f'subj{args.sn}.{H}.pial.32k.surf.gii')
            mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
            amap_tmp = am.AtlasMapSurf(atlas_hem.vertex[0], white, pial, mask)

            print(f'building hemisphere: {H}')
            amap_tmp.build()

            # add hem name
            amap_tmp.name = H

            # add number of voxels
            amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

            amap.append(amap_tmp)

        amap = am.exclude_overlapping_voxels(amap, exclude=[(0, 1)])
        for amap_tmp in amap:
            print(f'saving hemisphere {amap_tmp.name}')
            mask_out = amap_tmp.save_as_image(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                                           f'Hem.{H}.nii'))




if __name__ == '__main__':
    start = time.time()
    main()
    finish = time.time()

    print(f'Execution time:{finish-start} s')

