import os
import SensoriMotorPrediction.globals as gl
import nitools as nt
import nibabel as nb
from nibabel.processing import resample_from_to
import numpy as np
import subprocess

import imaging_pipelines.rois as rois

import Functional_Fusion.atlas_map as am




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


def thalamus_segmentation(sn, experiment='smp2'):
    """
    Run FreeSurfer thalamic-nuclei segmentation for one subject.

    Parameters
    ----------
    sn : int or str
        Subject number, e.g. 101 or 102 -> folder 'subj101'.
    """
    # subj = f"subj{sn}"

    # segment_subregions looks for <SUBJECTS_DIR>/<subj>/mri/, so point it at base_dir.
    # (--sd does the same thing as exporting SUBJECTS_DIR, just inline.)
    subjects_fs_dir = os.path.join(gl.baseDir, experiment, 'surfaceFreesurfer', f'subj{sn}',)
    cmd = ["segment_subregions", "thalamus", "--cross", f'subj{sn}', "--sd", subjects_fs_dir]

    print("SUBJECTS_DIR =", subjects_fs_dir)
    print("running     :", " ".join(cmd))
    
    subprocess.run(cmd)

    # FreeSurfer thalamic-nuclei label -> name (from FreeSurferColorLUT.txt)
    LUT = {
        8103: "L.AV",   8104: "L.CeM",  8105: "L.CL",   8106: "L.CM",
        8108: "L.LD",   8109: "L.LGN",  8110: "L.LP",   8111: "L.L.Sg",
        8112: "L.MDl",  8113: "L.MDm",  8115: "L.MGN",  8116: "L.MV(Re)",
        8117: "L.Pc",   8118: "L.Pf",   8119: "L.Pt",   8120: "L.PuA",
        8121: "L.PuI",  8122: "L.PuL",  8123: "L.PuM",  8125: "L.R",
        8126: "L.VA",   8127: "L.VAmc", 8128: "L.VLa",  8129: "L.VLp",
        8130: "L.VM",   8133: "L.VPL",  8134: "L.PaV",  8135: "L.PuMm",
        8136: "L.PuMl",
        8203: "R.AV",  8204: "R.CeM", 8205: "R.CL",  8206: "R.CM",
        8208: "R.LD",  8209: "R.LGN", 8210: "R.LP",  8211: "R.L.Sg",
        8212: "R.MDl", 8213: "R.MDm", 8215: "R.MGN", 8216: "R.MV(Re)",
        8217: "R.Pc",  8218: "R.Pf",  8219: "R.Pt",  8220: "R.PuA",
        8221: "R.PuI", 8222: "R.PuL", 8223: "R.PuM", 8225: "R.R",
        8226: "R.VA",  8227: "R.VAmc",8228: "R.VLa", 8229: "R.VLp",
        8230: "R.VM",  8233: "R.VPL", 8234: "R.PaV", 8235: "R.PuMm",
        8236: "R.PuMl",
    }

    ref_path = os.path.join(gl.baseDir, 'smp2', gl.roiDir, f'subj{sn}', 'Hem.L.nii')
    seg_path = os.path.join(gl.baseDir, 'smp2', 'surfaceFreesurfer', f'subj{sn}', f'subj{sn}', 'mri', 'ThalamicNuclei.mgz')
    out_dir = os.path.join(gl.baseDir, 'smp2', gl.roiDir, f'subj{sn}',)

    # os.makedirs(out_dir, exist_ok=True)
    seg = nb.load(seg_path)
    ref = nb.load(ref_path)

    # Resample the label volume onto the functional grid.
    seg_r = resample_from_to(seg, (ref.shape, ref.affine), order=0)
    data = np.asanyarray(seg_r.dataobj).astype(np.int32)

    # Build a header that matches the reference (uint8, 0/1), so masks are
    # byte-compatible with your other functional-space ROIs.
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.uint8)

    present = [int(l) for l in np.unique(data) if l != 0]
    mask_tot = {'L': np.zeros_like(data),
                'R': np.zeros_like(data) }
    for label in present:
        mask = (data == label).astype(np.uint8)
        mask_tot[LUT[label][0]] += mask
        n = int(mask.sum())
        fname = os.path.join(out_dir, f'Thalamus.{LUT[label]}.nii')
        if n < 10:                      # cannot happen here, kept for safety
            print(f'skipping {LUT[label]}, not enough voxels...'); continue
        img = nb.Nifti1Image(mask, ref.affine, hdr)
        nb.save(img, fname)

    for H in ['L', 'R']:
        img = nb.Nifti1Image(mask_tot[H], ref.affine, hdr)
        fname = os.path.join(out_dir, f'Thalamus.{H}.nii')
        nb.save(img, fname)


# if __name__ == '__main__':
#     start = time.time()

#     parser = argparse.ArgumentParser()

#     parser.add_argument('what', nargs='?', default=None)
#     parser.add_argument('--experiment', type=str, default='smp2')
#     parser.add_argument('--sn', type=int, default=None)
#     parser.add_argument('--atlas', type=str, default='ROI')
#     parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,])
#     parser.add_argument('--glm', type=int, default=12)

#     args = parser.parse_args()

#     main(args)
#     finish = time.time()

#     print(f'Execution time:{finish-start} s')

