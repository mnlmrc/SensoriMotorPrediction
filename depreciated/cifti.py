import argparse
from nitools import spm
from nibabel import cifti2
import nitools as nt
import nibabel as nb
import numpy as np

# def get_ciftis(mask=(None, None), SPM=None, TR=1000, extra=None):
#     """
#     Creates Cifti images to store the output of rerun_glm from an ROI defined in both hemispheres
#
#     Args:
#         mask ((str, str) tuple):
#             Path to the L and R (in this order) .nii ROI mask.
#             E.g. ('path/to/left_S1.nii', 'path/to/right_S1.nii')
#         SPM (SpmGlm):
#             Object containing first-levels GLMs estimated in SPM
#         TR (int):
#             Temporal resolution in milliseconds
#         extra (dict):
#             Extra information to store in Cifti images.
#             E.g., name of ROI: extra['name'] = 'S1'
#     Returns:
#         img_beta (CiftiImage): beta coefficients (PxQ), save as *.dscalar.nii
#         img_raw (CiftiImage): raw time series data (TxP), save as *.dtseries.nii
#         img_filt (CiftiImage): filtered time series data (TxP), save as *.dtseries.nii
#         img_hat (CiftiImage): predicted time series data (TxP), save as *.dtseries.nii
#             This is predicted only using regressors of interest (without the constant or other nuisance regressors)
#         img_adj (CiftiImage): adjusted time series data (TxP), save as *.dtseries.nii
#             This is filtered timeseries with constants and other nuisance regressors substrated out
#         img_res (CiftiImage): residuals (TxP), save as *.dtseries.nii
#     """
#
#     data = {name: [] for name in ["vox", "raw", "beta", "y_filt", "y_hat", "y_adj", "residuals"]}
#     struct = ['cortex_left', 'cortex_right']
#     for m, s in zip(mask, struct):
#         if m is not None:
#             # load mask
#             mask_img = nb.load(m)
#             coords = nt.get_mask_coords(mask_img)
#
#             # get raw data
#             raw_tmp = nt.sample_images(SPM.rawdata_files, coords)
#             data['raw'].append(raw_tmp)
#
#             # rerun glm
#             beta_tmp, info, y_filt_tmp, y_hat_tmp, y_adj_tmp, residuals_tmp = SPM.rerun_glm(raw_tmp)
#             data['beta'].append(beta_tmp)
#             data['y_filt'].append(y_filt_tmp)
#             data['y_hat'].append(y_hat_tmp)
#             data['y_adj'].append(y_adj_tmp)
#             data['residuals'].append(residuals_tmp)
#
#             data['vox'].append(cifti2.BrainModelAxis(
#                 name=s,
#                 voxel=nt.coords_to_voxelidxs(coords, mask_img).T,
#                 affine=mask_img.affine,
#                 volume_shape=mask_img.shape)
#             )
#
#     raw = np.hstack(data['raw'])
#     beta = np.hstack(data['beta'])
#     y_filt = np.hstack(data['y_filt'])
#     y_hat = np.hstack(data['y_hat'])
#     y_adj = np.hstack(data['y_adj'])
#     residuals = np.hstack(data['residuals'])
#
#     vox = data['vox'][0] + data['vox'][1]
#
#     series = cifti2.SeriesAxis(start=0, step=TR, size=raw.shape[0])
#
#     header = cifti2.Cifti2Header.from_axes((series, vox))
#     img_raw = Cifti2Image(dataobj=raw, header=header, extra=extra)
#     img_filt = Cifti2Image(dataobj=y_filt, header=header, extra=extra)
#     img_hat = Cifti2Image(dataobj=y_hat, header=header, extra=extra)
#     img_adj = Cifti2Image(dataobj=y_adj, header=header, extra=extra)
#     img_res = Cifti2Image(dataobj=residuals, header=header, extra=extra)
#
#     scalar = cifti2.ScalarAxis(name=info.reg_name)
#     header = cifti2.Cifti2Header.from_axes((scalar, vox))
#     img_beta = Cifti2Image(dataobj=beta, header=header, extra=extra)
#
#     return img_raw, img_beta, img_filt, img_hat, img_adj, img_res

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=103)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    if args.what == 'save_rois_timeseries':
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1']

        SPM = spm.SpmGlm(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}'))  #
        SPM.get_info_from_spm_mat()


if __name__ == '__main__':

    main()
