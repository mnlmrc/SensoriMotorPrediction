import os
import nibabel as nb
import SensoriMotorPrediction.globals as gl
import nitools as nt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    experiment = 'smp2'
    atlas_name = 'ROI'
    GLMs = [12, 17]

    for glm in GLMs:

        con_dict = {'con': [],
                    'psc': [],
                    'condition': [],
                    'sn': [],
                    'roi': [],
                    'Hem': [],
                    'epoch': []}
        regr_idx = {12: [2, 10, 7, 4, 0,  3, 11, 8, 5,  12, 9, 6, 1],
                    17: [1, 4, 3, 2, 0,  6,  5]}
        for sn in gl.sns:
            print(f'Processing subj{sn}')
            path_glm = os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}')
            path_rois = os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}')
            cifti_psc = nb.load(path_glm + '/' + 'psc.dscalar.nii')
            cifti_con = nb.load(path_glm + '/' + 'contrast.dscalar.nii')
            regr = cifti_psc.header.get_axis(0).name[regr_idx[glm]]
            vol_psc = nt.volume_from_cifti(cifti_psc, struct_names=gl.struct)
            vol_con = nt.volume_from_cifti(cifti_con, struct_names=gl.struct)
            for H in gl.Hem:
                for roi in gl.rois[atlas_name]:
                    mask = os.path.join(path_rois, f'ROI.{H}.{roi}.nii')
                    coords = nt.get_mask_coords(mask)
                    con = nt.sample_image(vol_con, coords[0], coords[1], coords[2],0)
                    psc = nt.sample_image(vol_psc, coords[0], coords[1], coords[2],0)
                    con_avg = np.nanmean(con, axis=0)[regr_idx[glm]]
                    psc_avg = np.nanmean(psc, axis=0)[regr_idx[glm]]
                    for i, (c, p) in enumerate(zip(con_avg, psc_avg)):
                        con_dict['con'].append(c)
                        con_dict['psc'].append(p)
                        con_dict['condition'].append(regr[i])
                        con_dict['sn'].append(str(sn))
                        con_dict['roi'].append(roi)
                        con_dict['Hem'].append(H)
                        epoch = 'exec' if ('index' in regr[i]) or ('ring' in regr[i]) else 'plan'
                        con_dict['epoch'].append(epoch)
        con = pd.DataFrame.from_dict(con_dict)
        con.to_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', 'ROI.activation.tsv'), sep='\t', index=False)