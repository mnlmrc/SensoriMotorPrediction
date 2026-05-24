import numpy as np
import pandas as pd
import SensoriMotorPrediction.globals as gl
import os
import nibabel as nb
import nitools as nt
from surfAnalysisPy.map import vol_to_surf

if __name__=="__main__":

    atlas = 'ROI'
    glm = 16
    experiment = 'smp2'
    rois = gl.rois[atlas]
    cols = gl.stimFinger + gl.cues
    surf_array = np.zeros((len(gl.sns), 2, 32492, 7))
    corr = np.zeros((len(gl.sns), 2, 7, 7, 7))
    Dict = {'sn': [],
            'roi': [],
            'Hem': [],
            'index': [],
            'ring': [],
            '100-0%': [],
            '75-25%': [],
            '50-50%': [],
            '25-75%': [],
            '0-100%': [],
            'corr_index_with_ring': [],
            'corr_index_with_100-0%': [],
            'corr_index_with_75-25%': [],
            'corr_index_with_50-50%': [],
            'corr_index_with_25-75%': [],
            'corr_index_with_0-100%': [],
            'corr_ring_with_100-0%': [],
            'corr_ring_with_75-25%': [],
            'corr_ring_with_50-50%': [],
            'corr_ring_with_25-75%': [],
            'corr_ring_with_0-100%': [],}
    
    for s, sn in enumerate(gl.sns):
        path_glm = os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'subj{sn}')
        cifti = nb.load(os.path.join(path_glm, f'W.regr_out_preact_ancova.dscalar.nii'))
        vol = nt.volume_from_cifti(cifti, struct_names=['CortexLeft', 'CortexRight'])        
        for h, H in enumerate(gl.Hem):
            white = os.path.join(gl.baseDir, experiment, gl.surfDir, f'subj{sn}', f'subj{sn}.{H}.white.32k.surf.gii')
            pial = os.path.join(gl.baseDir, experiment, gl.surfDir, f'subj{sn}', f'subj{sn}.{H}.pial.32k.surf.gii')
            vol_list = []
            for v in range(vol.shape[-1]):
                vol_tmp = nb.Nifti2Image(vol.get_fdata()[:, :, :, v], affine=vol.affine, header=vol.header)
                vol_list.append(vol_tmp)

            surf_array[s, h] = vol_to_surf(vol_list, white, pial,)

            for r, roi in enumerate(rois):
                print(f'doing participant {sn}, {H}, {roi}...')
                path_roi = os.path.join(gl.baseDir, 'smp2', f'roi', f'subj{sn}')
                mask = nb.load(os.path.join(path_roi, f'{atlas}.{H}.{roi}.nii'))
                coords = nt.get_mask_coords(mask)
                data = nt.sample_image(vol, coords[0], coords[1], coords[2], 0).T
                data = data[:, ~np.isnan(data).all(axis=0)]
                corr[s, h, r] = np.corrcoef(data)
                
    giftiL = nt.make_func_gifti(surf_array[:, 0].mean(axis=0), anatomical_struct='CortexLeft', column_names=cols)
    giftiR = nt.make_func_gifti(surf_array[:, 1].mean(axis=0), anatomical_struct='CortexRight', column_names=cols)
    
    nb.save(giftiL, os.path.join(gl.baseDir, experiment, gl.surfDir, f'W.regr_out_preact_ancova.glm{glm}.plan.L.func.gii'))
    nb.save(giftiR, os.path.join(gl.baseDir, experiment, gl.surfDir, f'W.regr_out_preact_ancova.glm{glm}.plan.R.func.gii'))

    for h, H in enumerate(gl.Hem):
        for r, roi in enumerate(rois):
            np.save(os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'W.regr_out_preact_ancova.corr.{H}.{roi}.npy'), corr[:, h, r])
