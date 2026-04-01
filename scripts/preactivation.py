import numpy as np
import pandas as pd
import SensoriMotorPrediction.globals as gl
import os
import nibabel as nb
import nitools as nt

if __name__=="__main__":

    atlas = 'ROI'
    glm = 12
    rois = gl.rois[atlas]
    Dict = {'sn': [],
            'roi': [],
            'Hem': [],
            'diff': [],
            #'ring': [],
            '100-0%': [],
            '75-25%': [],
            '50-50%': [],
            '25-75%': [],
            '0-100%': [],
            'corr_with_100-0%': [],
            'corr_with_75-25%': [],
            'corr_with_50-50%': [],
            'corr_with_25-75%': [],
            'corr_with_0-100%': [],}
    
    for sn in gl.sns:
        path_glm = os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'subj{sn}')
        cifti = nb.load(os.path.join(path_glm, f'W.regr_out_preact_ancova.dscalar.nii'))
        vol = nt.volume_from_cifti(cifti, struct_names=['CortexLeft', 'CortexRight'])
        for H in gl.Hem:
            for r, roi in enumerate(rois):
                print(f'doing participant {sn}, {H}, {roi}...')
                path_roi = os.path.join(gl.baseDir, 'smp2', f'roi', f'subj{sn}')
                mask = nb.load(os.path.join(path_roi, f'{atlas}.{H}.{roi}.nii'))
                coords = nt.get_mask_coords(mask)
                data = nt.sample_image(vol, coords[0], coords[1], coords[2], 0).T
                data = data[:, ~np.isnan(data).all(axis=0)]
                corr = np.corrcoef(data)
                Dict['sn'].append(sn)
                Dict['roi'].append(roi)
                Dict['Hem'].append(H)
                Dict['diff'].append(np.nanmean(data[0]))
                Dict['100-0%'].append(np.nanmean(data[1]))
                Dict['75-25%'].append(np.nanmean(data[2]))
                Dict['50-50%'].append(np.nanmean(data[3]))
                Dict['25-75%'].append(np.nanmean(data[4]))
                Dict['0-100%'].append(np.nanmean(data[5]))
                Dict['corr_with_100-0%'].append(corr[0, 1])
                Dict['corr_with_75-25%'].append(corr[0, 2])
                Dict['corr_with_50-50%'].append(corr[0, 3])
                Dict['corr_with_25-75%'].append(corr[0, 4])
                Dict['corr_with_0-100%'].append(corr[0, 5])

                pass
    
    df = pd.DataFrame(Dict)
    df.to_csv(os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'preactivation_weights.tsv'), sep='\t', index=False)