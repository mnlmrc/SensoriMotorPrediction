import nitools as nt
import nibabel as nb
import os
import numpy as np
import SensoriMotorPrediction.globals as gl
from sklearn.preprocessing import MinMaxScaler
from nibabel.gifti import GiftiImage, GiftiDataArray

if __name__=='__main__':

    mclip = .2
    threshold = .05 / mclip
    scaler = MinMaxScaler()

    experiment = 'smp2'

    for H, struct in zip(gl.Hem, gl.struct):

        gifti = nb.load(os.path.join(gl.baseDir, experiment, gl.surfDir, f'searchlight.var_expl.plan.{H}.func.gii'))
        data = nt.get_gifti_data_matrix(gifti)
        raw_max = np.nanmax(data)
        data = data / raw_max#scaler.fit_transform(data)
        raw_min = 0 #scaler.inverse_transform([[threshold * mclip, threshold * mclip]])[0,0]
        data = np.clip(data / mclip, 0, 1)

        sulc = nt.get_gifti_data_matrix(nb.load(os.path.join('atlases', 'fs_LR.32k.LR.sulc.dscalar.gii')))
        sulc = sulc[:len(data)]
        sulc_norm = MinMaxScaler((0.3, 0.7)).fit_transform(sulc.reshape(-1, 1)).flatten()

        rgba = np.zeros((len(sulc_norm), 4))
        rgba[:, 0] = sulc_norm  # red = grey
        rgba[:, 1] = sulc_norm  # green = grey
        rgba[:, 2] = sulc_norm  # blue = grey
        rgba[:, 3] = 1.0        # opaque background

        overlay_mask = (data[:, 0] >= threshold) | (data[:, 1] >= threshold)

        rgba[overlay_mask, 0] = data[overlay_mask, 0]  # red
        rgba[overlay_mask, 1] = 0                    # green stays off for 2-color blend
        rgba[overlay_mask, 2] = data[overlay_mask, 1]  # blue

        darray = GiftiDataArray(
            data=(rgba[:, :3] * 255).astype(np.uint8),
            intent=nb.nifti1.intent_codes['NIFTI_INTENT_RGB_VECTOR'],
            datatype='NIFTI_TYPE_UINT8'
        )

        #img = nt.make_func_gifti(darray, anatomical_struct=struct)
        img = GiftiImage(darrays=[darray])
        nb.save(img, os.path.join(gl.baseDir, experiment, gl.surfDir, f'searchlight.var_expl.plan.rgba.{H}.func.gii'))

