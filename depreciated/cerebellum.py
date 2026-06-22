# Import the Suit package
import SUITPy.flatmap as flatmap
import Functional_Fusion.atlas_map as am
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import nitools as nt
import os
import globals as gl

experiment = 'smp2'
glm = 12
sn = 102
H = 'R'

atlas, _ = am.get_atlas('SUIT1')
rois = [f'M1{H}', f'M2{H}', f'M3{H}', f'A1{H}', f'A2{H}', f'A3{H}']
_,_,labels = nt.read_lut(os.path.join(gl.baseDir, experiment, 'SUIT', 'atl-NettekovenSym32.lut'))
label_value = []
for roi in rois:
    label_value.append(labels.index(roi))
subatlas = atlas.get_subatlas_image(os.path.join(gl.baseDir, experiment, 'SUIT', 'atl-NettekovenSym32_space-SUIT_dseg.nii'), label_value)

deform = os.path.join(gl.baseDir, experiment, 'SUIT', 'anatomicals', f'subj{sn}', f'y_subj{sn}_anatomical_suitdef.nii')
mask = os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}', 'mask.nii')
amap = am.AtlasMapDeform(subatlas.world,None,mask)
amap.build(interpolation=1)  # Using Trilinear interpolation (0 for nearest neighbor, 2 for smoothing)
img = os.path.join(gl.baseDir, experiment, 'SUIT', 'ROI', f'subj{sn}', f'cerebellum.{H}.nii')
amap.save_as_image(img)

