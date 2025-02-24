import sys
import os
import globals as gl
import nitools as nt
import nibabel as nb

import argparse

sys.path.append('/Users/mnlmrc/Documents/GitHub/Functional_Fusion')
sys.path.append('/home/ROBARTS/memanue5/Documents/GitHub/Functional_Fusion')

import Functional_Fusion.atlas_map as am

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=108)
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    atlas, _ = am.get_atlas('fs32k')

    Hem = ['L', 'R']

    struct = ['CortexLeft', 'CortexRight']

    for h, H in enumerate(Hem):

        g_atlas = nb.load(os.path.join(gl.atlas_dir, f'{args.atlas}.32k.{Hem[h]}.label.gii'))

        labels = {
            ele.key: getattr(ele, 'label', '')
            for ele in g_atlas.labeltable.labels
        }

        for nlabel, label in enumerate(labels.values()):

            print(f'ROI: {label}')

            atlas_hem = atlas.get_hemisphere(h)
            subatlas = atlas_hem.get_subatlas_image(os.path.join(gl.atlas_dir,
                                                                     f'{args.atlas}.32k.{H}.label.gii'), nlabel)

            subj_dir = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{args.sn}')
            white = os.path.join(subj_dir, f'subj{args.sn}.L.white.32k.surf.gii')
            pial = os.path.join(subj_dir, f'subj{args.sn}.L.pial.32k.surf.gii')
            mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
            amap = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
            amap.build()

            mask = amap.save_as_image(os.path.join(gl.baseDir, args.experiment, gl.roiDir, f'subj{args.sn}',
                                                   f'{args.atlas}.{H}.{label}.nii'))