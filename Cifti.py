import argparse

import nibabel as nb
import nitools as nt
import os
import globals as gl
import numpy as np

def main(args):

    Hem = ['L', 'R']

    if args.what=='gifti2cifti_glm':
        print(f'Processing participant {args.sn}')
        path = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
        giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)
        nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
    if args.what=='gifti2cifti_glm_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='gifti2cifti_glm',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                dtype=args.dtype,
            )
            main(args)
    if args.what=='save_surface_cifti_avg':
        data = []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            img = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                               f'glm{args.glm}.{args.dtype}.dscalar.nii')
            cifti = nb.load(img)
            data.append(cifti.dataobj)

            if args.snS.index(sn) == 0:
                brain_axis = cifti.header.get_axis(1)
                row_axis = cifti.header.get_axis(0)
    if args.what=='save_smoothed_surface_cifti_avg':
        nt.smooth_cifti(os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                               f'glm{args.glm}.{args.dtype}.dscalar.nii'),
                        os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                     f'glm{args.glm}.{args.dtype}.smooth.dscalar.nii'),
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.flat.surf.gii',
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.flat.surf.gii'
                        )

        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        # save y_raw
        print(f'saving average surface for glm{args.glm} - dtype: {args.dtype}')
        cifti = nb.Cifti2Image(
            dataobj=np.array(data).mean(axis=0),
            header=header,
        )
        nb.save(cifti, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'glm{args.glm}.{args.dtype}.dscalar.nii'))


    if args.what == 'gifti2cifti_tessellation':

        for h, H in enumerate(Hem):

            T, T_col_names = list(), list()
            theta_comp, tc_col_names = list(), list()
            theta_feat, tf_col_names = list(), list()
            for sn in args.snS:

                print(f'Hemisphere: {H}, subj{sn}')

                g_T = nb.load(os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                          f'ML.Icosahedron{args.ntessels}.glm{args.glm}.pcm.exec.{H}.func.gii'))
                T.append(nt.get_gifti_data_matrix(g_T))
                T_col_names = nt.get_gifti_column_names(g_T)

                g_theta_comp = nb.load(os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                           f'theta.Icosahedron{args.ntessels}.component.glm{args.glm}.pcm.exec.{H}.func.gii'))
                theta_comp.append(nt.get_gifti_data_matrix(g_theta_comp))
                tc_col_names = nt.get_gifti_column_names(g_theta_comp)

                g_theta_feat = nb.load(os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                                           f'theta.Icosahedron{args.ntessels}.feature.glm{args.glm}.pcm.exec.{H}.func.gii'))
                theta_feat.append(nt.get_gifti_data_matrix(g_theta_feat))
                tf_col_names = nt.get_gifti_column_names(g_theta_feat)

            gifti_img_T = nt.make_func_gifti(np.array(T).mean(axis=0), anatomical_struct=struct[h],
                                             column_names=T_col_names)
            gifti_img_theta_comp = nt.make_func_gifti(np.array(theta_comp).mean(axis=0), anatomical_struct=struct[h],
                                                           column_names=tc_col_names)
            gifti_img_theta_feat = nt.make_func_gifti(np.array(theta_feat).mean(axis=0), anatomical_struct=struct[h],
                                                         column_names=tf_col_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    parser.add_argument('--dtype', type=str, default='con')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)