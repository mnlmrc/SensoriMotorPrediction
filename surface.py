import argparse

import nibabel as nb
import nitools as nt
import os
import globals as gl
import numpy as np
import SUITPy.flatmap as flatmap

def main(args):

    Hem = ['L', 'R']

    if args.what=='gifti2cifti':
        print(f'Processing participant {args.sn}')
        path = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
        giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)
        nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
    if args.what=='gifti2cifti_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='gifti2cifti',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                dtype=args.dtype,
            )
            main(args)
    if args.what == 'surface_cifti_avg':
        data = []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            img = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                               f'glm{args.glm}.{args.dtype}.dscalar.nii')
            cifti = nb.load(img)
            data.append(cifti.get_fdata())

            if args.snS.index(sn) == 0:
                brain_axis = cifti.header.get_axis(1)
                row_axis = cifti.header.get_axis(0)

        data = np.array(data).mean(axis=0)
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_img = nb.Cifti2Image(
            dataobj=data,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti_img, os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                               f'glm{args.glm}.{args.dtype}.dscalar.nii'))

    if args.what=='surface_cifti_avg_smoothed':
        nt.smooth_cifti(os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                               f'glm{args.glm}.{args.dtype}.dscalar.nii'),
                        os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                     f'glm{args.glm}.{args.dtype}.smooth.dscalar.nii'),
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.flat.surf.gii',
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.flat.surf.gii'
                        )
        cifti = nb.load(os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                     f'glm{args.glm}.{args.dtype}.smooth.dscalar.nii'))
        # giftis = nt.split_cifti_to_giftis(cifti)

        pass

    if args.what=='plan-exec':

        data = []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            img = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                               f'glm{args.glm}.{args.dtype}.dscalar.nii')
            cifti_img = nb.load(img)

            row_axis = cifti_img.header.get_axis(0).name
            plan_col_names = [col for col in row_axis if 'index' not in col and 'ring' not in col]
            exec_col_names = [col for col in row_axis if 'index' in col or 'ring' in col or 'exec' in col]

            data_tmp = cifti_img.get_fdata()

            im = np.array([x in plan_col_names for x in row_axis])
            data_plan = np.array(data_tmp[im]).mean(axis=0)

            im = np.array([x in exec_col_names for x in row_axis])
            data_exec = np.array(data_tmp[im]).mean(axis=0)

            data.append(np.vstack([data_plan, data_exec]))

            if args.snS.index(sn) == 0:
                brain_axis = cifti_img.header.get_axis(1)

        data = np.array(data).mean(axis=0)

        row_axis = nb.cifti2.ScalarAxis(['plan', 'exec'])
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti_img = nb.Cifti2Image(
            dataobj=data,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
        nb.save(cifti_img, os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                        f'glm{args.glm}.{args.dtype}.plan-exec.dscalar.nii'))

        nt.smooth_cifti(os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                     f'glm{args.glm}.{args.dtype}.plan-exec.dscalar.nii'),
                        os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                     f'glm{args.glm}.{args.dtype}.plan-exec.smooth.dscalar.nii'),
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_L/fs_LR.32k.L.flat.surf.gii',
                        '/home/UWO/memanue5/Documents/GitHub/surfAnalysisPy/standard_mesh/fs_R/fs_LR.32k.R.flat.surf.gii'
                        )

        giftis = nt.split_cifti_to_giftis(cifti_img, column_names=['plan', 'exec'], type='func')
        nb.save(giftis[0], os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                        f'glm{args.glm}.{args.dtype}.L.plan-exec.smooth.func.gii'))
        nb.save(giftis[1], os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                        f'glm{args.glm}.{args.dtype}.R.plan-exec.smooth.func.gii'))

    if args.what == 'save_smoothed_flatmap_cerebellum_avg':
        funcdata_across = []
        for sn in args.snS:
            path = os.path.join(gl.baseDir, args.experiment, 'SUIT', f'glm{args.glm}', f'subj{sn}')
            funcdata_within, col_name = [], []
            for root, dirs, file in os.walk(path):
                for f in file:
                    if 'con' in f:
                        print(f'Processing subject {root}/{f}')
                        funcdata_within.append(flatmap.vol_to_surf(os.path.join(root, f)))
                        col_name.append(f.split('.')[0])
            funcdata_across.append(np.array(funcdata_within).T)
        funcdata_across = np.array(funcdata_across).mean(axis=0).squeeze()
        gifti = nt.make_func_gifti(funcdata_across, anatomical_struct='Cerebellum', column_names=col_name)
        nb.save(gifti, os.path.join(gl.baseDir,args.experiment,'SUIT',f'glm{args.glm}.{args.dtype}.flat.surf.gii'))

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
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--dtype', type=str, default='con')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)