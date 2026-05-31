import argparse

import nibabel as nb
import nitools as nt
import os
import SensoriMotorPrediction.globals as gl
import numpy as np
from surfAnalysisPy.map import vol_to_surf
import SUITPy.flatmap as flatmap


def make_smooth_cifti(sns, glm, experiment='smp2'):
    data = []
    for sn in sns:
        print(f'Processing participant {sn}')

        # load cifti PSC
        path = os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}', 'psc.dscalar.nii')
        #giftis = [path + '/' + f'glm{glm}.con.{H}.func.gii' for H in gl.Hem]
        cifti_psc = nb.load(path)

        # convert to nifti list
        nifti_psc = nt.volume_from_cifti(cifti_psc, struct_names=gl.struct)
        vol_list_psc = [nb.Nifti2Image(nifti_psc.get_fdata()[..., i], nifti_psc.affine) for i in range(nifti_psc.shape[-1])]

        # load white and pial
        whiteL = os.path.join(gl.baseDir, experiment, gl.wbDir, f'subj{sn}', f'subj{sn}.L.white.32k.surf.gii')
        whiteR = os.path.join(gl.baseDir, experiment, gl.wbDir, f'subj{sn}', f'subj{sn}.R.white.32k.surf.gii')
        pialL = os.path.join(gl.baseDir, experiment, gl.wbDir, f'subj{sn}', f'subj{sn}.L.pial.32k.surf.gii')
        pialR = os.path.join(gl.baseDir, experiment, gl.wbDir, f'subj{sn}', f'subj{sn}.R.pial.32k.surf.gii')

        # retrieve names
        row_axis = cifti_psc.header.get_axis(0).name
        plan_col_names = [col for col in row_axis if 'index' not in col 
                                                    and 'ring' not in col 
                                                    # these two are because of GLM 17
                                                    and 'exec' not in col 
                                                    and 'plan' not in col]
        exec_col_names = [col for col in row_axis if 'index' in col or 'ring' in col or 'exec' in col]

        # project to surface
        giftiL = nt.gifti.make_func_gifti(vol_to_surf(vol_list_psc, whiteL, pialL), anatomical_struct='CortexLeft', column_names=row_axis) 
        giftiR = nt.gifti.make_func_gifti(vol_to_surf(vol_list_psc, whiteR, pialR), anatomical_struct='CortexRight', column_names=row_axis)
        cifti_img = nt.join_giftis_to_cifti([giftiL, giftiR])

        data_tmp = cifti_img.get_fdata()

        im = np.array([x in plan_col_names for x in row_axis])
        data_plan = np.array(data_tmp[im]).mean(axis=0)

        im = np.array([x in exec_col_names for x in row_axis])
        data_exec = np.array(data_tmp[im]).mean(axis=0)

        data.append(np.vstack([data_plan, data_exec]))

        if sns.index(sn) == 0:
            brain_axis = cifti_img.header.get_axis(1)

    data = np.array(data).mean(axis=0)

    row_axis = nb.cifti2.ScalarAxis(['plan', 'exec'])
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti_img = nb.Cifti2Image(
        dataobj=data,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    nb.save(cifti_img, os.path.join(gl.baseDir, experiment, gl.wbDir, f'glm{glm}.psc.plan-exec.dscalar.nii'))

    nt.smooth_cifti(os.path.join(gl.baseDir, experiment, gl.wbDir, f'glm{glm}.psc.plan-exec.dscalar.nii'),
                    os.path.join(gl.baseDir, experiment, gl.wbDir, f'glm{glm}.psc.plan-exec.smooth.dscalar.nii'),
                    'atlases/fs_LR.32k.L.flat.surf.gii', 'atlases/fs_LR.32k.R.flat.surf.gii')

    # giftis = nt.split_cifti_to_giftis(cifti_img, column_names=['plan', 'exec'], type='func')
    # nb.save(giftis[0], os.path.join(gl.baseDir, experiment, gl.wbDir, f'glm{glm}.psc.L.plan-exec.smooth.func.gii'))
    # nb.save(giftis[1], os.path.join(gl.baseDir, experiment, gl.wbDir, f'glm{glm}.psc.R.plan-exec.smooth.func.gii'))



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
    if args.what=='smooth_cifti':
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