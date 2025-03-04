import os
import nibabel as nb
import globals as gl

import argparse

import nitools as nt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--Hem', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--ntessels', type=int, default=362, choices=[42, 162, 362, 642, 1002, 1442])

    args = parser.parse_args()

    Hem = ['L', 'R']
    struct = ['CortexLeft', 'CortexRight']

    if args.what == 'save_tessel_execution':

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

            nb.save(gifti_img_T, os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                              f'ML.Icosahedron{args.ntessels}.glm{args.glm}.pcm.exec.{H}.func.gii'))
            nb.save(gifti_img_theta_comp, os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                                            f'theta.Icosahedron{args.ntessels}.component.glm{args.glm}.pcm.exec.{H}.func.gii'))
            nb.save(gifti_img_theta_feat, os.path.join(gl.baseDir, args.experiment, gl.wbDir,
                                                          f'theta.Icosahedron{args.ntessels}.feature.glm{args.glm}.pcm.exec.{H}.func.gii'))
