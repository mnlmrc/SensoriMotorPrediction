import argparse
import os

import mat73
import nitools as nt
import nitools.gifti
import numpy as np
import scipy
from matplotlib import pyplot as plt
import globals as gl
import pandas as pd
from force import Force
from pcm import make_Z_all, FixedModel, make_Z_cue
from plot import make_colors, make_tAx, make_yref
# import tkinter as tk
import seaborn as sns
from rsa import plot_rdm, draw_contours
import nibabel as nb
import rsatoolbox as rsa

import PcmPy as pcm

import sys

sys.path.append('/Users/mnlmrc/Documents/GitHub')
sys.path.append('/home/ROBARTS/memanue5/Documents/GitHub')

import surfAnalysisPy as surf

def main(what, experiment=None, session=None, sn=None, GoNogo=None, stimFinger=None, cue=None,
         glm=None, Hem=None, regressor=None, roi=None, derivs=None, prefix=None,
         fig=None, axs=None, vsep=None, xlim=None, ylim=None, vmin=None, vmax=None, ref_len=None):

    # if participant_id is None:
    #     participant_id = gl.participants[experiment]

    match what:

        # region FORCE:mov2npz
        case 'FORCE:mov2npz':
            for p in participant_id:
                force = Force(experiment, session, p)
                force_segmented, descr = force.segment_mov()

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}.npz'),
                         data_array=force_segmented, descriptor=descr, allow_pickle=False)
        # endregion

        # region FORCE:timec_avg
        case 'FORCE:timec_avg':

            force_avg = list()
            for p in participant_id:
                force = Force(experiment, session, p)
                force_avg.append(force.calc_avg_timec(GoNogo=GoNogo))

            force_avg = np.array(force_avg)

            return force_avg
        # endregion

        # region FORCE:timec2bins
        case 'FORCE:timec2bins':

            df = pd.DataFrame()
            for p in participant_id:
                force = Force(experiment, session, p)
                df_tmp = force.calc_bins()
                df_tmp['participant_id'] = p
                df = pd.concat([df, df_tmp])

            df.to_csv(os.path.join(Force(experiment, session).get_path(), 'bins.force.csv'))

            return df
        # endregion

        # region FORCE:timec_dist
        case 'FORCE:timec_dist':

            for p in participant_id:
                force = Force(experiment, session, p)
                dist, descr = force.calc_dist_timec(method='crossnobis', GoNogo=GoNogo)

                print(f"Saving participant {p}, session {session}...")
                np.savez(os.path.join(force.get_path(), p, f'{force.experiment}_{force.sn}_dist_{GoNogo}.npz'),
                         data_array=dist, descriptor=descr, allow_pickle=False)
        # endregion

        # region RDM:roi
        case 'RDM:roi':

            # RDMs = list()
            # for p in participant_id:
            # load ROI file into python
            print('loading R...')
            mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, participant_id,
                                                f'{participant_id}_ROI_region.mat'))
            R_cell = mat['R'][0]
            R = list()
            for r in R_cell:
                R.append({field: r[field].item() for field in r.dtype.names})

            # find roi where to calc RDM
            R = R[[True if (r['name'].size > 0) and (r['name'] == roi) and (r['hem'] == Hem)
                   else False for r in R].index(True)]

            # R_index = [((r['name'] == roi), (r['hem'] == Hem)) for r in R]

            print(f'region:{roi}, hemisphere:{Hem}, {len(R["data"])} voxels')

            # define path
            pathGlm = os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', participant_id)

            # # load SPM
            # print('loading iB...')
            # try:
            #     iB = mat73.loadmat(os.path.join(pathGlm, f'iB.mat'))['iB'].squeeze()
            # except:
            #     iB = scipy.io.loadmat(os.path.join(pathGlm, f'iB.mat'))['iB'].squeeze()

            # load reginfo
            print('loading reginfo...')

            reginfo = pd.read_csv(os.path.join(pathGlm, f'{participant_id}_reginfo.tsv'), sep='\t')

            # load residual mean squared for univariate pre-whitening
            ResMS = nb.load(os.path.join(pathGlm, 'ResMS.nii'))
            res = nt.sample_image(ResMS, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)

            betas_prewhitened = list()
            betas = list()
            for n_regr in np.arange(0, reginfo.shape[0], derivs.sum() + 1):
                vol = nb.load(os.path.join(pathGlm, f'{prefix}beta_{n_regr+1:04d}.nii'))
                beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
                betas.append(beta)
                betas_prewhitened.append(beta / np.sqrt(res))

            betas_prewhitened = np.array(betas_prewhitened)
            betas = np.array(betas)
            dataset = rsa.data.Dataset(
                betas_prewhitened,
                channel_descriptors={'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
                obs_descriptors={'conds': reginfo.name.iloc[::derivs.sum() + 1].reset_index(drop=True),
                                 'run': reginfo.run[::derivs.sum() + 1].reset_index(drop=True)})
            rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run')
            rdm.rdm_descriptors = {'roi': R["name"], 'hem': R["hem"], 'index': [0]}
            rdm.reorder(np.argsort(rdm.pattern_descriptors['conds']))
            rdm.reorder(gl.rdm_index[f'glm{glm}'])
            rdm.pattern_descriptors['conds'] = [c.replace(" ", "") for c in rdm.pattern_descriptors['conds']]
            rdm.save(os.path.join(gl.baseDir, experiment, gl.rdmDir, participant_id, f'glm{glm}.{Hem}.{roi}.hdf5'),
                     overwrite=True, file_type='hdf5')
            # RDMs.append(rdm)

            # RDMs = rsa.rdm.concat(RDMs).mean()
            # RDMs.pattern_descriptors['conds'] = [c.replace(" ", "") for c in RDMs.pattern_descriptors['conds']]
            # RDMs.save(os.path.join(gl.baseDir, experiment, gl.rdmDir, f'glm{glm}.{Hem}.{roi}.hdf5'),
            #           overwrite=True, file_type='hdf5')
            #
            # return RDMs
        # endregion

        # region RDM:rois
        case 'RDM:rois':
            rois = gl.rois['ROI']
            Hem = ['L', 'R']
            for H in Hem:
                for r in rois:
                    main('RDM:roi', experiment=experiment, participant_id=participant_id, roi=r, Hem=H, glm=glm,
                         derivs=derivs, prefix=prefix)

        # endregion

        # region PCM:FixedModel
        case 'PCM:FixedModel':

            reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                               f'subj{sn}_reginfo.tsv'), sep="\t")
            reginfo[['cue', 'stimFinger']] = reginfo['name'].str.split(',', expand=True)
            reginfo.cue = reginfo.cue.str.replace(" ", "")
            reginfo.stimFinger = reginfo.stimFinger.fillna('nogo')
            reginfo = reginfo[reginfo['run'] == 1]

            Z_all = pcm.matrix.indicator(reginfo.name)
            Z_cue = pcm.matrix.indicator(reginfo.cue)
            Z_stimFinger = pcm.matrix.indicator(reginfo.stimFinger)

            M_all, G_all = FixedModel('all', Z_all)
            M_cue, G_cue = FixedModel('cue', Z_cue)
            M_stimFinger, G_stimFinger = FixedModel('stimFinger', Z_stimFinger)
            MC = pcm.model.ComponentModel('cue+stimFinger', [M_cue + M_stimFinger])

            mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}',
                                                f'subj{sn}_ROI_region.mat'))
            R_cell = mat['R'][0]
            R = list()
            for r in R_cell:
                R.append({field: r[field].item() for field in r.dtype.names})

            # find roi where to calc RDM
            R = R[[True if (r['name'].size > 0) and (r['name'] == 'M1') and (r['hem'] == 'L')
                   else False for r in R].index(True)]

            reginfo = pd.read_csv(
                os.path.join(gl.baseDir, 'smp2', gl.glmDir + '12', f'subj{sn}', f'subj{sn}_reginfo.tsv'),
                sep='\t')

            betas = list()
            for n_regr in np.arange(0, reginfo.shape[0]):
                print(f'loading regressor #{n_regr + 1}')

                vol = nb.load(
                    os.path.join(gl.baseDir, 'smp2', gl.glmDir + '12', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
                beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
                betas.append(beta)

            betas = np.array(betas)

            dataset = pcm.dataset.Dataset(
                betas,
                obs_descriptors={'cond_vec': reginfo.name,
                                 'part_vec': reginfo.run})

            T, theta = pcm.inference.fit_model_individ(dataset, [M_all, M_cue, M_stimFinger, MC], fixed_effect='block')

            print(T)

        # endregion

        # region PLOT:timec_force
        case 'PLOT:timec_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            force = main('FORCE:timec_avg', experiment, session, participant_id, GoNogo=GoNogo)  # dimord: (subj, cue, stimFinger, channel, time)
            # clamp = np.load(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped.npy')).mean(axis=0)[[1, 3]]

            tAx = make_tAx(force) if GoNogo == 'go' else make_tAx(force, (0, 0))  # in nogo trials tAx is not corrected for the latency of perturbation

            colors = make_colors(5)
            palette = {cue: color for cue, color in zip(gl.clabels, colors)}

            sf = gl.stimFinger.index(stimFinger) if GoNogo == 'go' else None

            for col, color in enumerate(palette):
                for c, ch in enumerate(gl.channels['mov']):
                    if GoNogo == 'go':
                        # for sf, stimF in enumerate(['index', 'ring']):
                            # axs[sf].set_title(f'{stimF} perturbation')

                        y = force.mean(axis=0)[:, sf, c] + c * vsep
                        yerr = force.std(axis=0)[:, sf, c] / np.sqrt(force.shape[0])

                        axs.plot(tAx[sf], y[col], color=palette[color])
                        axs.fill_between(tAx[sf], y[col] - yerr[col], y[col] + yerr[col],
                                             color=palette[color], lw=0, alpha=.2)

                    elif GoNogo == 'nogo':

                        # axs.set_title(f'nogo trials')

                        y = force.mean(axis=0)[:, c] + c * vsep
                        yerr = force.std(axis=0)[:, c] / np.sqrt(force.shape[0])
                        axs.plot(tAx[0], y[col], color=palette[color])
                        axs.fill_between(tAx[0], y[col] - yerr[col], y[col] + yerr[col],
                                         color=palette[color], lw=0, alpha=.2)

            return fig, axs
        # endregion

        # region PLOT:bins_force
        case 'PLOT:bins_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots(len(gl.channels['mov']), len(gl.stimFinger_code),
                                        figsize=(5, 8), sharex=True, sharey=True)

            timew = ['Pre', 'LLR', 'Vol']

            df = pd.read_csv(os.path.join(Force(experiment, session).get_path(), 'bins.force.csv'))
            df = df[df['GoNogo'] == GoNogo]

            value_vars = [f'{time}/{ch}' for time in timew for ch in gl.channels['mov']]

            df = df.groupby(['participant_id', 'cue', 'stimFinger'])[value_vars].mean().reset_index()

            df = df.melt(id_vars=['cue', 'stimFinger', 'participant_id'],
                         value_vars=value_vars,
                         var_name='timew/finger',
                         value_name='force (N)')
            df[['timew', 'finger']] = df['timew/finger'].str.split('/', expand=True)
            df = df.drop(columns=['timew/finger'])

            df['cue'] = df['cue'].map(gl.cue_mapping)

            colors = make_colors(5)
            palette = {cue: color for cue, color in zip(gl.clabels, colors)}

            for c, ch in enumerate(gl.channels['mov']):
                for sF, stimF in enumerate(gl.stimFinger_code):
                    df_tmp = df[(df['finger'] == ch) & (df['stimFinger'] == stimF) & df['timew'].isin(timew)]

                    sns.boxplot(data=df_tmp, ax=axs[c, sF], x='timew', y='force (N)',
                                hue='cue', hue_order=gl.clabels, palette=palette)
                    axs[c, sF].legend_.remove()  # Remove the individual legend
                    axs[c, sF].set_title(ch)

                    axs[c, sF].set_ylabel('')
                    axs[c, sF].set_xlabel('')

                    # axs[c, sF].set_yscale(yscale)

            # Create a single legend at the top
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=3)

            # fig.suptitle(title)
            # fig.supylabel(ylabel)
            # fig.suptitle(title, y=.92)
            # fig.subplots_adjust(top=.85, hspace=.4, bottom=.05)

            return fig, axs
        # endregion

        # region PLOT:timec_dist_force
        case 'PLOT:timec_dist_force':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            dist = list()
            for p in participant_id:
                path = Force(experiment, session, p).get_path()
                sn = int(''.join([c for c in p if c.isdigit()]))
                npz = np.load(os.path.join(path, p, f'{experiment}_{sn}_dist_{GoNogo}.npz'))
                dist.append(npz['data_array'])

            dist = np.array(dist)
            y = dist.mean(axis=0)
            yerr = dist.std(axis=0) / np.sqrt(len(participant_id))

            latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'),
                                  sep='\t').mean(axis=1).to_numpy()

            color = ['darkorange', 'darkviolet']
            label = ['stimFinger', 'cue']
            tAx = np.linspace(-gl.prestim, gl.poststim, int(gl.fsample_mov * (gl.poststim + gl.prestim))) - latency

            for i in range(2):
                axs.plot(tAx, y[i], color=color[i], label=label[i])
                axs.fill_between(tAx, y[i] - yerr[i], y[i] + yerr[i], color=color[i], alpha=0.2, lw=0)
                # axs.plot(tAx, dist[:, i].T, color=color[i], alpha=0.2)

            axs.axvline(x=0, ls='-', color='k', lw=.8)
            axs.axhline(y=0, ls='-', color='k', lw=.8)

            axs.set_yscale('symlog', linthresh=.1)
            axs.set_ylim([-.1, 200])
            axs.set_xlim([-.3, .5])

            axs.set_title(f'expectation effect, {session}, N={dist.shape[0]}')

            axs.set_xlabel('time relative to perturbation (s)')
            axs.set_ylabel('cross-validated multivariate distance (a.u.)')
            axs.legend()

            return fig, axs
        # endregion

        # region PLOT:rdm_force
        case 'PLOT:rdm_force':

            timew = [(-.5, 0), (.1, .4), (.4, 1)]

            if fig is None or axs is None:
                fig, axs = plt.subplots(1, len(timew), sharey=True, sharex=True, figsize=(10, 5))

            colors = ['purple', 'darkorange', 'darkgreen']
            symmetry = [1, 1, -1]
            masks = [gl.mask_stimFinger, gl.mask_cue]

            for T, t in enumerate(timew):
                rdm = list()
                for p in participant_id:
                    force = Force(experiment, session, p)
                    rdm.append(force.calc_rdm(t, GoNogo=GoNogo))

                cax, axs[T] = plot_rdm(rdm, ax=axs[T], vmin=vmin, vmax=vmax)
                axs[T].set_title(f'{t[0]} to {t[1]}s')

                if GoNogo == 'go':
                    draw_contours(masks, symmetry, colors, axs=axs[T])

            cbar = fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=.02)
            cbar.set_label('cross-validated multivariate distance (a.u.)')

            fig.subplots_adjust(bottom=.35)
            fig.suptitle(f'{session}, {GoNogo} trials (N={len(participant_id)})')

            plt.show()

            return fig, axs
        # endregion

        # region PLOT:timec_dist_force_session
        case 'PLOT:timec_dist_force_session':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            if GoNogo == 'go':
                sessions = ['behavioural', 'training', 'scanning', 'pilot']
            elif GoNogo == 'nogo':
                sessions = ['training', 'scanning', 'pilot']

            dist = list()
            for S, s in enumerate(sessions):

                if s == 'behavioural':
                    exp = 'smp0'
                elif (s == 'training') or (s == 'scanning'):
                    exp = 'smp1'
                elif s == 'pilot':
                    exp = 'smp2'

                participants = gl.participants[exp]

                dist_tmp = list()
                for P, p in enumerate(participants):
                    path = Force(exp, s, p).get_path()
                    sn = int(''.join([c for c in p if c.isdigit()]))
                    npz = np.load(os.path.join(path, p, f'{exp}_{sn}_dist_{GoNogo}.npz'))

                    dist_tmp.append(npz['data_array'][1])

                dist.append(np.array(dist_tmp))

            palette = sns.color_palette("husl", len(sessions))

            latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'),
                                  sep='\t').mean(axis=1).to_numpy()

            tAx = np.linspace(-gl.prestim, gl.poststim, int(gl.fsample_mov * (gl.poststim + gl.prestim))) - latency

            for i in range(len(sessions)):
                axs.plot(tAx, dist[i].mean(axis=0), color=palette[i], label=sessions[i])
                axs.fill_between(tAx, dist[i].mean(axis=0) - dist[i].std(axis=0) / np.sqrt(dist[i].shape[0]),
                                 dist[i].mean(axis=0) + dist[i].std(axis=0) / np.sqrt(dist[i].shape[0]),
                                 color=palette[i], alpha=0.3, lw=0)

            axs.set_ylabel('cross-validated multivariate distance (a.u.)')
            axs.set_xlabel('time relative to perturbation (s)')

            axs.axvline(x=0, ls='-', color='k', lw=.8)
            axs.axhline(y=0, ls='-', color='k', lw=.8)

            axs.set_yscale('symlog', linthresh=.1)
            axs.set_ylim([-.1, 200])
            axs.set_xlim([-.3, .5])

            axs.legend(loc='upper left')

            return fig, axs
        # endregion

        # region PLOT:flatmap
        case 'PLOT:flatmap':
            if fig is None or axs is None:
                fig, axs = plt.subplots()

            darray_avg_subj = list()
            for p in participant_id:
                data = os.path.join(gl.baseDir, experiment, gl.wbDir, p, f'glm{glm}.spmT.{Hem}.func.gii')

                D = nb.load(data)
                darray = nitools.get_gifti_data_matrix(D)

                # make indeces to plot
                col_names = nitools.get_gifti_column_names(D)
                regressor_tmp = [f'spmT_{r}.nii' for r in regressor]
                im = np.array([x in regressor_tmp for x in col_names])

                # avg darray
                darray_avg_subj.append(darray[:, im].mean(axis=1))

            darray_avg = np.array(darray_avg_subj).mean(axis=0)
            plt.sca(axs)
            surf.plot.plotmap(darray_avg, f'fs32k_{Hem}',
                              underlay=None,
                              borders=gl.borders[Hem],
                              cscale=[vmin, vmax],
                              cmap='jet',
                              underscale=[-1.5, 1],
                              alpha=.5,
                              new_figure=False,
                              colorbar=False,
                              frame=[xlim[0], xlim[1], ylim[0], ylim[1]])

            return fig, axs
        # endregion

        # region PLOT:hrf_roi
        case 'PLOT:hrf_roi':

            if fig is None or axs is None:
                fig, axs = plt.subplots()

            y_adj_go, y_hat_go, y_adj_nogo, y_hat_nogo, y_raw_go, y_raw_nogo = [], [], [], [], [], []
            for p in participant_id:
                mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, p, f'hrf_glm{glm}.mat'))
                T = mat['T'][0, 0]
                T_fields = T.dtype.names
                T_dict = {field: T[field] for field in T_fields}

                y_adj_go.append(np.nanmean(T_dict['y_adj'][((T_dict['name'] == roi) &
                                                            (T_dict['eventname'] == 'go') &
                                                            (T_dict['hem'] == Hem)).flatten()], axis=0))
                y_hat_go.append(np.nanmean(T_dict['y_hat'][((T_dict['name'] == roi) &
                                                            (T_dict['eventname'] == 'go') &
                                                            (T_dict['hem'] == Hem)).flatten()], axis=0))
                y_adj_nogo.append(np.nanmean(T_dict['y_adj'][((T_dict['name'] == roi) &
                                                              (T_dict['eventname'] == 'nogo') &
                                                              (T_dict['hem'] == Hem)).flatten()], axis=0))
                y_hat_nogo.append(np.nanmean(T_dict['y_hat'][((T_dict['name'] == roi) &
                                                              (T_dict['eventname'] == 'nogo') &
                                                              (T_dict['hem'] == Hem)).flatten()], axis=0))
                y_raw_go.append(np.nanmean(T_dict['y_raw'][((T_dict['name'] == roi) &
                                                            (T_dict['eventname'] == 'go') &
                                                            (T_dict['hem'] == Hem)).flatten()], axis=0))
                y_raw_nogo.append(np.nanmean(T_dict['y_raw'][((T_dict['name'] == roi) &
                                                            (T_dict['eventname'] == 'nogo') &
                                                            (T_dict['hem'] == Hem)).flatten()], axis=0))

            y_adj_go = np.array(y_adj_go).mean(axis=0)
            y_hat_go = np.array(y_hat_go).mean(axis=0)
            y_adj_nogo = np.array(y_adj_nogo).mean(axis=0)
            y_hat_nogo = np.array(y_hat_nogo).mean(axis=0)
            y_raw_go = np.array(y_raw_go).mean(axis=0)
            y_raw_nogo = np.array(y_raw_nogo).mean(axis=0)

            tAx = np.linspace(-10, 10, 21)

            axs.plot(tAx, y_adj_go, color='magenta', label='go adj', ls='-')
            axs.plot(tAx, y_raw_go - y_raw_go.mean(), color='magenta', label='go raw', ls='-', alpha=0.3)
            axs.plot(tAx, y_hat_go, color='magenta', label='go hat', ls='--')
            axs.plot(tAx, y_raw_nogo - y_raw_nogo.mean(), color='green', label='nogo raw', ls='-', alpha=0.3)
            axs.plot(tAx, y_adj_nogo, color='green', label='nogo adj', ls='-')
            axs.plot(tAx, y_hat_nogo, color='green', label='nogo hat', ls='--')

            return fig, axs
        # endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', default='smp2', help='')
    parser.add_argument('--session', default='training', help='')
    parser.add_argument('--sn', default=None, help='')
    parser.add_argument('--GoNogo', default=None, help='')
    parser.add_argument('--stimFinger', default=None, help='')
    parser.add_argument('--cue', default=None, help='')
    parser.add_argument('--glm', default=None, help='')
    parser.add_argument('--Hem', default=None, help='')
    parser.add_argument('--regressor', nargs='+', default=None, help='')
    parser.add_argument('--roi', default=None, help='')
    parser.add_argument('--xlim', default=None, help='')
    parser.add_argument('--ylim', default=None, help='')
    parser.add_argument('--vsep', default=None, help='')
    parser.add_argument('--derivs', nargs='+', type=int, default=[0, 0], help='')
    parser.add_argument('--prefix', default='', help='')

    args = parser.parse_args()

    what = args.what
    experiment = args.experiment
    session = args.session
    sn = args.sn
    GoNogo = args.GoNogo
    stimFinger = args.stimFinger
    cue = args.cue
    glm = args.glm
    Hem = args.Hem
    regressor = args.regressor
    roi = args.roi
    xlim = args.xlim
    ylim = args.ylim
    vsep = args.vsep
    derivs = np.array(args.derivs)
    prefix = args.prefix

    pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

    main(what=what, experiment=experiment, session=session, sn=sn, GoNogo=GoNogo,
         stimFinger=stimFinger, cue=cue, glm=glm, Hem=Hem, regressor=regressor, roi=roi, derivs=derivs, prefix=prefix,
         vsep=8, xlim=[-1, 1], ylim=[0, 40])

    plt.show()
