import argparse
from SensoriMotorPrediction import force, pcm_cortical, searchlight, spike, pcm_lfp, pcm_spike, pcm_thalamus, pcm_models, hrf, betas, surface, kinematics, rois, pcm_emg
import time
import SensoriMotorPrediction.globals as gl

def main(args):
    
    ### HUMANS ### ------------------------------------------------------------------------------------
    # calc single-trial behavioural metrics
    if args.what == 'single_trial_behaviour':
        for sn in args.sns:
            force.single_trial_behaviour(experiment=args.experiment, sn=sn)

    # segment thalamic nuclei
    elif args.what == 'thalamus_segmentation':
        for sn in args.sns:
            rois.thalamus_segmentation(sn)

    # save raw and predicted BOLD timeseries for each voxel
    elif args.what == 'save_BOLD':
        for sn in args.sns:
            hrf.save_bold_rois(sn, args.glm, rois=args.rois)

    # optimise HRF parameters
    elif args.what == 'optimise_hrf':
        for sn in args.sns:
            HRF = hrf.Optimise_HRF(sn, args.glm, rois=gl.rois[args.atlas_name], H='L')
            HRF.gridsearch()
    
    # convert SPM.mat file to mat7
    elif args.what == "spm_as_mat7":
        for sn in args.sns:
            betas.save_spm_as_mat7(sn=sn, glm=args.glm)

    # searchlight definitions for each participant
    elif args.what == "make_searchlight":
        for sn in args.sns:
            searchlight.make_searchlight(sn)

    # make cifti files of beta coefficients
    elif args.what == "make_cifti_beta":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='beta')

    # make cifti files of contrasts (a.u.)
    elif args.what == "make_cifti_contrast":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='contrast')

    # make cifti files of intercept for each run
    elif args.what == "make_cifti_intercept":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='intercept')

    # make cifti files of activation in % signal change
    elif args.what == "make_cifti_psc":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='psc')

    # make cifti file of residual time series (4D)
    elif args.what == "make_cifti_residual":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='residual')

    # make smoothed gifti activation maps
    elif args.what == "smooth_surf_activation":
        surface.make_smooth_cifti(sns=args.sns, glm=args.glm)
    
    # regress out finger pre-activation from raw betas
    elif args.what == 'regress_out_preactivation':
        for sn in args.sns:
            pcm_cortical.regress_out_preactivation(sn, args.glm, method=args.method)

    # create pcm models (G matrix) for preparation and execution
    elif args.what == 'pcm_models':
        pcm_models.make_models(experiment=args.experiment, epoch=args.epoch)

    # fit component model to betas for preparation and execution in ROI
    elif args.what == 'component_model_cortical':
        pcm_cortical.component_model(experiment=args.experiment, sns=args.sns, glm=args.glm, atlas=args.atlas_name, epoch=args.epoch)

    # fit component model to betas for preparation and execution in thalamic regions
    elif args.what == 'component_model_thalamus':
        pcm_thalamus.component_model(experiment=args.experiment, sns=args.sns, glm=args.glm, atlas=args.atlas_name, epoch=args.epoch)

    # fit component model to betas for preparation and execution through continuous searchlight
    elif args.what == 'component_model_searchlight':
        pcm_cortical.searchlight(experiment=args.experiment, sns=args.sns, glm=args.glm, epoch=args.epoch)

    # estimate correlation between preparation and execution in cortical activity
    elif args.what == 'correlation_plan-exec_cortical':
        pcm_cortical.correlation(experiment=args.experiment, sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], corr='plan-exec')

    # estimate correlation between expectation and sensory input in cortical activity
    elif args.what == 'correlation_cue-finger_cortical':
        pcm_cortical.correlation(experiment=args.experiment, sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], corr='cue-finger')

    ### EMG ### ------------------------------------------------------------------------------------
    # time-align EMG activity to perturbation
    elif args.what == 'align_emg':
        for sn in args.sns:
            emg.align_emg(experiment=args.experiment, sns=args.sns)

    # calc G matrix within time windows
    elif args.what == 'calc_G_emg':
        pcm_emg.calc_G(experiment=args.experiment, sns=args.sns)

    # estimate correlation between expectation and sensory input in EMG
    elif args.what == 'correlation_cue-finger_emg':
        pcm_emg.correlation(experiment=args.experiment, sns=args.sns)
        
    ### NHP ### ------------------------------------------------------------------------------------
    # time-align elbow angle to cue and perturbation 
    elif args.what == 'align_kinematics':
        for mon in gl.monkey:
            for rec in gl.recordings[mon]:
                kinematics.align_kinematics(mon, rec)

    # time-align spiking activity
    elif args.what == 'align_spike':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    spike.align_spike(roi=roi, monkey=mon, rec=rec)

    # time-align LFPs
    elif args.what == 'align_slfp':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    spike.align_spike(roi=roi, monkey=mon, rec=rec)

    # calc variance of cross-validated and non-cross-validated G matrix in LFPs
    elif args.what == 'tot_variance_lfp':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_lfp.tot_variance(roi=roi, monkey=mon, rec=rec)

    # calc variance of cross-validated and non-cross-validated G matrix in spiking activity
    elif args.what == 'tot_variance_spike':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_spike.tot_variance(roi=roi, monkey=mon, rec=rec)

    # fit component model for spiking activity
    elif args.what == 'component_model_spike':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)[:-1]
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_spike.component_model('plan', mon, M=M, roi=roi, rec=rec)

    # fit component model for LFPs
    elif args.what == 'component_model_spike':
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "rb")
        M = pickle.load(f)[:-1]
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_lfp.component_model('plan', mon, M=M, roi=roi, rec=rec)

    # run cluster based permutation on total signal and bayes factor in spiking activity
    elif args.what == 'cluster_based_perm_spike':
        for roi in ['PMd', 'M1', 'S1']:
            pcm_spike.cluster_based_perm(roi=roi)

    # run cluster based permutation on total signal and bayes factor in LFPs
    elif args.what == 'cluster_based_perm_lfp':
        for roi in ['PMd', 'M1', 'S1']:
            pcm_lfp.cluster_based_perm(roi=roi)

    # estimate correlation between expectation and sensory input in spiking activity
    elif args.what == 'correlation_cue-direction_spike':
        for fband in ['alpha', 'beta', 'gamma']:
            pcm_spike.correlation(rois=['M1', 'S1'])

    # estimate correlation between expectation and sensory input in LFPs
    elif args.what == 'correlation_cue-direction_lfp':
        for fband in ['alpha', 'beta', 'gamma']:
            pcm_lfp.correlation(fband, rois=['M1', 'S1'])

    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', default='smp2')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.sns)
    parser.add_argument('--rois', nargs='+', type=str, default=gl.rois)
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=12)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--method', type=str, default='raw')

    args = parser.parse_args()

    if args.rois==None:
        args.rois = gl.rois[args.atlas_name]

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')



