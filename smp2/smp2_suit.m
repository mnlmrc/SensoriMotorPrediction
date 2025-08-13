function varargout = smp2_suit(what, varargin)

    localPath = '/Users/mnlmrc/Documents/';
    cbsPath = '/home/UWO/memanue5/Documents/';
    % Directory specification
    if isfolder(localPath)
        path = localPath;
    elseif isfolder(cbsPath)
        path = cbsPath;
    end
    
    addpath([path 'GitHub/sensori-motor-prediction/'])
    addpath([path 'GitHub/spmj_tools/'])
    addpath([path 'GitHub/dataframe/util/'])
    addpath([path 'GitHub/dataframe/kinematics/'])
    addpath([path 'GitHub/dataframe/graph/'])
    addpath([path 'GitHub/dataframe/pivot/'])
    addpath([path 'GitHub/surfAnalysis/'])
    addpath([path 'MATLAB/spm12/'])
    addpath([path 'GitHub/rwls/'])
    addpath([path 'GitHub/surfing/surfing/'])
    addpath([path 'GitHub/suit/'])
    addpath([path 'GitHub/rsatoolbox_matlab/'])
    addpath([path 'GitHub/surfing/toolbox_fast_marching/'])
    addpath([path 'GitHub/region/'])
    
    % Use a different baseDir when using your local machine or the cbs
    % server. Add more directory if needed.
    if isfolder("/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/")
        baseDir = "/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp2/";
    elseif isfolder("/cifs/diedrichsen/data/SensoriMotorPrediction/smp2")
        workdir = "/cifs/diedrichsen/data/SensoriMotorPrediction/smp2";
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    baseDir         = (sprintf('%s/',workdir));
    bidsDir = 'BIDS'; % Raw data post AutoBids conversion
    anatomicalDir   = 'anatomicals'; 
    imagingRawDir = 'imaging_data_raw'; % Temporary directory for raw functional data
    imagingDir = 'imaging_data'; % Preprocesses functional data
    fmapDir = 'fieldmaps'; % Fieldmap dir after moving from BIDS and SPM make fieldmap

    pinfo = dload(fullfile(baseDir,'participants.tsv'));
    
    % handling input args:
    sn = [];
    type = 'betas'; % 'betas' or 'con' or 'ResMS' or 'cerebellarGrey' or 'anatomical'
    glm = 12;
    vararginoptions(varargin,{'sn', 'type', 'glm'})
    if isempty(sn)
        error('BIDS:move_unzip_raw_func -> ''sn'' must be passed to this function.')
    end

    % get participant row from participant.tsv
    if isscalar(sn)
        subj_row=getrow(pinfo, pinfo.sn== sn);
    
        % get subj_id
        subj_id = subj_row.subj_id{1};
        % get runs (FuncRuns column needs to be in participants.tsv)    
        runs = spmj_dotstr2array(subj_row.FuncRuns{1});
    end

    

    switch(what)
    case 'SUIT:isolate_segment'
            % Segment cerebellum into grey and white matter
         
            
            fprintf('- Isolate and segment the cerebellum for %s\n', ...
                subj_id)
            spm_jobman('initcfg')
            
            % Get the file of subjects anatomical
            anat_subj_dir  = fullfile(baseDir, anatomicalDir, subj_id);
            anat_name = [subj_id '_anatomical.nii'];
            
            % Define suit folder
            suit_dir = fullfile(baseDir, 'SUIT/anatomicals',subj_id);
            spmj_dircheck(suit_dir)
            
            % Copy anatomical_raw file to suit folder
            source = fullfile(anat_subj_dir, anat_name);
            dest   = fullfile(suit_dir, anat_name);
            copyfile(source, dest);
            
            % go to subject directory for suit and isolate segment
            suit_isolate_seg({dest}, 'keeptempfiles', 1);
            
    case 'SUIT:normalise_dartel' % SUIT normalization using dartel
        % Launch spm fmri before running
        spm fmri
        
        suit_subj_dir = fullfile(baseDir, 'SUIT/anatomicals' , subj_id);
        
        job.subjND.gray       = {fullfile(suit_subj_dir, ...
            sprintf('c_%s_anatomical_seg1.nii', subj_id))};
        job.subjND.white      = {fullfile(suit_subj_dir, ...
            sprintf('c_%s_anatomical_seg2.nii', subj_id))};
        job.subjND.isolation  = {fullfile(suit_subj_dir, ...
            sprintf('c_%s_anatomical_pcereb.nii', subj_id))};
        suit_normalize_dartel(job);
        
    case 'SUIT:save_dartel_def'
        % Saves the dartel flow field as a deformation file.
        currentDir = pwd;
        cd(fullfile(baseDir,'SUIT/anatomicals', subj_id));
        anat_name = [subj_id '_anatomical'];
        suit_save_darteldef(anat_name);

        cd(currentDir) 
        
    case 'SUIT:reslice'

        % run the case with 'anatomical' to check the suit normalization
        % Example usage: nishimoto_imana('SUIT:reslice','type','ResMS', 'mask', 'pcereb_corr')
        % make sure that you reslice into 2mm^3 resolution
        currentDir = pwd;
        mask = ['c_' subj_id '_anatomical_pcereb']; % 'cereb_prob_corr_grey' or 'cereb_prob_corr' or 'dentate_mask' or 'pcereb'
        glm  = sprintf('glm%d', glm);             % glm number. Used for reslicing betas and contrasts
        glmSubjDir = fullfile(baseDir,glm,subj_id);
        switch type
            case 'betas'
                images='beta_0';
                source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                cd(glmSubjDir);
            case 'con'
                images='con_';
                source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                cd(glmSubjDir);
            case 'spmT'
                images='spmT_';
                source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                cd(glmSubjDir)
            case 'ResMS'
                images = 'ResMS';
                source=dir(fullfile(glmSubjDir,'ResMS.nii')); % images to be resliced
                cd(glmSubjDir)
            case 'mask'
                images = 'mask';
                source=dir(fullfile(glmSubjDir,'mask.nii')); % images to be resliced
                cd(glmSubjDir)
            case 'residual'
                source=dir(fullfile(glmSubjDir,'residual.nii')); % images to be resliced
                cd(glmSubjDir)
        end
        job.subj.affineTr = {fullfile(baseDir,'SUIT/anatomicals',subj_id,['Affine_c_',subj_id ,'_anatomical_seg1.mat'])};
        job.subj.flowfield= {fullfile(baseDir,'SUIT/anatomicals',subj_id,['u_a_c_',subj_id, '_anatomical_seg1.nii'])};
        job.subj.resample = {source.name};
        job.subj.mask     = {fullfile(baseDir,'SUIT/anatomicals',subj_id,sprintf('%s.nii',mask))};
        job.vox           = [1 1 1];
        %             Replace Nans with zeros to avoid big holes in the the data
        for i=1:length(source)
            V=spm_vol(source(i).name);
            X=spm_read_vols(V);
            X(isnan(X))=0;
            spm_write_vol(V,X);
        end
        suit_reslice_dartel(job);
        source=fullfile(glmSubjDir,'*wd*');
        destination=fullfile(baseDir,'SUIT',glm,subj_id);
        if ~exist(destination,"dir")
            mkdir(destination);
        end
        movefile(source,destination);
        fprintf('%s have been resliced into suit space \n',type);
        cd(currentDir)
        
    case 'SUIT:across_participants'
        for s = sn
            smp2_suit('SUIT:isolate_segment', 'sn', s)
            smp2_suit('SUIT:normalise_dartel', 'sn', s)
            smp2_suit('SUIT:save_dartel_def', 'sn', s)
            smp2_suit('SUIT:reslice', 'type', 'betas','sn', s,  'glm', glm)
            smp2_suit('SUIT:reslice', 'type', 'con','sn', s, 'glm', glm)
            smp2_suit('SUIT:reslice', 'type', 'spmT','sn', s, 'glm', glm)
            smp2_suit('SUIT:reslice', 'type', 'ResMS','sn', s, 'glm', glm)
        end
    end
end
        