function varargout = smp2_imana(what,varargin)
    % Template function for preprocessing of the fMRI data.
    % Rename this function to <experiment_name>_imana.m 
    % Don't forget to add path the required tools!
    
    localPath = '/Users/mnlmrc/Documents/';
    cbsPath = '/home/ROBARTS/memanue5/Documents/';
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
    % addpath([path 'GitHub/suit/'])
    addpath([path 'GitHub/rsatoolbox_matlab/'])
    addpath([path 'GitHub/surfing/toolbox_fast_marching/'])
    addpath([path 'GitHub/region/'])
    % Define the data base directory 
    
    % automatic detection of datashare location:
    % After mounting the diedrichsen datashare on a mac computer.
    if isfolder("/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2")
        workdir = "/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2";
    elseif isfolder("/cifs/diedrichsen/data/SensoriMotorPrediction/smp2")
        workdir = "/cifs/diedrichsen/data/SensoriMotorPrediction/smp2";
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    baseDir         = (sprintf('%s/',workdir));                            % Base directory of the project
    bidsDir         = 'BIDS';                                              % Raw data post AutoBids conversion
    behavDir        = 'behavioural';       
    imagingRawDir   = 'imaging_data_raw';                                  % Temporary directory for raw functional data
    imagingDir      = 'imaging_data';                                      % Preprocesses functional data
    anatomicalDir   = 'anatomicals';                                       % Preprocessed anatomical data (LPI + center AC + segemnt)
    fmapDir         = 'fieldmaps';                                         % Fieldmap dir after moving from BIDS and SPM make fieldmap
    glmEstDir       = 'glm';
    regDir          = 'ROI';
    suitDir         = 'suit';
    wbDir           = 'surfaceWB';
    freesurferDir = 'surfaceFreesurfer'; % freesurfer reconall output
    numDummys       = 5;                                                   % number of dummy scans at the beginning of each run
    
    
    %% subject info
    
    % Read info from participants .tsv file 
    % pinfo = dload(fullfile(baseDir,'participants.tsv'));    
    pinfo = dload(fullfile(baseDir,'participants.tsv'));
    
    %% MAIN OPERATION 
    switch(what)
        case 'BIDS:move_unzip_raw_anat'    

            % Moves, unzips and renames anatomical images from BIDS
            % directory to anatomicalDir. After you run this function you 
            % will find a <subj_id>_anatomical_raw.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.
                        
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_anat -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % path to the subj anat data:
            anat_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.03d',sn), 'anat',[pinfo.AnatRawName{pinfo.sn==sn}, '.nii.gz']);
    
            % destination path:
            output_folder = fullfile(baseDir,anatomicalDir,subj_id);
            output_file = fullfile(output_folder,[subj_id '_anatomical_raw.nii.gz']);
    
            if ~exist(output_folder,"dir")
                mkdir(output_folder);
            end
    
            % copy file to destination:
            [status,msg] = copyfile(anat_raw_path,output_file);
            if ~status  
                error('ANAT:move_anatomical -> subj %d raw anatmoical was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
    
            % unzip the .gz files to make usable for SPM:
            gunzip(output_file);
    
            % delete the compressed file:
            delete(output_file);
    
        case 'ANAT:reslice_LPI'          
            % Reslice anatomical image within LPI coordinate systems. This
            % function creates a <subj_id>_anatomical.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.


            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:reslice_LPI -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % (1) Reslice anatomical image to set it within LPI co-ordinate frames
            source = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical_raw.nii']);
            dest = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            spmj_reslice_LPI(source,'name', dest);
    
    
        case 'ANAT:centre_AC'            
            % Description:
            % Recenters the anatomical data to the Anterior Commissure
            % coordiantes. Doing that, the [0,0,0] coordinate of subject's
            % anatomical image will be the Anterior Commissure.
    
            % You should manually find the voxel coordinates 
            % (1-based index --> fslyes starts from 0) AC for each from 
            % their anatomical scans and add it to the participants.tsv 
            % file under the loc_ACx loc_ACy loc_ACz columns.
    
            % This function runs for all subjects and sessions.
    
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:centre_AC -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % path to the raw anatomical:
            anat_raw_file = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            if ~exist(anat_raw_file,"file")
                error('ANAT:centre_AC -> file %s was not found.',anat_raw_file)
            end
            
            % Get header info for the image:
            V = spm_vol(anat_raw_file);
            % Read the volume:
            dat = spm_read_vols(V);
            
            % changing the transform matrix translations to put AC near [0,0,0]
            % coordinates:
            R = V.mat(1:3,1:3);
            AC = [pinfo.locACx(pinfo.sn==sn),pinfo.locACy(pinfo.sn==sn),pinfo.locACz(pinfo.sn==sn)]';
            t = -1 * R * AC;
            V.mat(1:3,4) = t;
            sprintf('ACx: %d, ACy: %d, ACz: %d', pinfo.locACx(pinfo.sn==sn), pinfo.locACy(pinfo.sn==sn), pinfo.locACz(pinfo.sn==sn))
    
            % writing the image with the changed header:
            spm_write_vol(V,dat);
    
        
        case 'ANAT:segmentation'
            % Segmentation + Normalization. Manually check results when
            % done. This step creates five files named 
            % c1<subj_id>_anatomical.nii, c2<subj_id>_anatomical.nii, 
            % c3<subj_id>_anatomical.nii, c4<subj_id>_anatomical.nii, 
            % c5<subj_id>_anatomical.nii, in the 
            % <project_id>/anatomicals/<subj_id>/ directory. Each of these
            % files contains a segment (e.g., white matter, grey matter) of
            % the centered anatomical image.

            % The output images correspond to the native parameter. For the
            % first five tissues, native is set to [1 0], which means that
            % the native space segmented images are saved. For the sixth
            % tissue (background), native is set to [0 0], which means that
            % no native space segmented image is saved for this tissue.

            % Thus, the code is designed to create segmentation for six tissue classes,
            % but only the first five are saved as output files (c1 to c5). The sixth
            % tissue class (background) does not produce an output image because its
            % native parameter is set to [0 0]. This is why you only see five output
            % images, despite the code handling six tissue classes.
    
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:segmentation -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            anat_path = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii,1']);

            % spmj_segmentation(anat_path);
            SPMhome=fileparts(which('spm.m'));
            J=[];
            % for s=sn WE DONT NEED THIS FOR LOOP 
            J.channel.vols = {fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii,1'])};
            J.channel.biasreg = 0.001;
            J.channel.biasfwhm = 60;
            J.channel.write = [0 0];
            J.tissue(1).tpm = {fullfile(SPMhome,'tpm/TPM.nii,1')}; % grey matter
            J.tissue(1).ngaus = 1;
            J.tissue(1).native = [1 0];
            J.tissue(1).warped = [0 0];
            J.tissue(2).tpm = {fullfile(SPMhome,'tpm/TPM.nii,2')}; % white matter
            J.tissue(2).ngaus = 1;
            J.tissue(2).native = [1 0];
            J.tissue(2).warped = [0 0];
            J.tissue(3).tpm = {fullfile(SPMhome,'tpm/TPM.nii,3')}; % CSF
            J.tissue(3).ngaus = 2; 
            J.tissue(3).native = [1 0];
            J.tissue(3).warped = [0 0];
            J.tissue(4).tpm = {fullfile(SPMhome,'tpm/TPM.nii,4')}; % soft tissue
            J.tissue(4).ngaus = 3;
            J.tissue(4).native = [1 0];
            J.tissue(4).warped = [0 0];
            J.tissue(5).tpm = {fullfile(SPMhome,'tpm/TPM.nii,5')}; % bone
            J.tissue(5).ngaus = 4;
            J.tissue(5).native = [1 0];
            J.tissue(5).warped = [0 0];
            J.tissue(6).tpm = {fullfile(SPMhome,'tpm/TPM.nii,6')}; % NOT SAVED
            J.tissue(6).ngaus = 2;
            J.tissue(6).native = [0 0];
            J.tissue(6).warped = [0 0];
            J.warp.mrf = 1;
            J.warp.cleanup = 1;
            J.warp.reg = [0 0.001 0.5 0.05 0.2];
            J.warp.affreg = 'mni';
            J.warp.fwhm = 0;
            J.warp.samp = 3;
            J.warp.write = [0 0];
            matlabbatch{1}.spm.spatial.preproc=J;
            spm_jobman('run',matlabbatch);

        case 'GLM:make_event'

            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};
    
            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});
            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end 
            
            operation  = sprintf('GLM:make_glm%d', glm);
            
            events = smp2_imana(operation, 'sn', sn);
            events = events(ismember(events.BN, runs), :);
            
            %% export
            output_folder = fullfile(baseDir, behavDir, subj_id);
            writetable(events, fullfile(output_folder, sprintf('glm%d_events.tsv', glm)), 'FileType', 'text', 'Delimiter','\t')

            if ~isfolder(fullfile(baseDir, [glmEstDir num2str(glm)] , subj_id))
                mkdir(fullfile(baseDir, [glmEstDir num2str(glm)], subj_id))
            end

        case 'GLM:calculate_VIF'

            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            SPM = load(fullfile(baseDir,[glmEstDir num2str(glm)],subj_id, 'SPM.mat')); SPM = SPM.SPM;

            X=(SPM.xX.X(:,SPM.xX.iC)); % Assuming 'X' is the field holding the design matrix
            
            % % Number of predictors
            % [n, p] = size(X);
            
            % Initialize VIF array
            vif_values = zeros(length(SPM.Sess), length(SPM.Sess(1).col));

            for s = 1:length(SPM.Sess)
                idx = 1;
                for i = SPM.Sess(s).col
                    % Separate the ith predictor
                    Xi = X(:, i);
                    % Remaining predictors
                    other_predictors = X(:, setdiff(SPM.Sess(s).col, i));
                    
                    % Fit a regression model of Xi on the other predictors
                    model = fitlm(other_predictors, Xi);
                    
                    % Get R-squared value
                    R_squared = model.Rsquared.Ordinary;
                    
                    % Compute VIF
                    vif_values(s, idx) = 1 / (1 - R_squared);
                    idx = idx + 1;
                end

            end

            figure
            bar([SPM.Sess(1).U.name], mean(vif_values, 1))
            set(groot, 'defaultTextInterpreter', 'none');
            hold on
            errorbar(categorical([SPM.Sess(1).U.name]), mean(vif_values, 1), std(vif_values, 1), 'k', 'linestyle', 'none')
             
            % Display VIF values
            vif_table = array2table( vif_values, 'VariableNames', [SPM.Sess(1).U.name]);
            disp(vif_table);  

        case 'GLM:make_glm10'

            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp2_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");

            %% planning 0% (nogo)
            plan0.BN = D.BN(~go & D.cue==93);
            plan0.TN = D.TN(~go & D.cue==93);
            plan0.cue = D.cue(~go & D.cue==93);
            plan0.stimFinger = D.stimFinger(~go & D.cue==93);
            plan0.Onset = D.startTimeReal(~go & D.cue==93) + D.baselineWait(~go & D.cue==93);
            plan0.Duration = zeros(length(plan0.BN), 1);
            plan0.eventtype = repmat({'0%'}, [length(plan0.BN), 1]);

            %% planning 25% (nogo)
            plan25.BN = D.BN(~go & D.cue==12);
            plan25.TN = D.TN(~go & D.cue==12);
            plan25.cue = D.cue(~go & D.cue==12);
            plan25.stimFinger = D.stimFinger(~go & D.cue==12);
            plan25.Onset = D.startTimeReal(~go & D.cue==12) + D.baselineWait(~go & D.cue==12);
            plan25.Duration = zeros(length(plan25.BN), 1);
            plan25.eventtype = repmat({'25%'}, [length(plan25.BN), 1]);

            %% planning 50% (nogo)
            plan50.BN = D.BN(~go & D.cue==44);
            plan50.TN = D.TN(~go & D.cue==44);
            plan50.cue = D.cue(~go & D.cue==44);
            plan50.stimFinger = D.stimFinger(~go & D.cue==44);
            plan50.Onset = D.startTimeReal(~go & D.cue==44) + D.baselineWait(~go & D.cue==44);
            plan50.Duration = zeros(length(plan50.BN), 1);
            plan50.eventtype = repmat({'50%'}, [length(plan50.BN), 1]);

            %% planning 75% (nogo)
            plan75.BN = D.BN(~go & D.cue==21);
            plan75.TN = D.TN(~go & D.cue==21);
            plan75.cue = D.cue(~go & D.cue==21);
            plan75.stimFinger = D.stimFinger(~go & D.cue==21);
            plan75.Onset = D.startTimeReal(~go & D.cue==21) + D.baselineWait(~go & D.cue==21);
            plan75.Duration = zeros(length(plan75.BN), 1);
            plan75.eventtype = repmat({'75%'}, [length(plan75.BN), 1]);

            %% planning 100% (nogo)
            plan100.BN = D.BN(~go & D.cue==39);
            plan100.TN = D.TN(~go & D.cue==39);
            plan100.cue = D.cue(~go & D.cue==39);
            plan100.stimFinger = D.stimFinger(~go & D.cue==39);
            plan100.Onset = D.startTimeReal(~go & D.cue==39) + D.baselineWait(~go & D.cue==39);
            plan100.Duration = zeros(length(plan100.BN), 1);
            plan100.eventtype = repmat({'100%'}, [length(plan100.BN), 1]);

            %% ring 0% (go)
            ring0.BN = D.BN(go & D.cue==93 & D.stimFinger==99919);
            ring0.TN = D.TN(go & D.cue==93 & D.stimFinger==99919);
            ring0.cue = D.cue(go & D.cue==93 & D.stimFinger==99919);
            ring0.stimFinger = D.stimFinger(go & D.cue==93 & D.stimFinger==99919);
            ring0.Onset = D.startTimeReal(go & D.cue==93 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==93 & D.stimFinger==99919) ...
                + D.planTime(go & D.cue==93 & D.stimFinger==99919);
            ring0.Duration = zeros(length(ring0.BN), 1);
            ring0.eventtype = repmat({'0%,ring'}, [length(ring0.BN), 1]);

            %% ring 25% (go)
            ring25.BN = D.BN(go & D.cue==12 & D.stimFinger==99919);
            ring25.TN = D.TN(go & D.cue==12 & D.stimFinger==99919);
            ring25.cue = D.cue(go & D.cue==12 & D.stimFinger==99919);
            ring25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==99919);
            ring25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==12 & D.stimFinger==99919);
            ring25.Duration = zeros(length(ring25.BN), 1);
            ring25.eventtype = repmat({'25%,ring'}, [length(ring25.BN), 1]);

            %% ring 50% (go)
            ring50.BN = D.BN(go & D.cue==44 & D.stimFinger==99919);
            ring50.TN = D.TN(go & D.cue==44 & D.stimFinger==99919);
            ring50.cue = D.cue(go & D.cue==44 & D.stimFinger==99919);
            ring50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==99919);
            ring50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==99919)... 
                + D.baselineWait(go & D.cue==44 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==44 & D.stimFinger==99919);
            ring50.Duration = zeros(length(ring50.BN), 1);
            ring50.eventtype = repmat({'50%,ring'}, [length(ring50.BN), 1]);

            %% ring 75% (go)
            ring75.BN = D.BN(go & D.cue==21 & D.stimFinger==99919);
            ring75.TN = D.TN(go & D.cue==21 & D.stimFinger==99919);
            ring75.cue = D.cue(go & D.cue==21 & D.stimFinger==99919);
            ring75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==99919);
            ring75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==21 & D.stimFinger==99919);
            ring75.Duration = zeros(length(ring75.BN), 1);
            ring75.eventtype = repmat({'75%,ring'}, [length(ring75.BN), 1]);

            %% index 25% (go)
            index25.BN = D.BN(go & D.cue==12 & D.stimFinger==91999);
            index25.TN = D.TN(go & D.cue==12 & D.stimFinger==91999);
            index25.cue = D.cue(go & D.cue==12 & D.stimFinger==91999);
            index25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==91999);
            index25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==12 & D.stimFinger==91999);
            index25.Duration = zeros(length(index25.BN), 1);
            index25.eventtype = repmat({'25%,index'}, [length(index25.BN), 1]);

            %% index 50% (go)
            index50.BN = D.BN(go & D.cue==44 & D.stimFinger==91999);
            index50.TN = D.TN(go & D.cue==44 & D.stimFinger==91999);
            index50.cue = D.cue(go & D.cue==44 & D.stimFinger==91999);
            index50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==91999);
            index50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==44 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==44 & D.stimFinger==91999);
            index50.Duration = zeros(length(index50.BN), 1);
            index50.eventtype = repmat({'50%,index'}, [length(index50.BN), 1]);

            %% index 75% (go)
            index75.BN = D.BN(go & D.cue==21 & D.stimFinger==91999);
            index75.TN = D.TN(go & D.cue==21 & D.stimFinger==91999);
            index75.cue = D.cue(go & D.cue==21 & D.stimFinger==91999);
            index75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==91999);
            index75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==21 & D.stimFinger==91999);
            index75.Duration = zeros(length(index75.BN), 1);
            index75.eventtype = repmat({'75%,index'}, [length(index75.BN), 1]);

            %% index 100% (go)
            index100.BN = D.BN(go & D.cue==39 & D.stimFinger==91999);
            index100.TN = D.TN(go & D.cue==39 & D.stimFinger==91999);
            index100.cue = D.cue(go & D.cue==39 & D.stimFinger==91999);
            index100.stimFinger = D.stimFinger(go & D.cue==39 & D.stimFinger==91999);
            index100.Onset = D.startTimeReal(go & D.cue==39 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==39 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==39 & D.stimFinger==91999);
            index100.Duration = zeros(length(index100.BN), 1);
            index100.eventtype = repmat({'100%,index'}, [length(index100.BN), 1]);
            
            %% make table
            
            plan0 = struct2table(plan0);
            plan25 = struct2table(plan25);
            plan50 = struct2table(plan50);
            plan75 = struct2table(plan75);
            plan100 = struct2table(plan100);
            ring0 = struct2table(ring0);
            ring25 = struct2table(ring25);
            ring50 = struct2table(ring50);
            ring75 = struct2table(ring75);
            index25 = struct2table(index25);
            index50 = struct2table(index50);
            index75 = struct2table(index75);
            index100 = struct2table(index100);

            events = [plan0; plan25; plan50; plan75; plan100; ring0; ring25; ring50; ring75; index25; index50; index75; index100];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;                 
                       
            varargout{1}= events;

        case 'GLM:make_glm11'

            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp2_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");

            %% planning 0%
            plan0.BN = D.BN(D.cue==93);
            plan0.TN = D.TN(D.cue==93);
            plan0.cue = D.cue(D.cue==93);
            plan0.stimFinger = D.stimFinger(D.cue==93);
            plan0.Onset = D.startTimeReal(D.cue==93) + D.baselineWait(D.cue==93);
            plan0.Duration = zeros(length(plan0.BN), 1);
            plan0.eventtype = repmat({'0%'}, [length(plan0.BN), 1]);

            %% planning 25%
            plan25.BN = D.BN(D.cue==12);
            plan25.TN = D.TN(D.cue==12);
            plan25.cue = D.cue(D.cue==12);
            plan25.stimFinger = D.stimFinger(D.cue==12);
            plan25.Onset = D.startTimeReal(D.cue==12) + D.baselineWait(D.cue==12);
            plan25.Duration = zeros(length(plan25.BN), 1);
            plan25.eventtype = repmat({'25%'}, [length(plan25.BN), 1]);

            %% planning 50% 
            plan50.BN = D.BN(D.cue==44);
            plan50.TN = D.TN(D.cue==44);
            plan50.cue = D.cue(D.cue==44);
            plan50.stimFinger = D.stimFinger(D.cue==44);
            plan50.Onset = D.startTimeReal(D.cue==44) + D.baselineWait(D.cue==44);
            plan50.Duration = zeros(length(plan50.BN), 1);
            plan50.eventtype = repmat({'50%'}, [length(plan50.BN), 1]);

            %% planning 75% 
            plan75.BN = D.BN(D.cue==21);
            plan75.TN = D.TN(D.cue==21);
            plan75.cue = D.cue(D.cue==21);
            plan75.stimFinger = D.stimFinger(D.cue==21);
            plan75.Onset = D.startTimeReal(D.cue==21) + D.baselineWait(D.cue==21);
            plan75.Duration = zeros(length(plan75.BN), 1);
            plan75.eventtype = repmat({'75%'}, [length(plan75.BN), 1]);

            %% planning 100% 
            plan100.BN = D.BN(D.cue==39);
            plan100.TN = D.TN(D.cue==39);
            plan100.cue = D.cue( D.cue==39);
            plan100.stimFinger = D.stimFinger( D.cue==39);
            plan100.Onset = D.startTimeReal( D.cue==39) + D.baselineWait( D.cue==39);
            plan100.Duration = zeros(length(plan100.BN), 1);
            plan100.eventtype = repmat({'100%'}, [length(plan100.BN), 1]);

            %% ring 0% (go)
            ring0.BN = D.BN(go & D.cue==93 & D.stimFinger==99919);
            ring0.TN = D.TN(go & D.cue==93 & D.stimFinger==99919);
            ring0.cue = D.cue(go & D.cue==93 & D.stimFinger==99919);
            ring0.stimFinger = D.stimFinger(go & D.cue==93 & D.stimFinger==99919);
            ring0.Onset = D.startTimeReal(go & D.cue==93 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==93 & D.stimFinger==99919) ...
                + D.planTime(go & D.cue==93 & D.stimFinger==99919);
            ring0.Duration = zeros(length(ring0.BN), 1);
            ring0.eventtype = repmat({'0%,ring'}, [length(ring0.BN), 1]);

            %% ring 25% (go)
            ring25.BN = D.BN(go & D.cue==12 & D.stimFinger==99919);
            ring25.TN = D.TN(go & D.cue==12 & D.stimFinger==99919);
            ring25.cue = D.cue(go & D.cue==12 & D.stimFinger==99919);
            ring25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==99919);
            ring25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==12 & D.stimFinger==99919);
            ring25.Duration = zeros(length(ring25.BN), 1);
            ring25.eventtype = repmat({'25%,ring'}, [length(ring25.BN), 1]);

            %% ring 50% (go)
            ring50.BN = D.BN(go & D.cue==44 & D.stimFinger==99919);
            ring50.TN = D.TN(go & D.cue==44 & D.stimFinger==99919);
            ring50.cue = D.cue(go & D.cue==44 & D.stimFinger==99919);
            ring50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==99919);
            ring50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==99919)... 
                + D.baselineWait(go & D.cue==44 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==44 & D.stimFinger==99919);
            ring50.Duration = zeros(length(ring50.BN), 1);
            ring50.eventtype = repmat({'50%,ring'}, [length(ring50.BN), 1]);

            %% ring 75% (go)
            ring75.BN = D.BN(go & D.cue==21 & D.stimFinger==99919);
            ring75.TN = D.TN(go & D.cue==21 & D.stimFinger==99919);
            ring75.cue = D.cue(go & D.cue==21 & D.stimFinger==99919);
            ring75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==99919);
            ring75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==21 & D.stimFinger==99919);
            ring75.Duration = zeros(length(ring75.BN), 1);
            ring75.eventtype = repmat({'75%,ring'}, [length(ring75.BN), 1]);

            %% index 25% (go)
            index25.BN = D.BN(go & D.cue==12 & D.stimFinger==91999);
            index25.TN = D.TN(go & D.cue==12 & D.stimFinger==91999);
            index25.cue = D.cue(go & D.cue==12 & D.stimFinger==91999);
            index25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==91999);
            index25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==12 & D.stimFinger==91999);
            index25.Duration = zeros(length(index25.BN), 1);
            index25.eventtype = repmat({'25%,index'}, [length(index25.BN), 1]);

            %% index 50% (go)
            index50.BN = D.BN(go & D.cue==44 & D.stimFinger==91999);
            index50.TN = D.TN(go & D.cue==44 & D.stimFinger==91999);
            index50.cue = D.cue(go & D.cue==44 & D.stimFinger==91999);
            index50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==91999);
            index50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==44 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==44 & D.stimFinger==91999);
            index50.Duration = zeros(length(index50.BN), 1);
            index50.eventtype = repmat({'50%,index'}, [length(index50.BN), 1]);

            %% index 75% (go)
            index75.BN = D.BN(go & D.cue==21 & D.stimFinger==91999);
            index75.TN = D.TN(go & D.cue==21 & D.stimFinger==91999);
            index75.cue = D.cue(go & D.cue==21 & D.stimFinger==91999);
            index75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==91999);
            index75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==21 & D.stimFinger==91999);
            index75.Duration = zeros(length(index75.BN), 1);
            index75.eventtype = repmat({'75%,index'}, [length(index75.BN), 1]);

            %% index 100% (go)
            index100.BN = D.BN(go & D.cue==39 & D.stimFinger==91999);
            index100.TN = D.TN(go & D.cue==39 & D.stimFinger==91999);
            index100.cue = D.cue(go & D.cue==39 & D.stimFinger==91999);
            index100.stimFinger = D.stimFinger(go & D.cue==39 & D.stimFinger==91999);
            index100.Onset = D.startTimeReal(go & D.cue==39 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==39 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==39 & D.stimFinger==91999);
            index100.Duration = zeros(length(index100.BN), 1);
            index100.eventtype = repmat({'100%,index'}, [length(index100.BN), 1]);
            
            %% make table
            
            plan0 = struct2table(plan0);
            plan25 = struct2table(plan25);
            plan50 = struct2table(plan50);
            plan75 = struct2table(plan75);
            plan100 = struct2table(plan100);
            ring0 = struct2table(ring0);
            ring25 = struct2table(ring25);
            ring50 = struct2table(ring50);
            ring75 = struct2table(ring75);
            index25 = struct2table(index25);
            index50 = struct2table(index50);
            index75 = struct2table(index75);
            index100 = struct2table(index100);

            events = [plan0; plan25; plan50; plan75; plan100; ring0; ring25; ring50; ring75; index25; index50; index75; index100];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;                 
                       
            varargout{1}= events;
            
        case 'GLM:make_glm12'

            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp2_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");

            %% planning 0%
            plan0.BN = D.BN(D.cue==93);
            plan0.TN = D.TN(D.cue==93);
            plan0.cue = D.cue(D.cue==93);
            plan0.stimFinger = D.stimFinger(D.cue==93);
            plan0.Onset = D.startTimeReal(D.cue==93) + D.baselineWait(D.cue==93);
            plan0.Duration = zeros(length(plan0.BN), 1);
            plan0.eventtype = repmat({'0%'}, [length(plan0.BN), 1]);

            %% planning 25%
            plan25.BN = D.BN(D.cue==12);
            plan25.TN = D.TN(D.cue==12);
            plan25.cue = D.cue(D.cue==12);
            plan25.stimFinger = D.stimFinger(D.cue==12);
            plan25.Onset = D.startTimeReal(D.cue==12) + D.baselineWait(D.cue==12);
            plan25.Duration = zeros(length(plan25.BN), 1);
            plan25.eventtype = repmat({'25%'}, [length(plan25.BN), 1]);

            %% planning 50% 
            plan50.BN = D.BN(D.cue==44);
            plan50.TN = D.TN(D.cue==44);
            plan50.cue = D.cue(D.cue==44);
            plan50.stimFinger = D.stimFinger(D.cue==44);
            plan50.Onset = D.startTimeReal(D.cue==44) + D.baselineWait(D.cue==44);
            plan50.Duration = zeros(length(plan50.BN), 1);
            plan50.eventtype = repmat({'50%'}, [length(plan50.BN), 1]);

            %% planning 75% 
            plan75.BN = D.BN(D.cue==21);
            plan75.TN = D.TN(D.cue==21);
            plan75.cue = D.cue(D.cue==21);
            plan75.stimFinger = D.stimFinger(D.cue==21);
            plan75.Onset = D.startTimeReal(D.cue==21) + D.baselineWait(D.cue==21);
            plan75.Duration = zeros(length(plan75.BN), 1);
            plan75.eventtype = repmat({'75%'}, [length(plan75.BN), 1]);

            %% planning 100% 
            plan100.BN = D.BN(D.cue==39);
            plan100.TN = D.TN(D.cue==39);
            plan100.cue = D.cue( D.cue==39);
            plan100.stimFinger = D.stimFinger( D.cue==39);
            plan100.Onset = D.startTimeReal( D.cue==39) + D.baselineWait( D.cue==39);
            plan100.Duration = zeros(length(plan100.BN), 1);
            plan100.eventtype = repmat({'100%'}, [length(plan100.BN), 1]);

            %% ring 0% (go)
            ring0.BN = D.BN(go & D.cue==93 & D.stimFinger==99919);
            ring0.TN = D.TN(go & D.cue==93 & D.stimFinger==99919);
            ring0.cue = D.cue(go & D.cue==93 & D.stimFinger==99919);
            ring0.stimFinger = D.stimFinger(go & D.cue==93 & D.stimFinger==99919);
            ring0.Onset = D.startTimeReal(go & D.cue==93 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==93 & D.stimFinger==99919) ...
                + D.planTime(go & D.cue==93 & D.stimFinger==99919);
            ring0.Duration = zeros(length(ring0.BN), 1);
            ring0.eventtype = repmat({'0%,ring'}, [length(ring0.BN), 1]);

            %% ring 25% (go)
            ring25.BN = D.BN(go & D.cue==12 & D.stimFinger==99919);
            ring25.TN = D.TN(go & D.cue==12 & D.stimFinger==99919);
            ring25.cue = D.cue(go & D.cue==12 & D.stimFinger==99919);
            ring25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==99919);
            ring25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==12 & D.stimFinger==99919);
            ring25.Duration = zeros(length(ring25.BN), 1);
            ring25.eventtype = repmat({'25%,ring'}, [length(ring25.BN), 1]);

            %% ring 50% (go)
            ring50.BN = D.BN(go & D.cue==44 & D.stimFinger==99919);
            ring50.TN = D.TN(go & D.cue==44 & D.stimFinger==99919);
            ring50.cue = D.cue(go & D.cue==44 & D.stimFinger==99919);
            ring50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==99919);
            ring50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==99919)... 
                + D.baselineWait(go & D.cue==44 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==44 & D.stimFinger==99919);
            ring50.Duration = zeros(length(ring50.BN), 1);
            ring50.eventtype = repmat({'50%,ring'}, [length(ring50.BN), 1]);

            %% ring 75% (go)
            ring75.BN = D.BN(go & D.cue==21 & D.stimFinger==99919);
            ring75.TN = D.TN(go & D.cue==21 & D.stimFinger==99919);
            ring75.cue = D.cue(go & D.cue==21 & D.stimFinger==99919);
            ring75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==99919);
            ring75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==99919)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==99919)...
                + D.planTime(go & D.cue==21 & D.stimFinger==99919);
            ring75.Duration = zeros(length(ring75.BN), 1);
            ring75.eventtype = repmat({'75%,ring'}, [length(ring75.BN), 1]);

            %% index 25% (go)
            index25.BN = D.BN(go & D.cue==12 & D.stimFinger==91999);
            index25.TN = D.TN(go & D.cue==12 & D.stimFinger==91999);
            index25.cue = D.cue(go & D.cue==12 & D.stimFinger==91999);
            index25.stimFinger = D.stimFinger(go & D.cue==12 & D.stimFinger==91999);
            index25.Onset = D.startTimeReal(go & D.cue==12 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==12 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==12 & D.stimFinger==91999);
            index25.Duration = zeros(length(index25.BN), 1);
            index25.eventtype = repmat({'25%,index'}, [length(index25.BN), 1]);

            %% index 50% (go)
            index50.BN = D.BN(go & D.cue==44 & D.stimFinger==91999);
            index50.TN = D.TN(go & D.cue==44 & D.stimFinger==91999);
            index50.cue = D.cue(go & D.cue==44 & D.stimFinger==91999);
            index50.stimFinger = D.stimFinger(go & D.cue==44 & D.stimFinger==91999);
            index50.Onset = D.startTimeReal(go & D.cue==44 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==44 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==44 & D.stimFinger==91999);
            index50.Duration = zeros(length(index50.BN), 1);
            index50.eventtype = repmat({'50%,index'}, [length(index50.BN), 1]);

            %% index 75% (go)
            index75.BN = D.BN(go & D.cue==21 & D.stimFinger==91999);
            index75.TN = D.TN(go & D.cue==21 & D.stimFinger==91999);
            index75.cue = D.cue(go & D.cue==21 & D.stimFinger==91999);
            index75.stimFinger = D.stimFinger(go & D.cue==21 & D.stimFinger==91999);
            index75.Onset = D.startTimeReal(go & D.cue==21 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==21 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==21 & D.stimFinger==91999);
            index75.Duration = zeros(length(index75.BN), 1);
            index75.eventtype = repmat({'75%,index'}, [length(index75.BN), 1]);

            %% index 100% (go)
            index100.BN = D.BN(go & D.cue==39 & D.stimFinger==91999);
            index100.TN = D.TN(go & D.cue==39 & D.stimFinger==91999);
            index100.cue = D.cue(go & D.cue==39 & D.stimFinger==91999);
            index100.stimFinger = D.stimFinger(go & D.cue==39 & D.stimFinger==91999);
            index100.Onset = D.startTimeReal(go & D.cue==39 & D.stimFinger==91999)...
                + D.baselineWait(go & D.cue==39 & D.stimFinger==91999)...
                + D.planTime(go & D.cue==39 & D.stimFinger==91999);
            index100.Duration = zeros(length(index100.BN), 1);
            index100.eventtype = repmat({'100%,index'}, [length(index100.BN), 1]);
            
            %% make table
            
            plan0 = struct2table(plan0);
            plan25 = struct2table(plan25);
            plan50 = struct2table(plan50);
            plan75 = struct2table(plan75);
            plan100 = struct2table(plan100);
            ring0 = struct2table(ring0);
            ring25 = struct2table(ring25);
            ring50 = struct2table(ring50);
            ring75 = struct2table(ring75);
            index25 = struct2table(index25);
            index50 = struct2table(index50);
            index75 = struct2table(index75);
            index100 = struct2table(index100);

            events = [plan0; plan25; plan50; plan75; plan100; ring0; ring25; ring50; ring75; index25; index50; index75; index100];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;                 
                       
            varargout{1}= events;

        case 'GLM:design'

            % Import globals from spm_defaults 
            global defaults; 
            if (isempty(defaults)) 
                spm_defaults;
            end 
            
            currentDir = pwd;

            sn = [];
            glm = [];
            hrf_params = [5 14 1 1 6 0 32];
            derivs = [0 0];
            vararginoptions(varargin,{'sn', 'glm', 'hrf_params', 'derivs'})

            if isempty(sn)
                error('GLM:design -> ''sn'' must be passed to this function.')
            end

            if isempty(sn)
                error('GLM:design -> ''glm'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};
    
            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});
            run_list = {}; % Initialize as an empty cell array
            for run = runs
                run_list{end+1} = sprintf('run_%02d', run);
            end

            % Load data once, outside of session loop
            % D = dload(fullfile(baseDir,behavDir,subj_id, sprintf('smp2_%d.dat', sn)));
            events_file = sprintf('glm%d_events.tsv', glm);

            Dd = dload(fullfile(baseDir,behavDir,subj_id, events_file));
            regressors = unique(Dd.eventtype);
            nRegr = length(regressors); 

            % init J
            J = [];
            T = [];
            J.dir = {fullfile(baseDir,sprintf('glm%d', glm),subj_id)};
            J.timing.units = 'secs';
            J.timing.RT = 1;

            % number of temporal bins in which the TR is divided,
            % defines the discrtization of the HRF inside each TR
            J.timing.fmri_t = 16;

            % slice number that corresponds to that acquired halfway in
            % each TR
            J.timing.fmri_t0 = 1;
        
            for run = runs
                % Setup scans for current session
                J.sess(run).scans = {fullfile(baseDir, imagingDir, subj_id, sprintf('%s_run_%02d.nii', subj_id, run))};
        
        
                % Preallocate memory for conditions
                J.sess(run).cond = repmat(struct('name', '', 'onset', [], 'duration', []), nRegr, 1);
                
                for regr = 1:nRegr
                    % cue = Dd.cue(regr);
                    % stimFinger = Dd.stimFinger(regr);
                    rows = find(Dd.BN == run & strcmp(Dd.eventtype, regressors(regr)));
                    % cue_id = unique(Dd.cue_id(rows));
                    % stimFinger_id = unique(Dd.stimFinger_id(rows));
                    % epoch = unique(Dd.epoch(rows));
                    % instr = unique(Dd.instruction(rows));
                    
                    % Regressor name
                    J.sess(run).cond(regr).name = regressors{regr};
                    
                    % Define durationDuration(regr));
                    J.sess(run).cond(regr).duration = Dd.Duration(rows); % needs to be in seconds
                    
                    % Define onset
                    J.sess(run).cond(regr).onset  = Dd.Onset(rows);
                    
                    % Define time modulator
                    % Add a regressor that account for modulation of
                    % betas over time
                    J.sess(run).cond(regr).tmod = 0;
                    
                    % Orthogonalize parametric modulator
                    % Make the parametric modulator orthogonal to the
                    % main regressor
                    J.sess(run).cond(regr).orth = 0;
                    
                    % Define parametric modulators
                    % Add a parametric modulators, like force or
                    % reaction time. 
                    J.sess(run).cond(regr).pmod = struct('name', {}, 'param', {}, 'poly', {});

                    %
                    % filling in "reginfo"
                    TT.sn        = sn;
                    TT.run       = run;
                    TT.name      = regressors(regr);
                    
                    T = addstruct(T, TT);
                    
                    if sum(derivs) == 1
                        TT.sn        = sn;
                        TT.run       = run;
                        TT.name      = [regressors{regr} ',deriv1'];
                        
                         T = addstruct(T, TT);
                         
                    elseif sum(derivs) == 2
                        TT.sn        = sn;
                        TT.run       = run;
                        TT.name      = [regressors{regr} ',deriv1'];
                        
                        T = addstruct(T, TT);
                        
                        TT.sn        = sn;
                        TT.run       = run;
                        TT.name      = [regressors{regr} ',deriv2'];
                        
                        T = addstruct(T, TT);
                        
                    end
                    % TT.cue       = cue_id;
                    % TT.epoch     = epoch;
                    % TT.stimFinger = stimFinger_id;
                    % TT.instr = instr;       

                    

                end

                % Specify high pass filter
                J.sess(run).hpf = Inf;

                % J.sess(run).multi
                % Purpose: Specifies multiple conditions for a session. Usage: It is used
                % to point to a file (.mat or .txt) that contains multiple conditions,
                % their onsets, durations, and names in a structured format. If you have a
                % complex design where specifying conditions manually within the script is
                % cumbersome, you can prepare this information in advance and just
                % reference the file here. Example Setting: J.sess(run).multi =
                % {'path/to/multiple_conditions_file.mat'}; If set to {' '}, it indicates
                % that you are not using an external file to specify multiple conditions,
                % and you will define conditions directly in the script (as seen with
                % J.sess(run).cond).
                J.sess(run).multi     = {''};                        

                % J.sess(run).regress
                % Purpose: Allows you to specify additional regressors that are not
                % explicitly modeled as part of the experimental design but may account for
                % observed variations in the BOLD signal. Usage: This could include
                % physiological measurements (like heart rate or respiration) or other
                % variables of interest. Each regressor has a name and a vector of values
                % corresponding to each scan/time point.
                J.sess(run).regress   = struct('name', {}, 'val', {});                        

                % J.sess(run).multi_reg Purpose: Specifies a file containing multiple
                % regressors that will be included in the model as covariates. Usage: This
                % is often used for motion correction, where the motion parameters
                % estimated during preprocessing are included as regressors to account for
                % motion-related artifacts in the BOLD signal. Example Setting:
                % J.sess(run).multi_reg = {'path/to/motion_parameters.txt'}; The file
                % should contain a matrix with as many columns as there are regressors and
                % as many rows as there are scans/time points. Each column represents a
                % different regressor (e.g., the six motion parameters from realignment),
                % and each row corresponds to the value of those regressors at each scan.
                J.sess(run).multi_reg = {''};
                
                % Specify factorial design
                J.fact             = struct('name', {}, 'levels', {});

                % Specify hrf parameters for convolution with
                % regressors
                J.bases.hrf.derivs = derivs;
                J.bases.hrf.params = hrf_params;  % positive and negative peak of HRF - set to [] if running wls (?)
                defaults.stats.fmri.hrf=J.bases.hrf.params; 
                
                % Specify the order of the Volterra series expansion 
                % for modeling nonlinear interactions in the BOLD response
                % *Example Usage*: Most analyses use 1, assuming a linear
                % relationship between neural activity and the BOLD
                % signal.
                J.volt = 1;

                % Specifies the method for global normalization, which
                % is a step to account for global differences in signal
                % intensity across the entire brain or between scans.
                J.global = 'None';

                % remove voxels involving non-neural tissue (e.g., skull)
                J.mask = {fullfile(baseDir, anatomicalDir, subj_id, 'rmask_noskull.nii')};
                
                % Set threshold for brightness threshold for masking 
                % If supplying explicit mask, set to 0  (default is 0.8)
                J.mthresh = 0.;

                % Create map where non-sphericity correction must be
                % applied
                J.cvi_mask = {fullfile(baseDir, anatomicalDir, subj_id, 'rmask_gray.nii')};

                % Method for non sphericity correction
                J.cvi =  'fast';
                
            end

            % T.cue000 = strcmp(T.cue, 'cue0');
            % T.cue025 = strcmp(T.cue, 'cue25');
            % T.cue050 = strcmp(T.cue, 'cue50');
            % T.cue075 = strcmp(T.cue, 'cue75');
            % T.cue100 = strcmp(T.cue, 'cue100');
            % 
            % T.index = strcmp(T.stimFinger, 'index');
            % T.ring = strcmp(T.stimFinger, 'ring');
            % 
            % T.plan = strcmp(T.epoch, 'plan');
            % T.exec = strcmp(T.epoch, 'exec');
            % 
            % T.go = strcmp(T.instr, 'go');
            % T.nogo = strcmp(T.instr, 'nogo');
            % 
            % T.rest = strcmp(T.name, 'rest');


            % remove empty rows (e.g., when skipping runs)
            J.sess = J.sess(~arrayfun(@(x) all(structfun(@isempty, x)), J.sess));
            
            if ~exist(J.dir{1},"dir")
                mkdir(J.dir{1});
            end            
            
            dsave(fullfile(J.dir{1},sprintf('%s_reginfo.tsv', subj_id)), T);
            spm_rwls_run_fmri_spec(J);

            cd(currentDir)
            
            % fprintf('- estimates for glm_%d session %d has been saved for %s \n', glm, ses, subj_str{s});

        case 'GLM:design_matrix'
            
            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            if isempty(sn)
                error('GLM:visualize_design_matrix -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            runs = str2double(split(pinfo.runsSess1{pinfo.sn==sn}, '.')); 

            SPM = load(fullfile(baseDir,[glmEstDir num2str(glm)],subj_id, 'SPM.mat'));SPM=SPM.SPM;

            reginfo = dload(fullfile(baseDir,[glmEstDir num2str(glm)],subj_id, sprintf('subj%d_reginfo.tsv', sn)));
            
            xTicks = SPM.xX.iC;
            X = SPM.xX.X(:, xTicks); % Assuming 'X' is the field holding the design matrix
            
            force = [];
            for run = 1:length(runs)
                force_tmp = forceFromMov(fullfile(baseDir, behavDir, subj_id, sprintf('smp2_%d_%02d.mov', sn, runs(run))));
                force_tmp(:, 1) = force_tmp(:, 1) + force_tmp(:, 2) / 1000 + 366 * (runs(run) -1);
                force = [force; force_tmp];
            end
            
            figure
            % Subplot 1: Force Plot
            subplot(121)
            plot(force(:, [4,6]), force(:,1))
            xlabel('X-axis Label'); % Add appropriate label
            ylabel('Y-axis Label'); % Add appropriate label
            title('Force Plot'); % Add appropriate title
            legend('index', 'ring')
            set(gca, 'YDir', 'reverse', 'ylim', [0, size(X,1)]); % Flip the y-axis direction
            % Subplot 2: Design Matrix
            subplot(122)
            imagesc(X); % Plot the design matrix
            colormap('gray'); % Optional: Use a grayscale colormap for better visibility
            colorbar; % Optional: Include a colorbar to indicate scaling
            xlabel('Regressors');
            ylabel('Scans');
            title('Design Matrix');
            xticks(xTicks)
            xticklabels(reginfo.name)
            % Link the x and y axes of the two subplots
            h1 = subplot(121);
            h2 = subplot(122);
            linkaxes([h1, h2], 'y');
            xtickangle(90)
        
        case 'GLM:estimate'      % estimate beta values

            currentDir = pwd;
            
            sn = [];
            glm = [];
            vararginoptions(varargin, {'sn', 'glm'})

            if isempty(sn)
                error('GLM:estimate -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:estimate -> ''glm'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
             
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                fprintf('- Doing glm%d estimation for subj %s\n', glm, subj_id);
                subj_est_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id);                
                SPM = load(fullfile(subj_est_dir,'SPM.mat'));
                SPM.SPM.swd = subj_est_dir;

                iB = SPM.SPM.xX.iB;
                regr_name = SPM.SPM.xX.name;

                save(fullfile(subj_est_dir, "iB.mat"), "iB");
                
                save(fullfile(subj_est_dir, "regr_name.mat"), "regr_name");
            
                spm_rwls_spm(SPM.SPM);
            end

            cd(currentDir)
            
        case 'GLM:combine_beta_derivs'
            
            sn = [];
            glm = [];            
            derivs = [0 0];
            
            vararginoptions(varargin, {'sn', 'glm', 'derivs'})
                        
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            % get the subject id folder name
            fprintf('combine betas from non-derivs and derivs for participant %s\n', subj_id)
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id);
            
            T = dload(fullfile(glm_dir, sprintf('%s_reginfo.tsv', subj_id)));
            sT = [];
            runs  = T.run;
            regressors = T.name;
            
            for regr = 1:sum(derivs) + 1:length(regressors)                
                for d = 0:sum(derivs)
                    
                    fprintf('run %d, computing regressor %s\n', runs(regr), regressors{regr});
                    
                    beta_path = fullfile(glm_dir, sprintf('beta_%04d.nii', regr + d));
                                        
                    V = spm_vol(beta_path);                    
                    Y_tmp = spm_read_vols(V);
                    
                    if d==0
                        Y = zeros(size(Y_tmp));
                        V1 = V;
                        beta_sign = sign(Y_tmp);
                    end
                    
                    Y = Y + Y_tmp.^2;
                    
                end
                
                V_sum = V1; % Use the metadata of the first beta file
                V_sum.fname = fullfile(glm_dir, sprintf('sbeta_%04d.nii', regr)); % Output file name
                
                Y_sum = beta_sign .* sqrt(Y);
                
                spm_write_vol(V_sum, Y_sum);
                
                TT.sn        = sn;
                TT.run       = runs(regr);
                TT.name      = regressors(regr);

                sT = addstruct(sT, TT);
                
            end
            
            dsave(fullfile(glm_dir,sprintf('s%s_reginfo.tsv', subj_id)), sT);

        case 'GLM:T_contrasts'
            
            currentDir = pwd;

            sn             = [];    % subjects list
            glm            = [];              % glm number
            replace_xCon   = true;

            vararginoptions(varargin, {'sn', 'glm', 'condition', 'baseline', 'replace_xCon'})

            if isempty(sn)
                error('GLM:T_contrasts -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:T_contrasts -> ''glm'' must be passed to this function.')
            end
            
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            % get the subject id folder name
            fprintf('Contrasts for participant %s\n', subj_id)
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

            if replace_xCon
                SPM  = rmfield(SPM,'xCon');
            end

            T = dload(fullfile(glm_dir, sprintf('%s_reginfo.tsv', subj_id)));
            contrasts = unique(T.name);

            for c = 1:length(contrasts)
 
                contrast_name = contrasts{c};
                xcon = zeros(size(SPM.xX.X,2), 1);
                xcon(strcmp(T.name, contrast_name)) = 1;
                xcon = xcon / sum(xcon);
                if ~isfield(SPM, 'xCon')
                    SPM.xCon = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = 1;
                elseif sum(strcmp(contrast_name, {SPM.xCon.name})) > 0
                    idx = find(strcmp(contrast_name, {SPM.xCon.name}));
                    SPM.xCon(idx) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = idx;
                else
                    SPM.xCon(end+1) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = length(SPM.xCon);
                end
                SPM = spm_contrasts(SPM,1:length(SPM.xCon));
                save('SPM.mat', 'SPM','-v7.3');
                % SPM = rmfield(SPM,'xVi'); % 'xVi' take up a lot of space and slows down code!
                % save(fullfile(glm_dir, 'SPM_light.mat'), 'SPM')
    
                % rename contrast images and spmT images
                conName = {'con','spmT'};
                for n = 1:numel(conName)
                    oldName = fullfile(glm_dir, sprintf('%s_%2.4d.nii',conName{n},cname_idx));
                    newName = fullfile(glm_dir, sprintf('%s_%s.nii',conName{n},SPM.xCon(cname_idx).name));
                    movefile(oldName, newName);
                end % conditions (n, conName: con and spmT)

            end

            cd(currentDir)

        case 'GLM:calc_PSC'


            sn             = [];    % subjects list
            glm            = [];    % glm number

            vararginoptions(varargin, {'sn', 'glm'})

            if isempty(sn)
                error('GLM:T_contrast -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:T_contrast -> ''glm'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};            
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

%             psc = readtable(fullfile(baseDir, sprintf('glm%d', glm), 'psc.txt'));
%             contr = {SPM.xCon.name};
% 
%             intercept = {};
%             for k = 1:SPM.nscan
%                 intercept{end+1} = fullfile(glm_dir, sprintf('beta_%d.nii', 220+k));  
%             end
%             
%             for c = 1:size(psc, 1)
% 
%                 p = [psc.condition(c) '-'];
%                 nRegr = SPM.xCon(find(strcmp({SPM.xCon.name}, p))).c > 0;
% 
%                 P = fullfile(glm_dir, ['con_' p '.nii']);
%                 P = [P, intercept];
% 
%                 imean = sprintf('(i1 ./ %f)', nRegr);
%                 cmean = '((i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + i10 + i11) ./ 10)';
% 
%                 formula = sprintf('100 .* (%s - %s) ./ %s', imean, cmean, cmean);
% 
%                 A = [];
%                 A.input = P;
%                 A.output = ['psc_' p];
%                 A.outdir = {glm_dir};
%                 A.expression = formula;
%                 A.var = struct('name', {}, 'value', {});
%                 A.options.dmtx = 0;
%                 A.options.mask = 0;
%                 A.options.interp = 1;
%                 A.options.dtype = 4;               
%     
%                 matlabbatch{1}.spm.util.imcalc=A;
%                 spm_jobman('run', matlabbatch);
%             end
            X=(SPM.xX.X(:,SPM.xX.iC)); % Design matrix - raw

            P={};
            numB=length(SPM.xX.iB);     % Partitions - runs
            for p=SPM.xX.iB
                P{end+1,1}=fullfile(baseDir, glmEstDir,subj_id, ...
                    sprintf('beta_%4.4d.nii',p));  % get the intercepts and use them to calculate the baseline (mean images)
            end
            
            t_con_name = extractfield(SPM.xCon, 'name');
            t_con_name = t_con_name(endsWith(t_con_name, '-'));

            % Create string with formula
            con_div_intercepts = '';
            for r=1:numB
                if r == numB
                    con_div_intercepts = sprintf('i%d./((%si%d)/%d)', ...
                        r+1, con_div_intercepts, r, numB);
                else
                    con_div_intercepts = sprintf('%si%d+', ...
                        con_div_intercepts, r);
                end
            end

            maxX = max(X);
            for con=1:length(t_con_name)  % all contrasts
                idx = find(strcmp({SPM.xCon.name}, t_con_name(1)));
                c = SPM.xCon(idx).c(SPM.xX.iC);
                h = min(maxX(c>0));

                P{numB+1,1}=fullfile(baseDir, glmEstDir, subj_id, ...
                    sprintf('con_%s.nii', t_con_name{con}));
                outname=fullfile(baseDir, glmEstDir, subj_id, ...
                    sprintf('psc_%s.nii', t_con_name{con}));

                formula=sprintf('100.*%f.*%s', h, con_div_intercepts);
                    
                A = [];
                A.input = P;
                A.output = outname;
                A.outdir = {glm_dir};
                A.expression = formula;
                A.var = struct('name', {}, 'value', {});
                A.options.dmtx = 0;
                A.options.mask = 0;
                A.options.interp = 1;
                A.options.dtype = 4;               

                matlabbatch{1}.spm.util.imcalc=A;
                spm_jobman('run', matlabbatch);

            end

        case 'GLM:all'

            sn = [];
            glm = [];
            hrf_params = [6 12 1 1 6 0 32]; % best 6 14
            derivs = [0, 0];
            vararginoptions(varargin,{'sn', 'glm', 'hrf_params', 'derivs'})
            
            spm_get_defaults('cmdline', true);  % Suppress GUI prompts, no request for overwirte

            for s=sn
                
                % Check for and delete existing SPM.mat file
                spm_file = fullfile(baseDir, [glmEstDir num2str(glm)], ['subj' num2str(s)], 'SPM.mat');
                if exist(spm_file, 'file')
                    delete(spm_file);
                end
                
                smp2_imana('GLM:make_event', 'sn', s, 'glm', glm)
                smp2_imana('GLM:design', 'sn', s, 'glm', glm, 'hrf_params', hrf_params, 'derivs', derivs)
                smp2_imana('GLM:estimate', 'sn', s, 'glm', glm)
                smp2_imana('GLM:T_contrasts', 'sn', s, 'glm', glm)
                smp2_imana('SURF:vol2surf', 'sn', s, 'glm', glm, 'type', 'spmT')
                smp2_imana('SURF:vol2surf', 'sn', s, 'glm', glm, 'type', 'beta')
                smp2_imana('SURF:vol2surf', 'sn', s, 'glm', glm, 'type', 'res')
                smp2_imana('SURF:vol2surf', 'sn', s, 'glm', glm, 'type', 'con')
                smp2_imana('HRF:ROI_hrf_get', 'sn', s, 'glm', glm, 'hrf_params', hrf_params)
            end
            
%         case 'SURF:reconall' % Freesurfer reconall routine
%             % Calls recon-all, which performs, all of the
%             % FreeSurfer cortical reconstruction process
%         
%             sn=[];
%             vararginoptions(varargin,{'sn'})
%             if isempty(sn)
%                 error('SURF:reconall -> ''sn'' must be passed to this function.')
%             end
% 
%             subj_row=getrow(pinfo,pinfo.sn== sn );
%             subj_id = subj_row.subj_id{1};   
%         
%             % recon all inputs
%             fs_dir = fullfile(baseDir,freesurferDir, subj_id);
%             anatomical_dir = fullfile(baseDir,anatomicalDir);
%             anatomical_name = sprintf('%s_anatomical.nii', subj_id);
%             
%             % Get the directory of subjects anatomical;
%             freesurfer_reconall(fs_dir, subj_id, ...
%                 fullfile(anatomical_dir, subj_id, anatomical_name));


        case 'SURF:fs2wb'          % Resampling subject from freesurfer fsaverage to fs_LR
            
            sn   = []; % subject list
            res  = 32;          % resolution of the atlas. options are: 32, 164
            % hemi = [1, 2];      % list of hemispheres
           
            vararginoptions(varargin, {'sn'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id);

            % dircheck(outDir);
            surf_resliceFS2WB(subj_id, fsDir, fullfile(baseDir, wbDir), 'resolution', sprintf('%dk', res))

        case 'SURF:vol2surf'
            
            currentDir = pwd;

            sn   = []; % subject list
            filename = [];
            res  = 32;          % resolution of the atlas. options are: 32, 164
            type = 'spmT';
            id = [];
            glm = [];
            % hemi = [1, 2];      % list of hemispheres
           
            vararginoptions(varargin, {'sn', 'glm', 'type'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            glmEstDir = [glmEstDir num2str(glm)];
            
            V = {};
            cols = {};
            if strcmp(type, 'spmT')
%                 filename = ['spmT_' id '.func.gii'];
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'spmT_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'beta')
                SPM = load(fullfile(baseDir, glmEstDir, subj_id, 'SPM.mat')); SPM=SPM.SPM;
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'beta_*.nii'));
                files = files(SPM.xX.iC);
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'psc')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'psc_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'con')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'con_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'res')
                V{1} = fullfile(baseDir, glmEstDir, subj_id, 'ResMS.nii');
                cols{1} = 'ResMS';
            end

            hemLpial = fullfile(baseDir, wbDir, subj_id,  [subj_id '.L.pial.32k.surf.gii']);
            hemRpial = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.pial.32k.surf.gii']);
            hemLwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.L.white.32k.surf.gii']);
            hemRwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.white.32k.surf.gii']);
            
            hemLpial = gifti(hemLpial);
            hemRpial = gifti(hemRpial);
            hemLwhite = gifti(hemLwhite);
            hemRwhite = gifti(hemRwhite);

            c1L = hemLpial.vertices;
            c2L = hemLwhite.vertices;
            c1R = hemRpial.vertices;
            c2R = hemRwhite.vertices;

            GL = surf_vol2surf(c1L,c2L,V,'anatomicalStruct','CortexLeft', 'exclude_thres', 0.9, 'faces', hemLpial.faces);
            GL = surf_makeFuncGifti(GL.cdata,'anatomicalStruct', 'CortexLeft', 'columnNames', cols);
    
            save(GL, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.'                                               type '.L.func.gii']))
    
            GR = surf_vol2surf(c1R,c2R,V,'anatomicalStruct','CortexRight', 'exclude_thres', 0.9, 'faces', hemRpial.faces);
            GR = surf_makeFuncGifti(GR.cdata,'anatomicalStruct', 'CortexRight', 'columnNames', cols);

            save(GR, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.' type '.R.func.gii']))
            
            cd(currentDir)
            

%         case 'SURF:resample_labelFS2WB'
% 
%             subj_id = '';
%             atlas_dir = [];
%             resolution = '32k';
%             atlas = 'aparc';
% 
%             vararginoptions(varargin, {'sn', 'resolution', 'atlas'});
%             
%             subj_id = pinfo.subj_id{pinfo.sn==sn};
%  
%             if isempty(atlas_dir)
% %                 repro_dir=fileparts(which('surf_label2label'));
%                 atlas_dir = [path 'GitHub/surfAnalysis/standard_mesh/'];
%             end
% 
%             fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id, subj_id);
%             
%             out_dir = fullfile(baseDir, wbDir, subj_id);
%             
% %             cd(fullfile(subject_dir,subj_id)); 
%             
%             hem = {'lh', 'rh'};
%             Hem = {'L', 'R'};
%             aStruct = {'CortexLeft', 'CortexRight'};
%             
%             for h = 1:length(hem)
%             
%                 reg_sphere = [fsDir '/surf/' hem{h} '.sphere.reg.surf.gii'];
%                 label = [fsDir '/label/' hem{h} '.' atlas '.annot'];
%                 surf = [fsDir '/surf/' hem{h} '.pial'];
%                 out_fs = [fsDir '/label/' hem{h} '.label.gii'];
%                 source_annot = [fsDir '/label/' hem{h} '.label.gii'];
%                 out_name = fullfile(out_dir,[subj_id '.' Hem{h} '.' resolution '.' atlas '.label.gii']); 
%                 atlas_name = fullfile(atlas_dir,'resample_fsaverage',['fs_LR-deformed_to-fsaverage.' Hem{h} '.sphere.' resolution '_fs_LR.surf.gii']);
%             
%                 system(['mris_convert --annot ' label ' ' surf ' ' out_fs]);
%             
%                 system(['wb_command -label-resample ' source_annot ' ' reg_sphere ' ' atlas_name ' BARYCENTRIC ' out_name]);
% 
%                 A = gifti(out_name);
%                 cdata = A.cdata;
%                 keys = A.labels.key;
%                 name = A.labels.name;
%                 
%                 rgba = A.labels.rgba;
%                 G = surf_makeLabelGifti(cdata, 'anatomicalStruct', aStruct{h},'columnNames', {atlas}, 'labelNames', name, 'labelRGBA', rgba, 'keys', keys);
%                 save(G, out_name)
%             
%             end
%         
%         case 'SURF:group'
%             sn = [];
%             glm=[];
%             vararginoptions(varargin,{'sn', 'glm'})
%             
%             PL = {};
%             PR = {};
%             participants = {};
%             namesL = {};
%             namesR = {};
%             
%             G = gifti(fullfile(baseDir, wbDir, pinfo.subj_id{pinfo.sn==sn(1)}, sprintf('glm%d.con.L.func.gii', glm)));
% 
%             names = surf_getGiftiColumnNames(G);
%             for n=1:length(names)
%                 namesL{n} = [names{n}(1:end-3) 'L.func.gii'];
%                 namesR{n} = [names{n}(1:end-3) 'R.func.gii'];
%             end
% 
%             for s=1:length(sn)
%                 subj_id = pinfo.subj_id{pinfo.sn==sn(s)};
% 
%                 participants = [participants, subj_id];
% 
%                 PL = [PL, fullfile(baseDir, wbDir, subj_id, sprintf('glm%d.con.L.func.gii', glm))];
%                 PR = [PR, fullfile(baseDir, wbDir, subj_id, sprintf('glm%d.con.R.func.gii', glm))];
% 
%             end
% 
%             surf_groupGiftis(PL, 'outcolnames',participants, 'outfilenames', fullfile(baseDir, wbDir, 'group', namesL))
%             surf_groupGiftis(PR, 'outcolnames',participants, 'outfilenames', fullfile(baseDir, wbDir, 'group', namesR))
% 
%             meanL = [];
%             tvalL = [];
%             pvalL = [];
% 
%             meanR = [];
%             tvalR = [];
%             pvalR = [];
% 
%             for n = 1:length(names)
%                 
%                 PL = gifti(fullfile(baseDir, wbDir, 'group', namesL{n}));
%                 cdata = PL.cdata;
%                 meanL = [meanL nansum(cdata, 2)];
%                 [~, pval, ~, tval] = ttest(cdata');
%                 tvalL = [tvalL tval.tstat' ];
%                 pvalL = [pvalL pval' ];
% 
%                 PR = gifti(fullfile(baseDir, wbDir, 'group', namesR{n}));
%                 cdata = PR.cdata;
%                 meanR = [meanR nansum(cdata, 2)];
%                 [~, pval, ~, tval] = ttest(cdata');
%                 tvalR = [tvalR tval.tstat' ];
%                 pvalR = [pvalR pval' ];
%                 
%                 
% 
%             end
% 
%             meanExecL = mean(meanL(:, contains(namesL, 'ring') | contains(namesL, 'index')), 2);
%             meanExecR = mean(meanR(:, contains(namesR, 'ring') | contains(namesR, 'index')), 2);
% 
%             meanPlanL = mean(meanL(:, ~contains(namesL, 'ring') & ~contains(namesL, 'index')), 2);
%             meanPlanR = mean(meanR(:, ~contains(namesR, 'ring') & ~contains(namesR, 'index')), 2);
% 
% 
%             meanL = surf_makeFuncGifti([meanL meanExecL meanPlanL],'anatomicalStruct', 'CortexLeft','columnNames', [namesL, {'exec.L.func.gii'}, {'plan.L.func.gii'}]);
%             tvalL = surf_makeFuncGifti(tvalL,'anatomicalStruct','CortexLeft','columnNames', namesL);
%             pvalL = surf_makeFuncGifti(pvalL,'anatomicalStruct','CortexLeft','columnNames', namesL);
% 
%             meanR = surf_makeFuncGifti([meanR meanExecR meanPlanR],'anatomicalStruct','CortexRight','columnNames', [namesR, {'exec.R.func.gii'}, {'plan.R.func.gii'}]);
%             tvalR = surf_makeFuncGifti(tvalR,'anatomicalStruct','CortexRight','columnNames', namesR);
%             pvalR = surf_makeFuncGifti(pvalR,'anatomicalStruct','CortexRight','columnNames', namesR);
% 
%             save(meanL, fullfile(baseDir, wbDir, 'group', 'mean.L.func.gii'))
%             save(tvalL, fullfile(baseDir, wbDir, 'group', 'tval.L.func.gii'))
%             save(pvalL, fullfile(baseDir, wbDir, 'group', 'pval.L.func.gii'))
% 
%             save(meanR, fullfile(baseDir, wbDir, 'group', 'mean.R.func.gii'))
%             save(tvalR, fullfile(baseDir, wbDir, 'group', 'tval.R.func.gii'))
%             save(pvalR, fullfile(baseDir, wbDir, 'group', 'pval.R.func.gii'))

        case 'SEARCH:define' 

            glm=[];
            sn=[];
            rad=12;
            vox=100;
            res='32';
            vararginoptions(varargin,{'sn','glm','rad','vox','surf'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            mask = fullfile(baseDir, glmEstDir, subj_id, 'mask.nii');
            Vmask = spm_vol(mask);
            Vmask.data = spm_read_vols(Vmask);
            Mask = rsa.readMask(Vmask);
            
            % directory for pial and white
            surfDir = fullfile(baseDir, wbDir,subj_id);  

            white = {fullfile(surfDir, sprintf('%s.L.white.%sk.surf.gii', subj_id, res)),...
                fullfile(surfDir, sprintf('%s.R.white.%sk.surf.gii', subj_id, res))};
            pial = {fullfile(surfDir, sprintf('%s.L.pial.%sk.surf.gii', subj_id, res)),...
                fullfile(surfDir, sprintf('%s.R.pial.%sk.surf.gii', subj_id, res))};
            
            S = rsa.readSurf(white, pial);  S = [S{:}];
            
            L = rsa.defineSearchlight_surface(S, Mask, 'sphere', [rad vox]);
            save(fullfile(baseDir, anatomicalDir, subj_id, sprintf('%s_searchlight_%d.mat',subj_id,vox)),'-struct','L');
            varargout={L};
        
        case 'HRF:ROI_hrf_get'                   % Extract raw and estimated time series from ROIs
            
            currentDir = pwd;
            
            sn = [];
            ROI = 'all';
            pre=10;
            post=10;
            atlas = 'ROI';
            glm = 9;
            hrf_params = [5, 14];

            vararginoptions(varargin,{'ROI','pre','post', 'glm', 'sn', 'atlas', 'hrf_params'});

            glmDir = fullfile(baseDir, [glmEstDir num2str(glm)]);
            T=[];

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            fprintf('%s\n',subj_id);

            % load SPM.mat
            cd(fullfile(glmDir,subj_id));
            SPM = load('SPM.mat'); SPM=SPM.SPM;
            
            TR = SPM.xY.RT;
            nScan = SPM.nscan(1);

            % % make (dummy) regressors
            % hrf = spm_hrf(1, hrf_params);
            % events = dload(fullfile(baseDir,behavDir,subj_id ,sprintf('glm%d_events.tsv', glm)));
            % eventtype = unique(events.eventtype);
            % regr = zeros(2760, length(eventtype));
            % 
            % for e = 1:length(eventtype)
            % 
            %     onset = events.Onset(strcmp(eventtype(e), events.eventtype));
            %     block = events.BN(strcmp(eventtype(e), events.eventtype));
            %     onset = round(onset) + (block - 1) * 276;
            % 
            %     regr(onset, e) = 1;
            %     regrC(:, e) = conv(regr(:, e), hrf);             
            % end
            
            % load ROI definition (R)
            R = load(fullfile(baseDir, regDir,subj_id,[subj_id '_' atlas '_region.mat'])); R=R.R;
            
            % extract time series data
            [y_raw, y_adj, y_hat, y_res,B] = region_getts(SPM,R);
            
%             D = spmj_get_ons_struct(SPM);
            Dd = dload(fullfile(baseDir, behavDir, subj_id, sprintf('smp2_%d.dat', sn)));
            
            D = [];
            D.ons = (Dd.startTimeReal / 1000) / TR;
            D.ons = D.ons + (Dd.BN - 1) * nScan;
            D.block = Dd.BN;
            D.GoNogo = Dd.GoNogo;     
            D.cue = Dd.cue;
            D.stimFinger = Dd.stimFinger;
            
            for r=1:size(y_raw,2)
                for i=1:size(D.block,1)
                    D.y_adj(i,:)=cut(y_adj(:,r),pre,round(D.ons(i)),post,'padding','nan')';
                    D.y_hat(i,:)=cut(y_hat(:,r),pre,round(D.ons(i)),post,'padding','nan')';
                    D.y_res(i,:)=cut(y_res(:,r),pre,round(D.ons(i)),post,'padding','nan')';
                    D.y_raw(i,:)=cut(y_raw(:,r),pre,round(D.ons(i)),post,'padding','nan')';
%                     D.regr(i, :, :)=cut(regrC,pre,round(D.ons(i)),post,'padding','nan')';
                end
                
                % Add the event and region information to tje structure. 
                len = size(D.ons,1);                
                D.SN        = ones(len,1)*sn;
                D.region    = ones(len,1)*r;
                D.name      = repmat({R{r}.name},len,1);
                D.hem       = repmat({R{r}.hem},len,1);
%                 D.type      = D.event; 
                T           = addstruct(T,D);
            end
            
            save(fullfile(baseDir,regDir, subj_id, sprintf('hrf_glm%d.mat', glm)),'T'); 
            varargout{1} = T;
            varargout{2} = y_adj;
            
            cd(currentDir)

        case 'ROI:define'
            
            sn = [];
            glm = 10;
            atlas = 'ROI';
            
            vararginoptions(varargin,{'sn', 'atlas', 'glm'});
            
            if isfolder('/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32')
                atlasDir = '/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32';
            elseif isfolder('/cifs/diedrichsen/data/Atlas_templates/fs_LR_32')
                atlasDir = '/cifs/diedrichsen/data/Atlas_templates/fs_LR_32';
            end
            atlasH = {sprintf('%s.32k.L.label.gii', atlas), sprintf('%s.32k.R.label.gii', atlas)};
            atlas_gii = {gifti(fullfile(atlasDir, atlasH{1})), gifti(fullfile(atlasDir, atlasH{1}))};

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            Hem = {'L', 'R'};
            R = {};
            r = 1;
            for h = 1:length(Hem)
                for reg = 1:length(atlas_gii{h}.labels.name)

                    R{r}.white = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.white.32k.surf.gii']);
                    R{r}.pial = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.pial.32k.surf.gii']);
                    R{r}.image = fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'mask.nii');
                    R{r}.linedef = [5 0 1];
                    key = atlas_gii{h}.labels.key(reg);
                    R{r}.location = find(atlas_gii{h}.cdata==key);
                    R{r}.hem = Hem{h};
                    R{r}.name = atlas_gii{h}.labels.name{reg};
                    R{r}.type = 'surf_nodes_wb';

                    r = r+1;
                end
            end

            R = region_calcregions(R, 'exclude', [2 3; 2 4; 2 5; 4 5; 8 9; 2 8;...
                11 12; 11 13; 11 14; 13 14; 17 18; 11 17], 'exclude_thres', .8);
            
            output_path = fullfile(baseDir, regDir, subj_id);
            if ~exist(output_path, 'dir')
                mkdir(output_path)
            end
            
            Vol = fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'mask.nii');
            for r = 1:length(R)
                img = region_saveasimg(R{r}, Vol, 'name',fullfile(baseDir, regDir, subj_id, sprintf('%s.%s.%s.nii', atlas, R{r}.hem, R{r}.name)));
            end       
            
            save(fullfile(output_path, sprintf('%s_%s_region.mat',subj_id, atlas)), 'R');

        case 'ROI:define_all'
            sn = [];
            atlas = 'ROI';
            
            vararginoptions(varargin,{'sn', 'atlas'});
            
            for s=sn
                smp2_imana('ROI:define', 'sn', s)
                clc
            end

        case 'HRF:ROI_hrf_plot'                 % Plot extracted time series
            sn = [];
            roi = {'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'};
            atlas = 'ROI';
            hem = 'L';
            eventname = [];
            regr = [];
            glm = [];
            p = true;

            vararginoptions(varargin,{'sn', 'roi', 'atlas', 'eventname', 'hem', 'glm', 'regr', 'p'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            

                % figure
    
            for r=1:length(roi)

                T = load(fullfile(baseDir, regDir, subj_id, sprintf('hrf_glm%d.mat', glm))); T=T.T;
                T = getrow(T,strcmp(T.name, roi{r}));
                pre = 10;
                post = 10;
                y_adj(r, :) = nanmean(T.y_adj(strcmp(T.eventname, eventname), :), 1);
                y_hat(r, :) = nanmean(T.y_hat(strcmp(T.eventname, eventname), :), 1);
                
                % Select a specific subset of things to plot 

                % subset = find(strcmp(T.eventname, eventname) & strcmp(T.hem, hem));

                if p==true

                    subplot(2, 4, r)
                    
                    % yyaxis left
                    % traceplot([-pre:post],T.y_hat, 'subset', subset ,'split', [], 'linestyle','--');
                    xAx = linspace(-pre, post, pre+post+1);
                    
                    plot(xAx, y_adj(r, :), 'linestyle','-', 'Color', 'red')
                    hold on;
                    % traceplot([-pre:post],T.y_res,  'subset', subset ,'split', [], 'linestyle',':');
                    xline(0);
                    yline(0);
                    % ax = gca;
                    % ax.YColor = 'b';
                    
        
                    % yyaxis right
                    
                    plot(xAx, y_hat(r, :), 'linestyle','--', 'Color', 'blue')
                    % traceplot([-pre:post],T.y_adj,'leg',[],'subset', subset , 'leglocation','bestoutside', 'linestyle','-', 'linecolor', [1 0 0]);
                    % ax = gca;
                    % ax.YColor = 'r';
                    
                    % yyaxis right
                    % customColors = [
                    %     0.9290, 0.6940, 0.1250;  % Yellow
                    %     0.4940, 0.1840, 0.5560;  % Purple
                    %     0.4660, 0.6740, 0.1880;  % Green
                    %     0.3010, 0.7450, 0.9330;  % Cyan
                    % ];
                    % plot(xAx, 10 * squeeze(mean(T.regr(strcmp(T.eventname, eventname), regr, :), 1)), 'linestyle','-', 'LineWidth', 2)
                    % set(gca, 'ColorOrder', customColors);
                    
                    if r==1
                        legend({'y_{adj}', 'y_{hat}'}, 'Location','northwest')
                    end
                
                    hold off;
                    xlabel('TR relative to startTrialReal');
                    ylabel('activation');
    
                    ylim([-2, 2])
        
                    title(roi{r})

                
    
                    sgtitle(sprintf('%s\nhemisphere:%s, eventname:%s', subj_id, hem, eventname), 'interpreter', 'none')
                    set(gcf, 'Position', [100 100, 1400, 800])
        
                    drawnow;
        
                    fig = gcf;
        
                    saveas(fig, fullfile(baseDir, 'figures', subj_id, sprintf('hrf.%s.glm%d.%s.%s.png', atlas, glm, hem, eventname)))

                end
            
            end

            varargout{1} = y_adj;
            varargout{2} = y_hat;

            
        case 'HRF:plot_all'

            sn = [];
            glm = 9;
            eventname = 'go';
            roi = {'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'};
            atlas = 'ROI';
            hem = 'L';

            vararginoptions(varargin,{'sn', 'glm', 'eventname'});

            min_sn = min(sn);

            for s=sn
                [y_adj(s - min_sn + 1, :, :), y_hat(s- min_sn + 1, :, :)] = smp2_imana('HRF:ROI_hrf_plot', 'sn', s, 'glm', glm, 'eventname', eventname, 'p', false);

            end

            figure

            for r=1:length(roi)

                pre = 10;
                post = 10;
                
                % Select a specific subset of things to plot 

                % subset = find(strcmp(T.eventname, eventname) & strcmp(T.hem, hem));

                h(r) = subplot(2, 4, r);
                
                % yyaxis left
                % traceplot([-pre:post],T.y_hat, 'subset', subset ,'split', [], 'linestyle','--');
                xAx = linspace(-pre, post, pre+post+1);
                p1 = plot(xAx, squeeze(mean(y_adj(:, r, :), 1)), 'linestyle','-', 'Color', 'red', 'LineWidth', 2)
                hold on
                plot(xAx, squeeze(y_adj(:, r, :)), 'linestyle','-', 'Color', [1 0 0 .2])
                % traceplot([-pre:post],T.y_res,  'subset', subset ,'split', [], 'linestyle',':');
                xline(0);
                yline(0);
                % ax = gca;
                % ax.YColor = 'b';
                
    
                % yyaxis right
                p2 = plot(xAx, squeeze(mean(y_hat(:, r, :),1)), 'linestyle','-', 'Color', 'blue', 'LineWidth', 2);
                plot(xAx, squeeze(y_hat(:, r, :)), 'linestyle','-', 'Color', [0 0 1 .2], 'LineWidth', 2)
                % traceplot([-pre:post],T.y_adj,'leg',[],'subset', subset , 'leglocation','bestoutside', 'linestyle','-', 'linecolor', [1 0 0]);
                % ax = gca;
                % ax.YColor = 'r';
                
                % yyaxis right
                % customColors = [
                %     0.9290, 0.6940, 0.1250;  % Yellow
                %     0.4940, 0.1840, 0.5560;  % Purple
                %     0.4660, 0.6740, 0.1880;  % Green
                %     0.3010, 0.7450, 0.9330;  % Cyan
                % ];
                % plot(xAx, 10 * squeeze(mean(T.regr(strcmp(T.eventname, eventname), regr, :), 1)), 'linestyle','-', 'LineWidth', 2)
                % set(gca, 'ColorOrder', customColors);
                
                if r==1
                    legend([p1, p2], {'y_{adj}', 'y_{hat}'}, 'Location','northwest')
                end

                hold off;
                xlabel('TR relative to startTrialReal');
                ylabel('activation');

                ylim([-2, 2])
    
                title(roi{r})

            end

            sgtitle(sprintf('%s\nhemisphere:%s, eventname:%s', 'group', hem, eventname), 'interpreter', 'none')
            set(gcf, 'Position', [100 100, 1400, 800])

            % Link the x and y axes of the two subplots

            linkaxes(h, 'y');

            drawnow;

            fig = gcf;

            % saveas(fig, fullfile(baseDir, 'figures', subj_id, sprintf('hrf.%s.glm%d.%s.%s.png', atlas, glm, hem, eventname)))

    
        case 'HRF:plot_all2'

            sn = [];
            glm = 9;
            roi = {'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'};
            atlas = 'ROI';
            hem = 'L';

            vararginoptions(varargin,{'sn', 'glm', 'eventname'});

            min_sn = min(sn);

            for s=sn
                [y_adj_go(s - min_sn + 1, :, :), y_hat_go(s- min_sn + 1, :, :)] = smp2_imana('HRF:ROI_hrf_plot', 'sn', s, 'glm', glm, 'eventname', 'go', 'p', false);
                [y_adj_nogo(s - min_sn + 1, :, :), y_hat_nogo(s- min_sn + 1, :, :)] = smp2_imana('HRF:ROI_hrf_plot', 'sn', s, 'glm', glm, 'eventname', 'nogo', 'p', false);
                
            end
            
            figure

            for r=1:length(roi)

                pre = 10;
                post = 10;
                
                % Select a specific subset of things to plot 

                % subset = find(strcmp(T.eventname, eventname) & strcmp(T.hem, hem));

                h(r) = subplot(2, 4, r);
                
                % yyaxis left
                % traceplot([-pre:post],T.y_hat, 'subset', subset ,'split', [], 'linestyle','--');
                xAx = linspace(-pre, post, pre+post+1);
                p1 = plot(xAx, squeeze(mean(y_adj_go(:, r, :), 1)), 'linestyle','-', 'Color', 'magenta');
                hold on
                plot(xAx, squeeze(mean(y_hat_go(:, r, :), 1)), 'linestyle','--', 'Color', 'magenta');
                % plot(xAx, squeeze(y_adj_go(:, r, :)), 'linestyle','-', 'Color', [1 0 0 .2])
                % traceplot([-pre:post],T.y_res,  'subset', subset ,'split', [], 'linestyle',':');
                xline(0);
                yline(0);
                % ax = gca;
                % ax.YColor = 'b';
                
    
                % yyaxis right
                p2 = plot(xAx, squeeze(mean(y_adj_nogo(:, r, :),1)), 'linestyle','-', 'Color', 'green');
                plot(xAx, squeeze(mean(y_hat_nogo(:, r, :),1)), 'linestyle','--', 'Color', 'green');
                % plot(xAx, squeeze(y_adj_nogo(:, r, :)), 'linestyle','-', 'Color', [0 0 1 .2], LineWidth=2)
                % traceplot([-pre:post],T.y_adj,'leg',[],'subset', subset , 'leglocation','bestoutside', 'linestyle','-', 'linecolor', [1 0 0]);
                % ax = gca;
                % ax.YColor = 'r';
                
                % yyaxis right
                % customColors = [
                %     0.9290, 0.6940, 0.1250;  % Yellow
                %     0.4940, 0.1840, 0.5560;  % Purple
                %     0.4660, 0.6740, 0.1880;  % Green
                %     0.3010, 0.7450, 0.9330;  % Cyan
                % ];
                % plot(xAx, 10 * squeeze(mean(T.regr(strcmp(T.eventname, eventname), regr, :), 1)), 'linestyle','-', 'LineWidth', 2)
                % set(gca, 'ColorOrder', customColors);
                
                if r==1
                    legend({'go adj', 'go hat', 'nogo adj', 'nogo hat'}, 'Location','northwest')
                end

                hold off;
                xlabel('TR relative to startTrialReal');
                ylabel('activation');

                ylim([-2, 2])
    
                title(roi{r})

            end

            sgtitle(sprintf('%s\nhemisphere:%s', 'group', hem), 'interpreter', 'none')
            set(gcf, 'Position', [100 100, 1400, 800])

            % Link the x and y axes of the two subplots

            linkaxes(h, 'y');

            drawnow;

            fig = gcf;



    end
end

