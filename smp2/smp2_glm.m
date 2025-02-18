function smp2_glm(what, varargin)

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
    
    switch what
        
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
                J.sess(run).hpf = 128;

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
             
            fprintf('- Doing glm%d estimation for subj %s\n', glm, subj_id);
            subj_est_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id);                
            SPM = load(fullfile(subj_est_dir,'SPM.mat'));
            SPM.SPM.swd = subj_est_dir;

            iB = SPM.SPM.xX.iB;
            regr_name = SPM.SPM.xX.name;

            save(fullfile(subj_est_dir, "iB.mat"), "iB");

            save(fullfile(subj_est_dir, "regr_name.mat"), "regr_name");

            spm_rwls_spm(SPM.SPM);

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
                
            % Check for and delete existing SPM.mat file
            spm_file = fullfile(baseDir, [glmEstDir num2str(glm)], ['subj' num2str(sn)], 'SPM.mat');
            if exist(spm_file, 'file')
                delete(spm_file);
            end

            smp2_glm('GLM:make_event', 'sn', sn, 'glm', glm)
            smp2_glm('GLM:design', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params, 'derivs', derivs)
            smp2_glm('GLM:estimate', 'sn', sn, 'glm', glm)
            smp2_glm('GLM:T_contrasts', 'sn', sn, 'glm', glm)
            smp2_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'spmT')
            smp2_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'beta')
            smp2_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'res')
            smp2_glm('SURF:vol2surf', 'sn', sn, 'glm', glm, 'type', 'con')
%             smp2_glm('HRF:ROI_hrf_get', 'sn', sn, 'glm', glm, 'hrf_params', hrf_params)
            
        case 'HRF:ROI_hrf_get'                   % Extract raw and estimated time series from ROIs
            
            currentDir = pwd;
            
            sn = [];
            ROI = 'all';
            pre=10;
            post=10;
            atlas = 'ROI';
            glm = 12;
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
            
        case 'ROI:define'
            
            sn = [];
            glm = [];
            atlas = 'ROI';
            vararginoptions(varargin, {'sn', 'glm', 'atlas'});
            
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
%                     R{r}.image = fullfile(baseDir, anatomicalDir, subj_id, 'rmask_gray.nii');
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
        
    end

end