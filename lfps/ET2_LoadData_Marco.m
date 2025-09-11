clear
close all
for monkeyList = {'Malfoy','Pert'}
    monkey = monkeyList{1};

    Lyle = false;
    Marco = true;

    %% CHANGE THIS TO WHERE YOU WANT THE DATA SAVED
    if strcmp(computer, 'GLNXA64')
        baseDir = ['/cifs/pruszynski/JAM/ET2/' monkey '/'];
        driveDir = ['/cifs/pruszynski/' monkey '/'];
    else
        baseDir = ['/Users/jonathanamichaels/Desktop/jmichaels/Projects/ET2_Neuropixels/Results/' monkey '/'];
        driveDir = ['/Users/jonathanamichaels/' monkey '/'];
    end


    if strcmp(monkey, 'Malfoy')
        % specify only the relevant days for this experiment
        expFolders = {'040622','040722','040822','041422','042022','042222','042822', '042922',...
            '050322','050422','050522','050622','051222','051322','051622','051822','052522','052622',...
            '052722','121422','121522','020223','020323','020723','020923','021023','021423','021523',...
            '021623','021723','022423','022823', '030123', '030323'};
    elseif strcmp(monkey, 'Pert')
        expFolders = {'122223', '010924', '011024', '011124', '011224', '011724', '011824', '011924', '012324', '012424', ...
            '012624', '012924', '013024', '013124', '020124', '020224', '020524', '020624', '020724', '020824', '020924', ...
            '021924', '022024', '022124', '022724', '022824', '032924'};
    end


    doLFP = false;
    if Lyle
        if strcmp(monkey, 'Malfoy')
            expFolders = expFolders([6, 16, 18, 20, 22, 23, 26, 30, 31, 33]);
        elseif strcmp(monkey, 'Pert')
            expFolders = expFolders([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 27]);
        end
        stability_threshold = 2;
    elseif Marco
        stability_threshold = 2;
        if strcmp(monkey, 'Pert')
            expFolders = expFolders(9);
        elseif strcmp(monkey, 'Malfoy')
            expFolders = expFolders(end);
        end
        doLFP = true;
    else
        stability_threshold = 2;
        if strcmp(monkey, 'Pert')
            expFolders = expFolders([6:18 20:24 26:27]);
        elseif strcmp(monkey, 'Malfoy')
            expFolders = expFolders([1:2 5:8 10:23 25:31 33:34]);
        end
    end

    saveGPFA = false;
    for subject = 1:length(expFolders)
        disp(['Processing Recording #' num2str(subject)])
        loadDir = [driveDir '/' expFolders{subject}];
        dataset = expFolders{subject};
        config = yaml.loadFile([loadDir '/config.yaml']);
        brain_area_list = config.Session.brain_area_list;
        brain_coord_list = config.Session.brain_coord_list;
        myo_muscle_list = config.Session.myo_muscle_list;

        kinarm_file = dir([loadDir '/*.kinarm']);
        data = zip_load([loadDir '/' kinarm_file.name]);									% Loads the named file into a new structure called 'data'.
        data = KINARM_add_hand_kinematics(data);						% Add hand velocity, acceleration and commanded forces to the data structure
        data = KINARM_add_sho_elb(data);
       
        %data = KINARM_add_trough_inertia(data, 'trough_db', DB, 'trough_size', trough_size, 'trough_location', trough_location);
        data_f = filter_double_pass(data, 'enhanced', 'fc', 20);
        %data_f = KINARM_add_torques(data_f);


        date = datetime([data.c3d(1).EXPERIMENT.START_DATE_TIME(1:10) ' ' data.c3d(1).EXPERIMENT.START_DATE_TIME(12:16)], ...
            'InputFormat', 'yyyy-MM-dd HH:mm');

        % How many neuropixels?
        neuropix = dir([loadDir '/*_g0/*imec*']);
        num_neuropixels = length(neuropix);

        % Any muscle?
        myo = dir([loadDir '/*_myo']);
        is_myo = length(myo);
        is_myo = false;

        trialStartShifted = []; trialNum = cell(1,2); trialStart = [];
        bitPoints = fliplr(1325 : 300 : 6000);
        if num_neuropixels > 0
            sync_file = dir([loadDir '/*_g0/*imec0/sync.mat']);
            if isfile([sync_file.folder '/' sync_file.name])
                load([sync_file.folder '/' sync_file.name]);
            else
                error('No sync');
            end
            trialSignal = sync;
            ii = find(diff(trialSignal) > 0.2) + 1;
            for i = 1:length(ii)
                dd(i) = sum(abs(trialSignal(ii(i)-8000 : ii(i)-1)) > 0.2);
                if dd(i) == 0
                    if ii(i) + bitPoints < length(trialSignal)
                        trialStart(end+1) = ii(i);
                        temp = trialSignal(ii(i) + bitPoints);
                        trialNum{1}(end+1) = bin2dec(sprintf('%d',temp));
                    end
                end
            end
            trialStartShifted = trialStart;
        end

        trialStartMyo = [];
        if is_myo
            myoDirs = dir([loadDir '/*_myo/sorted*']);
            load([myoDirs(1).folder '/sync'])

            ii = find(diff(sync) > 0.2) + 1;
            for i = 1:length(ii)
                dd(i) = sum(abs(sync(ii(i)-8000 : ii(i)-1)) > 0.2);
                if dd(i) == 0
                    trialStartMyo(end+1) = ii(i);
                    temp = sync(ii(i) + bitPoints);
                    trialNum{2}(end+1) = bin2dec(sprintf('%d',temp));
                end
            end
        end
        neuralSync{1} = trialStartShifted;
        neuralSync{2} = trialStartMyo;

        stability = cell(1,2); stabilityBig = cell(1,2); unitLabel = cell(1,2);
        spikeTimes = cell(1,2); type1 = []; type2 = [];
        allMaxAmp = cell(1,2); allSpatial = cell(1,2);
        if is_myo
            type2 = 2;
        end
        if num_neuropixels > 0
            type1 = 1;
        end
        typeList = [type2, type1];
        templates = cell(1,2);
        brainID = []; brainCoords = []; muscleID = []; allConsistency = [];
        cutOffTrial = [Inf Inf];
        for type = typeList
            % Let's check if we have multiple myomatrixes
            if type == 2
                iters = length(myoDirs);
            elseif type == 1
                iters = num_neuropixels;
            end
            bigT = []; bigI = []; clustID = []; bigConsistency = struct('R',[],'wave',[],'channel',[]);
            D = []; maxAmpAll = []; spatialAllAll = [];
            for rr = 1:iters
                if type == 1
                    if strcmp(monkey, 'Malfoy')
                        sortDir = dir([loadDir '/*_g0/*imec' num2str(rr-1) '/kilosort4']);
                        if length(sortDir) <= 2
                            sortDir = dir([loadDir '/*_g0/*imec' num2str(rr-1) '/sorted']);
                        end
                    else
                        %sortDir = dir([loadDir '/*_g0/*imec' num2str(rr-1) '/kilosort2.0/sorter_output']);
                        sortDir = dir([loadDir '/*_g0/*imec' num2str(rr-1) '/kilosort4']);
                    end
                    sortDir = [sortDir(1).folder '/'];
                else
                    sortDir = [myoDirs(rr).folder '/' myoDirs(rr).name '/'];
                end

                if type == 1
                    % Load required files
                    T = readNPY([sortDir '/spike_times.npy']);                 % [nSpikes x 1]
                    I = readNPY([sortDir '/spike_clusters.npy']);              % [nSpikes x 1]
                    TT = readNPY([sortDir '/templates.npy']);                  % [nTemplates x nTimepoints x nChannels]
                    INV = readNPY([sortDir '/whitening_mat_inv.npy']);         % [nChannels x nChannels]
                    spikeTemplates = readNPY([sortDir '/spike_templates.npy']);% [nSpikes x 1]

                    % Read cluster labels
                    clusterGroup = tdfread([sortDir '/cluster_KSLabel.tsv']);

                    % Get template dimensions
                    [nTemplates, nTime, nChan] = size(TT);

                    % Precompute unwhitened template amplitudes
                    C_maxChannel = zeros(1, nTemplates);
                    C_maxAmp = zeros(1, nTemplates);
                    C_spatial = zeros(1, nTemplates);

                    for j = 1:nTemplates
                        template = reshape(TT(j, :, :), [nTime, nChan]);         % [T x C]
                        template_unwhitened = template * INV;                    % [T x C]
                        V = max(abs(template_unwhitened), [], 1);                % [1 x C]
                        V_uV = V * 0.195;                             % Convert to µV
                        [C_maxAmp(j), C_maxChannel(j)] = max(V_uV);

                        % Compute peak-to-peak amplitude on each channel
                        ptp = max(template_unwhitened) - min(template_unwhitened);                    % [1 x channels]


                        % Normalize ptp for computing spread
                        ptp_norm = ptp / sum(ptp);                      % energy distribution across channels

                        % Compute spatial spread: weighted std of channel indices
                        chan_inds = 1:length(ptp);                      % or use actual spatial positions if available
                        mu = sum(chan_inds .* ptp_norm);
                        C_spatial(j) = sqrt(sum(ptp_norm .* (chan_inds - mu).^2));
                    end

                    % Initialize outputs
                    keepInd = [];        % Cluster IDs
                    maxChannel = [];     % Max channel per cluster
                    maxAmp = [];         % Max amplitude (uV) per cluster
                    spatialAll = [];

                    % Get list of unique cluster IDs
                    clusterIDs = unique(I);

                    for k = 1:length(clusterIDs)
                        clusterID = clusterIDs(k);

                        % Check if it's labeled "good"
                        idxInLabel = find(clusterGroup.cluster_id == clusterID);
                        if isempty(idxInLabel) || ~strcmp(clusterGroup.KSLabel(idxInLabel, 1:3), 'goo')
                            continue
                        end

                        % Get spikes for this cluster
                        spikeIdx = find(I == clusterID);

                        % Get their templates
                        templatesForThisCluster = spikeTemplates(spikeIdx);

                        % Use the most common template
                        tmplIdx = mode(templatesForThisCluster) + 1;  % MATLAB indexing

                        % Save results
                        keepInd(end+1) = clusterID;
                        maxChannel(end+1) = C_maxChannel(tmplIdx);
                        maxAmp(end+1) = C_maxAmp(tmplIdx);
                        spatialAll(end+1) = C_spatial(tmplIdx);
                    end


                    if isempty(keepInd)
                        keepInd = 1;
                    end
                    I = uint32(I);
                    keepSpikes = find(ismember(I, keepInd));
                    I = I(keepSpikes);
                    T = T(keepSpikes);
                    C = keepInd;
                else
                    load([sortDir 'custom_merge/custom_merge'])
                end

                if doLFP
                    if strcmp(monkey, 'Pert')
                        sortDir2 = sortDir(1:end-12);
                    else
                        sortDir2 = sortDir(1:end-10);
                    end
                    ff = dir([sortDir2 '*lf.bin']);
                    f = fopen([sortDir2 ff(1).name], 'r');
                    LFP{rr} = fread(f, [385 Inf], '*int16');
                    LFP{rr} = LFP{rr}(1:384,:);
                    fclose(f);
                    disp('Loaded LFP')
                end

                disp(['Total good spikes: ' num2str(length(T))])

                bigT = cat(1, bigT, T);
                if isempty(bigI)
                    bigI = I;
                else
                    bigI = cat(1, bigI, I + max(bigI) + 1);
                end
                %bigConsistency.R = cat(3, bigConsistency.R, consistency.R);
                %bigConsistency.wave = cat(4, bigConsistency.wave, consistency.wave);
                %bigConsistency.channel = cat(2, bigConsistency.channel, consistency.channel);

                if type == 1
                    if iscell(brain_area_list{rr})
                        chanInd = 0;
                        brain_channel_list = config.Session.brain_channel_list{rr};
                        for j = 1:length(C)
                            for k = 1:length(brain_channel_list)
                                ir = brain_channel_list{k}{1} : brain_channel_list{k}{2};
                                if ismember(maxChannel(j), ir)
                                    brainID = cat(1, brainID, brain_area_list{rr}{k});
                                end
                            end
                        end
                    else
                        brainID = cat(1, brainID, repmat(brain_area_list{rr}, length(C), 1));
                    end
                    brainCoords = cat(1, brainCoords, repmat(cell2mat(brain_coord_list{rr}), length(C), 1));
                    maxAmpAll = cat(2, maxAmpAll, maxAmp);
                    spatialAllAll = cat(2, spatialAllAll, spatialAll);
                    %  D = cat(1, D, readNPY([sortDir '/alf/clusters.depths.npy']));
                else
                    muscleID = cat(1, muscleID, repmat(myo_muscle_list{rr}, length(C), 1));
                end
            end
            I = bigI;
            T = bigT;

            C = unique(I);
            SPK = cell(1,length(C));
            for i = 1:length(C)
                thisC = C(i);
                times = T(I == thisC);
                SPK{i} = double(times);
            end

            L = round(double(max(T)) / 30000);



            cutOffTrial(type) = length(neuralSync{type})+1;

            % in case we restricted spike sorting to a shorter range, let's use
            % last spike
            consRange = [neuralSync{type}(2) double(T(end))];
            % and let's make sure we don't use any trials beyond that limit
            temp = find(neuralSync{1} > double(T(end)),1) - 1;
            if ~isempty(temp)
                cutOffTrial(type) = temp;
            end

            chunk = 30; % seconds
            p = floor((consRange(2)-consRange(1)) / 30000 / chunk);
            bounds = round(linspace(consRange(1), consRange(2), p + 1));

            spkInside = zeros(length(SPK), p);
            for i = 1:length(SPK)
                for j = 1:p
                    temp = (SPK{i} >= bounds(j) & SPK{i} < bounds(j+1));
                    spkInside(i,j) = sum(temp) / (bounds(j+1) - bounds(j)) * 30000;
                end
            end

            stability{type} = var(spkInside, [], 2) ./ mean(spkInside,2);
            stabilityBig{type} = spkInside;

            allMaxAmp{type} = maxAmpAll;
            allSpatial{type} = spatialAllAll;

            if type == 1
                minimumRate = 0.1; % minimum spike rate in Hz
            else
                minimumRate = 0.0; % KEEP EM ALL
            end
            delInd = [];
            for i = 1:length(SPK)
                if length(SPK{i}) < (minimumRate * L) || stability{type}(i) > stability_threshold
                    delInd(end+1) = i;
                end
            end

            disp(['Have ' num2str(length(SPK))])
            disp(['Deleting ' num2str(length(delInd))])

            SPK(delInd) = [];
            spikeTimes{type} = SPK;
            allMaxAmp{type}(delInd) = [];
            allSpatial{type}(delInd) = [];
            stability{type}(delInd) = [];
            stabilityBig{type}(delInd,:) = [];
            if type == 1
                brainID(delInd) = [];
                brainCoords(delInd,:) = [];
                %            D(delInd) = [];
            else
                muscleID(delInd) = [];
            end
            figure(type)
            imagesc(stabilityBig{type})
            drawnow
        end
        cutOffTrial = min(cutOffTrial);

        sd = 20; % SD of kernel
        L = 601;
        s = (L-1) / (2*sd);
        g1 = gausswin(L,s);
        sd = 80; % SD of kernel
        s = (L-1) / (2*sd);
        g = gausswin(L,s);
        g(1:floor(L/2)) = g1(1:floor(L/2)); % make causal only
        R = g / sum(g);


        sd = 3; % SD of kernel
        L = 101;
        s = (L-1) / (2*sd);
        g = gausswin(L,s);
        R2 = g / sum(g);

        % let's get some trials
        numTrials = length(data.c3d);

        % Grab target positions for this session
        targPos = [data.c3d(1).TARGET_TABLE.X_GLOBAL([1 2 11]) data.c3d(1).TARGET_TABLE.Y_GLOBAL([1 2 11])];

        badInd = []; successInd = []; badEndpoint = []; all_block_row = zeros(1,numTrials); delayVel = []; movementTime = [];
        preMoveInd = [];
        for i = 1:numTrials
            pertTime = []; cueTime = []; rewardTime = [];
            for j = 1:length(data.c3d(i).EVENTS.LABELS)
                if strcmp(data.c3d(i).EVENTS.LABELS{j}(1:3), 'Cue')
                    cueTime = round(data.c3d(i).EVENTS.TIMES(j) * 1000) + 58;
                end
                if strcmp(data.c3d(i).EVENTS.LABELS{j}(1:4), 'Pert')
                    pertTime = round(data.c3d(i).EVENTS.TIMES(j) * 1000);
                end
                if strcmp(data.c3d(i).EVENTS.LABELS{j}(1:4), 'Rewa')
                    rewardTime = round(data.c3d(i).EVENTS.TIMES(j) * 1000);
                end
            end
            if ~isempty(pertTime) && ~isempty(cueTime)
                tRange = cueTime+100 : pertTime-50;
                delayVel(i) = max(sqrt(data_f.c3d(i).Right_HandXVel(tRange).^2 + data_f.c3d(i).Right_HandYVel(tRange).^2));
            else
                delayVel(i) = 0;
            end
            if ~isempty(pertTime) && ~isempty(rewardTime)
                tRange = pertTime : rewardTime;
                movementTime(i) = length(tRange);
            else
                movementTime(i) = 0;
            end
            if strcmp(monkey, 'Pert')
                delayThresh = 0.005;
            else
                delayThresh = 0.01;
            end
            if strcmp(data.c3d(i).EVENTS.LABELS{end}(1:5), 'Moved') ...
                    || length(data.c3d(i).EVENTS.TIMES) < 3 ...
                    || delayVel(i) > delayThresh || movementTime(i) > 1200
                badInd(end+1) = i;
            else
                successInd(end+1) = i;
            end
            if strcmp(data.c3d(i).EVENTS.LABELS{end}(1:5), 'Moved') ...
                    || delayVel(i) > delayThresh
                preMoveInd(end+1) = i;
            end
            all_block_row(i) = data.c3d(i).TRIAL.BLOCK_ROW;
        end
        successInd(successInd >= cutOffTrial) = [];
        disp(length(successInd)/numTrials)

        count = 1;
        position_in_block = zeros(1,length(all_block_row));
        for i = 2:length(all_block_row)
            position_in_block(i-1) = count;
            if all_block_row(i) - all_block_row(i-1) > 0
                count = 1;
            else
                count = count + 1;
            end
        end
        position_in_block(end) = count;


        latents = [];
        taskEvents = struct('target1On',[],'target1Off',[],'target2On',[],'target2Off',[],...
            'perturbation1On',[],'perturbation1Off',[],'perturbation2On',[],'perturbation2Off',[],'reward',[]);
        numBad = length(badInd);
        numSuccess = length(successInd)-1;
        nBack = 600;
        nForward = 300;
        tSample = 10;
        visualPresentationDelay = 58;
        trial = struct([]); handKin = cell(1,numSuccess); elbKin = cell(1,numSuccess); elbVel = cell(1,numSuccess);
        shoKin = cell(1,numSuccess); shoVel = cell(1,numSuccess);
        jointKin_Lyle = cell(numSuccess,3); handKin_Lyle = cell(numSuccess,3);
        jointTor = cell(1,numSuccess);
        spikes_s = cell(numSuccess,2);
        spikes_Lyle = cell(numSuccess,2);
        spikes_s_full = cell(numSuccess,2,2);
        spikes = cell(numSuccess,2);
        block_table = data.c3d(1).BLOCK_TABLE.TP_LIST;
        % LFP stuff
        lfp = cell(1,iters);
        allLFP = zeros(27500,32,numSuccess);
        pad_length = 7500;
        for i = 1:numSuccess
            if mod(i,200) == 0
                disp(num2str(i/numSuccess))
            end
            % find perturbation onset
            for j = 1:length(data.c3d(successInd(i)).EVENTS.LABELS)
                if strcmp(data.c3d(successInd(i)).EVENTS.LABELS{j}(1:4), 'Pert')
                    pertTime = round(data.c3d(successInd(i)).EVENTS.TIMES(j) * 1000);
                end
                if strcmp(data.c3d(successInd(i)).EVENTS.LABELS{j}(1:3), 'Cue')
                    probTime = round(data.c3d(successInd(i)).EVENTS.TIMES(j) * 1000) + visualPresentationDelay;
                end
                if strcmp(data.c3d(successInd(i)).EVENTS.LABELS{j}(1:3), 'Rew')
                    rewardTime = round(data.c3d(successInd(i)).EVENTS.TIMES(j) * 1000);
                end
            end
            TP = data.c3d(successInd(i)).TRIAL.TP;
            if TP <= 16
                trial(i).cond = TP;
            end

            tRangeNeural = probTime - nBack + 1 : rewardTime + nForward;
            if Lyle
                trial(i).probTime = nBack;
                trial(i).pertTime = pertTime - probTime + nBack;
                trial(i).rewardTime = rewardTime - probTime + nBack;
            else
                trial(i).probTime = floor((nBack / tSample) + 0.5);
                trial(i).pertTime = floor(((pertTime - probTime + nBack) / tSample) + 0.5);
                trial(i).rewardTime = floor(((rewardTime - probTime + nBack) / tSample) + 0.5);
                trial(i).moveTimeFull = (rewardTime - probTime + nBack) - (pertTime - probTime + nBack);
            end

            % get actual perturbation direction
            A = mod(TP, 2);
            if A == 1
                trial(i).pertDirection = 1;
            else
                trial(i).pertDirection = 2;
            end

            % get prob cond
            if TP <= 16
                A = mod(trial(i).cond, 8);
                if A == 1
                    trial(i).prob = 1;
                elseif A == 5 || A == 6
                    trial(i).prob = 2;
                elseif A == 3 || A == 4
                    trial(i).prob = 3;
                elseif A == 7 || A == 0
                    trial(i).prob = 4;
                elseif A == 2
                    trial(i).prob = 5;
                end
            end
            trial(i).AdaptationBlock = false;
            if TP > 16
                B = data.c3d(successInd(i)).TRIAL.BLOCK_ROW;
                BT = str2num(block_table{B});
                if sum(BT == 17) == sum(BT == 18)
                    trial(i).prob = 3;
                elseif sum(BT == 17) < sum(BT == 18)
                    trial(i).prob = 4;
                elseif sum(BT == 17) > sum(BT == 18)
                    trial(i).prob = 2;
                end
                if trial(i).prob == 2
                    trial(i).cond = 20 + trial(i).pertDirection;
                elseif trial(i).prob == 3
                    trial(i).cond = 18 + trial(i).pertDirection;
                elseif trial(i).prob == 4
                    trial(i).cond = 22 + trial(i).pertDirection;
                end
                trial(i).AdaptationBlock = true;
            end

            % get catch
            if (TP >= 9 && TP <= 16) || TP == 19
                trial(i).isCatch = true;
            else
                trial(i).isCatch = false;
            end

            if trial(i).isCatch && TP > 16
                trial(i).cond = 25;
            elseif trial(i).isCatch && TP <= 16
                trial(i).cond = 26; % !!!
            end

            % record block and position in block
            trial(i).block = all_block_row(successInd(i));
            trial(i).position_in_block = position_in_block(successInd(i));


            flanks = 300;
            tRange = tRangeNeural(1) - flanks : tRangeNeural(end) + flanks;
            tSteps = [1 tSample:tSample:length(tRangeNeural)];

            tempHandKin = [data_f.c3d(successInd(i)).Right_HandX(tRangeNeural) data_f.c3d(successInd(i)).Right_HandY(tRangeNeural)];
            tempElbKin = rad2deg(data_f.c3d(successInd(i)).Right_ElbAng(tRangeNeural));
            tempElbVel = rad2deg(data_f.c3d(successInd(i)).Right_ElbVel(tRangeNeural));
            tempShoKin = rad2deg(data_f.c3d(successInd(i)).Right_ShoAng(tRangeNeural));
            tempShoVel = rad2deg(data_f.c3d(successInd(i)).Right_ShoVel(tRangeNeural));
            %tempJointTor = [data_f.c3d(successInd(i)).Right_SHOTorIM(tRangeNeural), data_f.c3d(successInd(i)).Right_ELBTorIM(tRangeNeural)];

            handKin_Lyle{i,1} = [data.c3d(successInd(i)).Right_HandX, data.c3d(successInd(i)).Right_HandY] * 100;
            handKin_Lyle{i,2} = [data.c3d(successInd(i)).Right_HandXVel, data.c3d(successInd(i)).Right_HandYVel] * 100;
            handKin_Lyle{i,3} = [data.c3d(successInd(i)).Right_HandXAcc, data.c3d(successInd(i)).Right_HandYAcc] * 100;
            jointKin_Lyle{i,1} = rad2deg([data.c3d(successInd(i)).Right_ShoAng, data.c3d(successInd(i)).Right_ElbAng]);
            jointKin_Lyle{i,2} = rad2deg([data.c3d(successInd(i)).Right_ShoVel, data.c3d(successInd(i)).Right_ElbVel]);
            jointKin_Lyle{i,3} = rad2deg([data.c3d(successInd(i)).Right_ShoAcc, data.c3d(successInd(i)).Right_ElbAcc]);

            for t = 1:length(tSteps)-1
                handKin{i}(t,:) = mean(tempHandKin(tSteps(t):tSteps(t+1)-1,:),1);
                elbKin{i}(t,:) = mean(tempElbKin(tSteps(t):tSteps(t+1)-1),1);
                elbVel{i}(t,:) = mean(tempElbVel(tSteps(t):tSteps(t+1)-1),1);
                shoKin{i}(t,:) = mean(tempShoKin(tSteps(t):tSteps(t+1)-1),1);
                shoVel{i}(t,:) = mean(tempShoVel(tSteps(t):tSteps(t+1)-1),1);
                % jointTor{i}(t,:) = mean(tempJointTor(tSteps(t):tSteps(t+1)-1,:),1);
            end

            for type = typeList
                thisT = neuralSync{type}(successInd(i));
                if type == 100
                    if trial(i).targ == 1
                        taskEvents.target1On = cat(2, taskEvents.target1On, thisT + targTime*30);
                        % This is only correct for subject < 7!!!
                        taskEvents.target1Off = cat(2, taskEvents.target1Off, thisT + length(data.c3d(successInd(i)).Right_ShoAng)*30);
                    elseif trial(i).targ == 2
                        taskEvents.target2On = cat(2, taskEvents.target2On, thisT + targTime*30);
                        taskEvents.target2Off = cat(2, taskEvents.target2Off, thisT + length(data.c3d(successInd(i)).Right_ShoAng)*30);
                    end
                    if trial(i).isCatch == false
                        if trial(i).pertDirection == 1
                            taskEvents.perturbation1On = cat(2, taskEvents.perturbation1On, thisT + pertTime*30);
                            taskEvents.perturbation1Off = cat(2, taskEvents.perturbation1Off, thisT + length(data.c3d(successInd(i)).Right_ShoAng)*30);
                        elseif trial(i).pertDirection == 2
                            taskEvents.perturbation2On = cat(2, taskEvents.perturbation2On, thisT + pertTime*30);
                            taskEvents.perturbation2Off = cat(2, taskEvents.perturbation2Off, thisT + length(data.c3d(successInd(i)).Right_ShoAng)*30);
                        end
                    end
                    taskEvents.reward = cat(2, taskEvents.reward, thisT + rewardTime*30);

                end

                innerSpikes = zeros(length(tRange)-flanks*2,length(spikeTimes{type}), 'logical');
                innerSpikes_s_1 = zeros(length(tRange)-flanks*2,length(spikeTimes{type}));
                innerSpikes_s_2 = zeros(length(tRange)-flanks*2,length(spikeTimes{type}));
                for j = 1:length(spikeTimes{type})
                    theseTimes = round((spikeTimes{type}{j}(spikeTimes{type}{j} >= thisT & spikeTimes{type}{j} <= thisT + (tRange(end)*30 + 60000)) - thisT) / 30);
                    theseTimes = theseTimes - tRange(1);
                    theseTimes(theseTimes <= 0 | theseTimes > tRange(end)-tRange(1)) = [];
                    temp = zeros(1,length(tRange));
                    temp(theseTimes) = 1;
                    innerSpikes(:,j) = logical(temp(flanks+1 : end-flanks));
                    temp2 = conv(temp, R, 'same') * 1000;
                    innerSpikes_s_1(:,j) = temp2(flanks+1 : end-flanks);
                    temp2 = conv(temp, R2, 'same') * 1000;
                    innerSpikes_s_2(:,j) = temp2(flanks+1 : end-flanks);
                end
                for t = 1:length(tSteps)-1
                    spikes{i,type}(t,:) = sum(innerSpikes(tSteps(t):tSteps(t+1)-1,:),1);
                end
                spikes_Lyle{i,type} = innerSpikes;
                spikes_s{i,type} = innerSpikes_s_1(tSteps(1:end-1),:);
                endTime = pertTime - probTime + nBack + 400;
                if endTime > size(innerSpikes_s_1,1)
                    endTime = size(innerSpikes_s_1,1);
                end
                spikes_s_full{i,type,2} = single(innerSpikes_s_2(pertTime - probTime + nBack - 200 : endTime,:));
                spikes_s_full{i,type,1} = [];%single(innerSpikes_s_2(nBack - 200 : nBack + 500,:));

                if doLFP
                    thisT = round(neuralSync{1}(successInd(i)) / 30 * 2.5);
                    for j = 1:length(LFP)
                        tRangeLFP = thisT + tRangeNeural(1)*2.5 + 1 - pad_length: thisT + tRangeNeural(1)*2.5 + 5000*2.5 + pad_length;
                        temp_lfp = LFP{j}(:,round(tRangeLFP))';

                        temp_lfp = temp_lfp(:,1:12:384); % Just sample every 12th channel
                        allLFP(:,:,i) = temp_lfp;
                    end
                end
            end
        end
        clear LFP

        if doLFP
            % Define the FieldTrip data structure
            [numTime, numChan, numTrials] = size(allLFP);
            data = [];
            data.trial = cell(1, numTrials);
            data.time  = cell(1, numTrials);
            data.label = arrayfun(@(x) sprintf('chan%d', x), 1:numChan, 'UniformOutput', false);
            data.fsample = 2500;  % specify your actual sampling rate here

            timeVec = (0:numTime-1) / data.fsample;

            for i = 1:numTrials
                % Transpose each trial data to channels x time
                data.trial{i} = squeeze(allLFP(:, :, i))';
                data.time{i}  = timeVec;
            end

            % Bandpass filtering configuration
            cfg_bp = [];
            cfg_bp.demean = 'yes';
            cfg_bp.bpfilter = 'yes';
            cfg_bp.bpfreq = [1 400];
            cfg_bp.bpfiltord = 3;
            cfg_bp.feedback = 'no';

            % Apply the bandpass filter
            data_filtered = ft_preprocessing(cfg_bp, data);

            % Frequency analysis configuration
            cfg = [];
            cfg.method = 'mtmconvol';
            cfg.taper = 'hanning';
            cfg.output = 'pow';
            numFreqs = 50;  % Or whatever number of frequencies you prefer
            cfg.foi = round(logspace(log10(1), log10(400), numFreqs), 2);
            cfg.t_ftimwin = 5 ./ cfg.foi; % time windows adjusted automatically
            cfg.toi = (pad_length / data.fsample) : 0.01 : ((numTime - pad_length) / data.fsample);
            cfg.pad = 'maxperlen';
            cfg.keeptrials = 'yes';  % Maintain trial dimension
            cfg.feedback = 'no';

            % Perform frequency analysis
            freq = ft_freqanalysis(cfg, data_filtered);

            % Rearrange to time x frequency x channel x trial
            lfp = permute(freq.powspctrm, [4, 3, 2, 1]);
        end

        if Lyle
            neural = []; kin = [];
            neural.spikes = spikes_Lyle;
            neural.spikes_description = 'Each row is a TxN matrix for each trial containing 1s at spike times. Column 1 is cortical data and column 2 is muscle data if available';
            neural.brainID = brainID;
            neural.brainID_description = 'Brain area label for each neuron';
            neural.muscleID = muscleID;
            neural.muscleID_description = 'Muscle label for each motor unit';
            kin.handKin = handKin_Lyle;
            kin.handKin_description = 'Rows are trials, columns are hand (x,y) pos in cm, vel in cm/s, and acc in cm/s^2. Each matrix is Tx2 and is in 1ms bins';
            kin.jointKin = jointKin_Lyle;
            kin.jointKin_description = 'Rows are trials, columns are joint (sho, elb) angle in deg, vel in deg/s, and acc in deg/s^2. Each matrix is Tx2 and is in 1ms bins';
            for j = 1:length(trial)
                trial(j).description = ['cond = condition' newline 'probTime = when probability cue appears' newline, ...
                    'pertTime = when perturbation happens' newline 'rewardTime = when reward is given' newline, ...
                    'pertDirection = direction of perturbation condition' newline 'prob = probability condition (1-5)' newline, ...
                    'AdaptationBlock = are we in an adaptation block (no visual cue)' newline, ...
                    'isCatch = is this a catch trial? If yes, there is no perturbation' newline, ...
                    'block = which block of the task are we in' newline 'position_in_block = which trial in the block are we at' newline];
            end
            save([baseDir '/Lyle/ET2-Recording-' num2str(subject)],'neural','kin','trial','dataset')
        else
            % Remap conditions into the desired order
            condRemap = [7, 3, 5, 1, 6, 4, 8, 2];
            condRemap(9:16) = condRemap(1:8) + 8;
            condRemap(17:24) = condRemap(1:8) + 16;

            cmap = repmat([0.722193294000000,0.813952739000000,0.976574709000000;0.552953156000000,0.688929332000000,0.995375608000000;0.383013340000000,0.509419040000000,0.917387822000000;0.229805700000000,0.298717966000000,0.753683153000000;0.958852946000000,0.769767752000000,0.678007945000000;0.958003065000000,0.602842431000000,0.481775914000000;0.869186849000000,0.378313092000000,0.300267182000000;0.705673158000000,0.0155561600000000,0.150232812000000], [3 1]);

            numPreMove = length(preMoveInd);

            if Marco
                save([baseDir '/Marco/Recording-' num2str(subject)], '-v7.3', 'elbKin', 'elbVel', 'shoKin', 'shoVel', ...
                    'jointTor', 'handKin', 'trial', 'condRemap',...
                    'numSuccess', 'numTrials', 'cmap', 'targPos', 'date', 'spikes', 'spikes_s', 'stability', 'stabilityBig', ...
                    'brainCoords', 'brainID', 'muscleID','spikes_s_full', 'numPreMove', 'allMaxAmp', 'allSpatial', 'lfp', 'cfg')
            else
                save([baseDir '/Recording-' num2str(subject)], '-v7.3', 'elbKin', 'elbVel', 'shoKin', 'shoVel', ...
                    'jointTor', 'handKin', 'trial', 'condRemap',...
                    'numSuccess', 'numTrials', 'cmap', 'targPos', 'date', 'spikes', 'spikes_s', 'stability', 'stabilityBig', ...
                    'brainCoords', 'brainID', 'muscleID','spikes_s_full', 'numPreMove', 'allMaxAmp', 'allSpatial')
            end
        end
    end
end
