%% Minimal ET2 loader that also saves LFP (FieldTrip power)
% Saves ONLY: trial, condRemap, numSuccess, numTrials, cmap, targPos, date,
%             brainCoords, brainID, numPreMove, lfp

clc
clear

%% --- Paths (only what we strictly need) ---
addpath('yaml/')
addpath('/home/UWO/memanue5/Documents/MATLAB/fieldtrip-20250106/')
ft_defaults
addpath('/home/UWO/memanue5/Documents/GitHub/dataframe/util/')
addpath('ReadC3D_3.1.2/')
addpath('npy-matlab-master/npy-matlab/')
% addpath('/path/to/fieldtrip'); ft_defaults;   % <- ensure FieldTrip is on path!

%% --- User/session inputs ---
baseDir = '/cifs/pruszynski/';
expDir = 'Marco/SensoriMotorPrediction/';
pinfo = dload(fullfile(baseDir, expDir, 'participants.tsv'));
pinfo = getrow(pinfo, find(pinfo.good==1));
monkeys = unique(pinfo.Monkey);

%% --- Loop over monkeys ---
for m = 1:length(monkeys)
    monkey = monkeys{m};
    rows = getrow(pinfo, find(strcmp(pinfo.Monkey, monkey)));
    recordings = rows.RecordingID;

    %% --- Loop over sessions
    for num_recording = recordings'
        %% --- Get session ---
        row = getrow(rows, rows.RecordingID==num_recording);
        session = datestr(row.Session, 'mmddyy');
        loadDir = fullfile(baseDir, monkey, session); %'/local/scratch';     % contains .kinarm, config.yaml, kilosort dir, sync.mat, *lf.bin
        outDir = fullfile(baseDir, expDir, 'Recordings/', monkey);
        if ~isfolder(outDir)
            mkdir(outDir);
        end
        %% --- Load config (brain area / coords) ---
        config = yaml.loadFile([loadDir '/config.yaml']);
        brain_area_list  = config.Session.brain_area_list;
        brain_coord_list = config.Session.brain_coord_list;
        if isfield(config.Session,'brain_channel_list')
            brain_channel_list = config.Session.brain_channel_list;
        else
            brain_channel_list = [];
        end
        regions = split(row.Region, ',');

        %% --- Loop over regions ---
        for roi = 1:length(regions)
            region = regions{roi};
            num_elec = find(strcmp(region, cellstr(brain_area_list))) - 1;

            %% --- Check that directory contains lf.bin ---
            fn = fullfile(loadDir, sprintf('/%s_g0/%s_g0_imec%d/%s_g0_t0.imec%d.lf.bin',...
                session, session, num_elec, session, num_elec));
            if ~isfile(fn)
                fprintf('%s, recording #%d in %s not found\n', monkey, num_recording, region)
                continue
            end
            
            %% --- Load KINARM & basic session info ---
            kinarm_file = dir([loadDir '/*.kinarm']);
            if isempty(kinarm_file), error('No .kinarm in %s', loadDir); end
            data = zip_load([loadDir '/' kinarm_file(1).name]);
            
            date = datetime([data.c3d(1).EXPERIMENT.START_DATE_TIME(1:10) ' ' ...
                             data.c3d(1).EXPERIMENT.START_DATE_TIME(12:16)], ...
                            'InputFormat','yyyy-MM-dd HH:mm');
            
            % targPos = [data.c3d(1).TARGET_TABLE.X_GLOBAL([1 2 11]) ...
            %            data.c3d(1).TARGET_TABLE.Y_GLOBAL([1 2 11])];
            
            numTrials = numel(data.c3d);
            
            data = KINARM_add_hand_kinematics(data);						% Add hand velocity, acceleration and commanded forces to the data structure
            data = KINARM_add_sho_elb(data);
            data_f = filter_double_pass(data, 'enhanced', 'fc', 20);
            
            %% --- Simple success / pre-move flags + block positions ---
            visualPresentationDelay = 58;  % ms

            successInd  = [];
            preMoveInd  = [];
            badInd      = [];
            
            % Block rows and position_in_block (unchanged)
            all_block_row = zeros(1, numTrials);
            for i = 1:numTrials
                all_block_row(i) = data.c3d(i).TRIAL.BLOCK_ROW;
            end
            position_in_block = zeros(1, numTrials);
            cnt = 1;
            for i = 2:numTrials
                position_in_block(i-1) = cnt;
                if all_block_row(i) - all_block_row(i-1) > 0
                    cnt = 1;
                else
                    cnt = cnt + 1;
                end
            end
            position_in_block(end) = cnt;
            
            % Threshold for early movement (matches old code)
            if strcmp(monkey,'Pert')
                delayThresh = 0.005;
            else
                delayThresh = 0.01;
            end
            
            delayVel     = zeros(1, numTrials);   % peak hand speed during delay
            movementTime = zeros(1, numTrials);   % ms from Pert to Reward
            
            for i = 1:numTrials
                cueTime = []; pertTime = []; rewardTime = [];
            
                % Parse events in ms
                for j = 1:length(data.c3d(i).EVENTS.LABELS)
                    lab = data.c3d(i).EVENTS.LABELS{j};
                    if strncmp(lab,'Cue',3)
                        cueTime   = round(data.c3d(i).EVENTS.TIMES(j)*1000) + visualPresentationDelay;
                    elseif strncmp(lab,'Pert',4)
                        pertTime  = round(data.c3d(i).EVENTS.TIMES(j)*1000);
                    elseif strncmp(lab,'Rew',3) || strncmp(lab,'Rewa',4)
                        rewardTime= round(data.c3d(i).EVENTS.TIMES(j)*1000);
                    end
                end
            
                % Early-movement metric (max speed) between Cue+100 and Pert-50
                if ~isempty(pertTime) && ~isempty(cueTime)
                    tRange = (cueTime+100) : (pertTime-50);
                    if ~isempty(tRange)
                        % Use filtered velocities if you have data_f; otherwise raw
                        % if exist('data_f','var') && ~isempty(data_f)
                        %     vx = data_f.c3d(i).Right_HandXVel(tRange);
                        %     vy = data_f.c3d(i).Right_HandYVel(tRange);
                        % else
                        vx = data.c3d(i).Right_HandXVel(tRange);
                        vy = data.c3d(i).Right_HandYVel(tRange);
                        % end
                        delayVel(i) = max(sqrt(vx.^2 + vy.^2));
                    else
                        delayVel(i) = 0;
                    end
                else
                    delayVel(i) = 0;
                end
            
                % Movement duration (Pert -> Reward)
                if ~isempty(pertTime) && ~isempty(rewardTime)
                    movementTime(i) = rewardTime - pertTime;   % ms
                else
                    movementTime(i) = 0;
                end
            
                % Old-code conditions
                lastLabelIsMoved = strncmp(data.c3d(i).EVENTS.LABELS{end}, 'Moved', 5);
                tooFewEvents     = (length(data.c3d(i).EVENTS.TIMES) < 3);
                earlyMove        = (delayVel(i) > delayThresh);
                tooLong          = (movementTime(i) > 1200);
            
                if lastLabelIsMoved || tooFewEvents || earlyMove || tooLong
                    badInd(end+1) = i; %#ok<AGROW>
                else
                    hasPert = ~isempty(pertTime);
                    hasRew  = ~isempty(rewardTime);
                    if hasPert && hasRew
                        successInd(end+1) = i; %#ok<AGROW>
                    else
                        badInd(end+1) = i; %#ok<AGROW>
                    end
                end
            
                % Keep pre-move flag like old code
                if lastLabelIsMoved || earlyMove
                    preMoveInd(end+1) = i; %#ok<AGROW>
                end
            end
            
            numPreMove = numel(preMoveInd);
            numSuccess = numel(successInd);
            
            % --- Build MINIMAL "trial" struct with uniform fields ---
            nBack   = 600;   % ref samples
            tSample = 10;    % downsample factor like original
            
            % 1) Prototype with ALL fields (ensure same set every iteration)
            proto = struct( ...
                'cond',           NaN, ...
                'pertDirection',  NaN, ...
                'prob',           NaN, ...
                'isCatch',        false, ...
                'AdaptationBlock',false, ...
                'block',          NaN, ...
                'position_in_block', NaN, ...
                'probTime',       NaN, ...
                'pertTime',       NaN, ...
                'rewardTime',     NaN, ...
                'moveTimeFull',   NaN);
            
            % 2) Preallocate
            trial = repmat(proto, 1, numSuccess);
            
            % 3) Fill each element
            for ii = 1:numSuccess
                i = successInd(ii);
            
                % Parse events
                probTime = NaN; pertTime = NaN; rewardTime = NaN; TP = data.c3d(i).TRIAL.TP;
                for j = 1:length(data.c3d(i).EVENTS.LABELS)
                    lab = data.c3d(i).EVENTS.LABELS{j};
                    if strncmp(lab,'Cue',3)
                        probTime = round(data.c3d(i).EVENTS.TIMES(j)*1000) + visualPresentationDelay;
                    elseif strncmp(lab,'Pert',4)
                        pertTime = round(data.c3d(i).EVENTS.TIMES(j)*1000);
                    elseif strncmp(lab,'Rew',3)
                        rewardTime = round(data.c3d(i).EVENTS.TIMES(j)*1000);
                    end
                end
            
                t = proto; % start from uniform prototype
            
                % Condition & direction
                if TP <= 16, t.cond = TP; end
                if mod(TP,2)==1, t.pertDirection = 1; else, t.pertDirection = 2; end
            
                % Probability bin
                if TP <= 16
                    A = mod(t.cond,8);
                    if A==1, t.prob=1;
                    elseif A==5 || A==6, t.prob=2;
                    elseif A==3 || A==4, t.prob=3;
                    elseif A==7 || A==0, t.prob=4;
                    elseif A==2, t.prob=5; end
                else
                    t.AdaptationBlock = true;  % always present now (default false)
                end
            
                % Catch status (and cond override)
                t.isCatch = ((TP>=9 && TP<=16) || TP==19);
                if t.isCatch && TP > 16, t.cond = 25; elseif t.isCatch && TP <= 16, t.cond = 26; end
            
                % Block info
                t.block = all_block_row(i);
                t.position_in_block = position_in_block(i);
            
                % Times (sampled)
                t.probTime = floor((nBack / tSample) + 0.5);
                if ~isnan(pertTime) && ~isnan(probTime)
                    t.pertTime = floor(((pertTime - probTime + nBack) / tSample) + 0.5);
                end
                if ~isnan(rewardTime) && ~isnan(probTime)
                    t.rewardTime = floor(((rewardTime - probTime + nBack) / tSample) + 0.5);
                    t.moveTimeFull = (rewardTime - probTime + nBack) - (pertTime - probTime + nBack);
                end
            
                trial(ii) = t;  % safe: identical field set every time
            end
            
            clear t probTime pertTime rewardTime TP i ii j lab
            
            %% --- Restore sync -> neuralSync{1} minimally (for LFP alignment) ---
            neuralSync = cell(1,2);
            neuralSync{2} = []; % no myo
            sync_file = dir([loadDir sprintf('/%s_g0/%s_g0_imec%d/sync.mat', session, session, num_elec)]);
            if isempty(sync_file), error('No sync.mat in %s (needed for LFP alignment)', loadDir); end
            load([sync_file.folder '/' sync_file.name]); % expects variable "sync"
            trialSignal = sync; clear sync
            
            bitPoints = fliplr(1325:300:6000);
            ii = find(diff(trialSignal) > 0.2) + 1;
            trialStart = []; trialNum = {};
            for i = 1:length(ii)
                dd = sum(abs(trialSignal(ii(i)-8000 : ii(i)-1)) > 0.2);
                if dd == 0
                    if ii(i) + bitPoints < length(trialSignal)
                        trialStart(end+1) = ii(i); %#ok<AGROW>
                        temp = trialSignal(ii(i) + bitPoints);
                        % trialNum{1}(end+1) = bin2dec(sprintf('%d',temp)); %#ok<AGROW>
                    end
                end
            end
            neuralSync{1} = trialStart;
            clear trialSignal ii dd temp bitPoints trialStart trialNum

            %% === Add spikes, spikes_s, kinematics, stability (memory-efficient) ===

            % ---------- PARAMETERS ----------
            stability_threshold = 2;   % same as your original
            minimumRate         = 0.1; % Hz for neural units
            % nBack   = 600;             % ms (matches your trial building)
            nForward= 300;             % ms
            % tSample = 10;              % ms binning for kinematics & spikes
            % visualPresentationDelay = 58; % ms (already used above)

            % ---------- GAUSSIAN KERNELS for spikes_s ----------
            % Wide kernel (causal-ish, like original R)
            sd = 20; L = 601; s = (L-1)/(2*sd);
            g1 = gausswin(L,s);
            sd = 80; L = 601; s = (L-1)/(2*sd);
            g  = gausswin(L,s);
            g(1:floor(L/2)) = g1(1:floor(L/2));
            R  = g/sum(g);                 % causal-ish, ~80ms SD tail
            % Narrow kernel (R2)
            sd = 3; L = 101; s = (L-1)/(2*sd);
            g  = gausswin(L,s);
            R2 = g/sum(g);
            
            % ---------- Locate Kilosort directory for this imec ----------
            ks_dir = fullfile(fileparts(fn), 'kilosort4');  % e.g. .../imec0/kilosort4
            if ~isfolder(ks_dir)
                ks_dir = fullfile(fileparts(fn), 'sorted'); % fallback
            end
            if ~isfolder(ks_dir)
                error('No kilosort dir found for %s', fn);
            end
            
            % ---------- Load basic Kilosort outputs ----------
            T  = readNPY(fullfile(ks_dir,'spike_times.npy'));        % int64 sample idx @30 kHz
            I  = readNPY(fullfile(ks_dir,'spike_clusters.npy'));     % cluster id per spike
            st = readNPY(fullfile(ks_dir,'spike_templates.npy'));    % template id per spike
            labT = tdfread(fullfile(ks_dir,'cluster_KSLabel.tsv'));  % has cluster_id & KSLabel
            
            % Keep only "good" clusters
            clusterIDs = unique(I);
            goodMask = false(size(clusterIDs));
            for k = 1:numel(clusterIDs)
                cid = clusterIDs(k);
                idx = find(labT.cluster_id == cid, 1);
                if ~isempty(idx) && strncmp(labT.KSLabel(idx,1:3), 'goo', 3)
                    goodMask(k) = true;
                end
            end
            goodClusters = clusterIDs(goodMask);
            
            % Subselect spikes from good clusters
            keep = ismember(I, goodClusters);
            I = I(keep); T = T(keep); st = st(keep);
            
            % Build SPK cell {unit} of spike times (double samples @30 kHz)
            C = unique(I);
            SPK = cell(1, numel(C));
            for iiU = 1:numel(C)
                c = C(iiU);
                SPK{iiU} = double(T(I==c));     % 30 kHz sample indices
            end
            
            % ---------- Stability computation (same logic, light memory) ----------
            if isempty(neuralSync{1})
                warning('No neuralSync{1}; skipping stability filter.');
                stability = [];
            else
                lastSpike = double(max(T));
                consRange = [neuralSync{1}(min(2, numel(neuralSync{1}))) lastSpike];  % start near 2nd trial
                chunkSec  = 30;                                     % seconds
                fs30      = 30000;
                p = floor((consRange(2)-consRange(1)) / fs30 / chunkSec);
                if p < 1, p = 1; end
                bounds = round(linspace(consRange(1), consRange(2), p+1));
            
                spkInside = zeros(numel(SPK), p);
                for iiU = 1:numel(SPK)
                    t = SPK{iiU};
                    for j = 1:p
                        sel = (t >= bounds(j) & t < bounds(j+1));
                        spkInside(iiU,j) = sum(sel) / ((bounds(j+1)-bounds(j))/fs30);
                    end
                end
                stability = var(spkInside, [], 2) ./ max(mean(spkInside,2), eps);
            end
            
            % Filter units by min rate & stability
            Lrec_sec = double(max(T))/30000;
            delIdx = false(1, numel(SPK));
            for iiU = 1:numel(SPK)
                rate = numel(SPK{iiU}) / max(Lrec_sec, eps);
                if rate < minimumRate || (~isempty(stability) && stability(iiU) > stability_threshold)
                    delIdx(iiU) = true;
                end
            end
            SPK(delIdx) = [];
            if ~isempty(stability), stability(delIdx) = []; end
            C = C(~delIdx);                 %#ok<NASGU>  % (kept only for reference)
            
            % ---------- Per-trial spikes & spikes_s (10 ms bins) ----------
            spikes   = cell(numSuccess, 2);   % {trial, type=1 neural ; 2 myo(not used)}
            spikes_s = cell(numSuccess, 2);   % smoothed (using R on 1ms grid then downsample)
            type = 1; % only neural here
            
            for iiTr = 1:numSuccess
                trIdx  = successInd(iiTr);
                % trial window in ms relative to probTime
                probTime = NaN; pertTime = NaN; rewardTime = NaN;
                for j = 1:length(data.c3d(trIdx).EVENTS.LABELS)
                    lab = data.c3d(trIdx).EVENTS.LABELS{j};
                    if strncmp(lab,'Cue',3)
                        probTime  = round(data.c3d(trIdx).EVENTS.TIMES(j)*1000) + visualPresentationDelay;
                    elseif strncmp(lab,'Pert',4)
                        pertTime  = round(data.c3d(trIdx).EVENTS.TIMES(j)*1000);
                    elseif strncmp(lab,'Rew',3)
                        rewardTime= round(data.c3d(trIdx).EVENTS.TIMES(j)*1000);
                    end
                end
                if isnan(probTime) || isnan(rewardTime) || isempty(neuralSync{1})
                    spikes{iiTr,type}   = [];
                    spikes_s{iiTr,type} = [];
                    continue
                end
            
                tRangeNeural = (probTime - nBack + 1) : (rewardTime + nForward); % ms
                flanks = 300;                                                    % ms for conv edges
                tRange = (tRangeNeural(1) - flanks) : (tRangeNeural(end) + flanks);
                tSteps = [1 tSample:tSample:length(tRangeNeural)];
            
                % Build binary spike trains on 1 ms grid (logical T x Nunits)
                innerSpikes   = false(length(tRange)-2*flanks, numel(SPK));
                innerSpikes_s = zeros(length(tRange)-2*flanks, numel(SPK), 'single');
            
                % anchor this trial in 30 kHz samples
                thisT_30k = neuralSync{1}(trIdx);
            
                for u = 1:numel(SPK)
                    % spike times in ms relative to trial anchor
                    tt_ms = round((SPK{u}(SPK{u} >= thisT_30k & SPK{u} <= thisT_30k + (tRange(end)*30 + 60000)) - thisT_30k) / 30);
                    tt_ms = tt_ms - tRange(1);
                    tt_ms(tt_ms <= 0 | tt_ms > (tRange(end)-tRange(1)+1)) = [];
                    v = zeros(1, length(tRange), 'single');
                    v(tt_ms) = 1;
            
                    % trim flanks
                    innerSpikes(:,u) = v(flanks+1:end-flanks) > 0;
            
                    % smoothed (Hz): conv on 1ms grid, then trim
                    vv = conv(v, R, 'same') * 1000;    % Hz
                    innerSpikes_s(:,u) = single(vv(flanks+1:end-flanks));
                end
            
                % Downsample to 10ms bins (counts) and 10ms samples (smoothed)
                sp = zeros(length(tSteps)-1, numel(SPK), 'single');
                sps= zeros(length(tSteps)-1, numel(SPK), 'single');
                for k = 1:length(tSteps)-1
                    seg = tSteps(k):tSteps(k+1)-1;
                    sp(k,:)  = sum(innerSpikes(seg,:),1,'native');
                    sps(k,:) = mean(innerSpikes_s(seg,:),1,'native');
                end
                spikes{iiTr,type}   = sp;
                spikes_s{iiTr,type} = sps;
            end
            
            % ---------- Kinematics (10 ms bins) ----------
            elbKin = cell(1,numSuccess); elbVel = cell(1,numSuccess);
            shoKin = cell(1,numSuccess); shoVel = cell(1,numSuccess);
            handKin= cell(1,numSuccess);
            jointTor = cell(1,numSuccess);   % keep as empty [] unless you compute torques
            
            for iiTr = 1:numSuccess
                trIdx  = successInd(iiTr);
                % get times again
                probTime = NaN; pertTime=NaN; rewardTime=NaN;
                for j = 1:length(data.c3d(trIdx).EVENTS.LABELS)
                    lab = data.c3d(trIdx).EVENTS.LABELS{j};
                    if strncmp(lab,'Cue',3)
                        probTime  = round(data.c3d(trIdx).EVENTS.TIMES(j)*1000) + visualPresentationDelay;
                    elseif strncmp(lab,'Pert',4)
                        pertTime  = round(data.c3d(trIdx).EVENTS.TIMES(j)*1000);
                    elseif strncmp(lab,'Rew',3)
                        rewardTime= round(data.c3d(trIdx).EVENTS.TIMES(j)*1000);
                    end
                end
                if isnan(probTime) || isnan(rewardTime)
                    elbKin{iiTr}=[]; elbVel{iiTr}=[]; shoKin{iiTr}=[]; shoVel{iiTr}=[]; handKin{iiTr}=[];
                    continue
                end
            
                tRangeNeural = (probTime - nBack + 1) : (rewardTime + nForward);
                tSteps = [1 tSample:tSample:length(tRangeNeural)];
            
                % pull from filtered KINARM signals (rad->deg; pos in whatever units you used)
                tmpElbKin = rad2deg(data.c3d(trIdx).Right_ElbAng(tRangeNeural));
                tmpElbVel = rad2deg(data.c3d(trIdx).Right_ElbVel(tRangeNeural));
                tmpShoKin = rad2deg(data.c3d(trIdx).Right_ShoAng(tRangeNeural));
                tmpShoVel = rad2deg(data.c3d(trIdx).Right_ShoVel(tRangeNeural));
            
                tmpHand   = [data.c3d(trIdx).Right_HandX(tRangeNeural), ...
                             data.c3d(trIdx).Right_HandY(tRangeNeural)];
            
                eK = zeros(length(tSteps)-1,1,'single');
                eV = zeros(length(tSteps)-1,1,'single');
                sK = zeros(length(tSteps)-1,1,'single');
                sV = zeros(length(tSteps)-1,1,'single');
                hK = zeros(length(tSteps)-1,2,'single');
            
                for k = 1:length(tSteps)-1
                    seg = tSteps(k):tSteps(k+1)-1;
                    eK(k)   = mean(tmpElbKin(seg));
                    eV(k)   = mean(tmpElbVel(seg));
                    sK(k)   = mean(tmpShoKin(seg));
                    sV(k)   = mean(tmpShoVel(seg));
                    hK(k,:) = mean(tmpHand(seg,:),1);
                end
                elbKin{iiTr} = eK; elbVel{iiTr} = eV;
                shoKin{iiTr} = sK; shoVel{iiTr} = sV;
                handKin{iiTr}= hK;
                jointTor{iiTr}= [];   % fill if you compute torques later
            end

            %% --- Extra trial selection criterium
            keepTrial = true(1, numel(successInd));   % start with all true

            if ~isempty(neuralSync{1}) && ~isempty(T)
                lastSpike30k = double(T(end));        % last spike timestamp in samples (30 kHz)
            
                % Find first trial whose sync starts AFTER the last spike
                ix = find(neuralSync{1} > lastSpike30k, 1);
            
                if ~isempty(ix)
                    cutoff = ix - 1;                  % last trial fully within spike window
                    keepTrial(successInd >= cutoff) = false;
                end
            end
            
            %% --- Low-memory LFP -> time–freq power (batched, streaming to disk) ---
            % Writes lfp to the same output .mat on disk; you will -append the other vars later.
            
            % ----------------- PARAMETERS YOU CAN TUNE -----------------
            pad_length   = 7500;   % LFP samples (2.5 kHz); shrink if memory is tight
            post_ms      = 5000;   % ms after probTime to include
            chan_step    = 12;     % keep every 12th channel -> 32 chans
            BATCH_TRIALS = 12;     % trials per FieldTrip batch
            numFreqs     = 40;     % fewer freqs = less memory/CPU
            toi_step_s   = 0.01;   % time step (sec) for TFR
            bp_band      = [1 200];% bandpass for FT preprocessing
            % -----------------------------------------------------------
            
            lfp = [];  % keep symbol in workspace; real data lives on disk
            info = dir(fn);
            bytesPerInt16 = 2;
            nChRaw = 385;
            nSamp  = info.bytes / (bytesPerInt16 * nChRaw);
            if abs(nSamp - round(nSamp)) > 1e-9
                error('Unexpected *lf.bin size: not divisible by 385*int16.');
            end
            nSamp = round(nSamp);
        
            mm = memmapfile(fn, ...
                "Format", {"int16", [nChRaw nSamp], "x"}, ...
                "Writable", false);
        
            % Channel selection (drop sync channel 385, decimate)
            keepChFull = 1:384;
            keepCh     = keepChFull(1:chan_step:end);  % e.g., 32 chans
            C          = numel(keepCh);
            fs         = 2500;                         % Hz
            % brainID    = unique(brainID);
        
            % ---------- Build per-trial sample windows (no big arrays) ----------
            nBack    = 600;                % must match earlier
            visualPresentationDelay = 58;  % ms
        
            trialRanges = cell(1, numSuccess); % [startSample endSample] in LFP samples
            maxT = 0;                          % longest raw epoch length in samples
        
            for ii = 1:numSuccess
                i = successInd(ii);
                probTime = NaN; rewardTime = NaN;
                for j = 1:length(data.c3d(i).EVENTS.LABELS)
                    lab = data.c3d(i).EVENTS.LABELS{j};
                    if strncmp(lab,'Cue',3)
                        probTime  = round(data.c3d(i).EVENTS.TIMES(j)*1000) + visualPresentationDelay;
                    elseif strncmp(lab,'Rew',3)
                        rewardTime = round(data.c3d(i).EVENTS.TIMES(j)*1000);
                    end
                end
                if isnan(probTime) || isempty(neuralSync{1})
                    trialRanges{ii} = [];
                    continue
                end
        
                % ms window (fixed length: pre nBack, post post_ms)
                tRange_ms = (probTime - nBack + 1) : (probTime + post_ms);
        
                % Convert sync (30 kHz) -> LFP samples (2.5 kHz)
                thisT = round(neuralSync{1}(successInd(ii)) / 30 * 2.5);
        
                s0 = thisT + round(tRange_ms(1)*2.5) + 1 - pad_length; % inclusive
                s1 = thisT + round(tRange_ms(end)*2.5)      + pad_length;
        
                % clamp to file
                s0 = max(1, s0);
                s1 = min(nSamp, s1);
                if s1 <= s0
                    trialRanges{ii} = [];
                else
                    trialRanges{ii} = [s0 s1];
                    maxT = max(maxT, s1 - s0 + 1);
                end
            end
        
            %% ---------- Prepare output .mat and batching ----------
            outPath = fullfile(outDir, sprintf('recording.%s-%d.mat',region, num_recording));
            mf = matfile(outPath, 'Writable', true);
            
            Tr = numSuccess;
            first_batch = true;
            Ttf = [];  % time bins of lfp after FT (filled on first batch)
            Fb  = [];  % number of freqs used (from cfg.foi)
        
            for startIdx = 1:BATCH_TRIALS:Tr
                endIdx   = min(Tr, startIdx + BATCH_TRIALS - 1);
                batchIdx = startIdx:endIdx;
                nb       = numel(batchIdx);
        
                % ---- Build FieldTrip structure for this batch from memmap ----
                ftdata = [];
                ftdata.trial   = cell(1, nb);
                ftdata.time    = cell(1, nb);
                ftdata.label   = arrayfun(@(x) sprintf('chan%d', x), 1:C, 'UniformOutput', false);
                ftdata.fsample = fs;
        
                for b = 1:nb
                    ii = batchIdx(b);
                    rngs = trialRanges{ii};
                    if isempty(rngs)
                        seg = zeros(C, maxT, 'single');        % empty trial -> zeros
                    else
                        s0 = rngs(1); s1 = rngs(2);
                        seg = single(mm.Data.x(keepCh, s0:s1)); % C x T
                        % right-pad/truncate to maxT for uniformity
                        if size(seg,2) < maxT
                            seg(:, end+1:maxT) = 0;
                        elseif size(seg,2) > maxT
                            seg = seg(:,1:maxT);
                        end
                    end
                    ftdata.trial{b} = seg;            % C x T
                    ftdata.time{b}  = (0:maxT-1)/fs;  % 1 x T (seconds)
                end
        
                % ---- Bandpass ----
                cfg_bp = [];
                cfg_bp.demean    = 'yes';
                cfg_bp.bpfilter  = 'yes';
                cfg_bp.bpfreq    = bp_band;
                cfg_bp.bpfiltord = 3;
                cfg_bp.feedback  = 'no';
                data_filt = ft_preprocessing(cfg_bp, ftdata);
        
                % ---- Time–frequency ----
                cfg = [];
                cfg.method     = 'mtmconvol';
                cfg.taper      = 'hanning';
                cfg.output     = 'pow';
                cfg.foi        = round(logspace(log10(max(2,bp_band(1))), log10(bp_band(2)), numFreqs), 2);
                cfg.t_ftimwin  = 5 ./ cfg.foi;
                cfg.toi        = (pad_length / fs) : toi_step_s : ((maxT - pad_length) / fs);
                cfg.pad        = 'maxperlen';
                cfg.keeptrials = 'yes';
                cfg.feedback   = 'no';
                freq = ft_freqanalysis(cfg, data_filt);   % trials x chan x freq x time
        
                % ---- Determine dims on first batch and preallocate on disk ----
                [tr_b, Cb, Fb_this, Tb] = size(freq.powspctrm);   % batch dims
                if first_batch
                    if Cb ~= C
                        error('Channel mismatch: FieldTrip=%d, expected=%d', Cb, C);
                    end
                    Fb  = Fb_this;
                    Ttf = Tb;
                    mf.lfp = zeros(Ttf, C, Fb, Tr, 'single');  % preallocate with REAL dims
                    first_batch = false;
                else
                    % If later batches differ in time bins, we’ll trim/pad below
                    if Cb ~= C || Fb_this ~= Fb
                        error('C/F mismatch across batches (Cb=%d,Fb=%d vs C=%d,Fb=%d).', Cb, Fb_this, C, Fb);
                    end
                end
        
                % ---- Rearrange to [time x chan x freq x trial] ----
                batch_lfp = single(permute(freq.powspctrm, [4 2 3 1])); % Tb x C x Fb x tr_b
        
                % Align time dimension (trim/pad to Ttf)
                if size(batch_lfp,1) > Ttf
                    batch_lfp = batch_lfp(1:Ttf,:,:,:);
                elseif size(batch_lfp,1) < Ttf
                    padT = Ttf - size(batch_lfp,1);
                    batch_lfp = cat(1, batch_lfp, zeros(padT, C, Fb, tr_b, 'single'));
                end
        
                % Safety check for trial count
                if tr_b ~= nb
                    error('Batch trial count mismatch: got %d, expected %d.', tr_b, nb);
                end
        
                % ---- Stream this batch to disk ----
                mf.lfp(:, :, :, batchIdx) = batch_lfp;
        
                % Cleanup batch
                clear ftdata data_filt cfg_bp freq batch_lfp
            end
            
            mf.trial       = trial;
            mf.numSuccess  = numSuccess;
            mf.numTrials   = numTrials;
            mf.shoKin      = shoKin;
            mf.elbKin      = elbKin;
            mf.shoVel      = shoVel;
            mf.elbVel      = elbVel;
            mf.spike       = spikes;
            mf.spike_s     = spikes_s;
            % mf.targPos     = targPos;
            mf.date        = date;
            mf.brainID     = brain_area_list{num_elec + 1};
            mf.numPreMove  = numPreMove;
            mf.cfg = cfg;
            
            % end
        end
    end
end


% %% --- Save EXACTLY the requested variables (plus lfp) ---
% outPath = fullfile(outDir, ['Recording-' num2str(num_recording) '.mat']);
% save(outPath, '-v7.3', ...
%      'trial','numSuccess','numTrials','targPos', 'cfg',...
%      'date','brainCoords','brainID','numPreMove','-append');
