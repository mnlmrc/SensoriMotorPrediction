%% Minimal ET2 loader that also saves LFP (FieldTrip power)
% Saves ONLY: trial, condRemap, numSuccess, numTrials, cmap, targPos, date,
%             brainCoords, brainID, numPreMove, lfp

clc
clear

%% --- Paths (only what we strictly need) ---
addpath('yaml/')
addpath('/home/UWO/memanue5/Documents/MATLAB/fieldtrip-20250106/')
addpath('ReadC3D_3.1.2/')
addpath('npy-matlab-master/npy-matlab/')
% addpath('/path/to/fieldtrip'); ft_defaults;   % <- ensure FieldTrip is on path!

%% --- User/session inputs ---
loadDir = '/local/scratch';     % contains .kinarm, config.yaml, kilosort dir, sync.mat, *lf.bin
baseDir = loadDir;              % change if you want a different output dir
subject = 1;

%% --- Load config (brain area / coords) ---
config = yaml.loadFile([loadDir '/config.yaml']);
brain_area_list  = config.Session.brain_area_list;
brain_coord_list = config.Session.brain_coord_list;
if isfield(config.Session,'brain_channel_list')
    brain_channel_list = config.Session.brain_channel_list;
else
    brain_channel_list = [];
end

%% --- Load KINARM & basic session info ---
kinarm_file = dir([loadDir '/*.kinarm']);
if isempty(kinarm_file), error('No .kinarm in %s', loadDir); end
data = zip_load([loadDir '/' kinarm_file(1).name]);

date = datetime([data.c3d(1).EXPERIMENT.START_DATE_TIME(1:10) ' ' ...
                 data.c3d(1).EXPERIMENT.START_DATE_TIME(12:16)], ...
                'InputFormat','yyyy-MM-dd HH:mm');

targPos = [data.c3d(1).TARGET_TABLE.X_GLOBAL([1 2 11]) ...
           data.c3d(1).TARGET_TABLE.Y_GLOBAL([1 2 11])];

numTrials = numel(data.c3d);

%% --- Simple success / pre-move flags + block positions ---
visualPresentationDelay = 58;  % ms
successInd  = [];
preMoveInd  = [];
all_block_row = zeros(1, numTrials);
for i = 1:numTrials, all_block_row(i) = data.c3d(i).TRIAL.BLOCK_ROW; end
position_in_block = zeros(1, numTrials);
cnt = 1;
for i = 2:numTrials
    position_in_block(i-1) = cnt;
    if all_block_row(i) - all_block_row(i-1) > 0, cnt = 1; else, cnt = cnt + 1; end
end
position_in_block(end) = cnt;

for i = 1:numTrials
    hasPert = false; hasRew = false; endMoved = false;
    for j = 1:length(data.c3d(i).EVENTS.LABELS)
        lab = data.c3d(i).EVENTS.LABELS{j};
        if strncmp(lab,'Pert',4), hasPert = true; end
        if strncmp(lab,'Rew', 3), hasRew  = true; end
        if j == length(data.c3d(i).EVENTS.LABELS) && strncmp(lab,'Moved',5), endMoved = true; end
    end
    if hasPert && hasRew && ~endMoved, successInd(end+1) = i; end %#ok<AGROW>
    if endMoved, preMoveInd(end+1) = i; end %#ok<AGROW>
end
numPreMove = numel(preMoveInd);
numSuccess = numel(successInd);

%% --- Build MINIMAL "trial" struct with uniform fields ---
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

%% --- Minimal brain area / coords from Kilosort (kept light) ---
brainID = []; brainCoords = [];
ks_dir = [];
cand = dir([loadDir '/kilosort4']);
if ~isempty(cand), ks_dir = cand(1).folder; end
if isempty(ks_dir)
    cand = dir([loadDir '/sorted']);
    if ~isempty(cand), ks_dir = cand(1).folder; end
end

if ~isempty(ks_dir)
    T  = readNPY([ks_dir '/spike_times.npy']);
    I  = readNPY([ks_dir '/spike_clusters.npy']);
    TT = readNPY([ks_dir '/templates.npy']);
    INV= readNPY([ks_dir '/whitening_mat_inv.npy']);
    st = readNPY([ks_dir '/spike_templates.npy']);
    grp= tdfread([ks_dir '/cluster_KSLabel.tsv']);

    [nTemplates, ~, ~] = size(TT);
    C_maxChannel = zeros(1,nTemplates);
    for j = 1:nTemplates
        template = reshape(TT(j,:,:), [], size(INV,1));
        template_unw = template * INV;
        V = max(abs(template_unw), [], 1);
        [~, C_maxChannel(j)] = max(V);
    end
    keep = false(size(I));
    uniqC = unique(I);
    for k = 1:numel(uniqC)
        cid = uniqC(k);
        idxLab = find(grp.cluster_id == cid, 1);
        if ~isempty(idxLab) && strncmp(grp.KSLabel(idxLab,1:3),'goo',3)
            keep = keep | (I == cid);
        end
    end
    I  = I(keep);
    st = st(keep);

    goodClusters = unique(I);
    maxChanPerCluster = zeros(numel(goodClusters),1);
    for k = 1:numel(goodClusters)
        cid = goodClusters(k);
        tmplIdx = mode(st(I == cid)) + 1;
        maxChanPerCluster(k) = C_maxChannel(tmplIdx);
    end

    for rr = 1:max(1, numel(brain_area_list))
        if iscell(brain_area_list{rr})
            this_ranges = brain_channel_list{rr};
            for k = 1:numel(maxChanPerCluster)
                ch = maxChanPerCluster(k);
                match = [];
                for r = 1:numel(this_ranges)
                    ir = this_ranges{r}{1} : this_ranges{r}{2};
                    if ismember(ch, ir), match = r; break; end
                end
                if ~isempty(match)
                    brainID     = [brainID; brain_area_list{rr}{match}]; %#ok<AGROW>
                    brainCoords = [brainCoords; cell2mat(brain_coord_list{rr})]; %#ok<AGROW>
                end
            end
        else
            brainID     = [brainID; repmat(brain_area_list{rr}, numel(maxChanPerCluster), 1)]; %#ok<AGROW>
            brainCoords = [brainCoords; repmat(cell2mat(brain_coord_list{rr}), numel(maxChanPerCluster), 1)]; %#ok<AGROW>
        end
    end
    clear T I TT INV st grp C_maxChannel maxChanPerCluster uniqC cid tmplIdx template template_unw V
end

%% --- Restore sync -> neuralSync{1} minimally (for LFP alignment) ---
neuralSync = cell(1,2);
neuralSync{2} = []; % no myo
sync_file = dir([loadDir '/sync.mat']);
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

%% --- Low-memory LFP -> time–freq power (batched, streaming to disk) ---
% Writes lfp to the same output .mat on disk; you will -append the other vars later.

% ----------------- PARAMETERS YOU CAN TUNE -----------------
pad_length   = 7500;   % LFP samples (2.5 kHz); shrink if memory is tight
post_ms      = 5000;   % ms after probTime to include
chan_step    = 12;     % keep every 12th channel -> 32 chans
BATCH_TRIALS = 12;     % trials per FieldTrip batch
numFreqs     = 40;     % fewer freqs = less memory/CPU
toi_step_s   = 0.02;   % time step (sec) for TFR
bp_band      = [1 200];% bandpass for FT preprocessing
% -----------------------------------------------------------

lfp = [];  % keep symbol in workspace; real data lives on disk

% Find *lf.bin from ks_dir; fall back one level up if needed
lfp_parent = fileparts(ks_dir);
if isempty(lfp_parent), lfp_parent = loadDir; end
ff = dir(fullfile(lfp_parent, '*lf.bin'));
if isempty(ff), ff = dir(fullfile(fileparts(lfp_parent), '*lf.bin')); end

if isempty(ff)
    warning('No *lf.bin found; lfp will be empty.');
else
    % ---------- Memory-map raw file (no full fread) ----------
    fn = fullfile(ff(1).folder, ff(1).name);
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

    % ---------- Prepare output .mat and batching ----------
    haveFT = exist('ft_preprocessing','file') == 2 && exist('ft_freqanalysis','file') == 2;
    outPath = fullfile(baseDir, ['Recording-' num2str(subject) '.mat']);
    mf = matfile(outPath, 'Writable', true);

    Tr = numSuccess;
    first_batch = true;
    Ttf = [];  % time bins of lfp after FT (filled on first batch)
    Fb  = [];  % number of freqs used (from cfg.foi)

    if haveFT
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
            clear ftdata data_filt cfg_bp cfg freq batch_lfp
        end
    else
        warning('FieldTrip not on path; leaving lfp empty. Add FT or ask for a pure-MATLAB STFT version.');
    end
end


%% --- Save EXACTLY the requested variables (plus lfp) ---
outPath = fullfile(baseDir, ['Recording-' num2str(subject) '.mat']);
save(outPath, '-v7.3', ...
     'trial','condRemap','numSuccess','numTrials','cmap','targPos', ...
     'date','brainCoords','brainID','numPreMove','-append');
