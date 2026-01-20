clc
clear
close all

% Get list of all Recording-*.mat files in the current folder
baseDir = fullfile('/cifs/pruszynski/Marco/SensoriMotorPreparation/');
monkey = 'Pert';
files = dir(fullfile(baseDir, 'Recordings', monkey, 'recording*.mat'));

%%
for i = 1:length(files)
    fprintf('Loading %s\n', files(i).name);
    load(fullfile(files(i).folder, files(i).name));
    name_parts = split(files(i).name, '.');
    name_parts = split(name_parts{2}, '-');
    region = name_parts{1};
    rec = str2double(name_parts{2});

    T = struct2table(trial);
    writetable(T, fullfile(baseDir, 'Recordings', monkey,  sprintf('trial_info-%d.tsv', rec)), 'FileType', 'text', 'Delimiter', '\t');
    save(fullfile(baseDir, 'Behavioural', monkey, sprintf('elbow_angle-%d.mat', rec)), 'elbKin', '-v7.3')
    save(fullfile(baseDir, 'LFPs', monkey, sprintf('lfp.%s-%d.mat', region, rec)), 'lfp', '-v7.3')
    save(fullfile(baseDir, 'LFPs',monkey, sprintf('cfg.%s-%d.mat', region, rec)), 'cfg', '-v7.3')
    save(fullfile(baseDir, 'spikes',monkey, sprintf('spike.%s-%d.mat', region, rec)), 'spike', '-v7.3')
    save(fullfile(baseDir, 'spikes',monkey, sprintf('spike_s.%s-%d.mat', region, rec)), 'spike_s', '-v7.3')
end


