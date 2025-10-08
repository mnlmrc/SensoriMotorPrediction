clc
clear
close all

% Get list of all Recording-*.mat files in the current folder
files = dir('Recording-*.mat');

% Loop through and load each file
for i = 1:length(files)
    filename = files(i).name;
    fprintf('Loading %s\n', filename);
    load(filename);

    % Extract number using regular expression
    tokens = regexp(filename, 'Recording-(\d+)\.mat', 'tokens');
    numberStr = tokens{1}{1};  % e.g., '21'

    T = struct2table(trial);
    writetable(T, sprintf('trial_info-%s.tsv', numberStr), 'FileType', 'text', 'Delimiter', '\t');

    save(sprintf('spike-%s.mat', numberStr), 'spikes_s')

    T = table(brainID, 'VariableNames', {'brainID'});
    writetable(T, sprintf('roi-%s.txt', numberStr), 'FileType', 'text', 'Delimiter', '\t');
end


