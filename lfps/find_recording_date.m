function  find_recording_date(monkey, num_recording)
    mf = matfile(sprintf('/cifs/pruszynski/Marco/SensoriMotorPrediction/spikes/%s/Recording-%d.mat', monkey, num_recording));
    disp(mf.date)
end