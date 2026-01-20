function filteredData = forceFromMov(filePath)
    % Read the data from the file
    data = readmatrix(filePath, 'Delimiter', '\t', 'FileType', 'text');
    
    % Filter rows where the 6th column is greater than 0
    filteredData = data(data(:, 6) > 0, [6:7,9:13]);

end