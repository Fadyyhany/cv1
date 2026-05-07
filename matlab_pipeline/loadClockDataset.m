function data = loadClockDataset(imageDir, labelFile)
%LOADCLOCKDATASET Load image files and convert labels to hand angles.

if ~isfolder(imageDir)
    error('Image directory not found: %s', imageDir);
end
if ~isfile(labelFile)
    error('Label file not found: %s', labelFile);
end

labels = readtable(labelFile);
varNames = lower(string(labels.Properties.VariableNames));

hourCol = find(contains(varNames, "hour"), 1);
minuteCol = find(contains(varNames, "minute"), 1);
secondCol = find(contains(varNames, "second"), 1);

numericCols = find(varfun(@isnumeric, labels, 'OutputFormat', 'uniform'));
if isempty(hourCol), hourCol = numericCols(1); end
if isempty(minuteCol), minuteCol = numericCols(2); end
if isempty(secondCol) && numel(numericCols) >= 3, secondCol = numericCols(3); end

H = double(labels{:, hourCol});
M = double(labels{:, minuteCol});
if isempty(secondCol)
    S = zeros(size(H));
    hasSecond = false;
else
    S = double(labels{:, secondCol});
    hasSecond = true;
end

anglesHour = mod((mod(H,12) + M./60 + S./3600) * 30, 360);
anglesMinute = mod((M + S./60) * 6, 360);
angles = [anglesHour, anglesMinute];
if hasSecond
    anglesSecond = mod(S * 6, 360);
    angles = [angles, anglesSecond]; %#ok<AGROW>
end

imds = imageDatastore(imageDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});

n = min(numel(imds.Files), size(angles,1));
if n == 0
    error('No usable samples found in %s.', imageDir);
end

data.Files = imds.Files(1:n);
data.Angles = single(angles(1:n,:));
data.HMS = [H(1:n), M(1:n), S(1:n)];
data.HasSecond = hasSecond;
data.NumOutputs = size(data.Angles,2);
data.NumSamples = n;
end
