function split = splitClockDataset(data, valFraction, seed)
%SPLITCLOCKDATASET Random train/validation split.

if nargin < 3
    seed = 42;
end

rng(seed);
n = data.NumSamples;
idx = randperm(n);
nVal = max(1, round(valFraction * n));
valIdx = idx(1:nVal);
trainIdx = idx(nVal+1:end);
if isempty(trainIdx)
    trainIdx = valIdx;
end

split.train.Files = data.Files(trainIdx);
split.train.Angles = data.Angles(trainIdx,:);
split.train.HMS = data.HMS(trainIdx,:);
split.train.HasSecond = data.HasSecond;
split.train.NumOutputs = data.NumOutputs;
split.train.NumSamples = numel(trainIdx);

split.val.Files = data.Files(valIdx);
split.val.Angles = data.Angles(valIdx,:);
split.val.HMS = data.HMS(valIdx,:);
split.val.HasSecond = data.HasSecond;
split.val.NumOutputs = data.NumOutputs;
split.val.NumSamples = numel(valIdx);
end
