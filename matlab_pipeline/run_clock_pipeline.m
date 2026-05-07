function artifacts = run_clock_pipeline(userCfg)
%RUN_CLOCK_PIPELINE End-to-end training/evaluation entry point.
%
% Design notes:
% - Predicting hand angles (hour/minute/optional second) is continuous and
%   generalizes better than classifying fixed time buckets.
% - Two-stage transfer learning improves stability: train head first, then
%   unfreeze for low-LR domain adaptation on real images.
% - Heavy augmentation and edge blending reduce shape/background overfitting.

if nargin < 1
    userCfg = struct();
end
cfg = mergeStructs(defaultClockConfig(), userCfg);

rng(cfg.randomSeed);
ensureDirs(cfg);

if gpuDeviceCount > 0
    gpuDevice;
    disp('GPU detected. Training on GPU where available.');
else
    disp('No GPU detected. Training on CPU.');
end

synthetic = loadClockDataset(cfg.dataset.syntheticImages, cfg.dataset.syntheticLabels);
realSet = loadClockDataset(cfg.dataset.realImages, cfg.dataset.realLabels);
numOutputs = min(synthetic.NumOutputs, realSet.NumOutputs);

% Keep labels consistent across datasets (2-angle or 3-angle mode).
synthetic.Angles = synthetic.Angles(:,1:numOutputs);
realSet.Angles = realSet.Angles(:,1:numOutputs);
synthetic.NumOutputs = numOutputs;
realSet.NumOutputs = numOutputs;

synSplit = splitClockDataset(synthetic, cfg.training.validationFraction, cfg.randomSeed);
realSplit = splitClockDataset(realSet, cfg.training.validationFraction, cfg.randomSeed + 1);

trainDS = makeClockDatastore(synSplit.train.Files, synSplit.train.Angles, cfg, true);
valDS = makeClockDatastore(synSplit.val.Files, synSplit.val.Angles, cfg, false);
fineTuneDS = makeClockDatastore(realSplit.train.Files, realSplit.train.Angles, cfg, true);

[trainedNet, trainInfo] = trainClockRegressor(trainDS, valDS, fineTuneDS, numOutputs, cfg);
visualizeTrainingHistory(trainInfo, cfg);

seenSummary = evaluateClockModel(trainedNet, synSplit.val, cfg, 'seen_validation');
unseenSummary = evaluateClockModel(trainedNet, realSplit.val, cfg, 'unseen_validation');

extraSeen = evaluateNamedSets(trainedNet, cfg, cfg.evaluation.seenSets, 'seen_case', numOutputs);
extraUnseen = evaluateNamedSets(trainedNet, cfg, cfg.evaluation.unseenSets, 'unseen_case', numOutputs);

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
modelPath = fullfile(cfg.output.models, ['ClockAngleRegressor_' timestamp '.mat']);
save(modelPath, 'trainedNet', 'cfg', 'trainInfo', 'seenSummary', 'unseenSummary', 'extraSeen', 'extraUnseen');

artifacts.modelPath = modelPath;
artifacts.seenSummary = seenSummary;
artifacts.unseenSummary = unseenSummary;
artifacts.extraSeen = extraSeen;
artifacts.extraUnseen = extraUnseen;

allSummaries = [seenSummary; unseenSummary];
if ~isempty(extraSeen)
    allSummaries = [allSummaries; extraSeen]; %#ok<AGROW>
end
if ~isempty(extraUnseen)
    allSummaries = [allSummaries; extraUnseen]; %#ok<AGROW>
end
writetable(allSummaries, fullfile(cfg.output.root, ['evaluation_summary_' timestamp '.csv']));
end

function out = evaluateNamedSets(net, cfg, sets, prefix, numOutputs)
if isempty(sets)
    out = [];
    return;
end
out = [];
for i = 1:numel(sets)
    ds = loadClockDataset(sets(i).imageDir, sets(i).labelFile);
    ds.Angles = ds.Angles(:,1:min(numOutputs, size(ds.Angles,2)));
    s = evaluateClockModel(net, ds, cfg, sprintf('%s_%s', prefix, sets(i).name));
    if isempty(out)
        out = s;
    else
        out = [out; s]; %#ok<AGROW>
    end
end
end

function ensureDirs(cfg)
folders = {cfg.output.root, cfg.output.figures, cfg.output.models, cfg.training.checkpointDir};
for i = 1:numel(folders)
    if ~isfolder(folders{i})
        mkdir(folders{i});
    end
end
end

function merged = mergeStructs(base, override)
merged = base;
if isempty(override)
    return;
end
fields = fieldnames(override);
for i = 1:numel(fields)
    f = fields{i};
    if isstruct(override.(f)) && isfield(base, f) && isstruct(base.(f))
        merged.(f) = mergeStructs(base.(f), override.(f));
    else
        merged.(f) = override.(f);
    end
end
end
