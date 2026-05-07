function [trainedNet, trainInfo] = trainClockRegressor(trainDS, valDS, fineTuneDS, numOutputs, cfg)
%TRAINCLOCKREGRESSOR Two-stage transfer learning with checkpointing and early stopping.

if ~isfolder(cfg.training.checkpointDir)
    mkdir(cfg.training.checkpointDir);
end

% Stage 1: frozen backbone to learn robust regression head.
lgraph1 = createClockRegressionModel(cfg, numOutputs, true);
opts1 = createTrainingOptions(cfg, valDS, cfg.training.stage1Epochs, cfg.training.stage1InitialLR, 'stage1');
[net1, info1] = trainNetwork(trainDS, lgraph1, opts1);

% Stage 2: unfreeze and fine-tune backbone with smaller LR.
if cfg.training.stage2Epochs > 0
    lgraph2 = layerGraph(net1);
    lgraph2 = setBackboneTrainable(lgraph2, true);
    opts2 = createTrainingOptions(cfg, valDS, cfg.training.stage2Epochs, cfg.training.stage2InitialLR, 'stage2');
    [trainedNet, info2] = trainNetwork(fineTuneDS, lgraph2, opts2);
else
    trainedNet = net1;
    info2 = struct([]);
end

trainInfo.stage1 = info1;
trainInfo.stage2 = info2;
end

function options = createTrainingOptions(cfg, valDS, epochs, initLR, stageName)
arguments
    cfg struct
    valDS
    epochs (1,1) double
    initLR (1,1) double
    stageName (1,:) char
end

stageCkpt = fullfile(cfg.training.checkpointDir, stageName);
if ~isfolder(stageCkpt)
    mkdir(stageCkpt);
end

commonArgs = {
    'MiniBatchSize', cfg.training.batchSize, ...
    'MaxEpochs', epochs, ...
    'InitialLearnRate', initLR, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', cfg.training.lrDropFactor, ...
    'LearnRateDropPeriod', cfg.training.lrDropPeriod, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valDS, ...
    'ValidationFrequency', max(10, floor(250/cfg.training.batchSize)), ...
    'ValidationPatience', cfg.training.validationPatience, ...
    'CheckpointPath', stageCkpt, ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress'};

if cfg.training.useMixedPrecision
    try
        options = trainingOptions(cfg.training.optimizer, commonArgs{:}, 'Precision', 'mixed');
        return;
    catch
    end
end

options = trainingOptions(cfg.training.optimizer, commonArgs{:});
end
