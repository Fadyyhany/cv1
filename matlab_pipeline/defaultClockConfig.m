function cfg = defaultClockConfig()
%DEFAULTCLOCKCONFIG Default configuration for analog clock regression pipeline.

projectRoot = fileparts(fileparts(mfilename('fullpath')));

cfg.randomSeed = 42;

cfg.dataset.syntheticImages = 'E:\\2nd Semester\\CV\\IDEA_CV\\50.000_Coloured_Real\\images\\images';
cfg.dataset.syntheticLabels = 'E:\\2nd Semester\\CV\\IDEA_CV\\50.000_Coloured_Real\\label.csv';
cfg.dataset.realImages      = 'E:\\2nd Semester\\CV\\IDEA_CV\\103_Real_perfect\\Images\\Images';
cfg.dataset.realLabels      = 'E:\\2nd Semester\\CV\\IDEA_CV\\103_Real_perfect\\label.csv';

cfg.model.inputSize = [224 224 3];
cfg.model.backbone = 'resnet50';
cfg.model.headWidth = [512 256];
cfg.model.dropout = 0.35;

cfg.training.batchSize = 16;
cfg.training.validationFraction = 0.2;
cfg.training.optimizer = 'adam';
cfg.training.stage1Epochs = 12;
cfg.training.stage2Epochs = 10;
cfg.training.stage1InitialLR = 2e-4;
cfg.training.stage2InitialLR = 4e-5;
cfg.training.lrDropFactor = 0.2;
cfg.training.lrDropPeriod = 4;
cfg.training.validationPatience = 5;
cfg.training.checkpointDir = fullfile(projectRoot, 'artifacts', 'checkpoints');
cfg.training.useMixedPrecision = true;

cfg.augmentation.rotation = 40;
cfg.augmentation.translate = 0.15;
cfg.augmentation.scale = [0.75 1.25];
cfg.augmentation.shear = 18;
cfg.augmentation.perspective = 0.08;
cfg.augmentation.brightness = 0.30;
cfg.augmentation.contrast = 0.35;
cfg.augmentation.colorJitter = 0.25;
cfg.augmentation.blurProbability = 0.35;
cfg.augmentation.noiseProbability = 0.35;
cfg.augmentation.occlusionProbability = 0.30;
cfg.augmentation.grayscaleProbability = 0.20;
cfg.augmentation.edgeBlend = 0.25;

cfg.output.root = fullfile(projectRoot, 'artifacts');
cfg.output.figures = fullfile(cfg.output.root, 'figures');
cfg.output.models = fullfile(cfg.output.root, 'models');

cfg.evaluation.seenSets = struct('name', {}, 'imageDir', {}, 'labelFile', {});
cfg.evaluation.unseenSets = struct('name', {}, 'imageDir', {}, 'labelFile', {});
end
