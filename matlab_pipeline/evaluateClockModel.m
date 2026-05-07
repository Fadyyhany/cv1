function summary = evaluateClockModel(net, data, cfg, splitName)
%EVALUATECLOCKMODEL Evaluate model and save visual diagnostics.

if nargin < 4
    splitName = 'eval';
end

outDir = fullfile(cfg.output.figures, splitName);
if ~isfolder(outDir)
    mkdir(outDir);
end

ds = makeClockDatastore(data.Files, data.Angles, cfg, false);
yPred = predict(net, ds, ...
    'MiniBatchSize', cfg.training.batchSize, ...
    'ExecutionEnvironment', 'auto');

yTrue = data.Angles;
metrics = computeClockMetrics(yTrue, yPred);

summary = table(string(splitName), data.NumSamples, ...
    metrics.timeMAESeconds, metrics.timeMedianAESeconds, metrics.timeWithin1Min, metrics.timeWithin5Min, ...
    'VariableNames', {'Split','NumSamples','TimeMAESeconds','TimeMedianAESeconds','Within1Min','Within5Min'});

fig1 = figure('Visible', 'off');
err = metrics.perSampleAngularError;
for k = 1:size(err,2)
    subplot(size(err,2),1,k);
    histogram(err(:,k), 30);
    xlabel('Absolute angular error (degrees)');
    ylabel('Count');
    title(sprintf('Hand %d angular error distribution', k));
end
saveas(fig1, fullfile(outDir, 'angular_error_hist.png'));
close(fig1);

fig2 = figure('Visible', 'off');
histogram(metrics.perSampleTimeErrorSeconds, 40);
xlabel('Absolute time error (seconds)');
ylabel('Count');
title(sprintf('%s time error distribution', splitName));
saveas(fig2, fullfile(outDir, 'time_error_hist.png'));
close(fig2);

nShow = min(9, data.NumSamples);
showIdx = randperm(data.NumSamples, nShow);
fig3 = figure('Visible', 'off');
for i = 1:nShow
    subplot(3,3,i);
    I = imread(data.Files{showIdx(i)});
    imshow(I);
    trueH = metrics.trueHMS(showIdx(i),:);
    predH = metrics.predHMS(showIdx(i),:);
    title(sprintf('GT %02d:%02d:%02d | Pred %02d:%02d:%02d', ...
        floor(trueH(1)), floor(trueH(2)), floor(trueH(3)), ...
        floor(predH(1)), floor(predH(2)), floor(predH(3))), 'FontSize', 8);
end
sgtitle(sprintf('Prediction examples (%s)', splitName));
saveas(fig3, fullfile(outDir, 'prediction_examples.png'));
close(fig3);
end
