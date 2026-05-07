function visualizeTrainingHistory(trainInfo, cfg)
%VISUALIZETRAININGHISTORY Plot stage-wise training and validation curves.

if ~isfolder(cfg.output.figures)
    mkdir(cfg.output.figures);
end

fig = figure('Visible', 'off');
tiledlayout(2,2);

nexttile;
plotField(trainInfo.stage1, 'TrainingLoss', 0, 'b-');
hold on;
plotField(trainInfo.stage1, 'ValidationLoss', 0, 'r-');
legendEntries = {'Train S1','Val S1'};
if ~isempty(trainInfo.stage2)
    e1 = numel(trainInfo.stage1.TrainingLoss);
    plotField(trainInfo.stage2, 'TrainingLoss', e1, 'b--');
    plotField(trainInfo.stage2, 'ValidationLoss', e1, 'r--');
    legendEntries = [legendEntries, {'Train S2','Val S2'}]; %#ok<AGROW>
end
grid on;
xlabel('Iteration'); ylabel('Loss'); title('Training/Validation Loss'); legend(legendEntries);

nexttile;
plotField(trainInfo.stage1, 'BaseLearnRate', 0, 'k-');
hold on;
if ~isempty(trainInfo.stage2)
    e1 = numel(trainInfo.stage1.BaseLearnRate);
    plotField(trainInfo.stage2, 'BaseLearnRate', e1, 'k--');
end
grid on;
xlabel('Iteration'); ylabel('Learning Rate'); title('Learning-rate schedule');

nexttile([1 2]);
if isfield(trainInfo.stage1, 'ValidationLoss')
    v1 = trainInfo.stage1.ValidationLoss;
    v1 = v1(~isnan(v1));
else
    v1 = [];
end
if ~isempty(trainInfo.stage2) && isfield(trainInfo.stage2, 'ValidationLoss')
    v2 = trainInfo.stage2.ValidationLoss;
    v2 = v2(~isnan(v2));
else
    v2 = [];
end
bar([mean(v1,'omitnan'), mean(v2,'omitnan')]);
set(gca,'XTickLabel',{'Stage1','Stage2'});
ylabel('Mean Validation Loss'); title('Stage comparison');

saveas(fig, fullfile(cfg.output.figures, 'training_curves.png'));
close(fig);
end

function plotField(info, fieldName, xOffset, style)
if isempty(info) || ~isfield(info, fieldName)
    return;
end
vals = info.(fieldName);
if isempty(vals)
    return;
end
x = (1:numel(vals)) + xOffset;
plot(x, vals, style, 'LineWidth', 1.5);
end
