function metrics = computeClockMetrics(yTrue, yPred)
%COMPUTECLOCKMETRICS Compute angular and time-domain regression metrics.

yTrue = double(yTrue);
yPred = double(yPred);

angularError = abs(mod(yPred - yTrue + 180, 360) - 180);
metrics.angularMAE = mean(angularError, 1, 'omitnan');
metrics.angularMedianAE = median(angularError, 1, 'omitnan');
metrics.angularRMSE = sqrt(mean(angularError.^2, 1, 'omitnan'));

[hTrue, mTrue, sTrue] = anglesToHMS(yTrue);
[hPred, mPred, sPred] = anglesToHMS(yPred);

secTrue = mod(hTrue,12) * 3600 + mTrue * 60 + sTrue;
secPred = mod(hPred,12) * 3600 + mPred * 60 + sPred;

diffSec = abs(secPred - secTrue);
diffSec = min(diffSec, 12*3600 - diffSec);

metrics.timeMAESeconds = mean(diffSec, 'omitnan');
metrics.timeMedianAESeconds = median(diffSec, 'omitnan');
metrics.timeP90Seconds = prctile(diffSec, 90);
metrics.timeWithin1Min = mean(diffSec <= 60);
metrics.timeWithin5Min = mean(diffSec <= 300);

metrics.perSampleAngularError = angularError;
metrics.perSampleTimeErrorSeconds = diffSec;
metrics.trueHMS = [hTrue, mTrue, sTrue];
metrics.predHMS = [hPred, mPred, sPred];
end

function [h, m, s] = anglesToHMS(angles)
% Convert hand angles to continuous clock time.

m = mod(angles(:,2) / 6, 60);
if size(angles,2) >= 3
    s = mod(angles(:,3) / 6, 60);
else
    s = zeros(size(m));
end

h = mod(angles(:,1) / 30 - m/60 - s/3600, 12);
end
