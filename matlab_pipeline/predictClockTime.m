function pred = predictClockTime(net, imageInput, cfg)
%PREDICTCLOCKTIME Inference utility for a single image.

if ischar(imageInput) || isstring(imageInput)
    img = imread(imageInput);
else
    img = imageInput;
end

x = im2single(img);
if size(x,3) == 1
    x = repmat(x,1,1,3);
end
x = imresize(x, cfg.model.inputSize(1:2));
angles = predict(net, x, 'ExecutionEnvironment', 'auto');
angles = double(reshape(angles, 1, []));

minute = mod(angles(2)/6, 60);
if numel(angles) >= 3
    second = mod(angles(3)/6, 60);
else
    second = 0;
end
hour = mod(angles(1)/30 - minute/60 - second/3600, 12);

pred.angles = angles;
pred.time = [hour, minute, second];
pred.formatted = sprintf('%02d:%02d:%02d', floor(hour), floor(minute), floor(second));
end
