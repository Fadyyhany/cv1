function out = preprocessClockSample(data, cfg, isTraining)
%PREPROCESSCLOCKSAMPLE Shared preprocessing for train/val/inference.

img = data{1};
angles = single(data{2});
if iscell(img), img = img{1}; end
if iscell(angles), angles = angles{1}; end

img = im2single(img);
if size(img,3) == 1
    img = repmat(img, 1, 1, 3);
end

if isTraining
    [img, angles] = augmentClockImage(img, angles, cfg.augmentation);
end

img = imresize(img, cfg.model.inputSize(1:2));

% Edge-emphasis helps reduce background bias and focus on hand geometry.
if cfg.augmentation.edgeBlend > 0
    gray = rgb2gray(img);
    edgeMap = edge(gray, 'Canny');
    edgeRGB = repmat(single(edgeMap), 1, 1, 3);
    alpha = cfg.augmentation.edgeBlend;
    img = (1 - alpha) .* img + alpha .* edgeRGB;
end

img = min(max(img, 0), 1);
out = {img, reshape(single(angles), 1, [])};
end
