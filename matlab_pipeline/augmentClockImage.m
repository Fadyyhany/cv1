function [img, angles] = augmentClockImage(img, angles, aug)
%AUGMENTCLOCKIMAGE Domain-randomized augmentation for robust generalization.

angles = double(reshape(angles, 1, []));

% Rotation with label correction in angular space.
rot = (rand * 2 - 1) * aug.rotation;
img = imrotate(img, rot, 'bilinear', 'crop');
angles = mod(angles + rot, 360);

% Affine perturbations.
scale = aug.scale(1) + rand * (aug.scale(2) - aug.scale(1));
tx = (rand * 2 - 1) * aug.translate * size(img,2);
ty = (rand * 2 - 1) * aug.translate * size(img,1);
shear = deg2rad((rand * 2 - 1) * aug.shear);
A = [scale tan(shear) 0; 0 scale 0; tx ty 1];
tformA = affine2d(A);
img = imwarp(img, tformA, 'OutputView', imref2d(size(img(:,:,1))), 'FillValues', rand(1,3));

% Perspective warp (when available).
if aug.perspective > 0
    try
        [h,w,~] = size(img);
        src = [1 1; w 1; w h; 1 h];
        delta = aug.perspective * min(h,w);
        dst = src + (rand(4,2) * 2 - 1) * delta;
        tformP = fitgeotform2d(src, dst, 'projective');
        img = imwarp(img, tformP, 'OutputView', imref2d([h w]), 'FillValues', rand(1,3));
    catch
        % Keep affine-only fallback for environments without projective fitting.
    end
end

% Color and illumination jitter.
contrast = 1 + (rand * 2 - 1) * aug.contrast;
brightness = (rand * 2 - 1) * aug.brightness;
img = img * contrast + brightness;

chScale = 1 + (rand(1,3) * 2 - 1) * aug.colorJitter;
for c = 1:3
    img(:,:,c) = img(:,:,c) * chScale(c);
end

if rand < aug.grayscaleProbability
    gray = rgb2gray(img);
    img = cat(3, gray, gray, gray);
end

if rand < aug.blurProbability
    sigma = 0.4 + rand * 2.0;
    img = imgaussfilt(img, sigma);
end

if rand < aug.noiseProbability
    img = imnoise(min(max(img,0),1), 'gaussian', 0, 0.002 + rand * 0.01);
end

% Random occlusion to simulate partial obstruction.
if rand < aug.occlusionProbability
    [h,w,~] = size(img);
    occW = max(8, round(w * (0.10 + rand * 0.25)));
    occH = max(8, round(h * (0.10 + rand * 0.25)));
    x0 = randi([1, max(1,w-occW+1)]);
    y0 = randi([1, max(1,h-occH+1)]);
    img(y0:y0+occH-1, x0:x0+occW-1, :) = rand(1,1,3);
end

img = min(max(img, 0), 1);
angles = single(angles);
end
