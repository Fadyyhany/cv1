function ds = makeClockDatastore(files, angles, cfg, isTraining)
%MAKECLOCKDATASTORE Build datastore with preprocessing and optional augmentation.

imds = imageDatastore(files, 'ReadFcn', @(f) imread(f));
labelDS = arrayDatastore(single(angles), 'IterationDimension', 1);
combined = combine(imds, labelDS);
ds = transform(combined, @(x) preprocessClockSample(x, cfg, isTraining));
end
