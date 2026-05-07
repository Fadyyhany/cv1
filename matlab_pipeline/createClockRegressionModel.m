function lgraph = createClockRegressionModel(cfg, numOutputs, freezeBackbone)
%CREATECLOCKREGRESSIONMODEL Build transfer-learning model for angle regression.

backbone = lower(string(cfg.model.backbone));
switch backbone
    case "resnet50"
        net = resnet50('Weights', 'imagenet');
        lgraph = layerGraph(net);
        lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
        featureLayer = 'avg_pool';
    case "mobilenetv2"
        net = mobilenetv2('Weights', 'imagenet');
        lgraph = layerGraph(net);
        lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});
        featureLayer = 'global_average_pooling2d_1';
    otherwise
        error('Unsupported backbone "%s". Use resnet50 or mobilenetv2.', cfg.model.backbone);
end

head = [
    fullyConnectedLayer(cfg.model.headWidth(1), 'Name', 'head_fc1', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'head_relu1')
    dropoutLayer(cfg.model.dropout, 'Name', 'head_drop1')
    fullyConnectedLayer(cfg.model.headWidth(2), 'Name', 'head_fc2', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'head_relu2')
    fullyConnectedLayer(numOutputs, 'Name', 'head_angles', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    regressionLayer('Name', 'clock_regression')
    ];

lgraph = addLayers(lgraph, head);
lgraph = connectLayers(lgraph, featureLayer, 'head_fc1');

if freezeBackbone
    lgraph = setBackboneTrainable(lgraph, false);
else
    lgraph = setBackboneTrainable(lgraph, true);
end
end
