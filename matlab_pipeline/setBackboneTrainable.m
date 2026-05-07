function lgraph = setBackboneTrainable(lgraph, isTrainable)
%SETBACKBONETRAINABLE Freeze/unfreeze all non-head layers.

layers = lgraph.Layers;
connections = lgraph.Connections;

factor = double(isTrainable);
for i = 1:numel(layers)
    layer = layers(i);
    if startsWith(layer.Name, 'head_') || strcmp(layer.Name, 'clock_regression')
        continue;
    end

    if isprop(layer, 'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = factor;
    end
    if isprop(layer, 'BiasLearnRateFactor')
        layer.BiasLearnRateFactor = factor;
    end
    if isprop(layer, 'ScaleLearnRateFactor')
        layer.ScaleLearnRateFactor = factor;
    end
    if isprop(layer, 'OffsetLearnRateFactor')
        layer.OffsetLearnRateFactor = factor;
    end
    layers(i) = layer;
end

lgraph = createLgraphUsingConnections(layers, connections);
end

function lgraph = createLgraphUsingConnections(layers, connections)
% Preserve DAG topology while replacing modified layers.

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers(i));
end
for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph, connections.Source{c}, connections.Destination{c});
end
end
