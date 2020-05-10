from .gnn_layers import MLPConcatLayer, MLPLinearConcatLayer

gnn_layer_dict = {
    'mlp_concat': MLPConcatLayer,
    'mlp_linear_concat': MLPLinearConcatLayer,
}