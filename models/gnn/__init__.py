from .gnn_layers import MLPConcatLayer, MLPLinearConcatLayer, MLPConcatScoreLayer, MLPConcatSeparateScoreLayer

gnn_layer_dict = {
    'mlp_concat': MLPConcatLayer,
    'mlp_concat_score': MLPConcatScoreLayer,
    'mlp_concat_score_separate': MLPConcatSeparateScoreLayer,
    'mlp_linear_concat': MLPLinearConcatLayer,
}