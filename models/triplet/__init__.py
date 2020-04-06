from .dist_mult import ConcatDot, ConcatLinearDot, DistMult, DistMultEntityOnly, RotatE

triplet_dict = {
    'concat_dot': ConcatDot,
    'concat_linear_dot': ConcatLinearDot,
    'distmult': DistMult,
    'distmult_entity_only': DistMultEntityOnly,
    'rotate': RotatE,
}
