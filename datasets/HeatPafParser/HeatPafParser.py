import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class HeatPafParser:
    def __init__(self, limb_sequence, max_number_person=100):
        self.max_number_person = max_number_person
        self.limb_sequence = limb_sequence

    def forward(self, heatmap, pafmap):
        if self.req[0] == req.null:
            return
        out = self.y
        number_of_parts, h0, w0 = heatmap.shape
        number_of_parts -= 1  # one channel for background
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            # (const T * p_heat, const T * p_paf, const T_index * limbs, const T_index number_of_parts,
            # const T_index number_of_limbs, const T_index image_width,
            # const T_index image_height, int * p_subsets_out)
            mobula.func.heat_paf_parser(heatmap, pafmap, self.limb_sequence, number_of_parts, len(self.limb_sequence), w0, h0, out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.heat_paf_parser(heatmap, pafmap, self.limb_sequence, number_of_parts, len(self.limb_sequence), w0, h0, self.y)

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # number of parts * Image Height * Image Width
        assert len(in_shape[1]) == 3  # number of limbs * Image Height * Image Width
        number_of_parts, h0, w0 = in_shape[0]
        number_of_limbs, h1, w1 = in_shape[1]
        assert h0 == h1
        assert w0 == w1
        return in_shape, [(self.max_number_person, number_of_parts + 2)]

    def backward(self):
        pass    # nothing need to do
