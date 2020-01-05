import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class PAFGen:
    def __init__(self, max_number_person=100):
        self.max_number_person = max_number_person

    def forward(self, heatmap, pafmap):
        if self.req[0] == req.null:
            return
        out = self.y
        number_of_parts, h0, w0 = heatmap.shape

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # number of parts * Image Height * Image Width
        assert len(in_shape[1]) == 2  # number of limbs * Image Height * Image Width
        number_of_parts, h0, w0 = in_shape[0]
        number_of_limbs, h1, w1 = in_shape[1]
        return in_shape, [(self.max_number_person, number_of_parts + 2)]

    def backward(self):
        pass    # nothing need to do
