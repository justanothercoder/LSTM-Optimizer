import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class IndexLayer(L.layers.Layer):
    def __init__(self, incoming, index, **kwargs):
        super(IndexLayer, self).__init__(incoming, **kwargs)
        self.index = index

    def get_output_for(self, input, **kwargs):
        return input[self.index]

