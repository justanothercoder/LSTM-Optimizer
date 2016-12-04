import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class GradLayer(L.layers.Layer):
    def __init__(self, incoming, function, **kwargs):
        super(GradLayer, self).__init__(incoming, **kwargs)
        self.function = function

    def get_output_for(self, input, **kwargs):
        function = self.function(input)
        grad = theano.gradient.disconnected_grad(theano.grad(function, input))
        return grad, function
