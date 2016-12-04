import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class PreprocessLayer(L.layers.Layer):
    def __init__(self, incoming, 
                 preprocess_input=True, p=10., 
                 use_function_values=False, **kwargs):
        super(PreprocessLayer, self).__init__(incoming, **kwargs)

        self.preprocess_input = preprocess_input
        self.p = p
        self.use_function_values = use_function_values

    def get_output_shape_for(self, input_shape):
        num_out = 1 + int(self.preprocess_input) + int(self.use_function_values)
        return (input_shape[0], num_out)

    def get_output_for(self, input, **kwargs):
        input = input.dimshuffle(0, 'x')

        if self.preprocess_input:
            lognorm = T.switch(T.abs_(input) > T.exp(-self.p), T.log(T.abs_(input)) / self.p, T.ones_like(input) * (-1))
            sign    = T.switch(T.abs_(input) > T.exp(-self.p), T.sgn(input), T.exp(self.p) * input)
            input = T.concatenate([lognorm, sign], axis=1)

        return input
        
        #if self.use_function_values:
        #    input = T.concatenate([input, T.ones_like(input) * func], axis=1)

        #if not deterministic and self.p_drop_grad > 0.0:
        #    grad_mask = self.grad_mask if self.fix_grad_coord_over_time else random_mask(input_n.shape, 1. - self.p_drop_grad)
        #    input_n = input_n * grad_mask
        #else:
        #    input_n = input_n * (1. - self.p_drop_grad)
        #
        #if not deterministic and self.p_drop_coord > 0.0:
        #    coord_mask = self.coord_mask if self.fix_drop_coord_over_time else random_mask(input_n.shape[:1], 1. - self.p_drop_coord)

