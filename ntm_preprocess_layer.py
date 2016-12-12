import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class NTMPreprocessLayer(L.layers.MergeLayer):
    def __init__(self, incoming, read_in, mem_in,
                 preprocess_input=True, p=10., 
                 use_function_values=False, **kwargs):
        incomings = [incoming, read_in, mem_in]
        super(NTMPreprocessLayer, self).__init__(incomings, **kwargs)

        self.read_in = read_in
        self.mem_in  = mem_in

        self.preprocess_input = preprocess_input
        self.p = p
        self.use_function_values = use_function_values

    def get_recurrent_inits(self, num_batch):
        mem_init  = T.eye(num_batch)
        read_init = T.zeros((num_batch, 1)) 
        return [(self.mem_in, mem_init), (self.read_in, read_init)]

    def get_output_shape_for(self, input_shapes):
        num_out = 1 + int(self.preprocess_input) + int(self.use_function_values) + 1
        return (input_shapes[0][0], num_out)

    def get_output_for(self, inputs, **kwargs):
        input, read, mem = inputs
        input = input.dimshuffle(0, 'x')

        if self.preprocess_input:
            lognorm = T.switch(T.abs_(input) > T.exp(-self.p), T.log(T.abs_(input)) / self.p, T.ones_like(input) * (-1))
            sign    = T.switch(T.abs_(input) > T.exp(-self.p), T.sgn(input), T.exp(self.p) * input)
            input = T.concatenate([lognorm, sign], axis=1)
        
        input_vector = mem.dot(read)
        input = T.concatenate([input, input_vector], axis=1)

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

