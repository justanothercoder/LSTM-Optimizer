import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class OptStep(L.layers.MergeLayer):
    def __init__(self, incoming, num_units, num_out, loglr=True,
                 W_hidden_to_output=L.init.GlorotUniform(),
                 **kwargs):

        incomings = [incoming]
        super(OptStep, self).__init__(incomings, **kwargs)

        self.num_out = num_out
        self.loglr = loglr
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, num_out), name='W_hidden_to_output', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (self.num_out,)
        
    def random_mask(self, shape, p): 
        return self._srng.binomial(shape, p=p, dtype=theano.config.floatX)

    def get_output_for(self, inputs, **kwargs):
        hid = inputs[0]
        
        out = hid.dot(self.W_hidden_to_output)
        if self.loglr:
            dtheta = T.exp(out[:, 1]) * out[:, 0]
        else:
            dtheta = out[:, 0]

        #if not deterministic and self.p_drop_delta > 0.0:
        #    delta_mask = self.delta_mask if self.fix_drop_delta_over_time else random_mask(dtheta.shape, 1. - self.p_drop_delta)
        #    dtheta = dtheta * delta_mask
        #else:
        #    dtheta = dtheta * (1. - self.p_drop_coord) * (1. - self.p_drop_delta)

        #theta = theta_previous + dtheta * self.scale_output
        #if not deterministic and self.p_drop_coord > 0.0:
        #    theta = T.switch(coord_mask, theta, theta_previous)

        #return theta, r, new_memory
        return dtheta
        
    def step_opt(theta_previous, read_vector, memory_matrix, *args):
        func = self.func(theta_previous)
        input_n = theano.gradient.disconnected_grad(theano.grad(func, theta_previous))

        grad = input_n
        out = step(theta_previous, read_vector, memory_matrix, grad, *args)
        return [out[0], func, out[1], out[2]] + out[3:]

    def step_opt_params(params_cur, theta_previous, *args):
        func = self.func(theta_previous, params_cur)
        input_n = theano.gradient.disconnected_grad(theano.grad(func, theta_previous))
        
        grad = input_n
        out = step(theta_previous, grad, *args)
        return [out[0], func] + out[1:]


    #def get_output_for(self, input, **kwargs):
        #return theta_out, loss_out, updates
