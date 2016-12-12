import numpy as np

import theano
import theano.tensor as T
import lasagne as L

from theano.printing import Print as TPP

class NTMOptStep(L.layers.MergeLayer):
    def __init__(self, incoming, memory_in, num_units, loglr=True, memdot=False, 
                 grad_in=None, hess_in=None,
                 W_hidden_to_output=L.init.GlorotUniform(), 
                 **kwargs):

        incomings = [incoming, memory_in]
            
        self.grad_in = grad_in
        self.hess_in = hess_in

        if grad_in is not None:
            incomings += [grad_in]
        elif hess_in is not None:
            incomings += [hess_in]

        super(NTMOptStep, self).__init__(incomings, **kwargs)

        self.memdot = memdot
        self.loglr = loglr
        self.num_out = 1 + int(loglr) + 3 # a, b, r + delta + loglr
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, self.num_out), name='W_hidden_to_output', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 1)
        
    def get_output_for(self, inputs, **kwargs):
        if self.grad_in is not None:
            hid, memory, grad = inputs
        elif self.hess_in is not None:
            hid, memory, hess = inputs
        else:
            hid, memory = inputs 
        
        out = hid.dot(self.W_hidden_to_output)
        if self.loglr:
            dtheta = T.exp(out[:, 1]) * out[:, 0]
            a, b, r = out[:, 2], out[:, 3], out[:, 4]
        else:
            dtheta = out[:, 0]
            a, b, r = out[:, 1], out[:, 2], out[:, 3]

        if self.memdot:
            dtheta = memory.dot(dtheta)

        if self.hess_in:
            new_memory = T.nlinalg.pinv(hess)
        else:
            new_memory = memory + T.dot(a, b.T)

        new_r = r.dimshuffle(0, 'x')

        if self.grad_in:
            dtheta = dtheta - grad

        return dtheta, new_r, new_memory
