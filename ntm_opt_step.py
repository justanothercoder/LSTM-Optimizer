import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class NTMOptStep(L.layers.MergeLayer):
    def __init__(self, incoming, memory_in, num_units, loglr=True,
                 W_hidden_to_output=L.init.GlorotUniform(),
                 **kwargs):

        incomings = [incoming, memory_in]
        super(NTMOptStep, self).__init__(incomings, **kwargs)

        self.loglr = loglr
        self.num_out = 1 + int(loglr) + 3 # a, b, r + delta + loglr
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, self.num_out), name='W_hidden_to_output', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 1)
        
    def get_output_for(self, inputs, **kwargs):
        hid, memory = inputs 
        
        out = hid.dot(self.W_hidden_to_output)
        if self.loglr:
            dtheta = T.exp(out[:, 1]) * out[:, 0]
            a, b, r = out[:, 2], out[:, 3], out[:, 4]
        else:
            dtheta = out[:, 0]
            a, b, r = out[:, 1], out[:, 2], out[:, 3]

        return dtheta, r.dimshuffle(0, 'x'), memory + T.dot(a, b.T)
