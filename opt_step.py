import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class OptStep(L.layers.MergeLayer):
    def __init__(self, incoming, num_units, loglr=True,
                 W_hidden_to_output=L.init.GlorotUniform(),
                 **kwargs):

        incomings = [incoming]
        super(OptStep, self).__init__(incomings, **kwargs)

        self.loglr = loglr
        self.num_out = 1 + int(loglr)
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, self.num_out), name='W_hidden_to_output', regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (self.num_out,)

    def get_output_for(self, inputs, **kwargs):
        hid = inputs[0]
        
        out = hid.dot(self.W_hidden_to_output)
        if self.loglr:
            dtheta = T.exp(out[:, 1]) * out[:, 0]
        else:
            dtheta = out[:, 0]

        return dtheta
