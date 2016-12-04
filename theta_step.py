import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class ThetaStep(L.layers.MergeLayer):
    def __init__(self, theta_in, delta_in, theta_init, scale_output=1.0, **kwargs):

        incomings = [theta_in, delta_in]
        super(ThetaStep, self).__init__(incomings, **kwargs)

        self.theta_in = theta_in
        self.delta_in = delta_in
        self.theta_init = theta_init
        self.scale_output = scale_output 

    def get_recurrent_inits(self, num_batch):
        return [(self.theta_in, self.theta_init)]
        
    def get_output_for(self, inputs, **kwargs):
        theta, delta = inputs
        return theta + self.scale_output * delta
