import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class LSTMStep(L.layers.MergeLayer):
    def __init__(self, incoming, cell_in, hid_in, num_units, n_gac=0,
                 ingate=L.layers.Gate(), forgetgate=L.layers.Gate(), 
                 outgate=L.layers.Gate(), nonlinearity=L.nonlinearities.tanh,
                 cell=L.layers.Gate(W_cell=None, nonlinearity=L.nonlinearities.tanh),
                 cell_init=L.init.Constant(0.), hid_init=L.init.Constant(0.), 
                 learn_init=False, grad_clipping=0, **kwargs):

        incomings = [incoming, cell_in, hid_in]
        super(LSTMStep, self).__init__(incomings, **kwargs)

        self.cell_in = cell_in
        self.hid_in = hid_in

        self.nonlinearity = nonlinearity or L.nonlinearities.identity

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping

        self.n_gac = n_gac

        input_shape = self.input_shapes[0]
        num_inputs = np.prod(input_shape[1:])
        
        def add_gate_params(gate, gate_name):
            W_in  = self.add_param(gate.W_in,  (num_inputs, num_units), name="W_in_to_{}".format(gate_name))
            W_hid = self.add_param(gate.W_hid, (num_units , num_units), name="W_hid_to_{}".format(gate_name))
            b     = self.add_param(gate.b,     (num_units ,)          , name="b_{}".format(gate_name), regularizable=False)
            return (W_in, W_hid, b, gate.nonlinearity)

        # Add in parameters from the supplied L.layers.Gate instances
        (self.W_in_to_ingate    , self.W_hid_to_ingate    , self.b_ingate    , self.nonlinearity_ingate    ) = add_gate_params(ingate    , 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate, self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        (self.W_in_to_cell      , self.W_hid_to_cell      , self.b_cell      , self.nonlinearity_cell      ) = add_gate_params(cell      , 'cell')
        (self.W_in_to_outgate   , self.W_hid_to_outgate   , self.b_outgate   , self.nonlinearity_outgate   ) = add_gate_params(outgate   , 'outgate')
        
        self.cell_init = self.add_param(cell_init, (1, num_units), name="cell_init", trainable=learn_init, regularizable=False)
        self.hid_init  = self.add_param(hid_init , (1, num_units), name="hid_init" , trainable=learn_init, regularizable=False)
        
        self.W_in_stacked  = T.concatenate([self.W_in_to_ingate , self.W_in_to_forgetgate , self.W_in_to_cell , self.W_in_to_outgate], axis=1)
        self.W_hid_stacked = T.concatenate([self.W_hid_to_ingate, self.W_hid_to_forgetgate, self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        self.b_stacked     = T.concatenate([self.b_ingate       , self.b_forgetgate       , self.b_cell       , self.b_outgate], axis=0)

    def get_output_shape_for(self, input_shapes):
        #return input_shapes[0][0], input_shapes[0][1]
        return input_shapes[0][0], self.num_units

    def get_recurrent_inits(self, num_batch):
        ones = T.ones((num_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)
        return [(self.cell_in, cell_init), (self.hid_in, hid_init)]
        
    def slice_w(self, x, n):
        return x[:, n * self.num_units:(n + 1) * self.num_units]

    def get_output_for(self, inputs, **kwargs):
        input, cell_prev, hid_prev = inputs

        input = T.dot(input, self.W_in_stacked) + self.b_stacked
        gates = input + T.dot(hid_prev, self.W_hid_stacked)

        if self.grad_clipping:
            gates = theano.gradient.grad_clip(gates, -self.grad_clipping, self.grad_clipping)

        ingate     = self.slice_w(gates, 0)
        forgetgate = self.slice_w(gates, 1)
        cell_input = self.slice_w(gates, 2)
        outgate    = self.slice_w(gates, 3)

        ingate     = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        cell = forgetgate * cell_prev + ingate * cell_input
        outgate = self.nonlinearity_outgate(outgate)

        if self.n_gac > 0:
            cell = T.set_subtensor(cell[:, :self.n_gac], cell[:, :self.n_gac].mean(axis=0))

        hid = outgate * self.nonlinearity(cell)
        return cell, hid
