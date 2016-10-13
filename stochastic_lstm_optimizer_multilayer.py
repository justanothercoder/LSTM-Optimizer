import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.printing import Print as TPP

class LSTMOptimizerLayer(L.layers.MergeLayer):
    def add_lstm_step(self, num_inp, ingate, forgetgate, outgate, cell, cell_init, hid_init):
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a L.layers.Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inp, self.num_units), name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (self.num_units, self.num_units), name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (self.num_units,), name="b_{}".format(gate_name), regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied L.layers.Gate instances
        (W_in_to_ingate, W_hid_to_ingate, b_ingate, self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (W_in_to_forgetgate, W_hid_to_forgetgate, b_forgetgate, self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        (W_in_to_cell, W_hid_to_cell, b_cell, self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (W_in_to_outgate, W_hid_to_outgate, b_outgate, self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        
        cell_init = self.add_param(cell_init, (1, self.num_units), name="cell_init", trainable=self.learn_init, regularizable=False)
        hid_init  = self.add_param(hid_init , (1, self.num_units), name="hid_init" , trainable=self.learn_init, regularizable=False)

        self.inits += [cell_init, hid_init]
        
        W_in_stacked = T.concatenate([W_in_to_ingate, W_in_to_forgetgate, W_in_to_cell, W_in_to_outgate], axis=1)
        W_hid_stacked = T.concatenate([W_hid_to_ingate, W_hid_to_forgetgate, W_hid_to_cell, W_hid_to_outgate], axis=1)
        b_stacked = T.concatenate([b_ingate, b_forgetgate, b_cell, b_outgate], axis=0)

        self.non_seqs += [W_in_stacked, W_hid_stacked, b_stacked]

        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        def step(input_n, cell_previous, hid_previous, *args):
            input_n = T.dot(input_n, W_in_stacked) + b_stacked
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                gates = theano.gradient.grad_clip(gates, -self.grad_clipping, self.grad_clipping)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            cell = forgetgate * cell_previous + ingate * cell_input
            outgate = self.nonlinearity_outgate(outgate)

            if self.n_gac > 0:
                cell = T.set_subtensor(cell[:, :self.n_gac], cell[:, :self.n_gac].mean(axis=0))

            hid = outgate * self.nonlinearity(cell)
            return [cell, hid]

        self.steps += [step]

    def __init__(self, incoming, num_units, function, n_steps, n_gac=0, use_function_values=False, n_layers=1, preprocess_input=True, p=10., scale_output=1.0,
                 ingate=L.layers.Gate(), 
                 forgetgate=L.layers.Gate(),
                 outgate=L.layers.Gate(), 
                 nonlinearity=L.nonlinearities.tanh,
                 cell=L.layers.Gate(W_cell=None, nonlinearity=L.nonlinearities.tanh),
                 cell_init=L.init.Constant(0.), 
                 hid_init=L.init.Constant(0.), 
                 gradient_steps=-1,
                 learn_init=False, grad_clipping=0, only_return_final=False, **kwargs):

        incomings = [incoming]
        super(LSTMOptimizerLayer, self).__init__(incomings, **kwargs)

        self.nonlinearity = nonlinearity or L.nonlinearities.identity

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        
        self.gradient_steps = gradient_steps
        
        self.func = function
        self.n_steps = n_steps
        self.n_gac = n_gac
        self.use_function_values = use_function_values
        self.preprocess_input = preprocess_input
        self.p = p 
        self.scale_output = scale_output

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        W_hidden_to_output = L.init.GlorotUniform().sample((num_units, 1))
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, 1), name='W_hidden_to_output', regularizable=False)

        self.inits = []
        self.steps = []
        self.non_seqs = []

        self.n_layers = n_layers
        num_inp = 1 + int(self.use_function_values) + int(self.preprocess_input)
        for _ in range(n_layers):
            self.add_lstm_step(num_inp, ingate, forgetgate, outgate, cell, cell_init,  hid_init)
            num_inp = num_units

        self.num_inp = num_inp

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return self.n_coords,
        else:
            return self.n_steps, self.n_coords

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]

        # Because scan iterates over the first dimension we dimshuffle to
        n_coords = input.shape[0]

        def step_opt(theta_previous, *args):
            func = self.func(theta_previous)
            input_n = theano.gradient.disconnected_grad(theano.grad(func, theta_previous))

            grad = input_n

            input_n = input_n.dimshuffle(0, 'x')

            if self.preprocess_input:
                lognorm = T.switch(T.abs_(input_n) > T.exp(-self.p), T.log(T.abs_(input_n)) / self.p, T.ones_like(input_n) * (-1))
                sign = T.switch(T.abs_(input_n) > T.exp(-self.p), T.sgn(input_n), T.exp(self.p) * input_n)

                input_n = T.concatenate([lognorm, sign], axis=1)

            if self.use_function_values:
                input_n = T.concatenate([input_n, T.ones_like(input_n) * func], axis=1)

            args = args[:-len(self.non_seqs)]

            cells = args[::2]
            hids = args[1::2]

            new_args = []

            for step, cell_previous, hid_previous in zip(self.steps, cells, hids):
                cell, hid = step(input_n, cell_previous, hid_previous)
                input_n = hid

                new_args += [cell, hid]

            dtheta = hid.dot(self.W_hidden_to_output).dimshuffle(0)

            theta = theta_previous + dtheta * self.scale_output
            return [theta, func, grad] + new_args

        # Dot against a 1s vector to repeat to shape (num_batch, num_units)
        inits = []
        ones = T.ones((n_coords, 1))

        for init in self.inits:
            inits.append(T.dot(ones, init))

        theta_init = input

        # The hidden-to-hidden weight matrix is always used in step
#        non_seqs = [W_hid_stacked]
#        non_seqs += [W_in_stacked, b_stacked, self.W_hidden_to_output]
        non_seqs = self.non_seqs + [self.W_hidden_to_output]

        theta_out, loss_out, grads_out = theano.scan(
            fn=step_opt,
            outputs_info=[theta_init, None, None] + inits,
            non_sequences=non_seqs,
            n_steps=self.n_steps,
            truncate_gradient=self.gradient_steps,
            strict=True)[0][:3]

        return theta_out, loss_out, grads_out

    def get_updates(self, loss, params):
        theta = T.concatenate([x.flatten() for x in params])
        #n_coords = theta.shape[0]
        n_coords = np.sum([np.prod(x.get_value().shape) for x in params])
        
        cells = []
        hids = []

        for _ in range(self.n_layers):
            cell_init = theano.shared(np.zeros((n_coords, self.num_units), dtype=np.float32))
            hid_init  = theano.shared(np.zeros((n_coords, self.num_units), dtype=np.float32))

            cells.append(cell_init)
            hids.append(hid_init)

        updates = OrderedDict()

        grads = theano.grad(loss, params)
        input_n = T.concatenate([x.flatten() for x in grads]).dimshuffle(0, 'x')
        if self.preprocess_input:
            lognorm = T.switch(T.abs_(input_n) > T.exp(-self.p), T.log(T.abs_(input_n)) / self.p, T.ones_like(input_n) * (-1))
            sign = T.switch(T.abs_(input_n) > T.exp(-self.p), T.sgn(input_n), T.exp(self.p) * input_n)

            input_n = T.concatenate([lognorm, sign], axis=1)

        if self.use_function_values:
            input_n = T.concatenate([input_n, T.ones_like(input_n) * func], axis=1)

        for step, cell_previous, hid_previous in zip(self.steps, cells, hids):
            cell, hid = step(input_n, cell_previous, hid_previous)
            input_n = hid

            updates[cell_previous] = cell
            updates[hid_previous] = hid

        dtheta = hid.dot(self.W_hidden_to_output).dimshuffle(0)
        new_theta = theta + dtheta * self.scale_output

        cur_pos = 0
        for p in params:
            next_pos = cur_pos + np.prod(p.get_value().shape)
            updates[p] = T.reshape(new_theta[cur_pos:next_pos], p.shape)
            cur_pos = next_pos

        return updates
