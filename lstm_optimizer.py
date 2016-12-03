import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class LSTMOptimizerLayer(L.layers.MergeLayer):
    def __init__(self, incoming, num_units, function, n_steps, 
                 n_gac=0, use_function_values=False, n_layers=1, 
                 preprocess_input=True, p=10., scale_output=1.0,
                 loglr=True,
                 p_drop_grad=0.0 , fix_drop_grad_over_time=False,
                 p_drop_delta=0.0, fix_drop_delta_over_time=False,
                 p_drop_coord=0.0, fix_drop_coord_over_time=False,
                 ingate=L.layers.Gate(), 
                 forgetgate=L.layers.Gate(), 
                 outgate=L.layers.Gate(), 
                 nonlinearity=L.nonlinearities.tanh,
                 cell=L.layers.Gate(W_cell=None, nonlinearity=L.nonlinearities.tanh),
                 cell_init=L.init.Constant(0.), hid_init=L.init.Constant(0.), 
                 gradient_steps=-1,
                 params_input=None,
                 learn_init=False, grad_clipping=0, only_return_final=False, **kwargs):

        incomings = [incoming]

        self.params_input_index = -1
        if params_input is not None:
            self.params_input_index = len(incomings)
            incomings.append(params_input)

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
        self.loglr = loglr

        self.preprocess_input = preprocess_input
        self.p = p 
        self.scale_output = scale_output
        
        self._srng = RandomStreams(L.random.get_rng().randint(1, 2147462579))

        self.p_drop_coord = p_drop_coord
        self.p_drop_grad = p_drop_grad
        self.p_drop_delta = p_drop_delta
        self.fix_drop_delta_over_time = fix_drop_delta_over_time
        self.fix_drop_coord_over_time = fix_drop_coord_over_time
        self.fix_drop_grad_over_time = fix_drop_grad_over_time

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        num_out = 1 + int(loglr)
        W_hidden_to_output = L.init.GlorotUniform().sample((num_units, num_out))
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, num_out), name='W_hidden_to_output', regularizable=False)

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
        if self.params_input_index != -1:
            params_input = inputs[self.params_input_index]

        # Because scan iterates over the first dimension we dimshuffle to
        n_coords = input.shape[0]

        deterministic = kwargs.get('deterministic', False)

        random_mask = lambda shape, p: self._srng.binomial(shape, p=p, dtype=theano.config.floatX)

        def step(theta_previous, input_n, *args):
            if not deterministic and self.p_drop_grad > 0.0:
                grad_mask = self.grad_mask if self.fix_grad_coord_over_time else random_mask(input_n.shape, 1. - self.p_drop_grad)
                input_n = input_n * grad_mask
            else:
                input_n = input_n * (1. - self.p_drop_grad)
                
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
            
            if not deterministic and self.p_drop_coord > 0.0:
                coord_mask = self.coord_mask if self.fix_drop_coord_over_time else random_mask(input_n.shape[:1], 1. - self.p_drop_coord)

            for lstm_step, cell_previous, hid_previous in zip(self.steps, cells, hids):
                cell, hid = lstm_step(input_n, cell_previous, hid_previous)
                if not deterministic and self.p_drop_coord > 0.0:
                    cell = T.switch(coord_mask.dimshuffle(0, 'x'), cell, cell_previous)
                    hid  = T.switch(coord_mask.dimshuffle(0, 'x'), hid, hid_previous)
                input_n = hid

                new_args += [cell, hid]

            dtheta = hid.dot(self.W_hidden_to_output)
            if self.loglr:
                dtheta = T.exp(dtheta[:, 1]) * dtheta[:, 0]
            else:
                dtheta = dtheta.dimshuffle(0)

            if not deterministic and self.p_drop_delta > 0.0:
                delta_mask = self.delta_mask if self.fix_drop_delta_over_time else random_mask(dtheta.shape, 1. - self.p_drop_delta)
                dtheta = dtheta * delta_mask
            else:
                dtheta = dtheta * (1. - self.p_drop_coord) * (1. - self.p_drop_delta)

            theta = theta_previous + dtheta * self.scale_output
            if not deterministic and self.p_drop_coord > 0.0:
                #coord_mask = self.coord_mask if self.fix_drop_coord_over_time else random_mask(input_n.shape[:1], 1. - self.p_drop_coord)
                theta = T.switch(coord_mask, theta, theta_previous)
            return [theta] + new_args
            
        def step_opt(theta_previous, *args):
            func = self.func(theta_previous)
            input_n = theano.gradient.disconnected_grad(theano.grad(func, theta_previous))

            grad = input_n
            out = step(theta_previous, grad, *args)
            return [out[0], func] + out[1:]

        def step_opt_params(params_cur, theta_previous, *args):
            func = self.func(theta_previous, params_cur)
            input_n = theano.gradient.disconnected_grad(theano.grad(func, theta_previous))
            
            grad = input_n
            out = step(theta_previous, grad, *args)
            return [out[0], func] + out[1:]

        # Dot against a 1s vector to repeat to shape (num_batch, num_units)
        inits = []
        ones = T.ones((n_coords, 1))

        for init in self.inits:
            inits.append(T.dot(ones, init))

        theta_init = input
        non_seqs = self.non_seqs + [self.W_hidden_to_output]

        if not deterministic:
            if self.p_drop_grad > 0.0 and self.fix_drop_grad_over_time:
                self.grad_mask = random_mask(input.shape[:1], 1. - self.p_drop_grad)
                non_seqs += [self.grad_mask]

            if self.p_drop_delta > 0.0 and self.fix_drop_delta_over_time:
                self.delta_mask = random_mask(input.shape[:1], 1. - self.p_drop_delta)
                non_seqs += [self.delta_mask]

            if self.p_drop_coord > 0.0 and self.fix_drop_coord_over_time:
                self.coord_mask = random_mask(input.shape[:1], 1. - self.p_drop_coord)
                non_seqs += [self.coord_mask]

        if self.params_input_index != -1:
            seqs = [params_input]
            step_fun = step_opt_params
        else:
            seqs = None
            step_fun = step_opt

        scan_out = theano.scan(
            sequences=seqs,
            fn=step_fun,
            outputs_info=[theta_init, None] + inits,
            non_sequences=non_seqs,
            n_steps=self.n_steps,
            truncate_gradient=self.gradient_steps,
            strict=True)

        theta_out, loss_out = scan_out[0][:2]
        updates = scan_out[1]

        return theta_out, loss_out, updates

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

