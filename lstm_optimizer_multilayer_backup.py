import theano
import theano.tensor as T
import lasagne as L

class NoGrads(theano.gof.Op):
    __props__ = ()
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x
    def infer_shape(self, node, i0_shapes):
        return i0_shapes
    def grad(self, inputs, output_grads):
        return [0 * output_grads[0]]

no_grads = NoGrads()

class LSTMStep(L.layers.MergeLayer):
    def __init__(self, incoming, num_inp, num_units, n_gac=0, ingate=L.layers.Gate(), forgetgate=L.layers.Gate(), 
                       cell=L.layers.Gate(W_cell=None, nonlinearity=L.nonlinearities.tanh),
                       outgate=L.layers.Gate(), nonlinearity=L.nonlinearities.tanh,
                       cell_init=L.init.Constant(0.), hid_init=L.init.Constant(0.),
                       learn_init=False, grad_clipping=0, **kwargs):
        incomings = [incoming]
        super(LSTMStep, self).__init__(incomings, **kwargs)

        self.nonlinearity = nonlinearity or L.nonlinearities.identity

        self.learn_init = learn_init
        self.num_units = num_units
        self.n_gac = n_gac
        self.grad_clipping = grad_clipping
        
        input_shape = self.input_shapes[0]

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a L.layers.Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inp, num_units), name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units), name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,), name="b_{}".format(gate_name), regularizable=False),
                    gate.nonlinearity)

        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate, self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate, self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell, self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate, self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        self.cell_init = self.add_param(cell_init, (1, self.num_units), name="cell_init", trainable=learn_init, regularizable=False)
        self.hid_init  = self.add_param(hid_init , (1, self.num_units), name="hid_init" , trainable=learn_init, regularizable=False)

        W_in_stacked = T.concatenate([self.W_in_to_ingate, self.W_in_to_forgetgate, self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        W_hid_stacked = T.concatenate([self.W_hid_to_ingate, self.W_hid_to_forgetgate, self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        b_stacked = T.concatenate([self.b_ingate, self.b_forgetgate, self.b_cell, self.b_outgate], axis=0)

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

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
        
        self.step = step
    
    def get_output_shape_for(self, input_shapes):
        return (self.num_units,)

class LSTMOptimizerLayer(L.layers.MergeLayer):
    def __init__(self, incoming, num_units, function, n_steps, 
                 n_gac=0, use_function_values=False, n_layers=1, 
                 gradient_steps=-1, grad_clipping=0, **kwargs):

        super(LSTMOptimizerLayer, self).__init__([incoming], **kwargs)
        
        W_hidden_to_output = L.init.GlorotUniform().sample((num_units, 1))
        self.W_hidden_to_output = self.add_param(W_hidden_to_output, (num_units, 1), name='W_hidden_to_output', regularizable=False)

        self.func = function
        self.n_steps = n_steps
        self.n_gac = n_gac
        self.use_function_values = use_function_values

        self.gradient_steps = gradient_steps
        
        num_inp = 1 + int(use_function_values)
        self.n_layers = n_layers

        inp_layer = incoming

        self.layers = []
        for _ in range(n_layers):
            inp_layer = LSTMStep(inp_layer, num_inp, num_units=num_units, grad_clipping=grad_clipping, n_gac=n_gac)
            self.layers.append(inp_layer)

    def get_output_shape_for(self, input_shapes):
        return self.n_steps, self.n_coords

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]

        n_coords = input.shape[0]

        def step_opt(theta_previous, *args):
            func = self.func(theta_previous)

            input_n = no_grads(theano.grad(func, theta_previous))

            if self.use_function_values:
                input_n = T.concatenate([input_n.dimshuffle(0, 'x'), T.ones((input_n.shape[0], 1)) * func], axis=1)
            else:
                input_n = input_n.dimshuffle(0, 'x')

            cells = args[::2]
            hids = args[1::2]

            new_args = []

            for layer, cell_previous, hid_previous in zip(self.layers, cells, hids):
                cell, hid = layer.step(input_n, cell_previous, hid_previous)
                input_n = hid

                new_args += [cell, hid]

            theta = theta_previous + hid.dot(self.W_hidden_to_output).dimshuffle(0)
            return [theta, func] + new_args

        theta_init = input

        # The hidden-to-hidden weight matrix is always used in step
#        non_seqs = [W_hid_stacked]
#        non_seqs += [W_in_stacked, b_stacked, self.W_hidden_to_output]
            
        ones = T.ones((n_coords, 1))

        inits = []
        for layer in self.layers:
            cell_init = T.dot(ones, layer.cell_init)
            hid_init = T.dot(ones, layer.hid_init)
            inits += [cell_init, hid_init]

        cell_out, hid_out, theta_out, loss_out = theano.scan(
            fn=step_opt,
            outputs_info=[theta_init, None] + inits,
#            non_sequences=non_seqs,
            n_steps=self.n_steps,
            truncate_gradient=self.gradient_steps,
            strict=True)[0]

        return theta_out, loss_out
