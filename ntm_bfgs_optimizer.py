import time
import numpy as np

import theano
import theano.tensor as T
import lasagne as L

from recurrence import Recurrence
from lstm_step import LSTMStep
from ntm_opt_step import NTMOptStep
from index_layer import IndexLayer
from grad_layer import GradLayer
from theta_step import ThetaStep
from ntm_preprocess_layer import NTMPreprocessLayer

from trainable_optimizer import TrainableOptimizer
        
class NTM_BFGS_Optimizer(TrainableOptimizer):
    def __init__(self, num_units, 
                 function, 
                 n_layers=2, n_gac=0,
                 use_function_values=False, scale_output=1.0,
                 preprocess_input=True, p=10., loglr=True,
                 p_drop_grad=0.0 , fix_drop_grad_over_time=False,
                 p_drop_delta=0.0, fix_drop_delta_over_time=False,
                 p_drop_coord=0.0, fix_drop_coord_over_time=False,
                 gradient_steps=-1, grad_clipping=0,
                 params_input=None, input_var=None, n_steps=None, 
                 **kwargs):

        super(NTM_BFGS_Optimizer, self).__init__()

        self.num_units = num_units
        input_var = input_var or T.vector()
        n_steps = n_steps or T.iscalar()

        l_input   = L.layers.InputLayer(shape=(None,), input_var=input_var, name='theta_in')
        l_read_in = L.layers.InputLayer(shape=(None,), name='read_in')
        l_mem_in  = L.layers.InputLayer(shape=(None, None), name='mem_in')

        l_grad = GradLayer(l_input, function)
        l_func = IndexLayer(l_grad, 1)

        l_lstm = IndexLayer(l_grad, 0)
        #l_lstm = L.layers.dropout(l_input, p=p_drop_grad) # fix over time
        l_lstm = NTMPreprocessLayer(l_lstm, l_read_in, l_mem_in, preprocess_input=preprocess_input, p=p, use_function_values=use_function_values, name='ntm_prep')

        recurrent_connections = { }

        for i in range(n_layers):
            l_lstm_cell = L.layers.InputLayer(shape=(None, num_units), name='cell_{}_in'.format(i))
            l_lstm_hid  = L.layers.InputLayer(shape=(None, num_units), name='hid_{}_in'.format(i))
            l_lstm = LSTMStep(l_lstm, l_lstm_cell, l_lstm_hid, num_units=num_units, n_gac=n_gac, grad_clipping=grad_clipping, name='lstm_{}'.format(i))

            l_cell = IndexLayer(l_lstm, 0)
            l_hid  = IndexLayer(l_lstm, 1)

            l_lstm = l_hid

            recurrent_connections[l_lstm_cell] = l_cell
            recurrent_connections[l_lstm_hid]  = l_hid

        l_opt = NTMOptStep(l_lstm, l_mem_in, num_units, loglr=loglr)
        l_read = IndexLayer(l_opt, 1, name='index_read')
        l_mem  = IndexLayer(l_opt, 2, name='index_mem')
        l_opt  = IndexLayer(l_opt, 0)

        recurrent_connections[l_read_in] = l_read
        recurrent_connections[l_mem_in] = l_mem
        #l_opt = L.layers.dropout(l_opt, p=p_drop_delta) # fix over time

        l_opt = ThetaStep(l_input, l_opt, input_var, scale_output=scale_output)
        recurrent_connections[l_input] = l_opt

        l_rec = Recurrence(
            l_input,
            n_steps=n_steps,
            recurrent_connections=recurrent_connections,
            outputs=[l_opt, l_func],
            gradient_steps=gradient_steps,
        )
        
        self.l_opt = l_opt
        self.l_rec = l_rec

        self.input_var = input_var
        self.n_steps = n_steps

    def get_params(self):
        return L.layers.get_all_params(self.l_rec)

    def get_params_values(self):
        return L.layers.get_all_param_values(self.l_rec)

    def get_output(self, **kwargs):
        return L.layers.get_output(self.l_rec, **kwargs)

    def get_net(self):
        return self.l_rec
    
    #def get_updates(self, loss, params):
    #    theta = T.concatenate([x.flatten() for x in params])
    #    #n_coords = theta.shape[0]
    #    n_coords = np.sum([np.prod(x.get_value().shape) for x in params])
    #    
    #    cells = []
    #    hids = []

    #    for _ in range(self.n_layers):
    #        cell_init = theano.shared(np.zeros((n_coords, self.num_units), dtype=np.float32))
    #        hid_init  = theano.shared(np.zeros((n_coords, self.num_units), dtype=np.float32))

    #        cells.append(cell_init)
    #        hids.append(hid_init)

    #    updates = OrderedDict()

    #    grads = theano.grad(loss, params)
    #    input_n = T.concatenate([x.flatten() for x in grads]).dimshuffle(0, 'x')
    #    if self.preprocess_input:
    #        lognorm = T.switch(T.abs_(input_n) > T.exp(-self.p), T.log(T.abs_(input_n)) / self.p, T.ones_like(input_n) * (-1))
    #        sign = T.switch(T.abs_(input_n) > T.exp(-self.p), T.sgn(input_n), T.exp(self.p) * input_n)

    #        input_n = T.concatenate([lognorm, sign], axis=1)

    #    if self.use_function_values:
    #        input_n = T.concatenate([input_n, T.ones_like(input_n) * func], axis=1)

    #    for step, cell_previous, hid_previous in zip(self.steps, cells, hids):
    #        cell, hid = step(input_n, cell_previous, hid_previous)
    #        input_n = hid

    #        updates[cell_previous] = cell
    #        updates[hid_previous] = hid

    #    dtheta = hid.dot(self.W_hidden_to_output).dimshuffle(0)
    #    new_theta = theta + dtheta * self.scale_output

    #    cur_pos = 0
    #    for p in params:
    #        next_pos = cur_pos + np.prod(p.get_value().shape)
    #        updates[p] = T.reshape(new_theta[cur_pos:next_pos], p.shape)
    #        cur_pos = next_pos

    #    return updates
