import time
import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

from recurrence import Recurrence
from lstm_step import LSTMStep
from opt_step import OptStep
from index_layer import IndexLayer
from grad_layer import GradLayer
from theta_step import ThetaStep
from preprocess_layer import PreprocessLayer

class LSTM_Optimizer:
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

        self.num_units = num_units
        input_var = input_var or T.vector()
        n_steps = n_steps or T.iscalar()

        l_input = L.layers.InputLayer(shape=(None,), input_var=input_var)
        l_grad = GradLayer(l_input, function)
        l_func = IndexLayer(l_grad, 1)

        l_lstm = IndexLayer(l_grad, 0)
        #l_lstm = L.layers.dropout(l_input, p=p_drop_grad) # fix over time
        l_lstm = PreprocessLayer(l_lstm, preprocess_input=preprocess_input, p=p, use_function_values=use_function_values)

        recurrent_connections = { }

        for i in range(n_layers):
            l_lstm_cell = L.layers.InputLayer(shape=(None, num_units))
            l_lstm_hid = L.layers.InputLayer(shape=(None, num_units))
            l_lstm = LSTMStep(l_lstm, l_lstm_cell, l_lstm_hid, num_units=num_units, n_gac=n_gac, grad_clipping=grad_clipping)

            l_cell = IndexLayer(l_lstm, 0)
            l_hid  = IndexLayer(l_lstm, 1)

            l_lstm = l_hid

            recurrent_connections[l_lstm_cell] = l_cell
            recurrent_connections[l_lstm_hid]  = l_hid

        l_opt = OptStep(l_lstm, num_units, loglr=loglr)
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
        
    def reset_network(self):
        L.layers.set_all_param_values(self.l_rec, self.params_init)

    def optimizer_loss(self, loss_history, loss_type='sum', M=np):
        if loss_type == 'sum':
            loss = loss_history.sum()
        elif loss_type == 'prod':
            loss = M.log(loss_history).sum()
        elif loss_type == 'weighted_prod':
            loss = (M.log(loss_history) * 0.9 ** M.arange(loss_history.shape[0])[::-1]).sum()
        elif loss_type == 'norm_sum':
            loss = loss_history[1:].sum() / loss_history[0]
        elif loss_type == 'rel_sum':
            loss = (loss_history[1:] / loss_history[:-1]).sum()

        return loss

    def prepare(self, func_params, start_lr=0.01, lambd=1e-5, loss_type='sum'):
        self.loss_type = loss_type
        (theta_history, loss_history), scan_updates = L.layers.get_output(self.l_rec)

        loss = self.optimizer_loss(loss_history, loss_type, M=T)
        loss += lambd * L.regularization.regularize_network_params(self.l_rec, L.regularization.l2)
                
        self.lr = theano.shared(np.array(0.01, dtype=np.float32))

        params = L.layers.get_all_params(self.l_rec)
        updates = L.updates.adam(loss, params, learning_rate=self.lr)
        updates.update(scan_updates)
        
        t = time.time()
        self.loss_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history, loss_history], allow_input_downcast=True, updates=scan_updates)
        print("Time compiling loss_fn: {}".format(time.time() - t))
        
        t = time.time()
        self.train_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history, loss_history], updates=updates, allow_input_downcast=True)
        print("Time compiling train_fn: {}".format(time.time() - t))
        
        (theta_history_det, loss_history_det), scan_updates_det = L.layers.get_output(self.l_rec, deterministic=True)
        self.loss_det_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history_det, loss_history_det], allow_input_downcast=True, updates=scan_updates_det)
        
        self.params_init = L.layers.get_all_param_values(self.l_rec)
        
    def train(self, sample_function, n_iter=100, n_epochs=50, batch_size=100, decay_rate=0.96, **kwargs):
        optimizer_loss = []
        optimizer_moving_loss = []
        moving_loss = None

        for epoch in range(n_epochs):
            t = time.time()    

            training_loss_history = []
            for j in range(batch_size):
                theta, params = sample_function()
 
                theta_history, loss_history = self.train_fn(theta, n_iter, *params)
                training_loss_history.append(loss_history)

                loss = self.optimizer_loss(loss_history, self.loss_type)
                optimizer_loss.append(loss)

                if moving_loss is None:
                    moving_loss = loss
                else:
                    moving_loss = 0.9 * moving_loss + 0.1 * loss

                optimizer_moving_loss.append(moving_loss)

            print("Epoch number {}".format(epoch))
            print("\tTime: {}".format(time.time() - t))
            print("\tOptimizer loss: {}".format(loss))
            print("\tMedian final loss: {}".format(np.median(training_loss_history, axis=0)[-1]))

            self.lr.set_value((self.lr.get_value() * decay_rate).astype(np.float32))
            
    def optimize(self, theta, func_params, n_iter):
        return self.loss_fn(theta, n_iter, *func_params)

    #def get_updates(self, loss, params):
    #    theta = T.concatenate([x.flatten() for x in params])
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

