import time
import numpy as np

import theano
import theano.tensor as T
import lasagne as L

class TrainableOptimizer:
    def __init__(self):
        pass

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
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

        return loss
    
    def prepare(self, func_params, start_lr=0.01, lambd=1e-5, loss_type='sum'):
        self.loss_type = loss_type
        #(theta_history, loss_history), scan_updates = L.layers.get_output(self.l_rec)
        (theta_history, loss_history), scan_updates = self.get_output()

        loss = self.optimizer_loss(loss_history, loss_type, M=T)
        #loss += lambd * L.regularization.regularize_network_params(self.l_rec, L.regularization.l2)
        loss += lambd * L.regularization.regularize_network_params(self.get_net(), L.regularization.l2)
                
        self.lr = theano.shared(np.array(0.01, dtype=np.float32))

        #params = L.layers.get_all_params(self.l_rec)
        params = self.get_params()
        updates = L.updates.adam(loss, params, learning_rate=self.lr)
        updates.update(scan_updates)
        
        t = time.time()
        self.loss_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history, loss_history], allow_input_downcast=True, updates=scan_updates)
        print("Time compiling loss_fn: {}".format(time.time() - t))
        
        t = time.time()
        self.train_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history, loss_history], updates=updates, allow_input_downcast=True)
        print("Time compiling train_fn: {}".format(time.time() - t))
        
        #(theta_history_det, loss_history_det), scan_updates_det = L.layers.get_output(self.l_rec, deterministic=True)
        (theta_history_det, loss_history_det), scan_updates_det = self.get_output(deterministic=True)
        self.loss_det_fn = theano.function([self.input_var, self.n_steps] + func_params, [theta_history_det, loss_history_det], allow_input_downcast=True, updates=scan_updates_det)
        
        #self.params_init = L.layers.get_all_param_values(self.l_rec)
        self.params_init = self.get_params_values()
        

    def train(self, sample_function, n_iter=100, n_epochs=50, batch_size=100, decay_rate=0.96, verbose=True, **kwargs):
        optimizer_loss = []

        for epoch in range(n_epochs):
            t = time.time()    

            training_loss_history = []
            for j in range(batch_size):
                theta, params = sample_function()
 
                theta_history, loss_history = self.train_fn(theta, n_iter, *params)
                training_loss_history.append(loss_history)

                loss = self.optimizer_loss(loss_history, self.loss_type)
                optimizer_loss.append(loss)

            if verbose:
                print("Epoch number {}".format(epoch))
                print("\tTime: {}".format(time.time() - t))
                print("\tOptimizer loss: {}".format(loss))
                print("\tMedian final loss: {}".format(np.median(training_loss_history, axis=0)[-1]))

            self.lr.set_value((self.lr.get_value() * decay_rate).astype(np.float32))

        return optimizer_loss
            
    def optimize(self, theta, func_params, n_iter):
        return self.loss_fn(theta, n_iter, *func_params)

    def reset_network(self):
        #L.layers.set_all_param_values(self.l_rec, self.params_init)
        L.layers.set_all_param_values(self.get_net(), self.params_init)

    def get_params(self):
        raise NotImplementedError

    def get_params_values(self):
        raise NotImplementedError

    def get_output(self, **kwargs):
        raise NotImplementedError
