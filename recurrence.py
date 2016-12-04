import numpy as np

import theano
import theano.tensor as T
import lasagne as L
from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.printing import Print as TPP

class Recurrence(L.layers.MergeLayer):
    def __init__(self, incoming, n_steps,
                 recurrent_connections, 
                 outputs,
                 gradient_steps=-1, **kwargs):
        
        incomings = [incoming]
        super(Recurrence, self).__init__(incomings, **kwargs)
        
        self.n_steps = n_steps
        self.recurrent_connections = recurrent_connections
        self.outputs = outputs
        self.gradient_steps = gradient_steps

    def get_params(self, **tags):
        #return L.layers.get_all_params(self.step, **tags)
        return sum((L.layers.get_all_params(out, **tags) for out in self.recurrent_connections.values()), [])

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        num_batch = input.shape[0]
        
        num_inits = len(self.recurrent_connections)
        input_layers, output_layers = [list(x) for x in zip(*self.recurrent_connections.items())]

        #layers = L.layers.get_all_layers(self.step)
        #inits = [l.get_recurrent_inits(num_batch) for l in layers if hasattr(l, 'get_recurrent_inits')]

        layer_inits = []
        for l in output_layers:
            for ll in L.layers.get_all_layers(l):
                if hasattr(ll, 'get_recurrent_inits'):
                    layer_inits += ll.get_recurrent_inits(num_batch)
        layer_inits = dict(layer_inits)

        inits = []
        for l in input_layers:
            inits.append(layer_inits[l])

        #inits = [l.get_recurrent_inits(num_batch, input) for l in output_layers if hasattr(l, 'get_recurrent_inits')]
        #inits = [item for sublist in inits for item in sublist]

        def step(*args):
            args = list(args)    
            
            previous_values = args[:num_inits]
            weights = args[num_inits:]
            
            step_map = {
                input_layer: input 
                for input_layer, input 
                in zip(input_layers, previous_values)
            }
            #step_map = dict(zip(input_layers, previous_values))
            for inp in step_map:
                print(inp.name, step_map[inp].ndim)

            outputs = L.layers.get_output(self.outputs + output_layers, step_map)

            return outputs

        non_seqs = [p for p in self.get_params() if 'init' not in p.name]

        out, updates = theano.scan(
            fn=step,
            outputs_info=[None] * len(self.outputs) + inits,
            non_sequences=non_seqs,
            truncate_gradient=self.gradient_steps,
            n_steps=self.n_steps,
            strict=True)

        return out[:len(self.outputs)], updates
