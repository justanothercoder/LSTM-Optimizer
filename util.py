import numpy as np

def iterate_minibatches(*arrays, **kwargs):
    batch_size = kwargs.get("batch_size", 100)
    shuffle = kwargs.get("shuffle", True)

    if shuffle:
        indices = np.arange(len(arrays[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(arrays[0]) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield [arr[excerpt] for arr in arrays]
