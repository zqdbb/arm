'''Numpy-based FPS replacement for TF1.x custom ops.
The original tf_sampling_so.so was compiled for TF1.x and won't load in TF2.
This provides the same API using numpy + tf.compat.v1.py_func.
'''

import tensorflow as tf
import numpy as np


def _farthest_point_sample_np(inp, npoint):
    """Pure numpy FPS implementation.

    Args:
        inp: (batch_size, ndataset, 3) float32 numpy array
        npoint: int, number of points to sample

    Returns:
        (batch_size, npoint) int32 array of indices
    """
    if isinstance(npoint, np.ndarray):
        npoint = int(npoint)
    batch_size, ndataset = inp.shape[:2]
    idx = np.zeros((batch_size, npoint), dtype=np.int32)
    distances = np.ones((batch_size, ndataset), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, ndataset, size=(batch_size,))
    batch_indices = np.arange(batch_size)

    for i in range(npoint):
        idx[:, i] = farthest
        centroid = inp[batch_indices, farthest, :].reshape(batch_size, 1, 3)
        dist = np.sum((inp - centroid) ** 2, axis=-1)
        np.minimum(distances, dist, out=distances)
        farthest = np.argmax(distances, axis=-1)

    return idx


def _gather_point_np(inp, idx):
    """Pure numpy gather_point implementation.

    Args:
        inp: (batch_size, ndataset, C) float32 numpy array
        idx: (batch_size, npoint) int32 array of indices

    Returns:
        (batch_size, npoint, C) float32 array
    """
    batch_size = inp.shape[0]
    npoint = idx.shape[1]
    batch_indices = np.tile(np.arange(batch_size).reshape(batch_size, 1), (1, npoint))
    return inp[batch_indices, idx, :]


def farthest_point_sample(npoint, inp):
    """TF op wrapper for numpy FPS.

    Args:
        npoint: int32 scalar, number of points to sample
        inp: (batch_size, ndataset, 3) float32 tensor

    Returns:
        (batch_size, npoint) int32 tensor of indices
    """
    result = tf.compat.v1.py_func(_farthest_point_sample_np, [inp, npoint], tf.int32)
    # Help TF infer the shape
    n = npoint if isinstance(npoint, int) else None
    result.set_shape([inp.shape[0] if inp.shape[0].value is not None else None, n])
    return result


def gather_point(inp, idx):
    """TF op wrapper for numpy gather_point.

    Args:
        inp: (batch_size, ndataset, C) float32 tensor
        idx: (batch_size, npoint) int32 tensor

    Returns:
        (batch_size, npoint, C) float32 tensor
    """
    result = tf.compat.v1.py_func(_gather_point_np, [inp, idx], tf.float32)
    npoint = idx.shape[1].value if idx.shape[1].value is not None else None
    channels = inp.shape[2].value if inp.shape[2].value is not None else None
    result.set_shape([inp.shape[0].value if inp.shape[0].value is not None else None,
                      npoint, channels])
    return result


def gather_point_grad(inp, idx, out_g):
    """Gradient for gather_point (stub, not needed for inference)."""
    raise NotImplementedError('gather_point_grad is not implemented in numpy fallback')


def prob_sample(inp, inpr):
    raise NotImplementedError('prob_sample is not implemented in numpy fallback')
