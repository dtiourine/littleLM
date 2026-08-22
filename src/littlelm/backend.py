import numpy as numpy

try:
    import cupy as cupy
except ImportError:  # CuPy is optional for CPU-only environments.
    cupy = None


def _cuda_available() -> bool:
    if cupy is None:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


GPU_AVAILABLE = _cuda_available()
xp = cupy if GPU_AVAILABLE else numpy


def to_cpu(array):
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return numpy.asarray(array)


def scalar(value):
    return value.item() if hasattr(value, "item") else value
