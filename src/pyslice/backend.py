# backend.py - Backend abstraction layer for NumPy/PyTorch support
import numpy as np


# First, try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Check for backend override via environment variable
import os
backend_override = os.environ.get('PYSLICE_BACKEND', '').lower()

if backend_override == 'numpy' or not TORCH_AVAILABLE:
    xp = np
    TORCH_BACKEND = False
else:
    xp = torch
    TORCH_BACKEND = True


def device_and_precision(device_spec=None):
    """
    Determine the device and precision (dtype) to use.
    
    Args:
        device_spec: Device specification ('cpu', 'cuda', 'mps', or None for auto)
    
    Returns:
        Tuple of (device, float_dtype, complex_dtype)
    """
    if not TORCH_BACKEND:
        return None, np.float64, np.complex128
    
    # Check for device override via environment variable
    device_override = os.environ.get('PYSLICE_DEVICE', '').lower()
    if device_override:
        device_spec = device_override
    
    # Auto-detect device if not specified
    if device_spec is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_spec)
    
    # Use float32 for MPS (doesn't support float64), float64 for CPU/CUDA
    if device.type == 'mps':
        float_dtype = torch.float32
        complex_dtype = torch.complex64
    else:
        float_dtype = torch.float64
        complex_dtype = torch.complex128
    
    return device, float_dtype, complex_dtype


# Determine default device and dtypes at import time
DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE = device_and_precision()

# Convenience aliases
float_dtype = DEFAULT_FLOAT_DTYPE
complex_dtype = DEFAULT_COMPLEX_DTYPE


def asarray(arraylike, dtype=None, device=None):
    """Convert array-like object to backend array (NumPy or PyTorch tensor)."""
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if TORCH_BACKEND:
        array = xp.tensor(arraylike, dtype=dtype, device=device)
    else:
        array = xp.asarray(arraylike, dtype=dtype)
    return array


def zeros(dims, dtype=None, device=None, type_match=None):
    """Create a zero-filled array."""
    if type_match is not None:
        # Infer dtype/device from type_match array
        if dtype is None:
            dtype = type_match.dtype
        if device is None and hasattr(type_match, "device"):
            device = type_match.device
        if type(type_match) in [np.memmap, np.ndarray]:
            return np.zeros(dims, dtype=dtype)
    
    # Set defaults
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    
    # Handle string dtypes
    if isinstance(dtype, str):
        dtype = DEFAULT_FLOAT_DTYPE if dtype == "float" else DEFAULT_COMPLEX_DTYPE
    
    # Create array
    if TORCH_BACKEND:
        array = xp.zeros(dims, dtype=dtype, device=device)
    else:
        array = xp.zeros(dims, dtype=dtype)
    return array


def ones(dims, dtype=None, device=None):
    """Create a one-filled array."""
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if TORCH_BACKEND:
        return xp.ones(dims, dtype=dtype, device=device)
    else:
        return xp.ones(dims, dtype=dtype)


def ones_like(x):
    """Create a one-filled array with same shape and dtype as input."""
    return xp.ones_like(x)


def fftfreq(n, d=1.0, dtype=None, device=None):
    """Compute FFT frequency array."""
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if TORCH_BACKEND:
        return xp.fft.fftfreq(n, d, dtype=dtype, device=device)
    else:
        return xp.fft.fftfreq(n, d).astype(dtype)


def fft2(x, **kwargs):
    """2D FFT with kwarg conversion (dim/axis)."""
    if TORCH_BACKEND and "axes" in kwargs:
        kwargs["dim"] = kwargs.pop("axes")
    if not TORCH_BACKEND and "dim" in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    return xp.fft.fft2(x, **kwargs)


def fft(x, **kwargs):
    """1D FFT with kwarg conversion."""
    if TORCH_BACKEND and "axes" in kwargs:
        kwargs["dim"] = kwargs.pop("axes")
    if not TORCH_BACKEND and "dim" in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    return xp.fft.fft(x, **kwargs)


def ifft2(x, **kwargs):
    """2D inverse FFT with kwarg conversion."""
    if TORCH_BACKEND and "axes" in kwargs:
        kwargs["dim"] = kwargs.pop("axes")
    if not TORCH_BACKEND and "dim" in kwargs:
        kwargs["axis"] = kwargs.pop("dim")
    return xp.fft.ifft2(x, **kwargs)


def fftshift(x, **kwargs):
    """FFT shift with kwarg conversion."""
    if TORCH_BACKEND and "axes" in kwargs:
        kwargs["dim"] = kwargs.pop("axes")
    if not TORCH_BACKEND and "dim" in kwargs:
        kwargs["axes"] = kwargs.pop("dim")
    return xp.fft.fftshift(x, **kwargs)


def ifftshift(x, **kwargs):
    """Inverse FFT shift with kwarg conversion."""
    if TORCH_BACKEND and "axes" in kwargs:
        kwargs["dim"] = kwargs.pop("axes")
    if not TORCH_BACKEND and "dim" in kwargs:
        kwargs["axes"] = kwargs.pop("dim")
    return xp.fft.ifftshift(x, **kwargs)


def meshgrid(*args, **kwargs):
    """Create mesh grids."""
    return xp.meshgrid(*args, **kwargs)


def sqrt(x):
    """Element-wise square root."""
    return xp.sqrt(x)


def exp(x):
    """Element-wise exponential."""
    return xp.exp(x)


def absolute(x):
    """Element-wise absolute value."""
    if TORCH_BACKEND and type(x) in [np.memmap, np.ndarray]:
        return np.absolute(x)
    return xp.absolute(x)


def real(x):
    """Take real part."""
    return xp.real(x)


def cos(x):
    """Element-wise cosine."""
    return xp.cos(x)


def arctan2(y, x):
    """Element-wise arctangent of y/x."""
    return xp.arctan2(y, x)


def amin(x):
    """Return minimum value."""
    return xp.amin(x)


def amax(x):
    """Return maximum value."""
    return xp.amax(x)


def sum(x, axis=None, **kwargs):
    """Sum elements along axis with kwarg conversion."""
    if TORCH_BACKEND and type(x) not in [np.memmap, np.ndarray]:
        if "axis" in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return xp.sum(x, dim=axis, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        return np.sum(x, axis=axis, **kwargs)


def mean(x, axis=None, **kwargs):
    """Mean of elements along axis with kwarg conversion."""
    is_torch = TORCH_BACKEND and type(x) not in [np.memmap, np.ndarray]
    
    if "keepdims" in kwargs and not is_torch:
        kwargs["keepdims"] = kwargs.pop("keepdims")
    elif "keepdims" in kwargs and is_torch:
        kwargs["keepdim"] = kwargs.pop("keepdims")
    
    if "keepdim" in kwargs and not is_torch:
        kwargs["keepdims"] = kwargs.pop("keepdim")
    
    if "axis" in kwargs and is_torch:
        kwargs["dim"] = kwargs.pop("axis")
    elif "dim" in kwargs and not is_torch:
        kwargs["axis"] = kwargs.pop("dim")
    
    if is_torch:
        return xp.mean(x, dim=axis, **kwargs)
    else:
        return np.mean(x, axis=axis, **kwargs)


def any(x):
    """Test whether any array elements along a given axis evaluate to True."""
    return xp.any(x)


def einsum(subscripts, *operands, **kwargs):
    """Einstein summation convention."""
    if TORCH_BACKEND:
        return xp.einsum(subscripts, *operands, **kwargs)
    else:
        return xp.einsum(subscripts, *operands, optimize=True, **kwargs)


def reshape(x, shape):
    """Reshape array."""
    if TORCH_BACKEND and type(x) == np.memmap:
        return np.reshape(x, shape)
    return xp.reshape(x, shape)


def expand_dims(x, axis):
    """Expand array dimensions."""
    if TORCH_BACKEND:
        return xp.unsqueeze(x, dim=axis)
    else:
        return np.expand_dims(x, axis)


def stack(arrays, axis=0, **kwargs):
    """Stack arrays."""
    if TORCH_BACKEND:
        if "axis" in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return xp.stack(arrays, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        return np.stack(arrays, axis=axis, **kwargs)


def to_cpu(x):
    """Convert tensor to CPU NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, (int, float, complex)):
        return x
    return np.asarray(x)


def isnan(x):
    """Test element-wise for NaN."""
    return xp.isnan(x)


def memmap(dims, dtype=None, filename=None):
    """Create memory-mapped array."""
    if filename is None:
        print("WARNING: memmap attempted without filename, falling back to zeros")
        return zeros(dims, dtype=dtype)
    
    # Convert torch dtypes to numpy for memmap
    if TORCH_BACKEND and dtype in [xp.complex128, xp.complex64, xp.float64, xp.float32]:
        dtype_map = {
            xp.complex128: np.complex128,
            xp.complex64: np.complex64,
            xp.float64: np.float64,
            xp.float32: np.float32,
        }
        dtype = dtype_map[dtype]
    
    return np.memmap(filename, dtype=dtype, mode='w+', shape=dims)


# Keep legacy function for backward compatibility
def argwhere(condition):
    """Return indices where condition is True."""
    return xp.argwhere(condition)
