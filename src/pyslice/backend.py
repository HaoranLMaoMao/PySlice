# backend.py - Backend abstraction layer for NumPy/PyTorch support
import numpy as np
import os


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

pi = xp.pi

def asarray(arraylike, dtype=None, device=None):
    """Convert array-like object to backend array (NumPy or PyTorch tensor)."""
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    # Handle complex to real casting
    if dtype is not None and 'float' in str(dtype).lower() and hasattr(arraylike, 'dtype') and 'complex' in str(arraylike.dtype).lower():
        arraylike = arraylike.real
    if TORCH_BACKEND:
        if hasattr(arraylike, 'detach'):  # it's already a tensor
            array = arraylike.detach().clone()
            if dtype is not None:
                array = array.to(dtype)
            if device is not None:
                array = array.to(device)
        else:
            array = xp.tensor(arraylike, dtype=dtype, device=device)
    else:
        array = xp.asarray(arraylike, dtype=dtype)
    return array

def astype(arraylike,dtype):
    if hasattr(arraylike,"to"): # torch
        return arraylike.to(dtype)
    return arraylike.astype(dtype) # numpy


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
        device=DEFAULT_DEVICE
    # string handling for dtype, "float" --> float
    if isinstance(dtype,str):
        dtype={"float":DEFAULT_FLOAT_DTYPE,"complex":DEFAULT_COMPLEX_DTYPE,"int":int}[dtype]
    # infer if we're using torch or numpy (numpy does not take device arg)
    if TORCH_BACKEND:
        array = xp.zeros(dims, dtype=dtype, device=device)
    else:
        array = xp.zeros(dims, dtype=dtype)
    return array

def memmap(dims,dtype=DEFAULT_FLOAT_DTYPE,filename=None):
    from numpy.lib.format import open_memmap
    if filename is None:
        print("WARNING: memmap attempted without filename, falling back to zeros")
        return zeros(dims,dtype)
    # cast to numpy dtypes so we can use numpy memmaps
    if xp != np and dtype in [ xp.complex128, xp.complex64, xp.float64, xp.float32 ]:
        dtype = { xp.complex128:np.complex128, xp.complex64:np.complex64,
                 xp.float64:np.float64, xp.float32:np.float32 }[ dtype ]
    mode = 'w+' #'r+' if os.path.exists(filename) else 'w+'
    #print("creating memmap",mode,dtype,dims,"->",filename)
    return open_memmap(filename, dtype=dtype, mode=mode, shape=dims)


def absolute(x):
    """Element-wise absolute value."""
    if TORCH_BACKEND and type(x) in [np.memmap, np.ndarray]:
        return np.absolute(x)
    return xp.absolute(x)


def reshape(array,shape):
    if xp != np and type(array) == np.memmap:
        return np.reshape(array,shape)
    return xp.reshape(array,shape)

def ones(dims, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    """Create a one-filled array."""
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
        print(device, dtype)
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

def fft(k,**kwargs):
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    if use_torch:
        return xp.fft.fft(k,**kwargs)
    return np.fft.fft(k,**kwargs)

def fftshift(k,**kwargs):
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "axes" in kwargs.keys():
        kwargs["dim"]=kwargs["axes"] ; del kwargs["axes"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axes"]=kwargs["dim"] ; del kwargs["dim"]
    if use_torch:
        return xp.fft.fftshift(k,**kwargs)
    return np.fft.fftshift(k,**kwargs)

def mean(k,**kwargs):
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "keepdims" in kwargs.keys():
        kwargs["keepdim"]=kwargs["keepdims"] ; del kwargs["keepdims"]
    if not use_torch and "keepdim" in kwargs.keys():
        kwargs["keepdims"]=kwargs["keepdim"] ; del kwargs["keepdim"]
    if use_torch and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    if not use_torch:
        return np.mean(k,**kwargs)
    return xp.mean(k,**kwargs)

def ifft2(k):
    return xp.fft.ifft2(k)

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
    #print([ (type(o),o.dtype) for o in operands])
    numpytypes = [ type(o) in [np.ndarray, np.memmap] for o in operands ]
    if TORCH_BACKEND and True not in numpytypes:
        return xp.einsum(subscripts, *operands, **kwargs)
    else:
        operands = [ to_cpu(o) for o in operands ]
        return np.einsum(subscripts, *operands, optimize=True, **kwargs)

def to_cpu(array):
    if type(array) in [ np.ndarray, np.memmap ]:
        return array
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        return np.stack(arrays, axis=axis, **kwargs)

# def to_cpu(x):
#     """Convert tensor to CPU NumPy array."""
#     if isinstance(x, np.ndarray):
#         return x
#     if hasattr(x, 'cpu'):
#         return x.cpu().numpy()
#     if isinstance(x, (int, float, complex)):
#         return x
#     return np.asarray(x)


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


def roll(x, shift, axis):
    """Roll array along axis."""
    if TORCH_BACKEND:
        return xp.roll(x, shifts=shift, dims=axis)
    else:
        return np.roll(x, shift, axis=axis)


def log(x):
    """Element-wise natural logarithm."""
    return xp.log(x)


def angle(x):
    """Element-wise angle of complex number."""
    return xp.angle(x)


def isnan(x):
    """Test element-wise for NaN."""
    return xp.isnan(x)

def midcrop(a,n): # e.g. unshifted ks: 0,1,2,3,4.....-4,-3,-,2-1, crop out 3 through -3, the inverse of a[n:-n]
    return xp.roll(xp.roll(a,len(a)//2)[n:-n],len(a)//2-n)

def ceil(v):
    if xp != np and type(v)==torch.Tensor:
        return int(xp.ceil(v))
    return int(np.ceil(v))

def cumsum(a):
    if xp != np and type(a)==torch.Tensor:
        return xp.cumsum(a,dim=0)
    return xp.cumsum(a)

def histogram(a,bins):
    #print(bins)
    #print(bins[1:]-bins[:-1])
    #return np.histogram(to_cpu(a),bins=to_cpu(bins))
    #if xp!=np and type(a)==torch.Tensor: # WHY ARE WE DOING THIS OURSELVES? not-implemented error for torch cuda
    #    #hist = zeros(len(bins)-1,type_match=bins)
    #    #mask = zeros(len(a),type_match=bins)
    #    #for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
    #    #    mask *= 0
    #    #    mask[a>=b1] = 1 ; mask[a>=b2] = 0
    #    #    hist[i] = xp.sum(mask)
    #    #return hist
    hist = zeros(len(bins)-1,type_match=a)
    for chunk in chunkIDs(len(bins)-1,1000):
        db = bins[chunk+1]-bins[chunk]
        diff = a[None,:]-bins[chunk,None]
        diff[diff>db[:,None]]=-1
        diff[diff<0]=-1
        diff[diff!=-1]=1
        diff[diff==-1]=0
        hist[chunk]=xp.sum(diff,axis=1)
    return hist
    #return np.histogram(to_cpu(a),bins=to_cpu(bins))[0]
    #return np.histogram(a,bins=bins)[0]

def randfloats(N):
    N=int(N)
    if xp != np:
        return xp.rand(N)
    return np.random.random(N)

def clone(a):
    if hasattr(a,"clone"):
        return a.clone()
    try:
        if hasattr(a,"copy"):
            return a.copy()
    except:
        pass
    return a

def chunkIDs(N,chunksize=1000):
    chunks = [] ; i=0
    while True:
        chunk = xp.arange(i*chunksize,min((i+1)*chunksize,N))
        chunks.append( chunk )
        i += 1
        if i*chunksize >= N:
            break
    return chunks


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
