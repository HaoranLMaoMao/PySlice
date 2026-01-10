import numpy as np
from tqdm import tqdm
import logging
from ..backend import zeros,mean,ones

try:
    import torch ; xp = torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'mps':
        complex_dtype = torch.complex64
        float_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        float_dtype = torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    xp = np
    print("PyTorch not available, falling back to NumPy")
    complex_dtype = np.complex128
    float_dtype = np.float64



logger = logging.getLogger(__name__)

m_electron = 9.109383e-31    # mass of an electron, kg
q_electron = 1.602177e-19    # charge of an electron, J / eV or kg m^2/s^2 / eV  
c_light = 299792458.0        # speed of light, m / s
h_planck = 6.62607015e-34    # m^2 kg / s


def m_effective(eV):
    """Relativistic correction: E=m*c^2, so m=E/c^2, in kg"""
    return m_electron + eV * q_electron / c_light**2

def wavelength(eV):
    return h_planck * c_light / ((eV * q_electron)**2 + 2 * eV * q_electron * m_electron * c_light**2)**0.5 * 1e10

class Probe:
    """
    PyTorch-accelerated probe class for electron microscopy.
    
    Generates probe wavefunctions on GPU for both plane wave and convergent beam modes.
    Significant speedup for large grid sizes through GPU-accelerated FFT operations.
    """
    
    def __init__(self, xs, ys, mrad, eV, array=None, device=None, gaussianVOA=0, preview=False, probe_xs=None, probe_ys=None, probe_positions=None):
        """
        Initialize GPU-accelerated probe wavefunction.
        
        Args:
            xs, ys: Real space coordinate arrays
            mrad: Convergence semi-angle in milliradians (0.0 = plane wave)
            eV: Electron energy in eV
            device: PyTorch device (None for auto-detection)
        """
        # TORCH DEVICES AND DTYPES
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified (same logic as Potential class)
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
            elif isinstance(device, str):
                device = torch.device(device)
            self.device = device
            self.use_torch = True

            # Use float32 for MPS compatibility (same as Potential class)
            self.dtype = torch.float32 if device.type == 'mps' else torch.float64
            self.complex_dtype = torch.complex64 if device.type == 'mps' else torch.complex128
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.device = None
            self.use_torch = False
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        
        # SET UP SPATIAL GRIDS
        # Convert coordinate arrays to tensors if using torch (same as Potential class)
        if self.use_torch:
            # Use as_tensor to avoid copy warning when input is already a tensor
            self.xs = torch.as_tensor(xs, dtype=self.dtype, device=self.device)
            self.ys = torch.as_tensor(ys, dtype=self.dtype, device=self.device)
        else:
            self.xs = xs
            self.ys = ys

        nx = len(xs) ; ny = len(ys)
        dx = xs[1] - xs[0] ; dy = ys[1] - ys[0]
        lx = nx*dx ; ly = ny*dy
        self.nx = nx ; self.dx = dx ; self.lx = lx
        self.ny = ny ; self.dy = dy ; self.ly = ly

        # HANDLE PROBE POSTIONS
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self.probe_positions = probe_positions

        # Preferred to pass probe_xs and probe_ys from which we will define a grid. copied from probe_grid (no defunct)
        if self.probe_xs is not None and self.probe_ys is not None:
            x,y = np.meshgrid(self.probe_xs,self.probe_ys)
            self.probe_positions = np.reshape([x,y],(2,len(x.flat))).T

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
            self.probe_xs = [lx/2] ; self.probe_ys = [ly/2]

        # HANDLE BEAM PARAMS
        self.mrad = mrad
        #if isinstance(eV,(float,int)):
        #    n = 1 if array is None else len(array)
        #    eV = [ eV ]*n
        self.eV = eV ; self.wavelength=wavelength(eV)
        self.eVs = np.asarray([eV])
        if self.use_torch:
            self.eVs = torch.as_tensor(self.eVs, dtype=self.dtype, device=self.device)
        self.wavelengths = wavelength(self.eVs)
        self.temporal_decoherence = None
        self.spatial_decoherence = None
        self.gaussianVOA = gaussianVOA
        
        # Set up device kwargs for unified xp interface (same as Potential class)
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        
        self.kxs = xp.fft.fftfreq(nx, d=dx, **device_kwargs)
        self.kys = xp.fft.fftfreq(ny, d=dy, **device_kwargs)

        if not array is None: # Allow construction of a Probe object with a passed array instead of building it below. used by create_batched_probes
            if self.use_torch and hasattr(array, 'to'):
                self._array = array.to(device=self.device, dtype=self.complex_dtype)
            else:
                self._array = xp.asarray(array)
        else:
            #self._array = zeros((len(self.eV),1,nx,ny))
            #for i,w in enumerate(self.wavelength):
            #   self._array[i,0,:,:] = self.generate_single_probe(mrad,w,gaussianVOA,preview=preview)
            self._array = zeros((1,1,nx,ny),dtype=complex_dtype)
            self._array[0,0,:,:] = self.generate_single_probe(mrad,self.wavelength,self.gaussianVOA,preview=preview)

        self.applyShifts()

    def generate_single_probe(self,mrad,wavelength,gaussianVOA,preview=False):
        nx,ny = len(self.kxs) , len(self.kys)
        if mrad == 0:
            return zeros((nx, ny))+1

        reciprocal = zeros((nx, ny))
        radius = (mrad * 1e-3) / wavelength  # Convert mrad to reciprocal space units
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        radii = xp.sqrt(kx_grid**2 + ky_grid**2)

        if gaussianVOA == 0:
            mask = radii < radius
            reciprocal[mask] = 1.0
        else:
            from scipy.special import erf
            reciprocal = 1-erf((radii-radius)/(gaussianVOA*radius))

        if preview:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots() ; print(radius)
            extent = (xp.min(self.kxs), xp.max(self.kxs), xp.min(self.kys), xp.max(self.kys))
            ax.imshow(xp.fft.fftshift(reciprocal.T), cmap="inferno",extent=extent)
            ax.set_xlabel("kx ($\\AA^{-1}$)")
            ax.set_ylabel("ky ($\\AA^{-1}$)")
            plt.show()

        return xp.fft.ifftshift(xp.fft.ifft2(reciprocal))
        #self.array_numpy = self.array.cpu().numpy()
    
    def copy(self):
        """Create a deep copy of the probe."""
        new_probe = ProbeTorch.__new__(ProbeTorch)
        new_probe.xs = self.xs.clone()
        new_probe.ys = self.ys.clone()
        new_probe.mrad = self.mrad
        new_probe.eV = self.eV
        new_probe.wavelength = self.wavelength
        new_probe.kxs = self.kxs.clone()
        new_probe.kys = self.kys.clone()
        new_probe._array = self._array.clone()
        new_probe.device = self.device
        new_probe.array_numpy = self.array_numpy.copy()
        return new_probe

    @property
    def array(self):
        return self.to_cpu()
    
    def to_cpu(self):
        """Convert probe array to CPU NumPy array."""
        if hasattr(self._array, 'cpu'):
            return self._array.cpu().numpy()
        return self._array
    
    def to_device(self, device):
        """Move probe to specified device (similar to Potential.to_device)."""
        if not self.use_torch:
            raise RuntimeError("to_device() requires PyTorch")

        # MPS doesn't support float64
        if hasattr(device, 'type') and device.type == 'mps':
            dtype, complex_dtype = torch.float32, torch.complex64
        else:
            dtype, complex_dtype = torch.float64, torch.complex128

        self._array = self._array.to(device=device, dtype=complex_dtype)
        self.xs = self.xs.to(device=device, dtype=dtype)
        self.ys = self.ys.to(device=device, dtype=dtype)
        self.kxs = self.kxs.to(device=device, dtype=dtype)
        self.kys = self.kys.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype
        return self

    def plot(self,filename=None,title=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # calling self.array should convert to CPU/numpy
        array = np.mean(np.absolute(self.array[:,:,:,:]),axis=1)[0,:,:] # positional,summable,x,y indices
        array=array.T # imshow convention: y,x. our convention: x,y
        plot_array = np.absolute(array)**.25

        # Convert extent values to CPU if needed (use xp for torch/numpy compatibility)
        xs_min = xp.amin(self.xs)
        xs_max = xp.amax(self.xs) # TODO technically this should be xs[-1]+dx??
        ys_min = xp.amin(self.ys)
        ys_max = xp.amax(self.ys)

        if hasattr(xs_min, 'cpu'):
            xs_min = xs_min.cpu()
            xs_max = xs_max.cpu()
            ys_min = ys_min.cpu()
            ys_max = ys_max.cpu()

        extent = (xs_min, xs_max, ys_min, ys_max)
        ax.imshow(plot_array, cmap="inferno",extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")
        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def defocus(self,dz): # POSITIVE DEFOCUS PUTS BEAM WAIST ABOVE SAMPLE, UNITS OF ANGSTROM
        if isinstance(dz,(int,float)):
            dz = zeros(len(self._array))+dz
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.wavelength * dz[:,None,None] * k_squared[None,:,:])
        nz = len(dz) ; nc,npt,nx,ny = self._array.shape #; print("nc,npt,nx,ny",nc,npt,nx,ny)
        self._array = xp.fft.ifft2( P[:,None,None,:,:] * xp.fft.fft2( self._array )[None,:,:,:,:] )
        self._array = self._array.reshape((nz*nc,npt,nx,ny))
        #print("defocus",dz,"new shape",self._array.shape)

    # ORDER OF OPERATIONS IS IMPORTANT. MUST DO: addTemporalDecoherence, addSpatialDecoherence, create_batched_probes
    # addTemporalDecoherence - creates new standard probes (must come first)
    # addSpatialDecoherence - applies defocus (applies to existing probe(s))
    # create_batched_probes - applied shift to each probe
    def addTemporalDecoherence(self,sigma_eV,N):
        nc,npt,nx,ny = self._array.shape #; print("addTemporalDecoherence shape was",nc,npt,nx,ny)
        if self.temporal_decoherence is not None:
            print("WARNING: calling addTemporalDecoherence twice will overwrite previous")
        self.temporal_decoherence = (sigma_eV,N)
        eV = self.eV
        self.eVs = np.linspace(eV-2*sigma_eV,eV+2*sigma_eV,N)
        if self.use_torch:
            self.eVs = torch.as_tensor(self.eVs, dtype=self.dtype, device=self.device)
        self.wavelengths = wavelength(self.eVs)
        amplitudes = np.exp(-(eV-self.eVs)**2/sigma_eV**2)
        self._array = zeros((N,1,nx,ny))
        for n,eV in enumerate(self.eVs):
            self._array[n,0,:,:] = amplitudes[n] * self.generate_single_probe(self.mrad,wavelength(eV),self.gaussianVOA)
        nc,npt,nx,ny = self._array.shape #; print("addTemporalDecoherence expands to",nc,npt,nx,ny)
        if self.spatial_decoherence is not None:
            self.addSpatialDecoherence(*self.spatial_decoherence)
        nc,npt,nx,ny = self._array.shape
        if npt==1:
            self.applyShifts()

    def addSpatialDecoherence(self,sigma_dz,N):
        nc,npt,nx,ny = self._array.shape #; print("addSpatialDecoherence shape was",nc,npt,nx,ny)
        if self.temporal_decoherence is not None:
            print("WARNING: calling addSpatialDecoherence twice will overwrite previous")
        self.spatial_decoherence = (sigma_dz,N)
        dzs = np.linspace(-2*sigma_dz,2*sigma_dz,N) # suppose N=25
        amplitudes = np.exp(-dzs**2/sigma_dz**2)
        nc,npt,nx,ny = self._array.shape            # suppose nc=10 (addTemporalDecoherence created 10 wavelengths)
        if self.use_torch:
            dzs = torch.as_tensor(dzs, dtype=self.dtype, device=self.device)
        self.defocus(dzs)                           # defocus starts with 25,10,npt,nx,ny --reshapes--> 250,npt,nx,ny
        for i in range(N):                         # reshape to flatten loops first index last: [[0,1],[2,3]] --> [0,1,2,3]
            for j in range(nc):
                self._array[i*nc+j] *= amplitudes[i]
        nc,npt,nx,ny = self._array.shape #; print("addSpatialDecoherence expands to",nc,npt,nx,ny)
        self.eVs = ones(N)[:,None]*self.eVs[None,:] # defocus expands into nz,nc then flattens to nz*nc
        self.eVs = self.eVs.reshape(nc)
        self.wavelengths = ones(N)[:,None]*self.wavelengths[None,:]
        self.wavelengths = self.wavelengths.reshape(nc)
        if npt==1:
            self.applyShifts()

    def applyShifts(self):
        nc,npt,nx,ny = self._array.shape ; print("applyShifts shape was",nc,npt,nx,ny)
        if npt>1: # TODO ALSO NEED SOMETHING TO DETERMINE IF SHIFTS HAVE ALREADY BEEN APPLIED. EG A LIST WHICH IS ALWAYS UPDATED WHEN ARRAY IS RESET?
            return
        self._array = self._array[:,0,None,:,:] * ones(len(self.probe_positions))[None,:,None,None]
        for i, (px,py) in enumerate(self.probe_positions):
            if px-self.lx/2 == 0 and py-self.ly/2 == 0:
                    continue
            # Create shifted probe using phase ramp in k-space
            probe_k = xp.fft.fft2(self._array[:,i,:,:]) # positional,summable,x,y

            # Apply phase ramp for spatial shift
            kx_shift = xp.exp(2j * xp.pi * self.kxs[None,:, None] * (px-self.lx/2) )
            ky_shift = xp.exp(2j * xp.pi * self.kys[None,None, :] * (py-self.ly/2) )
            probe_k_shifted = probe_k * kx_shift * ky_shift

            # Convert back to real space
            self._array[:,i,:,:] = xp.fft.ifft2(probe_k_shifted)
        nc,npt,nx,ny = self._array.shape ; print("applyShifts expands to",nc,npt,nx,ny)

    def aberrate(self,aberrations):
        dP = aberrationFunction(self.kxs,self.kys,self.wavelength,aberrations)
        # recall, in Probe.__init__, we created the real-space array via:
        # self.array = xp.fft.ifftshift(xp.fft.ifft2(reciprocal))
        # Aberrations are defined at the aperture plane, so we must apply them in reciprocal space. 
	    # (or do a convolution in real-space)
        reciprocal = xp.fft.fft2(xp.fft.fftshift(self._array))
        reciprocal *= dP
        self._array = xp.fft.ifftshift(xp.fft.ifft2(reciprocal))

# See Kirkland Eq 2.10: 
# χ(k,ϕ) = π/2 Cs λ³ k⁴ - π Δf λ k²
# + π fa2 λ k² sin(2*(ϕ-ϕa2)) + 2π/3 fa3 λ² k³ sin(3*(ϕ-ϕa3)) 
# + 2π/3 fc3 λ² k³ sin(ϕ-ϕc3)
# where fa is astig, fc is coma, with orientations ϕa or ϕc
# or https://doi-org.ornl.idm.oclc.org/10.1016/S0304-3991(99)00013-3 Eq. A1
# χ(rᵤ,rᵥ) = 2π/λ { C₁₀ ( rᵤ² + rᵥ² ) / 2
# + C₁₂ᵤ ( rᵤ² - rᵥ² ) / 2 + C₁₂ᵥ rᵤ rᵥ
# + C₂₁ᵤ rᵤ ( rᵤ² + rᵥ² ) / 3 + C₂₁ᵥ rᵥ ( rᵤ² + rᵥ² ) / 3 
# + C₂₃ᵤ rᵤ ( rᵤ² - 3 rᵥ² ) / 3 + C₂₃ᵥ rᵥ ( 3 rᵤ² - rᵥ² ) / 3
# + C₃₀ ( rᵤ² + rᵥ² )² / 4
# + C₃₂ᵤ ( rᵤ⁴ - rᵥ⁴ ) / 4 + C₃₂ᵥ 2 rᵤ rᵥ ( rᵤ² + rᵥ² ) / 4
# + C₃₄ᵤ ( rᵤ⁴ - 6 rᵤ² rᵥ² + rᵥ² ) / 4 + C₃₄ᵥ ( rᵤ³ rᵥ - rᵤ rᵥ³ ) } 
# where rᵤ = r cos(ϕ) and rᵥ = r sin(ϕ), r² = rᵤ² + rᵥ²
# χ(rᵤ,rᵥ) = 2π/λ { C₁₀ r² / 2
# + C₁₂ᵤ ( rᵤ² - rᵥ² ) / 2 + C₁₂ᵥ rᵤ rᵥ
# + C₂₁ᵤ rᵤ r² / 3 + C₂₁ᵥ rᵥ r² / 3 
# + C₂₃ᵤ rᵤ ( rᵤ² - 3 rᵥ² ) / 3 + C₂₃ᵥ rᵥ ( 3 rᵤ² - rᵥ² ) / 3
# + C₃₀ r⁴ / 4
# + C₃₂ᵤ ( rᵤ⁴ - rᵥ⁴ ) / 4 + C₃₂ᵥ 2 rᵤ rᵥ ( rᵤ² + rᵥ² ) / 4
# + C₃₄ᵤ ( rᵤ⁴ - 6 rᵤ² rᵥ² + rᵥ² ) / 4 + C₃₄ᵥ ( rᵤ³ rᵥ - rᵤ rᵥ³ ) } 
# or https://doi-org.ornl.idm.oclc.org/10.1016/j.ultramic.2010.04.006 Eq A1
# χ(u,v) = C₀₁ u + C₀₁ v
# + 1/2 [ C₁₀ ( u² + v² ) + C₁₂ᵤ ( u² - v² ) + 2 C₁₂ᵥ u v ]
# +1/3 [ C₂₃ᵤ
# ₀₁₂₃₄ᵤᵥ
# or comparing to: https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html
# χ(k,ϕ) = π/2/λ 1/(n+1) C ( k λ )^(n+1) cos(m*(ϕ-ϕa))
# Aberrations are an adjustment to the phase of the wave ("dPhi"), to be applied in reciprocal space.
# this is done by multiplying the complex wave (be it a probe or an exit wave) by xp.exp(-1j * dPhi)
def aberrationFunction(kxs,kys,wavelength,aberrations): # aberrations should be a dict of Cnm following https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html
    dPhi = xp.zeros((len(kxs),len(kys)))
    ks = xp.sqrt( kxs[:,None]**2 + kys[None,:]**2 )
    theta = xp.arctan2( kys[None,:] , kxs[:,None] )
    for k in aberrations.keys():
        n,m = int(k[1]),int(k[2]) # C03 --> 0,3
        C = aberrations[k] ; phi0 = 0
        if not isinstance(C,(int,float)):
            C,phi0 = C
        dPhi += 2*xp.pi/wavelength * \
            (1/(n+1)) * C * ( ks * wavelength ) ** (n+1) * \
            xp.cos( m * (theta-phi0) )
    return xp.exp(-1j * dPhi)

#def probe_grid(xlims,ylims,n,m):
#	x,y=np.meshgrid(np.linspace(*xlims,n),np.linspace(*ylims,m))
#	return np.reshape([x,y],(2,len(x.flat))).T


def create_batched_probes(base_probe, probe_positions, device=None):
    """
    Create a batch of shifted probes for vectorized processing.

    Args:
        base_probe: ProbeTorch object
        probe_positions: List of (x,y) positions
        device: PyTorch device

    Returns:
        probe object with an array of shape (n_probes, nx, ny)
    """
    # Move probe to correct device if needed
    if device is not None and TORCH_AVAILABLE:
        # Always move to ensure array is actually on the device
        # (checking base_probe.device may not reflect actual array device)
        base_probe.to_device(device)

    n_probes = len(probe_positions)
    probe_arrays = []

    nx = len(base_probe.xs)
    ny = len(base_probe.ys)

    # Compute dx, dy on same device as probe
    if TORCH_AVAILABLE and hasattr(base_probe.xs, 'device'):
        dx = (base_probe.xs[1] - base_probe.xs[0]).item()
        dy = (base_probe.ys[1] - base_probe.ys[0]).item()
    else:
        dx = base_probe.xs[1] - base_probe.xs[0]
        dy = base_probe.ys[1] - base_probe.ys[0]

    lx = nx*dx ; ly = ny*dy

    for px, py in probe_positions:
        # Create shifted probe using phase ramp in k-space
        probe_k = xp.fft.fft2(base_probe._array)

        # Apply phase ramp for spatial shift
        kx_shift = xp.exp(2j * xp.pi * base_probe.kxs[:, None] * (px-lx/2) )
        ky_shift = xp.exp(2j * xp.pi * base_probe.kys[None, :] * (py-ly/2) )
        probe_k_shifted = probe_k * kx_shift * ky_shift
        
        # Convert back to real space
        shifted_probe_array = xp.fft.ifft2(probe_k_shifted)
        probe_arrays.append(shifted_probe_array)
    
    # Stack into batch tensor
    if TORCH_AVAILABLE:
        array = torch.stack(probe_arrays, dim=0)
    else:
        array = xp.asarray(probe_arrays)

    return Probe(base_probe.xs, base_probe.ys, base_probe.mrad, base_probe.eV, array=array, device=base_probe.device)

def Propagate(probe, potential, device=None, progress=False, onthefly=True, store_all_slices=False):
    """
    PyTorch-accelerated multislice propagation function.
    Supports both single probe and batched multi-probe processing.

    Args:
        probe: ProbeTorch object or tensor with shape (n_probes, nx, ny)
        potential: Potential object (can be NumPy or PyTorch version)
        device: PyTorch device (None for auto-detection)
        progress: Show progress bar
        onthefly: If True, calculate potential slices on the fly. If False, build full array
        store_all_slices: If True, return wavefunction at each slice instead of just exit wave

    Returns:
        torch.Tensor: Exit wavefunction(s) after multislice propagation
                     If store_all_slices=True, shape is (n_slices, n_probes, nx, ny)
                     Otherwise, shape is (n_probes, nx, ny) or (nx, ny) for single probe
    """
    if device is not None and not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Please install PyTorch.")
    
    # Initialize wavefunction with probe(s) - shape: (n_probes, nx, ny)
    print(probe._array.shape)
    nc,npt,nx,ny = probe._array.shape #; print("nc,npt,nx,ny",nc,npt,nx,ny)
    array = probe._array.reshape((nc*npt,nx,ny)) # "flatten" first two indices
    probe_wavelengths = probe.wavelengths[:,None]*ones(npt)[None,:] # also expand wavelengths and eVs arrays to cover all probe positions npt
    probe_wavelengths = probe_wavelengths.reshape(nc*npt)
    probe_eVs = probe.eVs[:,None]*ones(npt)[None,:]
    probe_eVs = probe_eVs.reshape(nc*npt)

    # Calculate interaction parameter (Kirkland Eq 5.6)
    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (probe_wavelengths * probe_eVs) * \
            (E0_eV + probe_eVs) / (2 * E0_eV + probe_eVs) # wavelength and eVs now have length of n_probes

    #if TORCH_AVAILABLE:
    #    #sigma_dtype = torch.float32 if device.type == 'mps' else torch.float64
    #    sigma = torch.tensor(sigma, dtype=float_dtype, device=device)
    
    # Get slice thickness
    dz = potential.zs[1] - potential.zs[0] if len(potential.zs) > 1 else 0.5
    
    # Pre-compute propagation operator in k-space (Fresnel propagation)
    # All tensors should already be on the correct device from creation
    kx_grid, ky_grid = xp.meshgrid(potential.kxs, potential.kys, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2
    P = xp.exp(-1j * xp.pi * probe_wavelengths[:,None,None] * dz * k_squared[None,:,:])

    if progress:
        localtqdm = tqdm
        print("propagating through slices")
    else:
        def localtqdm(iterator):
            return iterator

    if not onthefly:
        potential.build()

    # More elegant approach: use list to accumulate slices if needed
    slice_wavefunctions = [] if store_all_slices else None

    # Vectorized multislice propagation through each slice
    for z in localtqdm(range(len(potential.zs))):
        # Transmission function: t = exp(iσV(x,y,z))
        # All tensors should already be on the correct device from creation
        if onthefly:
            potential_slice = potential.calculateSlice(z)
        else:
            potential_slice = potential._array[:, :, z]
        t = xp.exp(1j * sigma[:,None,None] * potential_slice[None,:,:])

        # Apply transmission to all probes: ψ' = t × ψ
        # Broadcasting: t[nx,ny] * array[n_probes,nx,ny] = array[n_probes,nx,ny]
        array = t * array

        # Store wavefunction at this slice if requested (after transmission)
        if store_all_slices:
            # Clone/copy to avoid reference issues
            if TORCH_AVAILABLE:
                slice_wavefunctions.append(array.clone())
            else:
                slice_wavefunctions.append(array.copy())

        # Fresnel propagation to next slice (except for last slice)
        if z < len(potential.zs) - 1:
            # Vectorized FFT over spatial dimensions for all probes
            kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
            fft_array = xp.fft.fft2(array, **kwarg)
            propagated_fft = P * fft_array
            array = xp.fft.ifft2(propagated_fft, **kwarg)

    # Return results based on what was requested
    if store_all_slices:
        # Stack the list into a tensor with slices as a new dimension
        # Shape will be (n_slices, n_probes, nx, ny) - more conventional ordering
        if TORCH_AVAILABLE:
            return torch.stack(slice_wavefunctions, dim=0)
        else:
            return xp.stack(slice_wavefunctions, axis=0)

    #array = array.reshape((nc,npt,nx,ny))

    # Return single probe result if input was single, otherwise return batch
    if array.shape[0] == 1:
        return array.squeeze(0)
    return array # okay for Propagate to return a Tensor. we probably don't want to move things off-gpu yet

