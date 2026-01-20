
#Imports: Typing
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pySEA.sea_eco.architecture.base_structure_numpy import Dimensions


#Imports: External
from numpy import array, linspace, tile, stack, meshgrid
from numpy.random import random

#Imports: Internal
from pySEA.sea_eco.models.components.gaussians import Gaussian1D, GaussianND

def _normalize_dimensions(dimensions: 'Dimensions' | Sequence) -> tuple[list, list[int]]:
    """Return a list of axis values and their sizes from Dimensions or raw arrays."""
    if hasattr(dimensions, "dimensions"):  # Dimensions instance
        axes = [array(dim.values) for dim in dimensions.dimensions]
    elif isinstance(dimensions, Sequence):
        axes = [array(getattr(dim, "values", dim)) for dim in dimensions]
    else:
        raise TypeError("dimensions must be a Dimensions object or a sequence of arrays.")

    if len(axes) < 2:
        raise ValueError("At least two axes are required to generate the SI-t dataset.")

    dims_n = [axis.shape[0] for axis in axes]
    return axes, dims_n


def generate_gaussian_SI_t_series_data(dimensions: 'Dimensions' | Sequence,
                                       kwargs_zlp:dict = dict(height=1e6, fwhm=0.235, mu=0),
                                       kwargs_loss:dict = dict(height=1e3, fwhm=[30, 30, 2], mu=[0,0,15]),
                                       add_noise: bool = True, noise_level: float = 0.1,
                                       print_info:bool = False):
    
    axes, dims_n = _normalize_dimensions(dimensions)

    # Create a ZLP
    e = axes[-1]
    zlp = Gaussian1D(**kwargs_zlp)
    zlp_data = zlp.evaluate(e)

    # Create a vacuum probe that decreases in intensity with time
    scale = linspace(100,1E-2,dims_n[0])[:, *(None,)*(len(dims_n)-1)]
    vac = scale * tile(zlp_data, (dims_n[:-1]+[1]))
    if print_info:
        print(f'Size of dimensions: {dims_n}')
        print(f'Shape of vacuum probe: {vac.shape}')
    sig_data = vac

    # Create loss function
    yxe = stack(meshgrid(*axes[1:], indexing='ij'), axis=-1)

    loss = GaussianND(**kwargs_loss)
    loss_data = loss.evaluate(yxe)
    loss_zlp = GaussianND(A=-loss.A, mu=[zlp.mu]*(len(dims_n)-1), sigma=list(loss.sigma[:-1])+[zlp.sigma])
    loss_zlp_data = loss_zlp.evaluate(yxe)
    loss = loss_data+loss_zlp_data
    
    loss = scale * loss[None,:]
    if print_info:
        print(f'Shape of loss: {loss.shape}')
    sig_data += loss

    # Combine probe, loss, and noise
    if add_noise:
        noise = random(dims_n)*noise_level
        sig_data += noise

    return sig_data
