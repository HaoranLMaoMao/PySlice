"""Gaussian model components used by SEA-eco fitting routines."""

#Imports: Typing
from numpy.typing import NDArray
from typing import Sequence, Dict, Literal

#Imports: External
import math
from warnings import warn
import numpy as np
#import sympy #TODO implement sympy parsing and functionality


# def _parse_sympy_expr(string):
#     splits = map(str.strip, string.split(";"))
#     expr = sympy.sympify(next(splits))
#     # We substitute one by one manually, as passing all at the same time does
#     # not work as we want (substitutions inside other substitutions do not work)
#     for sub in splits:
#         t = tuple(map(str.strip, sub.split("=")))
#         expr = expr.subs(t[0], sympy.sympify(t[1]))
#     return expr

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))

class Gaussian1D():
    r"""1D-Gaussian function.

    .. math::
        f(x) = \frac{A}{\sigma \sqrt{2\pi}}\exp\left[
               -\frac{\left(x-x_0\right)^{2}}{2\sigma^{2}}\right]

    ============== ===========
    Variable        Parameter
    ============== ===========
    :math:`A`       A
    :math:`\sigma`  sigma
    :math:`x_0`     mean
    ============== ===========


    Parameters
    ----------
    A : float, optional
        Amplitude that scales the normalized Gaussian (equal to the integral).
    height : float, optional
        Peak height of the Gaussian. Mutually exclusive with ``A``.
    sigma : float, optional
        Scale parameter of the Gaussian distribution.
    fwhm : float, optional
        Full width at half maximum; alternative to ``sigma``.
    mean : float
        Location of the Gaussian maximum (peak position).

    Attributes
    ----------
    fwhm : float
        Convenience attribute to get and set the full width at half maximum.
    height : float
        Convenience attribute to get and set the height using ``A`` and ``_norm``.
    """

    def __init__(self, A: float | None = None, sigma: float | None = None,
                 mu: float = 0.0, *, height: float | None = None,
                 fwhm: float | None = None) -> None:
        self.expression="A * (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)**2 \
                        / (2 * sigma**2))"
        self.name="Gaussian"
        self.position="mu"

        # Check the validity of sigma/fwhm and A/height kwargs
        if sigma is not None and fwhm is not None:
            raise ValueError("Provide only one of sigma or fwhm.")
        elif sigma is None and fwhm is None:
            sigma = 1.0
        if A is not None and height is not None:
            raise ValueError("Provide only one of A or height.")
        elif A is None and height is None:
            A = 1.0

        sigma_value = sigma if sigma is not None else (fwhm / sigma2fwhm if fwhm is not None else 1.0)
        self._sigma: float | None = None
        self._norm: float | None = None
        self.sigma = float(sigma_value)

        self.mu=mu
        if height is not None:
            self.A = float(height) / self._norm
        else:
            self.A = float(1.0 if A is None else A)

        self.parameters = {
            "A": self.A,
            "sigma": self.sigma,
            "mu": self.mu,
        }
        
        #self.function = _parse_sympy_expr(self.expression)
    def evaluate(self, coords: NDArray, 
                 store_values=True, return_values=True
                 ) -> NDArray:
        A = self.A
        mu = self.mu
        vals = A * self._norm * np.exp(-(coords - mu) ** 2 / (2 * self.sigma ** 2))
        if store_values: self._values = vals
        if return_values: return vals

    @property
    def sigma(self) -> float:
        return float(self._sigma)
    @sigma.setter
    def sigma(self, value: float):
        if value <= 0:
            raise ValueError("sigma must be positive.")
        self._sigma = float(value)
        self._update_norm()

    def _update_norm(self):
        self._norm = 1.0 / (self._sigma * sqrt2pi)

    @property
    def fwhm(self):
        return self.sigma * sigma2fwhm

    @fwhm.setter
    def fwhm(self, value):
        self.sigma = value / sigma2fwhm

    @property
    def height(self):
        return self.A * self._norm

    @height.setter
    def height(self, value):
        self.A = value / self._norm


class Gaussian2D():
    r"""
    2D-Gaussian with principal-axis parameterization and rotation.

    This class represents a normalized 2D Gaussian:
    .. math::
        f(x, y) = A * exp[-0.5 * (x - μ)^T Σ^{-1} (x - μ)]
    
    where the covariance matrix Σ is constructed from principal-axis standard
    deviations (σ₁, σ₂) and a rotation angle θ. The class allows evaluation
    on grids or point sets, extraction of the covariance matrix in the
    coordinate basis, and computation of projected standard deviations and
    correlation coefficients along the coordinate axes.

    Parameters
    ----------
    A : float, optional
        Amplitude that scales the normalized Gaussian. Default is 1.0.
    height : float, optional
        Peak height of the Gaussian. Mutually exclusive with ``A``.
    mu : Sequence[float]
        Center of the Gaussian.
    sigma : Sequence[float]
        Standard deviation along the principal axes.
    fwhm : Sequence[float] | float, optional
        Full width at half maximum along the principal axes; alternative to ``sigma``.
    theta : float
        Rotation angle in **radians**. ? = 0 aligns the first principal axis
        with the +x direction.
    Attributes
    ----------
    A : float, optional
        Amplitude that scales the normalized Gaussian. Default is 1.0.
    height : float
        Convenience attribute to get and set the peak height using ``A`` and ``_norm``.
    fwhm : Sequence[float]
        Convenience attribute to get and set the full width at half maximum.
    mu : Sequence[float]
        Center of the Gaussian.
    sigma : Sequence[float]
        Standard deviation along the principal axes.
    theta : float
        Rotation angle in **radians**. ? = 0 aligns the first principal axis
        with the +x direction.
    Sigma : ndarray of shape (2, 2)
        Covariance matrix in the (x, y) coordinate frame.
    Methods
    -------
    evaluate(coords)
        Evaluate the Gaussian on a grid or a collection of points.
    covariance()
        Return the 2×2 covariance matrix in the coordinate basis.
    get_covarianceariance_coefficient()
        Return the projected coordinate-axis standard deviations and
        correlation coefficient.
    from_covarianceariance(x0, y0, Sigma, amplitude=1.0)
        Build a `Gaussian2D` instance from a given 2×2 covariance matrix.

    Notes
    -----
    **1. Covariance construction**

    The Gaussian is parameterized by its principal-axis variances σ₁², σ₂²
    and rotation θ. The covariance in the coordinate frame is

    .. math::
        Σ = R(θ)
            \\begin{bmatrix}
                σ_1^2 & 0 \\\\
                0 & σ_2^2
            \\end{bmatrix}
            R(θ)^T
        \\tag{1}

    with the rotation matrix

    .. math::
        R(θ) =
            \\begin{bmatrix}
                \\cosθ & -\\sinθ \\\\
                \\sinθ &  \\cosθ
            \\end{bmatrix}.

    **2. Expanded components**

    From Eq. (1), the coordinate-frame covariances are:

    .. math::
        \\begin{aligned}
        Σ_{xx} &= σ_1^2 \\cos^2θ + σ_2^2 \\sin^2θ, \\\\
        Σ_{yy} &= σ_1^2 \\sin^2θ + σ_2^2 \\cos^2θ, \\\\
        Σ_{xy} &= (σ_1^2 - σ_2^2) \\sinθ \\cosθ.
        \\end{aligned}
        \\tag{2}

    **3. Projection to coordinate axes**

    The projected (marginal) standard deviations and correlation are:

    .. math::
        σ_x = \\sqrt{Σ_{xx}}, \\quad
        σ_y = \\sqrt{Σ_{yy}}, \\quad
        ρ = Σ_{xy} / (σ_x σ_y).
        \\tag{3}

    **4. Numerical stability**

    For evaluation, the class uses `numpy.linalg.solve` to compute
    Σ⁻¹·Δx without forming Σ⁻¹ explicitly. This is numerically stable
    and efficient, especially for repeated evaluations.

    Examples
    --------
    >>> g = Gaussian2D(0.0, 0.0, sigma1=3.0, sigma2=1.0,
    ...                theta=np.deg2rad(30), amplitude=1.0)
    >>> x = np.linspace(-10, 10, 201)
    >>> y = np.linspace(-10, 10, 201)
    >>> X, Y = np.meshgrid(x, y, indexing="xy")
    >>> Z = g.values((X, Y))  # evaluate on grid
    >>> g.covariance()
    array([[8.  , 2.6],
           [2.6 , 2.  ]])
    >>> g.get_covarianceariance_coefficient()
    {'sigma_x': 2.828..., 'sigma_y': 1.414..., 'rho': 0.46...}
    """

    def __init__(
        self,
        A: float | None = None,
        mu: Sequence[float] = (0.0, 0.0),
        sigma: Sequence[float] | float | None = None,
        principle_axes: NDArray | None = np.eye(2),
        theta: float | None = None,
        *,
        height: float | None = None,
        fwhm: Sequence[float] | float | None = None,
    ):
        # Check the validity of sigma/fwhm and A/height kwargs
        if sigma is not None and fwhm is not None:
            raise ValueError("Provide only one of sigma or fwhm.")
        elif sigma is None and fwhm is None:
            sigma = 1.0
        if A is not None and height is not None:
            raise ValueError("Provide only one of A or height.")
        elif A is None and height is None:
            A = 1.0

        self.ndim = 2
        self.mu = mu
        self._covariance: NDArray | None = None
        self._det: float | None = None
        self._norm: float | None = None
        self._values = None
        self._principle_axes: NDArray | None = None
        self._sigma: NDArray | None = None

        sigma_guess = sigma if sigma is not None else (np.asarray(fwhm) / sigma2fwhm if fwhm is not None else (1.0, 1.0))
        self.sigma = sigma_guess

        if not np.array_equal(principle_axes, np.eye(2)):
            if theta is not None:
                warn('Both principle axes and theta were provided. Using the principle_axes.')
            self.principle_axes = principle_axes
        elif theta is not None:
            self.theta = theta
        else:
            self.principle_axes = principle_axes

        self._ensure_covariance()
        if height is not None:
            self.A = float(height) / self._norm
        else:
            self.A = float(1.0 if A is None else A)
    
    def _invalidate_covariance(self):
        self._covariance = None
        self._det = None
        self._norm = None

    def _ensure_covariance(self):
        if self._covariance is None:
            R = self.principle_axes
            D = np.diag(np.asarray(self._sigma) ** 2)

            cov = R @ D @ R.T
            det = np.linalg.det(cov)
            if det <= 0: raise np.linalg.LinAlgError("Covariance matrix must be SPD (determinant > 0).")
            self._covariance = cov
            self._det = det
            self._norm = (2.0 * np.pi) ** (-self.ndim / 2) * det ** (-0.5)

    def _coerce_sigma(self, value: Sequence[float] | float) -> NDArray:
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            arr = np.repeat(arr, self.ndim)
        arr = arr.ravel()
        if arr.shape != (self.ndim,):
            raise ValueError("sigma must have length 2.")
        if np.any(arr <= 0):
            raise ValueError("sigma1 and sigma2 must be positive.")
        return arr

    @property
    def sigma(self) -> NDArray:
        return self._sigma
    @sigma.setter
    def sigma(self, value: Sequence[float] | float):
        self._sigma = self._coerce_sigma(value)
        self._invalidate_covariance()

    @property
    def fwhm(self):
        return self.sigma * sigma2fwhm
    @fwhm.setter
    def fwhm(self, value):
        self.sigma = np.asarray(value, dtype=float) / sigma2fwhm

    @property
    def height(self):
        self._ensure_covariance()
        return self.A * self._norm
    @height.setter
    def height(self, value):
        self._ensure_covariance()
        self.A = value / self._norm

    @property
    def theta(self) -> float:
        """Angle in radians of first lab frame axis to first principle axis"""
        return np.arctan2(self.principle_axes[1, 0], self.principle_axes[0, 0])
    @theta.setter
    def theta(self, angle:float) -> None:
        "Set the principle axes from the angle provided."
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s],
                      [s,  c]])
        self.principle_axes = R

    @property
    def principle_axes(self) -> NDArray:
        return self._principle_axes
    @principle_axes.setter
    def principle_axes(self, principle_axes: NDArray):
        axes = np.asarray(principle_axes, dtype=float)
        if axes.shape != (2, 2): raise ValueError("principle_axes must be shape (2, 2).")
        if not np.isclose(np.dot(axes[0], axes[1]), 0): raise ValueError('The provided principle_axes are not orthogonal.')
        norms = np.linalg.norm(axes, axis=0)
        if np.any(norms == 0):
            raise ValueError("principle_axes columns must be non-zero.")
        if any(norms!=1): axes = axes / norms
        self._principle_axes = axes
        self._invalidate_covariance()

    @property
    def covariance(self) -> NDArray:
        """Return the 2×2 covariance matrix Σ in the coordinate frame."""
        self._ensure_covariance()
        return self._covariance.copy()
    @covariance.setter
    def covariance(self, covariance: NDArray):
        """Set the 2×2 covariance matrix Σ in the coordinate frame."""
        covariance_dict = self.covariance_to_paramaters(covariance)
        self._covariance = covariance_dict['cov']
        self._det = covariance_dict['det']
        self._principle_axes = covariance_dict['principle_axes']
        self._sigma = covariance_dict['sigma']
        self._norm = (2.0 * np.pi) ** (-self.ndim / 2) * self._det ** (-0.5)

    @staticmethod
    def covariance_to_paramaters(covariance: NDArray) -> Dict[str, float]:
        value = np.asarray(covariance, dtype=float)

        if value.shape != (2, 2): raise ValueError("Covariance must be 2×2.")
        if not np.allclose(value, value.T, atol=1e-12): raise ValueError("Covariance must be symmetric.")

        det = np.linalg.det(value)
        if det <= 0: raise np.linalg.LinAlgError("Covariance matrix must be SPD (determinant > 0).")
        
        eigvals, eigvecs = np.linalg.eigh(value)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return {'cov': value, 'det': det, 'principle_axes': eigvecs, 
                'sigma': np.sqrt(eigvals), 'theta': np.arctan2(eigvecs[1, 0], eigvecs[0, 0])}

    @property
    def theta_deg(self) -> float:
        return np.degrees(self.theta)
    @theta_deg.setter
    def theta_deg(self, value: float):
        self.theta = np.radians(value)

    def get_covarianceariance_coefficient(self) -> Dict[str, float]:
        """
        Compute projected (marginal) statistics along the coordinate axes.

        Returns
        -------
        dict
            Dictionary containing:
            - 'sigma_x' : float, standard deviation along x-axis
            - 'sigma_y' : float, standard deviation along y-axis
            - 'rho'     : float, correlation coefficient between x and y
        """
        S = self.covariance
        sx = np.sqrt(S[0, 0])
        sy = np.sqrt(S[1, 1])
        rho = S[0, 1] / (sx * sy)
        return {"sigma_x": float(sx), "sigma_y": float(sy), "rho": float(rho)}

    def evaluate(self, coords: Sequence[NDArray] | NDArray, 
                 store_values=True, return_values=True,
                 mahal_type: Literal['solve', 'inv'] = 'inv') -> NDArray:
        """
        Evaluate the Gaussian on a grid or at arbitrary points.

        Parameters
        ----------
        coords : array-like
            Either:
              - (X, Y): sequence of two 2D arrays of identical shape, or
              - array of shape (..., 2) with stacked coordinates.

        Returns
        -------
        vals : ndarray
            Array of Gaussian values with the same leading shape as `coords`.
        """
        # Check that the corrdinates are valid and a stacked array
        if isinstance(coords, (list, tuple)):
            if len(coords) != 2:
                raise ValueError("coords sequence must contain (x, y).")
            x = np.asarray(coords[0], dtype=float)
            y = np.asarray(coords[1], dtype=float)
            if x.shape != y.shape:
                raise ValueError("x and y must have the same shape.")
            p = np.stack([x, y], axis=-1)
        else:
            p = np.asarray(coords, dtype=float)
            if p.shape[-1] != 2:
                raise ValueError("coords must have last dimension 2.")
            
        grid_shape = p.shape[:-1]

        # dx = p - np.asarray(self.mu)
        # M = dx.reshape(-1, 2)

        # y = np.linalg.solve(self.covariance, M.T)   # solves Σ y = Δxᵀ
        # mahal = np.sum(M * y.T, axis=1)
        # vals = np.exp(-0.5 * mahal) * self._norm * self.A
        # vals = vals.reshape(grid_shape)

        dx = p - np.asarray(self.mu)

        self._ensure_covariance()

        if mahal_type == 'solve':
            M = dx.reshape(-1, self.ndim)
            y = np.linalg.solve(self.covariance, M.T)   # solves Σ y = Δxᵀ
            mahal = np.sum(M * y.T, axis=1).reshape(p.shape[:-1])
        elif mahal_type == 'inv':
            mahal = np.sum(dx @ np.linalg.inv(self.covariance) * dx, axis=-1)
        else:
            raise ValueError("mahal_type must be 'solve' or 'inv'.")

        vals = self._norm * self.A * np.exp(-0.5 * mahal) 
        if store_values: self._values = vals
        if return_values: return vals

    @staticmethod
    def from_covarianceariance(covariance: NDArray, 
                        A: float = 1.0, mu: Sequence[float] = (0.0, 0.0)
                        ) -> "Gaussian2D":
        """
        Construct a `Gaussian2D` from a 2×2 covariance matrix.

        Parameters
        ----------
        covariance : ndarray of shape (2, 2)
            Symmetric positive-definite covariance matrix.
        mu : Sequence[float]
            Gaussian center coordinates. Default is (0.0, 0.0).
        amplitude : float, optional
            Amplitude of the Gaussian. Default is 1.0.

        Returns
        -------
        Gaussian2D
            New instance with parameters extracted from Σ.

        Notes
        -----
        The rotation θ and principal-axis variances (σ₁², σ₂²)
        are obtained from the eigen-decomposition Σ = VΛVᵀ.
        θ is the angle of the eigenvector corresponding to
        the largest eigenvalue with respect to the +x axis.
        """
        covariance_dict = Gaussian2D.covariance_to_paramaters(covariance)
        return Gaussian2D(A=A, mu=mu,
                          sigma=covariance_dict['sigma'], principle_axes=covariance_dict['principle_axes'])

class GaussianND():
    r"""
    2D-Gaussian with principal-axis parameterization and rotation.

    This class represents a normalized 2D Gaussian:
    .. math::
        f(x, y) = A * exp[-0.5 * (x - μ)^T Σ^{-1} (x - μ)]
    
    where the covariance matrix Σ is constructed from principal-axis standard
    deviations (σ₁, σ₂) and a rotation angle θ. The class allows evaluation
    on grids or point sets, extraction of the covariance matrix in the
    coordinate basis, and computation of projected standard deviations and
    correlation coefficients along the coordinate axes.

    Parameters
    ----------
    A : float, optional
        Amplitude that scales the normalized Gaussian. Default is 1.0.
    height : float, optional
        Peak height of the Gaussian. Mutually exclusive with ``A``.
    mu : Sequence[float]
        Center of the Gaussian.
    sigma : Sequence[float]
        Standard deviation along the principal axes.
    fwhm : Sequence[float] | float, optional
        Full width at half maximum along the principal axes; alternative to ``sigma``.
    Attributes
    ----------
    A : float, optional
        Amplitude that scales the normalized Gaussian. Default is 1.0.
    height : float
        Convenience attribute to get and set the peak height using ``A`` and ``_norm``.
    fwhm : Sequence[float]
        Convenience attribute to get and set the full width at half maximum.
    mu : Sequence[float]
        Center of the Gaussian.
    sigma : Sequence[float]
        Standard deviation along the principal axes.
    Sigma : ndarray of shape (2, 2)
        Covariance matrix in the (x, y) coordinate frame.
    Methods
    -------
    evaluate(coords)
        Evaluate the Gaussian on a grid or a collection of points.
    covariance()
        Return the N×N covariance matrix in the coordinate basis.
    get_covarianceariance_coefficient()
        Return the projected coordinate-axis standard deviations and
        correlation coefficient.
    from_covarianceariance(x0, y0, Sigma, amplitude=1.0)
        Build a `GaussianND` instance from a given N×N covariance matrix.

    Notes
    -----
    **1. Covariance construction**

    The Gaussian is parameterized by its principal-axis variances σ₁², σ₂²
    and rotation θ. The covariance in the coordinate frame is

    .. math::
        Σ = R(θ)
            \\begin{bmatrix}
                σ_1^2 & 0 \\\\
                0 & σ_2^2
            \\end{bmatrix}
            R(θ)^T
        \\tag{1}

    with the rotation matrix

    .. math::
        R(θ) =
            \\begin{bmatrix}
                \\cosθ & -\\sinθ \\\\
                \\sinθ &  \\cosθ
            \\end{bmatrix}.

    **2. Expanded components**

    From Eq. (1), the coordinate-frame covariances are:

    .. math::
        \\begin{aligned}
        Σ_{xx} &= σ_1^2 \\cos^2θ + σ_2^2 \\sin^2θ, \\\\
        Σ_{yy} &= σ_1^2 \\sin^2θ + σ_2^2 \\cos^2θ, \\\\
        Σ_{xy} &= (σ_1^2 - σ_2^2) \\sinθ \\cosθ.
        \\end{aligned}
        \\tag{2}

    **3. Projection to coordinate axes**

    The projected (marginal) standard deviations and correlation are:

    .. math::
        σ_x = \\sqrt{Σ_{xx}}, \\quad
        σ_y = \\sqrt{Σ_{yy}}, \\quad
        ρ = Σ_{xy} / (σ_x σ_y).
        \\tag{3}

    **4. Numerical stability**

    For evaluation, the class uses `numpy.linalg.solve` to compute
    Σ⁻¹·Δx without forming Σ⁻¹ explicitly. This is numerically stable
    and efficient, especially for repeated evaluations.

    Examples
    --------
    >>> g = Gaussian2D(0.0, 0.0, sigma1=3.0, sigma2=1.0,
    ...                theta=np.deg2rad(30), amplitude=1.0)
    >>> x = np.linspace(-10, 10, 201)
    >>> y = np.linspace(-10, 10, 201)
    >>> X, Y = np.meshgrid(x, y, indexing="xy")
    >>> Z = g.values((X, Y))  # evaluate on grid
    >>> g.covariance()
    array([[8.  , 2.6],
           [2.6 , 2.  ]])
    >>> g.get_covarianceariance_coefficient()
    {'sigma_x': 2.828..., 'sigma_y': 1.414..., 'rho': 0.46...}
    """

    def __init__(
        self,
        A: float | None = None,
        mu: Sequence[float] = (0.0, 0.0),
        sigma: Sequence[float] | float | None = None,
        #TODO "Include roation" project: Need to include a method to handle the rotation in N-dimensions. The best method I have seen is to 1) take k target direcions (a), 2) orthonormalize them (Gram-Schmidt) to get basis vectors (q_1...q_k), 3) Complete to an orthonormal basis (q_k+1...q_n), 4) form Q=[q1...qn] and if det(Q)<0 flip sign of last column, 5) compute the covariance matrix.
        # theta: Sequence[float] = (0.0),
        *,
        height: float | None = None,
        fwhm: Sequence[float] | float | None = None,
    ):
        # Check the validity of sigma/fwhm and A/height kwargs
        if sigma is not None and fwhm is not None:
            raise ValueError("Provide only one of sigma or fwhm.")
        elif sigma is None and fwhm is None:
            sigma = 1.0
        if A is not None and height is not None:
            raise ValueError("Provide only one of A or height.")
        elif A is None and height is None:
            A = 1.0

        self.mu = mu
        self.ndim = len(mu)
        if self.ndim == 0:
            raise ValueError("mu must have at least one dimension.")
        self._values = None
        self._covariance: NDArray | None = None
        self._det: float | None = None
        self._norm: float | None = None
        self._principle_axes: NDArray | None = None
        self._sigma: NDArray | None = None

        sigma_guess = sigma if sigma is not None else (np.asarray(fwhm) / sigma2fwhm if fwhm is not None else np.ones(self.ndim))
        self.sigma = sigma_guess

        self._ensure_covariance()
        if height is not None:
            self.A = float(height) / self._norm
        else:
            self.A = float(1.0 if A is None else A)
    
    def _invalidate_covariance(self):
        self._covariance = None
        self._det = None
        self._norm = None

    def _ensure_covariance(self):
        if self._covariance is None:
            Q = np.eye(self.ndim) #self.Q #TODO part of "Include roation" project
            D = np.diag(np.asarray(self._sigma)**2)

            cov = Q @ D @ Q.T
            det = np.linalg.det(cov)
            if det <= 0:
                raise np.linalg.LinAlgError("Covariance matrix must be SPD (determinant > 0).")
            self._covariance = cov
            self._det = det
            self._norm = (2.0 * np.pi)**(-self.ndim/2) * self._det**(-0.5)
            self._principle_axes = Q

    def _coerce_sigma(self, value: Sequence[float] | float) -> NDArray:
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            arr = np.repeat(arr, self.ndim)
        arr = arr.ravel()
        if arr.shape != (self.ndim,):
            raise ValueError(f"sigma must have length {self.ndim}.")
        if np.any(arr <= 0):
            raise ValueError("All sigma must be positive.")
        return arr

    @property
    def sigma(self) -> NDArray:
        return self._sigma
    @sigma.setter
    def sigma(self, value: Sequence[float] | float):
        self._sigma = self._coerce_sigma(value)
        self._invalidate_covariance()

    @property
    def fwhm(self):
        return self.sigma * sigma2fwhm
    @fwhm.setter
    def fwhm(self, value):
        self.sigma = np.asarray(value, dtype=float) / sigma2fwhm

    @property
    def height(self):
        self._ensure_covariance()
        return self.A * self._norm
    @height.setter
    def height(self, value):
        self._ensure_covariance()
        self.A = value / self._norm

    @property
    def covariance(self) -> NDArray:
        """Return and update the N×N covariance matrix Σ in the coordinate frame."""
        self._ensure_covariance()
        return self._covariance.copy()
    @covariance.setter
    def covariance(self, value: NDArray):
        """Set the N×N covariance matrix Σ in the coordinate frame."""
        covariance_dict = self.covariance_to_paramaters(value)

        self._covariance = covariance_dict['cov']
        self._det = covariance_dict['det']
        self._principle_axes = covariance_dict['principle_axes']
        self._sigma = covariance_dict['sigma']
        self._norm = (2.0 * np.pi)**(-self.ndim/2) * self._det**(-0.5)
    @staticmethod
    def covariance_to_paramaters(covariance: NDArray) -> Dict[str, float]:
        value = np.asarray(covariance, dtype=float)

        # Check value validity
        if not np.allclose(value, value.T, atol=1e-12): raise ValueError("Covariance must be symmetric.")

        # Check positive-definiteness
        det = np.linalg.det(value)
        if det <= 0:
            raise np.linalg.LinAlgError("Covariance matrix must be SPD (determinant > 0).")
        
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(value)  # eigh() for symmetric matrices #? Could this be done with SVD. Might be more practical for GaussianND.
        idx = np.argsort(eigvals)[::-1]# Sort by descending eigenvalue
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return {'cov': value, 'det': det, 'principle_axes': eigvecs, 
                'sigma': np.sqrt(eigvals)}
    def set_covarianceariance_derived_parameters(self):
        covariance_dict = self.covariance_to_paramaters(self._covariance)

        # Store covariance and determinant
        self._det = covariance_dict['det']
        self._principle_axes = covariance_dict['principle_axes']
        self._sigma = covariance_dict['sigma']
        self._norm = (2.0 * np.pi)**(-self.ndim/2) * self._det**(-0.5)

    def get_covarianceariance_coefficient(self) -> Dict[str, float]:
        """
        Compute projected (marginal) statistics along the coordinate axes.

        Returns
        -------
        dict
            Dictionary containing:
            - 'sigma' : Sequence[float], standard deviation of i
            - 'rho'   : Sequence[float], correlation coefficient between i and j
        """
        S = self.covariance
        s = np.sqrt(np.diag(S))
        D = np.diag(self.sigma)
        R = np.linalg.inv(D) @ S @ np.linalg.inv(D)
        return {"sigma": s, "rho": R}

    def evaluate(self, coords: Sequence[NDArray] | NDArray, 
                 store_values: bool = True, return_values: bool = True,
                 mahal_type: Literal['solve', 'inv'] = 'inv') -> NDArray:
        """
        Evaluate the Gaussian on a grid or at arbitrary points.

        Parameters
        ----------
        coords : array-like
            Either:
              - (X, Y): sequence of two 2D arrays of identical shape, or
              - array of shape (..., 2) with stacked coordinates.

        Returns
        -------
        vals : ndarray
            Array of Gaussian values with the same leading shape as `coords`.
        """
        if isinstance(coords, (list, tuple)):
            if len(coords) != self.ndim:
                raise ValueError(f"coords sequence must contain {self.ndim} dimensions.")
            coord_arrays = [np.asarray(c, dtype=float) for c in coords]
            grid_shape = coord_arrays[0].shape
            if any(arr.shape != grid_shape for arr in coord_arrays[1:]):
                raise ValueError("all coordinate arrays must have the same shape.")
            p = np.stack(coord_arrays, axis=-1)
        else:
            p = np.asarray(coords, dtype=float)
            if p.shape[-1] != self.ndim:
                raise ValueError(f"coords must have last dimension {self.ndim}.")

        dx = p - np.asarray(self.mu)

        self._ensure_covariance()

        if mahal_type == 'solve':
            M = dx.reshape(-1, self.ndim)
            y = np.linalg.solve(self.covariance, M.T)   # solves Σ y = Δxᵀ
            mahal = np.sum(M * y.T, axis=1).reshape(p.shape[:-1])
        elif mahal_type == 'inv':
            mahal = np.sum(dx @ np.linalg.inv(self.covariance) * dx, axis=-1)
        else:
            raise ValueError("mahal_type must be 'solve' or 'inv'.")

        vals = self._norm * self.A * np.exp(-0.5 * mahal) 
        if store_values: self._values = vals
        if return_values: return vals

    @staticmethod
    def from_covarianceariance(covariance: NDArray, 
                        A: float = 1.0, mu: Sequence[float] = ()
                        ) -> "GaussianND":
        """
        Construct a `GaussianND` from a N×N covariance matrix.

        Parameters
        ----------
        covariance : ndarray of shape (N, N)
            Symmetric positive-definite covariance matrix.
        mu : Sequence[float]
            Gaussian center coordinates. Default is ().
        amplitude : float, optional
            Amplitude of the Gaussian. Default is 1.0.

        Returns
        -------
        GaussianND
            New instance with parameters extracted from Σ.

        Notes
        -----
        The rotation θ and principal-axis variances (σ₁², σ₂²)
        are obtained from the eigen-decomposition Σ = VΛVᵀ.
        θ is the angle of the eigenvector corresponding to
        the largest eigenvalue with respect to the +x axis.
        """
        # Check validity of mu
        if len(mu) == 0: mu = (0.0,) * covariance.shape[0]
        elif len(mu) != covariance.shape[0]: raise ValueError(f"mu must have length {covariance.shape[0]}.")
        else: pass

        covariance_dict = GaussianND.covariance_to_paramaters(covariance)
        return GaussianND(A=A, mu=mu,
                          sigma=covariance_dict['sigma'])






#TODO: This should not be seperate from the Gaussian_nd class.
def gaussian_nd(x, A=1, mu=0, sigma=1):
    """
    Compute the probability density of an n-dimensional Gaussian distribution.
    
    Parameters:
    -----------
    x : array_like, shape (n,) or (m, n)
        Point(s) at which to evaluate the Gaussian. Can be a single point (1D array)
        or multiple points (2D array where each row is a point).
    mu : array_like, shape (n,)
        Mean vector of the Gaussian distribution.
    sigma : array_like, shape (n, n)
        Covariance matrix of the Gaussian distribution.
    
    Returns:
    --------
    pdf : float or array_like, shape (m,)
        Probability density at the given point(s).
    """
    x = np.atleast_2d(x)  # Ensure x is 2D for batch processing
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    n = len(mu)  # Dimensionality
    
    # Compute the normalization constant
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * np.linalg.det(sigma))
    
    # Compute the inverse of the covariance matrix
    sigma_inv = np.linalg.inv(sigma)
    
    # Compute the exponent for each point
    x_centered = x - mu
    exponent = -0.5 * np.sum(x_centered @ sigma_inv * x_centered, axis=-1)
    
    # Compute the PDF
    pdf = A * norm_const * np.exp(exponent)
    
    # Return scalar if input was 1D
    return pdf[0] if x.shape[0] == 1 else pdf

# class GaussianND():
#     def __init__(self,
#                  mean: NDArray,
#                  cov: NDArray,
#                  amplitude: float = 1.0,
#                  normalize: bool = False
#                  ) -> NDArray:
#         """
#         Evaluate an n-dimensional Gaussian:
#             f(x) = amplitude * exp(-0.5 * (x-mean)^T cov^{-1} (x-mean))
#         If normalize=True, multiplies by the normalizing constant
#             (2*pi)^{-n/2} det(cov)^{-1/2}.

#         Parameters
#         ----------
#         mean : ndarray, shape (n,)
#             Mean vector.
#         cov : ndarray, shape (n, n)
#             Symmetric positive-definite covariance matrix.
#         amplitude : float, optional
#             Scalar multiplier for the peak height (default 1.0).
#         normalize : bool, optional
#             If True, include the Gaussian normalizing constant.

#         Returns
#         -------
#         vals : ndarray
#             Array of Gaussian values with the same leading shape as `coords`
#             (for sequence input, the common grid shape).
#         """
#         self.mean = np.asarray(mean, dtype=float)
#         self.cov = np.asarray(cov, dtype=float)
#         self.amplitude = amplitude
#         self.normalize = normalize
#         self.ndim = self.mean.shape[0]

#         # Check the input covariance and mean shapes
#         if cov.ndim == 1:
#             cov = np.diag(cov)
#         elif cov.ndim ==2:
#             if cov.shape[0] != cov.shape[1]:
#                 raise ValueError("cov must be square (n x n).")
#         else:
#             raise ValueError("cov must be 1D or 2D array.")

#         if cov.shape[0] != (self.ndim,):
#             raise ValueError(f"mean must have shape ({self.ndim},).")

#     def evaluate(self, coords: NDArray | Sequence[NDArray], 
#                  store_values=True, return_values=True
#                  ) -> NDArray:
#         # Accept either stacked points (..., n) or a sequence of n arrays with same shape
#         if isinstance(coords, (list, tuple)):
#             # sequence of n arrays (grid-like)
#             if len(coords) != self.ndim:
#                 raise ValueError(f"coords sequence must have length {self.ndim}.")
#             coords = [np.asarray(c, dtype=float) for c in coords]
#             grid_shape = coords[0].shape
#             if any(c.shape != grid_shape for c in coords[1:]):
#                 raise ValueError("All coord arrays in the sequence must share the same shape.")
#             x = np.stack(coords, axis=-1)  # (..., n)
#         else:
#             x = np.asarray(coords, dtype=float)
#             if x.shape[-1] != self.ndim:
#                 raise ValueError(f"coords must have last dimension {self.ndim}.")
#             grid_shape = x.shape[:-1]

#         # meaned differences
#         dx = x - self.mean  # (..., n)
#         dx_2 = dx.reshape(-1, self.ndim)  # (M, n), where M = product of leading dims

#         # Solve cov * y = dx^T for y, then Mahalanobis = sum(DX * y^T, axis=-1)
#         # (avoids forming cov^{-1})
#         try:
#             y = np.linalg.solve(self.cov, dx_2.T)  # This solves for yi​=Σ^-1 (xi​−μ)
#         except np.linalg.LinAlgError as e:
#             raise np.linalg.LinAlgError("cov must be symmetric positive-definite (SPD).") from e

#         mahal = np.sum(dx_2 * y.T, axis=1)  # (M,)

#         vals = np.exp(-0.5 * mahal)

#         if self.normalize:
#             # Normalizing constant for N(mean, cov)
#             det_covariance = np.linalg.det(self.cov)
#             if det_covariance <= 0:
#                 raise np.linalg.LinAlgError("cov must have positive determinant.")
#             norm = (2.0 * np.pi) ** (-0.5 * self.ndim) * det_covariance ** (-0.5)
#             vals *= norm

#         vals *= float(self.amplitude)
#         vals = vals.reshape(grid_shape)
#         if store_values: self._values = vals
#         if return_values: return vals
