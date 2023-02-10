"""Routine functions for generation of Zernike polynomials
used to represent phase aberrations. 

Some funcitons are taken from 
https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
The library by Tim van Werkhoven (werkhoven@strw.leidenuniv.nl) is
availible under Creative Commons Attribution-Share Alike license,
see http://creativecommons.org/licenses/by-sa/3.0/
It is explicitly stated which funcitons are adapted.
Docstring style and comments were changed.

Example:
    from wavefronts import Zernike
    zernike = Zernike()
    wavefront = zernike.phase([0, 1, 0, 0, 0, 3])
"""
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


class Zernike(object):
    """Class for generating functions from Zernike decomposition coefficients.
    Precomputes and stores Zernike modes up to chosen index for efficiency.
    """
    
    def __init__(self, resolution: int = 256, n_modes: int = 50,
                 mask_corners: bool = True,
                 xgrid: Optional[npt.NDArray] = None,
                 ygrid: Optional[npt.NDArray] = None) -> None:
        """Instantiates class precomputing and storing polynomial generation settings.
        
        Args:
            resolution: Resolution of the discretization grid.
            mask_corners: Option to zero polynomial outside of the
                unit circle.
            xgrid: x-coordinate custom grid to generate polynomials, if None
                uniformely-spaced (resolution x resolution) meshgrid created.
            ygrid: y-coordinate grid, same as xgrid.
        """
        if xgrid is None or ygrid is None:
            xx, yy = np.mgrid[-1:1:resolution * 1j, -1:1:resolution * 1j]
            print('Generating Zernikes on uniform grid...')
        else:
            xx, yy = xgrid, ygrid
            
        rho = np.array(np.sqrt(xx ** 2 + yy ** 2), dtype=np.float64)
        phi = np.array(np.arctan2(yy, xx), dtype=np.float64)
        self.mask = np.asarray(rho <= 1.0, dtype=np.float)
        
        self.modes = []
        self.resolution = resolution
        self.n_modes = n_modes
        for noll in range(1, self.n_modes+1):
            zmode = self._generate_mode(noll, rho, phi)
            if mask_corners:
                zmode *= self.mask
            self.modes.append(zmode)
            print(f'Generating mode {noll} of {self.n_modes}...', end='\r')
        self.modes = np.stack(self.modes, axis=-1)
        
    def opd(self, coeffs: list) -> npt.NDArray:
        """Generates OPD (optical path difference) surface corresponding to
        the given coeffs expansion. Output is in wavelengths.
        """
        c = np.zeros(self.n_modes)
        c[:len(coeffs)] = coeffs
        return np.asarray(np.dot(self.modes, c), dtype=np.float)

    def phase(self, coeffs: list) -> npt.NDArray:
        """Generates wavefront phase offset corresponding to the given
        coefficients expansion. Output is in radians.
        """
        return self.opd(coeffs)*np.pi*2.0
    
    @staticmethod
    def _generate_mode(noll: int, rho: npt.NDArray, phi: npt.NDArray) -> npt.NDArray:
        """Generates normalized i-th Zernike mode with initiated parameters.
        
        Args:
            noll: Noll's index of the polynomial.
            rho: Meshgrid of polar radius coordinates.
            phi: Meshgrid of polar angle coordinates.
        
        Returns:
            NDArray: Zernike polynomial.
        """
        n, m = Zernike._noll_to_zern(noll)
        mode = Zernike._z(n, m, rho, phi) * Zernike._norm_coeff(n, m)
        return mode

    @staticmethod
    def _noll_to_zern(j: int) -> Tuple:
        """(c) Tim van Werkhoven, adapted from
        https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
        
        Converts linear Noll index to tuple of Zernike indices.
        
        Zernike polynomials, although funcitons of two parameters
        (radial/angular frequences), can be enumerated with a single index
        using Noll's enumeration. Info can be found at
        https://oeis.org/A176988/b176988.txt
        
        Args:
            j: Zernike mode Noll index.
        
        Returns:
            Tuple: (n, m) Zernike indices (radial and tangential frequencies),
                where n is the radial Zernike index and m is the azimuthal
                Zernike index.
        """
        if j == 0:
            raise ValueError("Noll indices start at 1, 0 is invalid.")
        n = 0
        j1 = j - 1
        while j1 > n:
            n += 1
            j1 -= n
        m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
        return n, m
    
    @staticmethod
    def _norm_coeff(n: int, m: int) -> float:
        """Computes normalization coefficient of the Zernike mode, such that
        RMS(mode) = 1.0.
        
        Args:
            n: Radial Zernike index.
            m: Azimuthal Zernike index.
            
        Returns:
            float: Normalization coefficient.
        """
        return np.sqrt((2*(n+1))/(1+int(m==0)))
    
    @staticmethod
    def _r(n: int, m: int, rho: npt.NDArray) -> npt.NDArray:
        """Generates radial component of the Zernike polynomial on given
        radius meshgrid.
        
        Args:
            n: Radial Zernike index.
            m: Azimuthal Zernike index.
            rho: Meshgrid of polar radius coordinates.
            
        Returns:
            NDArray: Radial component of the Zernike polynomial.
        """
        ans = np.zeros_like(rho, dtype=np.float64)
        for k in range(0, int((n - m) / 2) + 1):
            ans = ans + np.power(rho, int(n - 2 * k)) * ((-1) ** k) *\
                  np.math.factorial(n - k) / float(
                    np.math.factorial(k) * np.math.factorial(int((n + m) / 2) - k) *
                    np.math.factorial(int((n - m) / 2) - k)
                  )
        return ans

    @staticmethod
    def _z(n: int, m: int, rho: npt.NDArray, phi: npt.NDArray) -> npt.NDArray:
        """Generates Zernike polynomial with chosen radial and azimuthal
        frequency on given polar meshgrid.
        
        Args:
            n: Radial Zernike index.
            m: Azimuthal Zernike index.
            rho: Meshgrid of polar radius coordinates.
            phi: Meshgrid of polar angle coordinates.
        
        Returns:
            NDArray: Zernike polynomial.
        """
        if m >= 0:
            return Zernike._r(n, m, rho) * np.cos(m * phi)
        else:
            return Zernike._r(n, -m, rho) * np.sin(-m * phi)
        

def axicone(resolution: int = 256, magnitude: float = 1.0,
            center_xy: Tuple[float, float] = (0.0, 0.0),
            xgrid: Optional[npt.NDArray] = None, ygrid: Optional[npt.NDArray] = None) -> npt.NDArray:
    """Generates Axicone surface on uniform meshgrid of given resolution or custom meshgrid.

    Args:
        resolution: Resolution of generated surface.
        magnitude: Axicone magnitude, i.e. "strength".
        center_xy: Axicone center coordinates, default - no offset.
        xgrid: x-coordinate custom grid to generate axicone, if None
            uniformely-spaced (resolution x resolution) meshgrid created.
        ygrid: y-coordinate grid, same as xgrid.

    Returns:
        NDArray: Axicone surface.
    """
    x0, y0 = center_xy
    if xgrid is None or ygrid is None:
        yy, xx = np.mgrid[-1:1:1j * resolution, -1:1:1j * resolution]
    else:
        yy, xx = ygrid, xgrid
    rho = np.sqrt((xx-x0) ** 2 + (yy-y0) ** 2)
    return rho*magnitude


def vortex(resolution: int = 256, magnitude: float = 1.0,
           center_xy: Tuple[float, float] =(0.0, 0.0),
           xgrid: Optional[npt.NDArray] = None, ygrid: Optional[npt.NDArray] = None) -> npt.NDArray:
    """Generates phase vortex surface on uniform meshgrid of given resolution or custom meshgrid.

    Args:
        resolution: Resolution of generated surface.
        magnitude: Vortex topological charge, how many times phase warps along a
            circular loop, i.e. "strength". For non-integer values a seam appears.
        center_xy: Vortex center coordinates, default - no offset.
        xgrid: x-coordinate custom grid to generate axicone, if None
            uniformly-spaced (resolution x resolution) meshgrid created.
        ygrid: y-coordinate grid, same as xgrid.

    Returns:
        NDArray: Phase vortex surface.
    """
    x0, y0 = center_xy
    if xgrid is None or ygrid is None:
        yy, xx = np.mgrid[-1:1:1j * resolution, -1:1:1j * resolution]
    else:
        yy, xx = ygrid, xgrid
    angle = np.arctan2(yy-y0, xx-x0)
    return angle*magnitude
