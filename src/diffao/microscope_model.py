"""This is our implementation of the computatonal model of 2-photon
microscope in Tensorflow. The necessary functions and transformations are
interfaced as keras.layer.
"""
import numpy as np
import tensorflow as tf
from diffao.czt_tf import czt_factors, czt2d

from typing import Optional
import numpy.typing as npt

# Used physical units:
UM: float = 1.0          # micrometer, main unit
NM: float = UM / 1000.0  # nanometer
MM: float = UM * 1000.0  # millimeter
PX: int = 1              # pixel

# Precision settings
DT_FL = 'float32'        # float dtype
DT_CX = 'complex64'      # complex dtype


class MicroscopeModel(object):
    """Encapsulates microscope image formation model with CZT
    integrator utilizing tensorflow. Initialization parameters
    correspond to parameters of the 2p-microscope in Seelig lab.
    """
    BASE_FOV_SIZE = 200 * UM  # Size of FOV of the microscope at zoom level 1.

    PMT_NONE = 'pmt_none'
    PMT_NONLINEARITY_ONLY = 'pmt_nonlinearity_only'
    PMT_POISSON = 'pmt_poisson'

    def __init__(self, zoom_level: float,
                 simulation_resolution: int = 512 * PX,
                 z_planes: list = [0 * UM],
                 object_function: Optional[npt.NDArray] = None,
                 pmt_simulation: str = PMT_NONLINEARITY_ONLY,
                 pmt_gain: float = 1.0,
                 pmt_bgr_noise_lvl: float = 0.0,
                 pmt_max_count: float = 5.0,
                 crop_output: bool = True) -> None:
        """Init microscope simulation with parameters:

        Args:
            zoom_level: Zoom level by the scanning mirrors.
            simulation_resolution: Resolution of the simulation.
            z_planes: List of z-plane offsets to simulate.
            object_function: Initialize model with object funciton.
            pmt_simulation: Options for PMT modelling PMT_NONE - no PMT;
                 PMT_NONLINEARITY_ONLY - models pmt intensity capping;
                 PMT_POISSON - models PMT noise and capping.
            pmt_gain: Gain of the PMT.
            pmt_bgr_noise_lvl: Background noise level of the PMT.
            pmt_max_count: PMT capping intensity (photon count).
            crop_output: if True will crop the output images.
        """
        self.zoom_lvl = zoom_level
        self.fov_size = self.BASE_FOV_SIZE / self.zoom_lvl
        self.fov_res = simulation_resolution
        
        out_fov = self.fov_size / 2 if crop_output else self.fov_size
        out_res = self.fov_res // 2 if crop_output else self.fov_res
        self.xrange = np.linspace(-out_fov / 2, out_fov / 2, out_res)
        self.yrange = np.linspace(-out_fov / 2, out_fov / 2, out_res)
        self.zrange = z_planes  # TODO proper 3d simulation
        
        print(f'Microscope model is initialized with FOV of {out_fov}um at {out_res}px resolution.')
        
        self.mo = MicroscopeObjective(
            d_pupil=6.7 * MM,
            f=3.3 * MM,
            n_t=1.33,   # Water
            wavelength=920 * NM,
            fov_size=self.fov_size,
            fov_res=self.fov_res,
            simulated_planes_z=z_planes
        )

        if object_function is not None:
            self.set_object_function(object_function)

        # Constructing the model:
        inp = tf.keras.layers.Input((self.fov_res, self.fov_res), dtype=DT_CX)
        _ = CircularBeam(phase_diff=-np.pi / 2)(inp)
        # _ = LinearBeam()(inp)
        _ = self.mo(_)
        
        if crop_output:
            _ = ImageCropper(self.fov_res // 2)(_)
            
        if pmt_simulation == self.PMT_NONLINEARITY_ONLY:
            self.pmt = PmtSimulatorRelu(pmt_gain, pmt_max_count, 0.0001)
        elif pmt_simulation == self.PMT_POISSON:
            self.pmt = PmtSimulatorPoisson(pmt_gain, pmt_max_count, pmt_bgr_noise_lvl)
        elif pmt_simulation == self.PMT_NONE:
            self.pmt = None

        if self.pmt is not None:
            _ = self.pmt(_)

        self.simulation = tf.keras.models.Model(inp, _)

    def set_object_function(self, new_object_function: npt.NDArray) -> None:
        """Assigns object function to the model.

        Args:
            new_object_function: NDarray, should match simulation resolution in size.
        """
        assert new_object_function.shape == self.mo.object_function.shape,\
            'Object function should match simulation resolution.'
        self.mo.object_function.assign(new_object_function)

    def bead_object_function(self, bead_size: float) -> None:
        """Makes object function with centered bead of given size,
        taking zoom level and FOV into account.

        Args:
            bead_size: Size of the bead, for example 0.5 * UM.
        """
        bead_of = np.zeros([self.fov_res, self.fov_res], dtype=DT_FL)
        cc_x, cc_y = np.mgrid[-self.fov_size / 2:self.fov_size / 2:self.fov_res * 1j,
                              -self.fov_size / 2:self.fov_size / 2:self.fov_res * 1j]
        bead_of[cc_x ** 2 + cc_y ** 2 < (bead_size / 2) ** 2] = 1.0
        self.set_object_function(bead_of)


class MicroscopeObjective(tf.keras.layers.Layer):
    """Encapsulates microscope objective model with Debye-Wolf vectorial
    diffraction integral as described in Boruah et al. "Focal field computation
    of an arbitrarily polarized beam using fast Fourier transforms."
    Optics communications (2009).
    """
    def __init__(self, d_pupil: float,
                 f: float,
                 n_t: float,
                 wavelength: float,
                 fov_size: float,
                 fov_res: int = 512,
                 simulated_planes_z: list = [0]) -> None:
        """
        Precomutes model parameter from parameters of the focused beam and
        of the required region of simulation.

        Args:
            d_pupil: Diameter of the objective pupil.
            f: Focal length of the objective.
            n_t: Refractive index of the medium.
            wavelength: Light wavelength lambda.
            fov_size: Size of simulated imaged field of view.
            fov_res: Resolution of the rendered FOV discretization.
            simulated_planes_z: List of offsets to simulated planes. With
                [0] only focal plane is simulated. Example: [-2*um, 0, 2*um]
                simulates 2 more z-planes at +-2 um.
        """
        super(MicroscopeObjective, self).__init__()
        self.n_t = n_t
        self.f = f
        self.wl = wavelength
        self.d = d_pupil
        self.fov_size = fov_size
        self.fov_res = fov_res
        # Angle of convergence of the beam:
        self.max_th = np.math.atan(self.d / (2 * self.f))

        # Numerical aperture:
        numerical_aperture = self.n_t * np.sin(np.math.atan(self.d / (2 * self.f)))
        print(f'Initialized Debye model with NA={np.round(numerical_aperture, 4)},\n'
              f'Angular aperture alpha={np.round(2 * self.max_th, 4)}')

        self.fov_scale = tf.constant(self.fov_res / self.fov_size, dtype=DT_FL)  # [px/um]

        # Calculating wave frequencies at FOV given its resolution and scale:
        k_0 = self.n_t * 2 * np.pi / self.wl  # Wavenumber [rad/um]
        k0_px = k_0 / (2 * np.pi) / self.fov_scale  # Cyclic freq fft units [cycles/px]

        # Maximal radial spatial frequency possible in [cycles / px]
        k_r_px_max = (k0_px * np.sin(self.max_th)).numpy()
        self.k_r_px_max = k_r_px_max  # Bandwidth

        # Initializing k-space meshgrids corresponding to the bandwidth:
        k_px = np.linspace(-k_r_px_max, k_r_px_max, self.fov_res)
        kxx_px, kyy_px = np.meshgrid(k_px, k_px)    # Careful with meshgrid/mgrid!
        krr_px = np.sqrt(kxx_px ** 2 + kyy_px ** 2)
        self.kzz_px = np.sqrt(k0_px ** 2 - krr_px ** 2)
        self.kzz_px[np.isnan(self.kzz_px)] = 0
        # Meshgrid on k-space with corresponding theta angles:
        self.thh_k = np.arctan2(krr_px, self.kzz_px)
        # Meshgrid on k-space with corresponding phi angles:
        self.phh_k = np.arctan2(kyy_px, kxx_px)
        # Bandwidth is basically a mask where spatial frequencies exist in k-space:
        self.bandwidth = self.thh_k < self.max_th

        # These constants used to map pupil function onto k-space,
        # see Boruah 2009 for derivation and details:
        self.Gx = np.sqrt(np.cos(self.thh_k)) *\
            (np.cos(self.thh_k) + (1 - np.cos(self.thh_k)) *
             np.sin(self.phh_k) ** 2) / np.cos(self.thh_k)
        self.Gy = np.sqrt(np.cos(self.thh_k)) *\
            ((np.cos(self.thh_k) - 1) * np.cos(self.phh_k) *
             np.sin(self.phh_k)) / np.cos(self.thh_k)
        self.Gz = -np.sqrt(np.cos(self.thh_k)) *\
            np.sin(self.thh_k) * np.cos(self.phh_k) / np.cos(self.thh_k)

        ## Initializing TF constants and variables:
        self.Gx = tf.constant(self.Gx, dtype=DT_CX)
        self.Gy = tf.constant(self.Gy, dtype=DT_CX)
        self.Gz = tf.constant(self.Gz, dtype=DT_CX)

        self.z_planes = tf.constant(simulated_planes_z, dtype=DT_FL)
        self.bandwidth = tf.constant(self.bandwidth, dtype=DT_CX)
        self.kzz_px = tf.constant(self.kzz_px, dtype=DT_CX)

        self.apodization = tf.Variable(
            tf.ones([self.fov_res, self.fov_res]),
            trainable=True,
            name='ApodizationFunction')
        self.aberration = tf.Variable(
            tf.zeros([self.fov_res, self.fov_res]),
            trainable=True,
            name='PhaseAberration')

        # Initial object function, just a single point in the center:
        of = np.zeros([self.fov_res, self.fov_res])
        of[self.fov_res // 2, self.fov_res // 2] = 1.0
        self.object_function = tf.Variable(of, dtype=DT_FL)

        # Used to store PSF intensity as one of optimization targets:
        self.total_intensity = tf.Variable(0, dtype=DT_FL)

        # Init CZT transform
        self.czt_factors = czt_factors(
            self.fov_res, self.fov_res,
            w_begin=-k_r_px_max, w_end=k_r_px_max, L=1024)

    @tf.function
    def sim(self, z_offset: tf.Tensor, pupil_x: tf.Tensor, pupil_y: tf.Tensor) -> tf.Tensor:
        """Computes vectorial diffraction at specified plane for X- and Y
        polarizations of the pupil.
        Args:
            z_offset: Offset of the simulated plane from the focal.
            pupil_x: Tensor [y, x], complex, for pupil x-polarization
            pupil_y: Tensor [y, x], complex, for pupil y-polarization

        Returns:
            Tensor [] with PSF at the requested plane.
        """

        # Propagation factor to the target z-plane:
        z_px = tf.complex(z_offset * self.fov_scale, tf.zeros_like(z_offset))
        prop_exp = tf.exp(2j * np.pi * self.kzz_px * z_px)  # [!] 2*pi factor
        # as kzz_px is in [cycles/px] and exponent takes rads [!]

        abb = tf.complex(tf.abs(self.apodization), tf.zeros_like(self.apodization)) * tf.exp(
            tf.complex(tf.zeros_like(self.aberration), self.aberration))

        # Refer to Boruah 2009.
        # Compute x- y- z-components of electic field at destination plane
        # induced by X-polarization of the pupil function:
        l0x = pupil_x * abb * prop_exp * self.bandwidth
        EXx = czt2d(l0x * self.Gx, self.czt_factors)
        EXy = czt2d(l0x * self.Gy, self.czt_factors)
        EXz = czt2d(-l0x * self.Gz, self.czt_factors)
        # induced by Y-polarization of the pupil function:
        l0y = pupil_y * abb * prop_exp * self.bandwidth
        EYx = czt2d(-l0y * tf.image.rot90(tf.expand_dims(self.Gy, axis=-1), k=1)[..., 0],
                    self.czt_factors)
        EYy = czt2d(l0y * tf.image.rot90(tf.expand_dims(self.Gx, axis=-1), k=1)[..., 0],
                    self.czt_factors)
        EYz = czt2d(l0y * tf.image.rot90(tf.expand_dims(self.Gz, axis=-1), k=1)[..., 0],
                    self.czt_factors)

        # Aggregate all contributions to PSF intensity and  square for fluorescence
        psf = tf.square((tf.square(tf.abs(EXx + EYx)) + tf.square(tf.abs(EXy + EYy)) + tf.square(
            tf.abs(EXz + EYz))) / 10e10)  # overflow protection

        self.total_intensity.assign_add(tf.reduce_sum(psf))
        # TODO: individual intensities for each batch and slice

        return tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.fft2d(
                tf.complex(psf, tf.zeros_like(psf))) * tf.signal.fft2d(
            tf.complex(tf.abs(self.object_function),
                       tf.zeros_like(self.object_function)))))) / 1e5

    @tf.function
    def sim_vol(self, pupil_xy: tf.Tensor) -> tf.Tensor:
        """Computes image for every z-plane of the simulation:
        Args:
            pupil_xy: Tensor [y, x, polarization], complex.

        Returns: Tensor [z, y, x], float, intensity of the fluorescence.
        """
        pupil_x, pupil_y = pupil_xy[..., 0], pupil_xy[..., 1]
        return tf.map_fn(lambda d: self.sim(d, pupil_x, pupil_y), self.z_planes)

    @tf.function
    def call(self, batch_pupil_xy: tf.Tensor) -> tf.Tensor:
        """Computes volume simulation for whole input batch

        Args:
            batch_pupil_xy: Tensor [BS, y, x, polarization], complex.

        Returns:
            Tensor [BS, z, y, x] with simulated volume intensity for each pupil
                function of the batch.
        """
        self.total_intensity.assign(0)
        return tf.map_fn(self.sim_vol, batch_pupil_xy, dtype=DT_FL)

class CircularBeam(tf.keras.layers.Layer):
    """Helper to convert phase surface to x- and y- components of _circularly_
    polarized beam pupil function.
    """
    def __init__(self, phase_diff: float = np.pi / 2) -> None:
        """Initializes beam with select parameters.

        Args:
            phase_diff: Difference in phase between x- and y- polarizations.
                By default makes circular beam. With other parameters elliptic
                or linear polarizations possible.
        """
        super(CircularBeam, self).__init__()
        self.phase_diff_exp = tf.exp(tf.constant(1j * phase_diff, dtype=DT_CX))

    @tf.function
    def call(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Args:
            batch: Tensor [BS, y, x], complex. Batch of phase-modulated pupils.

        Returns:
            Tensor [BS, y, x, polarization], complex.
        """
        return tf.stack([batch, batch * self.phase_diff_exp], axis=-1)


class LinearBeam(tf.keras.layers.Layer):
    """Helper to convert phase surface to x- and y- components of _linearly_
     polarized beam pupil function.
     """
    def __init__(self, alpha=0.0) -> None:
        """Initializes beam with select parameters.

        Args:
            alpha: angle of linear polarization.
        """
        super(LinearBeam, self).__init__()
        self.ca = tf.constant(np.cos(alpha), dtype=DT_CX)
        self.sa = tf.constant(np.sin(alpha), dtype=DT_CX)

    @tf.function
    def call(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Args:
            batch: Tensor [BS, y, x], complex. Batch of phase-modulated pupils.

        Returns:
            Tensor [BS, y, x, polarization], complex.
        """
        return tf.stack([self.ca * batch, self.sa * batch], axis=-1)


class RadialBeam(tf.keras.layers.Layer):
    """Helper to convert phase surface to x- and y- components of _radially_
     polarized beam pupil function. In every part of radial beam light is
     linearly polarized in radial direction.
    """
    def __init__(self, ap_xx: tf.Tensor, ap_yy: tf.Tensor) -> None:
        super(RadialBeam, self).__init__()
        phi = np.arctan2(ap_yy, ap_xx)
        self.ca = tf.constant(np.cos(phi), dtype=DT_CX)
        self.sa = tf.constant(np.sin(phi), dtype=DT_CX)

    @tf.function
    def call(self, batch: tf.Tensor) -> tf.Tensor:
        """
        Args:
            batch: Tensor [BS, y, x], complex. Batch of phase-modulated pupils.

        Returns:
            Tensor [BS, y, x, polarization], complex.
        """
        return tf.stack([self.ca * batch, self.sa * batch], axis=-1)


class AzimuthalBeam(tf.keras.layers.Layer):
    """Helper to convert phase surface to x- and y- components of _azimuthally_
     polarized beam pupil function. In every part of radial beam light is
     linearly polarized in tangential direction.
    """
    def __init__(self, ap_xx, ap_yy) -> None:
        super(AzimuthalBeam, self).__init__()
        phi = np.arctan2(ap_yy, ap_xx)+np.pi/2
        self.ca = tf.constant(np.cos(phi), dtype=DT_CX)
        self.sa = tf.constant(np.sin(phi), dtype=DT_CX)

    @tf.function
    def call(self, batch):
        """
        Args:
            batch: Tensor [BS, y, x], complex. Batch of phase-modulated pupils.

        Returns:
            Tensor [BS, y, x, polarization], complex.
        """
        return tf.stack([self.ca * batch, self.sa * batch], axis=-1)


class ImageCropper(tf.keras.layers.Layer):
    """Helper function to crop simulated images batch. Crops both
    x and y.
    """
    def __init__(self, crop_size: int) -> None:
        """
        Args:
            crop_size: Size to which crop the output.
        """
        super(ImageCropper, self).__init__()
        self.crop_size = crop_size
        print(f'ImageCropper crops to {self.crop_size}')

    @tf.function
    def call(self, images: tf.Tensor):
        """Crops last 2 dimensions whcih correspond to y and x of images to
        the specified size.

        Args:
            images: Input

        Returns:
            Tensor [... , y_cropped, x_cropped] - cropped input.
        """
        h, w = images.shape[-2:]
        cropped = images[...,
                   h // 2 - self.crop_size // 2:h // 2 + self.crop_size // 2,
                   w // 2 - self.crop_size // 2:w // 2 + self.crop_size // 2
                   ]
        return cropped


class PmtSimulatorRelu(tf.keras.layers.Layer):
    """Simulates PMT nonlinearity without shot noise. Multiplies
    image intensity by gain and caps max intensity. Leaky RelU for gradient
    preservation at capped intensities specified by alpha.
    """
    def __init__(self, gain: float, intensity_cap: float, relu_alpha: float) -> None:
        """Initialize PMT simulation parameters:

        Args:
            gain: Input intensity multiplier.
            intensity_cap: Intensity cap.
            relu_alpha: Leakiness of the intensity cap.
        """
        super(PmtSimulatorRelu, self).__init__()
        self.gain = tf.Variable(gain)
        self.relu = tf.keras.layers.LeakyReLU(relu_alpha)
        self.pmt_cap = tf.Variable(intensity_cap, dtype=DT_FL)

    @tf.function
    def call(self, intensities: tf.Tensor) -> tf.Tensor:
        return -self.relu(-intensities * self.gain + self.pmt_cap) + self.pmt_cap


class PmtSimulatorPoisson(tf.keras.layers.Layer):
    """Simulates PMT nonlinearity and shot noise using Poisson process.
    Multiplies intensities by gain and caps max intensity after drawing
    Poisson-distributed random output intensity. Models background noise.
    """
    def __init__(self, gain: float, intensity_cap: int, bgr_noise_lvl: float = 0.0):
        """Initialize PMT with poisson simulation parameters:

        Args:
            gain: Input intensity multiplier.
            intensity_cap: PMT intensity cap.
            bgr_noise_lvl: Background noise to add.
        """
        super(PmtSimulatorPoisson, self).__init__()
        self.gain = tf.Variable(gain, dtype=DT_FL)
        self.bgr = tf.Variable(bgr_noise_lvl, dtype=DT_FL)
        self.max_pmt_count = tf.Variable(intensity_cap, dtype=DT_FL)

    @tf.function
    def call(self, intensities):
        return tf.minimum(
            tf.random.poisson([1], (intensities + self.bgr) * self.gain),
            self.max_pmt_count)[0]
