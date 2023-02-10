"""This is a tensorflow implementation of "The chirp z-transform
algorithm" Rabiner L. IEEE 1969.

Allows gradient backpropagation. For performance the constant factors
are precomputed and saved as constants with chosen precision.

Example:
    
    import numpy as np
    from diffao.czt_tf import czt, czt_factors

    RES = 512

    signal = np.zeros(RES)
    signal[RES//10:-RES//10] = 1.0

    cf = czt_factors(RES, RES)
    transformed = czt(signal, cf)
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple


def czt_factors(N: int, M: int,
                w_begin: float = -0.5, w_end: float = 0.5,
                L: Optional[int] = None,
                float_precision: Optional[str] = None) -> Tuple:
    """Precomputes constant factors used in CZT 1d CZT for input len N
    and output len M for frequency band from w_begint to w_end.
    
    Args:
        N: Length of the input signal.
        M: Length of the output signal.
        w_begin: Beginning frequency (cycles/sample) of the transform contour. 
        w_end: End frequency (cycles/sample) of the transform contour.
        L: Positive integer or None, optional. Length of FFT transform to use.
            Should be L>=M+N-1 (default), but for performance better to use 
            next biggest power of 2 integer. Second best is a highly composite
            integer; worst performance on prime numbers.
        float_precision: precision to specify the precomputed factors dtypes.
        
    Returns:
        Tuple: Precomputed factors that can be didectly used by czt_tf().
    """
    if float_precision is None:
        float_precision = tf.keras.backend.floatx()
    assert float_precision in (tf.dtypes.float32, tf.dtypes.float64), \
        f'Unsupported precision {float_precision}!'
    if float_precision == tf.dtypes.float32:
        dt_fl, dt_cx = 'float32', 'complex64'
    elif float_precision == tf.dtypes.float64:
        dt_fl, dt_cx = 'float64', 'complex128'
    print(f'Instantitaing CZT with {dt_fl} ({dt_cx}) floating point precision.')
    
    if L is None:
        L = N + M
        print(f'Using L={L}. You may suggest a better value for performance.')
    else:
        assert L >= N + M - 1, "Unsuitable L value!"
        
    n = np.arange(0, N)  # indices to enumerate input signal samples
    k = np.arange(0, M)  # and transform points
    r = np.arange(L - N, L) # convolution window indices
    
    A = 1.0 * np.exp(2j * np.pi * w_begin)
    delta_w = -2j * np.pi * (w_end - w_begin) / M
    
    # Starting point A and step W of the z-plane contour:
    A = 1.0 * np.exp(2j * np.pi * w_begin)
    delta_w = -2j * np.pi * (w_end - w_begin) / M
    W = 1.0 * np.exp(delta_w)
    
    z = A * W ** (-k) # z-plane points of the transform
    y_factor = A ** (-n) * W ** (n ** 2 / 2)
    
    v_n = np.zeros([L], dtype=dt_cx)
    v_n[:M] = W ** (-k ** 2 / 2)
    v_n[L - N:L + 1] = W ** (-(L - r) ** 2 / 2)
    
    V = np.fft.fft(v_n)
    g_factor = W ** (k ** 2 / 2)
    
    return (tf.constant(y_factor, dtype=dt_cx), tf.constant(V, dtype=dt_cx),
            tf.constant(g_factor, dtype=dt_cx), L, M, N)


def czt(x: tf.Tensor, factors: Tuple) -> tf.Tensor:
    """Computes czt transform with precalucalted factor parameters
    along the last axis of signal x.
    """
    yf, V, gf, L, M, N = factors
    paddings = [[0, 0] for _ in range(len(x.shape))]
    paddings[-1][-1] = L-N
    y = tf.pad(x*yf, paddings)
    Y = tf.signal.fft(y)
    G = V * Y
    g = tf.signal.ifft(G)
    return g[...,:M]*gf


def czt2d(x: tf.Tensor, factors: Tuple) -> tf.Tensor:
    """Computes 2d czt transform with precalucalted factor parameters
    along the last 2 axes of signal x.
    """
    axes_order = [_ for _ in range(len(x.shape))]
    axes_order.append(axes_order.pop(-2))
    return tf.transpose(czt(
        tf.transpose(czt(x, factors), axes_order), factors), axes_order)
