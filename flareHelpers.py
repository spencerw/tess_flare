import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def FINDflare(flux, error, N1=3, N2=1, N3=3,
              avg_std=False, std_window=7,
              returnbinary=False, debug=False):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005
    Note: these equations originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.
    Note: this algorithm originally ran over sections without "changes" as
    defined by Change Point Analysis. May have serious problems for data
    with dramatic starspot activity. If possible, remove starspot first!
    Parameters
    ----------
    flux : numpy array
        data to search over
    error : numpy array
        errors corresponding to data.
    N1 : int, optional
        Coefficient from original paper (Default is 3)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare
    avg_std : bool, optional
        Should the "sigma" in this data be computed by the median of
        the rolling().std()? (Default is False)
        (Not part of original algorithm)
    std_window : float, optional
        If avg_std=True, how big of a window should it use?
        (Default is 25 data points)
        (Not part of original algorithm)
    returnbinary : bool, optional
        Should code return the start and stop indicies of flares (default,
        set to False) or a binary array where 1=flares (set to True)
        (Not part of original algorithm)
    '''

    med_i = np.nanmedian(flux)

    if debug is True:
        print("DEBUG: med_i = {}".format(med_i))

    if avg_std is False:
        sig_i = np.nanstd(flux) # just the stddev of the window
    else:
        # take the average of the rolling stddev in the window.
        # better for windows w/ significant starspots being removed
        sig_i = np.nanmedian(pd.Series(flux).rolling(std_window, center=True).std())
    if debug is True:
        print("DEBUG: sig_i = ".format(sig_i))

    ca = flux - med_i
    cb = np.abs(flux - med_i) / sig_i
    cc = np.abs(flux - med_i - error) / sig_i

    if debug is True:
        print("DEBUG: N0={}, N1={}, N2={}".format(sum(ca>0),sum(cb>N1),sum(cc>N2)))

    # pass cuts from Eqns 3a,b,c
    ctmp = np.where((ca > 0) & (cb > N1) & (cc > N2))

    cindx = np.zeros_like(flux)
    cindx[ctmp] = 1

    # Need to find cumulative number of points that pass "ctmp"
    # Count in reverse!
    ConM = np.zeros_like(flux)
    # this requires a full pass thru the data -> bottleneck
    for k in range(2, len(flux)):
        ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])

    # these only defined between dl[i] and dr[i]
    # find flare start where values in ConM switch from 0 to >=N3
    istart_i = np.where((ConM[1:] >= N3) &
                        (ConM[0:-1] - ConM[1:] < 0))[0] + 1

    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i] - 1)

    istart_i = np.array(istart_i, dtype='int')
    istop_i = np.array(istop_i, dtype='int')

    if returnbinary is False:
        return istart_i, istop_i
    else:
        bin_out = np.zeros_like(flux, dtype='int')
        for k in range(len(istart_i)):
            bin_out[istart_i[k]:istop_i[k]+1] = 1
        return bin_out

def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_function(x):
    """Estimate the normalized autocorrelation function of a 1-D series
    .. note:: This is from `emcee <https://github.com/dfm/emcee>`_.
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        The autocorrelation function of the time series.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf

def autocorr_estimator(
    x,
    y,
    yerr=None,
    min_period=None,
    max_period=None,
    oversample=2.0,
    smooth=2.0,
    max_peaks=10,
):
    """Estimate the period of a time series using the autocorrelation function
    .. note:: The signal is interpolated onto a uniform grid in time so that
        the autocorrelation function can be computed.
    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        oversample (Optional[float]): When interpolating, oversample the times
            by this factor (default: 2.0)
        smooth (Optional[float]): Smooth the autocorrelation function by this
            factor times the minimum period (default: 2.0)
        max_peaks (Optional[int]): The maximum number of peaks to identify in
            the autocorrelation function (default: 10)
    Returns:
        A dictionary with the computed autocorrelation function and the
        estimated period. For compatibility with the
        :func:`lomb_scargle_estimator`, the period is returned as a list with
        the key ``peaks``.
    """
    if gaussian_filter is None:
        raise ImportError("scipy is required to use the autocorr estimator")

    if min_period is None:
        min_period = np.min(np.diff(x))
    if max_period is None:
        max_period = x.max() - x.min()

    # Interpolate onto an evenly spaced grid
    dx = np.min(np.diff(x)) / float(oversample)
    xx = np.arange(x.min(), x.max(), dx)
    yy = np.interp(xx, x, y)

    # Estimate the autocorrelation function
    tau = xx - x[0]
    acor = autocorr_function(yy)
    smooth = smooth * min_period
    acor = gaussian_filter(acor, smooth / dx)

    # Find the peaks
    peak_inds = (acor[1:-1] > acor[:-2]) & (acor[1:-1] > acor[2:])
    peak_inds = np.arange(1, len(acor) - 1)[peak_inds]
    peak_inds = peak_inds[tau[peak_inds] >= min_period]

    result = dict(autocorr=(tau, acor), peaks=[])

    # No peaks were found
    if len(peak_inds) == 0 or tau[peak_inds[0]] > max_period:
        return result

    # Only one peak was found
    if len(peak_inds) == 1:
        result["peaks"] = [
            dict(period=tau[peak_inds[0]], period_uncert=np.nan)
        ]
        return result

    # Check to see if second peak is higher
    if acor[peak_inds[1]] > acor[peak_inds[0]]:
        peak_inds = peak_inds[1:]

    # The first peak is larger than the maximum period
    if tau[peak_inds[0]] > max_period:
        return result

    result["peaks"] = [dict(period=tau[peak_inds[0]], period_uncert=np.nan)]
    return result