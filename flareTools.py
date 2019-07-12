import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, splrep, splev
import time, sys
from IPython.display import clear_output

# Progress bar for long for loops in ipython notebook
def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def id_segments(time, dt_limit, dt_trim=0):
    '''
    Identify breaks in a 1D array of time values larger than dt_limit
    Divide the array into segments and trim some amount of data off of
    the end of each segment
    Parameters
    ----------
    time : numpy array
        time values to segment
    dt_limit : float
        Smallest time interval to consider a 'break'
    dt_trim : float, optional
        Time interval to trim off of the ends of each segment
    Returns
    -------
        istart, istop which are each arrays containing the indices of the
        start and stop times in the 'time' array
    '''
    # Identify segments
    i_seg = np.array([0])
    dt = np.diff(time)
    for idx, val in enumerate(dt):
        if val > dt_limit:
            i_seg = np.append(i_seg, idx)
            i_seg = np.append(i_seg, idx+1)
    i_seg = np.append(i_seg, len(time)-1)
    
    # Trim the ends of the segments by dt_trim
    i_start = np.array([], dtype='int')
    i_end = np.array([], dtype='int')
    for idx in range(0, len(i_seg), 2):
        diff = (time - time[i_seg[idx]])
        i_left_new = np.where(diff > dt_trim)[0][0]
        i_seg[idx] = i_left_new
        
        diff = (time[i_seg[idx+1]] - time)
        i_right_new = np.where(diff > dt_trim)[0][-1]
        i_seg[idx+1] = i_right_new
        
        i_start = np.append(i_start, i_left_new)
        i_end = np.append(i_end, i_right_new)
    return i_start, i_end
    
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

def IRLSSpline(time, flux, error, Q=400.0, ksep=0.4, numpass=10, order=3, debug=False):
    '''
    IRLS = Iterative Re-weight Least Squares
    Do a multi-pass, weighted spline fit, with iterative down-weighting of
    outliers. This is a simple, highly flexible approach. Suspiciously good
    at times...
    Originally described by DFM: https://github.com/dfm/untrendy
    Likley not adequately reproduced here.
    uses scipy.interpolate.LSQUnivariateSpline
    Parameters
    ----------
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    Q : float, optional
        the penalty factor to give outlier data in subsequent passes
        (deafult is 400.0)
    ksep : float, optional
        the spline knot separation, in units of the light curve time
        (default is 0.07)
    numpass : int, optional
        the number of passes to take over the data (default is 5)
    order : int, optional
        the spline order to use (default is 3)
    debug : bool, optional
        used to print out troubleshooting things (default=False)
    Returns
    -------
    the final spline model
    '''

    weight = 1. / (error**2.0)

    knots = np.arange(np.nanmin(time) + 3*ksep, np.nanmax(time) - 2*ksep, ksep)

    # s1 = UnivariateSpline(time, flux, w=weight, k=3)
    # knots = s1.get_knots()


    if debug is True:
        print('IRLSSpline: knots: ', np.shape(knots))
        print('IRLSSpline: time: ', np.shape(time), np.nanmin(time), time[0], np.nanmax(time), time[-1])
        print('IRLSSpline: <weight> = ', np.mean(weight))
        print(np.where((time[1:] - time[:-1] < 0))[0])

        plt.figure()
        plt.errorbar(time, flux, error)
        plt.scatter(knots, knots*0. + np.median(flux))
        plt.show()

    for k in range(numpass):
        # print('IRLSSpline: k=', k)

        spl = LSQUnivariateSpline(time, flux, knots, k=order, check_finite=True, w=weight)

        # spl = UnivariateSpline(time, flux, w=weight, k=order, check_finite=True,
        #                        s=len(flux)*2.)
        # spl_model = splrep(time, flux, k=order, w=weight)

        chisq = ((flux - spl(time))**2.) / (error**2.0)

        weight = Q / ((error**2.0) * (chisq + Q))

    return spl(time)
