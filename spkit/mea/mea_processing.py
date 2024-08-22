"""

MEA Processing Library
--------------------
Author @ Nikesh Bajaj
pdated on Date: 27 March 2023. Version : 0.0.2
updated on Date: 16 March 2023, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk | nikesh.bajaj@qmul.ac.uk

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys, pickle, copy, time
import warnings
from scipy.signal import find_peaks,find_peaks_cwt,peak_prominences, peak_widths
import seaborn as sns
#import io.read_hdf as read_hdf
#from io_utilis import read_hdf
from ..utils_misc.io_utils import read_hdf
from ..core.processing import fill_nans_2d, conv2d_nan, filterDC_sGolay
from ..core.processing import get_activation_time, get_repolarisation_time, agg_angles, show_compass, direction_flow_map
from ..utils import bcolors


def get_stim_loc(X,fs=25000,ch=0,fhz=1,N=None,method='max_dvdt',gradient_method='fdiff',sg_window=11,sg_polyorder=3,gauss_window=0,gauss_itr=1, plot=1,verbose=False,figsize=(15,3)):

    r"""


    Get Stimulus Locations in given multi-channel X (nch,n)
    -------------------------------------------------------

    This uses 'get_activation_time' from spkit.get_activation_time function. Check help(get_activation_time) for more details


    Parameters
    ----------
    X : nd-array of shape = (nch,n) = (number of channels, number of samples), multi-channel signal recording
    fs: sampling frequency of signal, default =25KHz for MEA
    ch: channel number of extract stimlus locations, default=0
    fhz: frequency of stimulus, e.g. fhz=1 means 1 stimulus per second
    N : Number of stimulus to be extracted, if None, N = number_samples/fs/fhz

    method : method to locate stimulus, could be one of {'max_dvdt', 'min_dvdt', 'abs_dvdt'}
           for MEA, max_dvdt - maximum gradient works well, which is a middle transition of stimulus from negative to positive,
           default = 'max_dvdt'

    gradient_method: method to compute gradient of signal, could be one of {"fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
           for MEA, fdiff works well.
           For more details, check help(signal_diff) from spkit.signal_diff

    (sg_window=11,sg_polyorder=3,gauss_window=0,gauss_itr=1): these parameters are used while gradient computation, only
        if gradient method is one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff)
           For more details, check help(signal_diff) from spkit.signal_diff


    plot: int
         if 0, no plot are shown,
         if 1, final locations of stimulus are plotted
         if 2, plot of each stimulus with signal are shown, N-plots are shown

    verbose: if False, no information is printed, default if False

    figsize: figure size for final figure (if plot>0), default figsize = (15,3)



    Returns
    -------

    stim_loc: list of locations (as index) of all the stimuli
    stim_mag: magnitude of deflection, at stim locatio. It can be used to determine the False Positive

    Example
    -------
    #TODO


    See Also
    --------
    get_activation_time, signal_diff


    """
    fs_stim = fs
    if fhz>1:
        fs_stim = fs/fhz
        if verbose: print(f'-cycle_dur= {fs_stim/fs} s')
    if N is None:
        N = round(X.shape[1]/fs_stim)
        if verbose: print(f'-#cycles= {N}')

    stim_loc = []
    stim_mag = []
    for k in range(N):
        if k<N-1:
            xi = X[ch,int(fs_stim*k):int(fs_stim*(k+1))]
            xj = X[:,int(fs_stim*k):int(fs_stim*(k+1))]
        else:
            xi = X[ch,int(fs_stim*k):]
            xj = X[:,int(fs_stim*k):]


        #at_i,loc_i,mag_i = get_activation_time(xi,fs,method=method,gradient_type=gradient_type)
        at_i,loc_i,mag_i, dx_i = get_activation_time(xi,fs,method=method,gradient_method=gradient_method,sg_window=sg_window,sg_polyorder=sg_polyorder,
                                                     gauss_window=gauss_window,gauss_itr=gauss_itr)

        loc_j = loc_i + int(fs_stim*k)

        stim_loc.append(loc_j)
        stim_mag.append([mag_i,max(xi),min(xi)])

        if plot>1:
            plt.figure(figsize=(10,3))
            t = np.arange(len(xi))/fs
            if plot==1:
                plt.plot(t,xi,'C0')
            else:
                plt.plot(t,xj.T,'C0',alpha=0.5)
            plt.axvline(t[loc_i],color='C3')
            plt.xlim([t[loc_i-1000],t[loc_i+1000]])
            plt.title(f'Cycle #{k+1}')
            plt.xlabel('time (s)')
            plt.ylabel('voltage')
            plt.grid()
            plt.show()
    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.vlines(stim_loc,0,1)
        plt.xticks(stim_loc)
        plt.xlabel('loc (index)')
        plt.ylim([0,1])
        plt.ylabel(' ')
        plt.title(f'Stimulus spikes loc., average duration = {np.diff(stim_loc).mean().round(2)} samples')

        stim_loc_time = np.array(stim_loc)/fs
        plt.subplot(212)
        plt.vlines(stim_loc_time,0,1)
        plt.xticks(stim_loc_time)
        plt.ylim([0,1])
        plt.xlabel('time (s)')
        plt.title(f'Stimulus spikes time (s), average duration = {np.diff(stim_loc_time).mean().round(2)} s')
        plt.tight_layout()
        plt.show()
    return stim_loc,stim_mag

def _num2str(x,n=5):
    r"""convert number to string of fixed length

    """
    str_i = str(x)
    if len(str_i)<n: str_i = ' '*(n-len(str_i)) + str_i
    return str_i[:n]

def find_bad_channels_idx_v0(X,thr=0.0001,fs=25000,mnmx=[None, None],plot=True,verbose=0):

    r"""

    Idenify the Bad Channels of MEA based on stimuli
    ------------------------------------------------

    This function compute the average time of stimuli over the maximum voltage.
    in MEA, ideally, a stimulus is provided with 1ms at negative max voltage (~ -1000) mV/µV and 1ms positive (~ +1000), in one cylce
    So, if stimulus last more than 2ms per cycle, it is either a bad electode or a bad recording.


    Parameters
    ----------
    X   : nd-array of shape = (nch,n) = (number of channels, number of samples), multi-channel signal recording
    fs  : sampling frequency of signal, default =25KHz for MEA
    thr : threshold duration for stimuli, if over the threshold on either side, channel is flagged as BAD

         NOTE: Lower the threshold, more channels will be flagged, higher the threshold less channels will be flagged


    stim_fhz : stimuli frequency, number of stimuli per second

    mnmx: min, max voltage of stimuli, if passed [None, None], default, then for each chennels, it is computed by its min, max value
    plot: if true, all channels are plotted with 'plot_dur' duration
    plot_dur: used for plooting channels
    verbose: 0, Off mode
           : 1, a few information printed
           : 2, list of all channel printed with computed values and BAD flag label


    Returns
    -------
    bad_channels_idx: list of channel index which are flagged as BAD

    Example
    -------
    #TODO

    See Also
    --------
    #TODO

    """
    CRED = '\033[91m'
    ENDC = '\033[0m'

    bad_channels_idx = []
    for k,x in enumerate(X):
        if mnmx[0] is None:
            mx,mn =max(x), min(x)
        else:
            mx,mn = mnmx[0],mnmx[1]
        mxf, mnf = np.mean(x>=mx)/(len(x)/fs), np.mean(x<=mn)/(len(x)/fs)

        bad=False
        if (mxf>thr) or (mnf>thr):
            bad_channels_idx.append(k)
            bad=True
        if verbose>1:
            if bad:
                print(f'{CRED}{k} - \t{_num2str(mn.round(1),n=5)}-{_num2str(mx.round(1),n=5)}|\t{_num2str(mnf.round(4),n=7)} - {_num2str(mxf.round(4),n=7)}|\t  --BAD?:{bad}{ENDC}')
            else:
                print(f'{k} - \t{_num2str(mn.round(1),n=5)}-{_num2str(mx.round(1),n=5)}|\t{_num2str(mnf.round(4),n=7)} - {_num2str(mxf.round(4),n=7)}|\t  --BAD?:{bad}')
        if plot:
            print(mn,mx,mnf,mxf)
            if bad:
                plt.plot(x[:int(fs*1)],'r')
            else:
                plt.plot(x[:int(fs*1)])
            plt.title(f'{k} - MN:{np.around(mnf,4)} - MX:{np.around(mxf,4)}')
            plt.show()
    return bad_channels_idx

def find_bad_channels_idx(X,fs=25000,thr=2,stim_fhz=1,mnmx=[None, None],plot=False,plot_dur=2,verbose=1):

    r"""


    Idenify the Bad Channels of MEA based on stimuli
    ------------------------------------------------

    This function compute the average time of stimuli over to the maximum voltage.
    In MEA, ideally, a stimulus is provided with 1ms at negative max voltage (~ -1000) mV/µV and 1ms positive (~ +1000), in one cylce
    So, if stimulus last more than 2ms (thr) per cycle on either side, it is flagged as a bad electode.




    Parameters
    ----------
    X   : nd-array of shape = (nch,n) = (number of channels, number of samples), multi-channel signal recording
    fs  : sampling frequency of signal, default =25KHz for MEA
    thr : threshold duration for stimuli, if over the threshold on either side, channel is flagged as BAD

    stim_fhz : stimuli frequency, number of stimuli per second

    mnmx: min, max voltage of stimuli, if passed [None, None], default, then for each chennels, it is computed by its min, max value
    plot: if true, all channels are plotted with 'plot_dur' duration
    plot_dur: used for plooting channels
    verbose: 0, Off mode
           : 1, a few information printed
           : 2, list of all channel printed with computed values and BAD flag label


    Returns
    -------
    bad_channels_idx: list of channel index which are flagged as BAD

    Example
    -------
    #TODO

    See Also
    --------
    #TODO

    """

    CRED = '\033[91m'
    ENDC = '\033[0m'

    bad_channels_idx = []

    total_dur_s = X.shape[1]/fs
    total_stim_n = round(total_dur_s*stim_fhz)

    if verbose:
        print(f'Total Duration of recording = {total_dur_s} s and number of stimuli N = {total_stim_n}')
        print(f'If average stimuli duration (dur/cycle) on +ve or on -ve side of volatage is > {thr} ms, it is BAD')

    for k,x in enumerate(X):
        if mnmx[0] is None:
            mx,mn =max(x), min(x)
        else:
            mx,mn = mnmx[0],mnmx[1]


        mxf, mnf = 1000*np.sum(x>=mx)/fs/total_stim_n, 1000*np.sum(x<=mn)/fs/total_stim_n

        bad=False
        if (mxf>=thr) or (mnf>=thr):
            bad_channels_idx.append(k)
            bad=True
        if verbose>1:
            if bad:
                print(f'{CRED}{k} - \t{_num2str(mn.round(1),n=5)}-{_num2str(mx.round(1),n=5)}|\t{_num2str(mnf.round(2),n=5)} - {_num2str(mxf.round(2),n=5)}|\t  --BAD?:{bad}{ENDC}')
            else:
                print(f'{k} - \t{_num2str(mn.round(1),n=5)}-{_num2str(mx.round(1),n=5)}|\t{_num2str(mnf.round(2),n=5)} - {_num2str(mxf.round(2),n=5)}|\t  --BAD?:{bad}')
        if plot:
            print(mn,mx,mnf,mxf)
            if bad:
                plt.plot(x[:int(fs*plot_dur)],'r')
                plt.title(f'{k} - MN:{np.around(mnf,2)} - MX:{np.around(mxf,2)} | BAD')
            else:
                plt.plot(x[:int(fs*plot_dur)])
                plt.title(f'{k} - MN:{np.around(mnf,2)} - MX:{np.around(mxf,2)} | GOOD')
            plt.show()
    return bad_channels_idx

def ch_label2idx(ch_list,ch_labels):
    r"""mapping channel labels to index

    """
    return [list(ch_labels).index(ch) for ch in ch_list]

def ch_idx2label(ch_idx,ch_labels):
    r"""mapping channel index to labels

    """
    return list(np.array(ch_labels)[ch_idx])

def align_cycles(X,stim_loc,fs=25000,exclude_first_dur=1,dur_after_spike=500,exclude_last_cycle=True,pad=np.nan,verbose=False,**kwargs):
    r"""


    Align Cycles
    ------------

    Aligning all the cycles with stimulus location.


    Parameters
    ----------
    X   : nd-array of shape = (nch,n) = (number of channels, number of samples), multi-channel signal recording
    fs  : sampling frequency of signal, default =25KHz for MEA

    stim_loc: list of locations (indexes) of stimuli

    exclude_first_dur : float, default=1
                        Exclude the duration (in ms) of signal after stimulus spike loc, default 1 ms
                        It depends on the method of finding stim loc, since stim is 1ms -ve, 1ms +v, and loc is
                        detected as middle transaction (max_dvdt), atleast 1ms of duration should be excluded.

    dur_after_spike   : float, int, defult=500,
                        Extract the duration (in ms) after stimulus spike to search for EGM
                        Default 500 ms, good for 1Hz stimuli cycle

    exclude_last_cycle: bool, if True, last cycle is excluded in aligned cycles. Usually, a good idea to exclude,
                        since last cylce might have a very  small number of samples after spike.
                        default = True

    pad:  In case of any cycle being shorter than other (usually last cycle), padding values to make all cycles in equal size
          default = np.nan, good to use nan, to avoid using those padded values.

    verbose: print information, if True. default=False

    Returns
    -------
    XB:  Aligned Cycles is shape of (n_ch, n_samples, n_cycles)

    Example
    -------
    #TODO

    See Also
    --------
    #TODO

    """

    XB = []
    spikes_ = stim_loc.copy()

    if exclude_last_cycle: spikes_ = spikes_[:-1]

    for loc in spikes_:
        dur_end = int(loc + fs*(dur_after_spike/1000))
        if dur_end<X.shape[1]:
            Xi = X[:,loc:dur_end]
        else:
            Xi = X[:,loc:]

        XB.append(Xi)

    if not(exclude_last_cycle):
        Xi = XB[-1]
        Xj = XB[-2]
        n = Xi.shape[1]
        m = Xj.shape[1]
        if n!=m:
            if n<m:
                Xii = np.c_[Xi,np.ones([Xi.shape[0],m-n])*pad]
                if verbose: print(Xii.shape)
                XB[-1] = Xii

    XB = np.array(XB).transpose([1,2,0])

    if exclude_first_dur>0:
        XB = XB[:,int(fs*exclude_first_dur/1000):,:]

    if verbose: print('Shape:',XB.shape)
    return XB

def activation_time_loc(X,fs=25000,t_range=[None,None],method='min_dvdt',gradient_method='fdiff',sg_window= 11,sg_polyorder=3,gauss_window=0,gauss_itr=1,plot=False,plot_dur=2,**kwargs):
    r"""


    Compute Activation Time of multi-channel signals
    -------------------------------------------------

    Same as 'Get Activation Time based on Gradient'

    Activation Time in cardiac electrograms refered to as time at which depolarisation of cells/tissues/heart occures.

    For biological signals (e.g. cardiac electorgram), an activation time in signal is reflected by maximum negative deflection,
    which is equal to min-dvdt, if signal is a volatge signal and function of time x = v(t)
    However, simply computing derivative of signal is sometime misleads the activation time location, due to noise, so derivative of
    a given signal has be computed after pre-processing

    Parameters
    ----------
    X  : (nch,n) - nch-channel signal, axis=0 is a channel.
    fs : sampling frequency,


    t_range: list of [t0 (ms),t1 (ms)]
             range of time to restrict the search of activation time during t0 ms to t1 ms
             if t_range=[None,None], whole input signal is considered for search
             if t_range=[t0,None], excluding signal before t0 ms
             if t_range=[None,t1], excluding signal after t1 ms for search

    method : Method to compute activation time, one of ("max_dvdt", "min_dvdt", "max_abs_dvdt")
             for more detail check help(get_activation_time)

    gradient_method: Method to compute gradient of signal
                    one of ("fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
                    check help(signal_diff) from sp.signal_diff

    Parameters for gradient_method:
    used if gradient_method in one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
        sg_window: sgolay-filter's window length
        sg_polyorder: sgolay-filter's polynomial order
        gauss_window: window size of gaussian kernel for smoothing,
        check help(signal_diff) from sp.signal_diff


    plot:  If true, plot 3 figures for each channel
           (1) Full signal trace with activation time
           (2) Segment of signal from loc-plot_dur to loc+plot_dur
           (3) Derivative of signal, with with activation time
           default=False


    plot_dur: duration in seconds to plot, if plot=True
              segment of signal from loc-plot_dur to loc+plot_dur
               default=2s

    Returns
    -------
    at_loc: 1d-array of length=nch, location of activation time as index
            to convert it in seconds, at_loc/fs


    Example
    -------
    #TODO

    See Also
    --------
    get_activation_time, get_repolarisation_time, activation_repol_time_loc

    """


    at_loc = []
    for k, x in enumerate(X):
        xi = x.copy()
        t0,t1 = 0,None
        if t_range[0] is not None:
            t0 = int(fs*t_range[0]/1000)
        if t_range[1] is not None:
            t1 = int(fs*t_range[1]/1000)

        xi = x[t0:t1] if t1 is not None else x[t0:]

        at_i,loc_i,mag_i,dx_i = get_activation_time(xi.copy(),fs,method=method,gradient_method=gradient_method,
                                                    sg_window=sg_window,sg_polyorder=sg_polyorder,
                                                    gauss_window=gauss_window,gauss_itr=gauss_itr)

        #loc_i+=t0
        at_loc.append(loc_i+t0)
        if plot:
            plt.figure(figsize=(15,3))
            tx1 = 1000*np.arange(len(xi))/fs
            plt.subplot(131)
            plt.plot(tx1,xi)
            plt.axvline(tx1[loc_i],color='C3',ls='--',alpha=0.5)
            plt.xlabel('time (ms)')
            plt.title(f'x: signal #{k}')

            plt.subplot(132)
            t0,t1 = loc_i-int(fs*plot_dur/1000),loc_i+int(fs*plot_dur/1000)
            if t0<0: t0=0
            if t1>len(xi): t1 = len(xi)

            tx = 1000*(t0+np.arange(len(xi[t0:t1])))/fs
            plt.plot(tx, xi[t0:t1])
            plt.axvline(tx[loc_i-t0],color='C3',alpha=0.5)
            plt.xlabel('time (ms)')
            plt.title(f'x: signal #{k}')

            plt.subplot(133)
            plt.plot(tx1,dx_i)
            plt.axvline(tx1[loc_i],color='C3',alpha=0.5,label='AT')
            plt.xlabel('time (ms)')
            plt.legend()
            plt.title(f'dx: derivative #{k}')
            plt.show()
    return np.array(at_loc)

def activation_repol_time_loc(X,fs=25000,at_range=[None,None], rt_range=[0.5, None],method='min_dvdt',
                              gradient_method='fdiff',sg_window= 11,sg_polyorder=3,gauss_window=0,gauss_itr=1,plot=False,plot_dur=2,**kwargs):

    r"""


    Computing Activation Time and Repolarisation Time of multi-channel signals
    --------------------------------------------------------------------------

    Activation Time in cardiac electrograms refered to as time at which depolarisation of cells/tissues/heart occures.
    In contrast to 'Activation Time' in cardiac electrograms, Repolarisation Time, also refered as Recovery Time,
    indicates a time at which repolarisation of cells/tissues/heart occures.

    For biological signals (e.g. cardiac electorgram), an activation time in signal is reflected by maximum negative deflection,
    which is equal to min-dvdt, if signal is a volatge signal and function of time x = v(t) and repolarisation time is again a reflected
    by maximum deflection (mostly negative), after activation occures.

    However, simply computing derivative of signal is sometime misleads the activation and reoolarisation time location,
    due to noise, so derivative of a given signal has be computed after pre-processing. Repolarisation Time is often very hard to detect
    reliably, due to very small electrogram, which is mostly lost in noise.

    Parameters
    ----------
    X  : (nch,n) - nch-channel signal, axis=0 is a channel.
    fs : sampling frequency,


    at_range: list of [t0 (ms),t1 (ms)]
    rt_range: list of [t0 (ms),t1 (ms)]
             range of time to restrict the search of activation time and repolarisation time
            check help(get_activation_time) and help(get_repolarisation_time) for more details

    method : Method to compute activation time, one of ("max_dvdt", "min_dvdt", "max_abs_dvdt")
             check help(get_activation_time) and help(get_repolarisation_time) for more details

    gradient_method: Method to compute gradient of signal
                    one of ("fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
                    check help(signal_diff) from sp.signal_diff

        NOTE: Same 'method' and 'gradient_method' is applied to both AT and RT computation.
             To use different methods use 'get_activation_time' and 'get_repolarisation_time' seperatly

    Parameters for gradient_method:
    used if gradient_method in one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
        sg_window: sgolay-filter's window length
        sg_polyorder: sgolay-filter's polynomial order
        gauss_window: window size of gaussian kernel for smoothing,
        check help(signal_diff) from sp.signal_diff


    plot:  If true, plot 4 figures for each channel
           (1) Full signal trace with activation time and repolarisation time
           (2) Segment of signal from loc-plot_dur to loc+plot_dur for activation time
           (3) Segment of signal from loc-plot_dur to loc+plot_dur for repolarisation time
           (4) Derivative of signal, with with activation time and repolarisation time

           default=False


    plot_dur: duration in seconds to plot, if plot=True
              segment of signal from loc-plot_dur to loc+plot_dur
               default=2s

    Returns
    -------
    at_loc: 1d-array of length=nch, location of activation time as index
            to convert it in seconds, at_loc/fs


    Example
    -------
    #TODO

    See Also
    --------
    get_activation_time, get_repolarisation_time, activation_time_loc

    """

    at_loc = []
    rt_loc = []
    for k, x in enumerate(X):
        #xi = x.copy()
        t0,t1 = 0,None
        if at_range[0] is not None:
            t0 = int(fs*at_range[0]/1000)
        if at_range[1] is not None:
            t1 = int(fs*at_range[1]/1000)

        x_at = x[t0:t1] if t1 is not None else x[t0:]

        at_i,loc_i,mag_i,dx_i = get_activation_time(x_at.copy(),fs,method=method,gradient_method=gradient_method,
                                                    sg_window=sg_window,sg_polyorder=sg_polyorder,
                                                    gauss_window=gauss_window,gauss_itr=gauss_itr)

        #loc_i+=t0
        at_loc.append(loc_i+t0)


        x_rt = x[t0:].copy()


        rt_j,loc_j,mag_j,dx_j = get_repolarisation_time(x_rt,fs,loc_i,t_range=rt_range,method=method,gradient_method=gradient_method,
                                                    sg_window=sg_window,sg_polyorder=sg_polyorder,
                                                    gauss_window=gauss_window,gauss_itr=gauss_itr)
        rt_loc.append(loc_j+t0)


        if plot:
            tx = 1000*np.arange(len(x))/fs


            plt.figure(figsize=(18,3))
            plt.subplot(141)
            plt.plot(tx,x,'C0',alpha=0.3)
            plt.plot(tx[t0:t0+len(x_at)],x_at,'C0')
            plt.axvline(tx[loc_i+t0],color='C2',ls='--',alpha=0.5)
            plt.axvline(tx[loc_j+t0],color='C3',ls='--',alpha=0.5)
            plt.fill_between(tx,x*0 + np.nanmax(x),np.nanmin(x),where=(tx>=tx[loc_i+t0]) & (tx<=tx[loc_j+t0]),alpha=0.2)
            plt.xlabel('time (ms)')
            plt.title(f'x: signal #{k}')

            plt.subplot(142)
            dur_samples = int(fs*plot_dur/1000)
            p0,p1 = loc_i-dur_samples,loc_i+dur_samples
            if p0<0: p0=0
            if p1>len(x_at): p1 = len(x_at)
            at_egm = x_at[p0:p1]
            tx_at_egm = 1000*(p0+np.arange(len(at_egm)))/fs


            plt.plot(tx_at_egm,at_egm)
            plt.axvline(tx_at_egm[loc_i-p0],color='C2',alpha=0.99)
            plt.xlabel('time (ms)')
            plt.title(f'x: Act EGM #{k}')

            plt.subplot(143)
            p0,p1 = loc_j-dur_samples,loc_j+dur_samples
            if p0<0: p0=0
            if p1>len(x_rt): p1 = len(x_rt)
            rt_egm = x_rt[p0:p1]
            tx_rt_egm = 1000*(p0+np.arange(len(rt_egm)))/fs


            plt.plot(tx_rt_egm,rt_egm)
            plt.axvline(tx_rt_egm[loc_j-p0],color='C3',alpha=0.99)
            plt.xlabel('time (ms)')
            plt.title(f'x: Rep EGM #{k}')


            plt.subplot(144)
            tx1 = 1000*np.arange(len(dx_j))/fs
            plt.plot(tx1,dx_j)
            plt.axvline(tx1[loc_i],color='C2',ls='--',alpha=0.5, label='AT')
            plt.axvline(tx1[loc_j],color='C3',ls='--',alpha=0.5, label='RT')
            plt.fill_between(tx1,dx_j*0 + np.nanmax(dx_j),np.nanmin(dx_j),where=(tx1>=tx1[loc_i]) & (tx1<=tx1[loc_j]),alpha=0.2)
            plt.xlabel('time (ms)')
            plt.title(f'dx: derivative #{k}')
            plt.legend()
            plt.show()
    return np.array(at_loc), np.array(rt_loc)

def extract_egm(X,act_loc,fs=25000,dur_from_loc=5,remove_drift=True,apply_after=True,sg_window=91,sg_polyorder=1,pad=np.nan,verbose=False,plot=False,plot_tOffset=True,figsize=(8,3)):

    """
    Extract EGMs (Electrograms) from each channel
    ----------------------------------------------

    Given a m-channels of a single cycle, EGM are extracted from each channel


    Parameters
    ----------
    X  : (nch,n) - nch = number of channels, n=number of samples of a SINGLE CYLCLE
         axis=0 is a channel. Each channel should have single stimulus location, which is given as a list of act_loc

    fs : sampling frequency,

    act_loc: list of activation time location as index, corresponding to each channel.
             Each channel should have single stimulus.

    dur_from_loc:  duration (ms) of signal extracted from each side of location,
                   extracted signal - EGM would be 2*dur_from_loc ms duration



    remove_drift: bool, default=True,
                 If True, Savitzky-Golay filter is applied to remove the drift

    apply_after: bool, default=True,
                 If True, Savitzky-Golay filter to remove drift is applied after extracting EGM
                 Else drift is removed from entire signal and EGM is extracted

    Parameters for Savitzky-Golay filter
        : sg_window=91,sg_polyorder=1
        : keep the window size large enough and polyorder low enough to remove only drift

     NOTE: Parameters of Savitzky-Golay filter should be choosen appropriately, which depends if
           it is applied before or after extracting EGM. In case of after EGM extraction,
           signal is a small length, so need to adjust sg_window accordingly

    pad: default=np.nan
         To pad values to EGM, in case is to shorter than others.
         Padding is done to make all EGMs of same shape-nd-array

    plot: bool, default False
          If True, two figures per channel are plotted
          Figure 1 shows, a raw EGM, computed drift and corrected EGM
          Figure 2 shows, Only corrected EGM with loc

    plot_tOffset: bool, deafult=True
                  If True, absolute time of EGM is shown on x-axis, as shifted to 0 -to -2*dur_from_loc is shown
                  Displaying absolute time is useful to analyse the locations

    figsize=(8,3):  size of figure

    verbose: bool, default=False
             If True, intermediate computations are printed,

    Returns
    -------
    XE  : All Exctracted EGMs as nd-array of shape (nch, n)
          n = 2*dur_from_loc*fs/1000

    ATloc : list of relative activation loc, corresponds to XE

    Example
    -------
    #TODO

    See Also
    --------
    egm_features, align_cycles


    """
    # same number of activation location as number of channels
    assert len(X)==len(act_loc)

    XE = []
    ATloc = []
    for k in range(len(X)):
        xi = X[k]
        pi = act_loc[k]

        xi_sg = xi.copy()
        if remove_drift and not(apply_after):
            #xi_sg = sp.filterDC_sGolay(xi.copy(), window_length=sg_win, polyorder=sg_porder)
            if sg_window<len(xi):
                xi_sg = filterDC_sGolay(xi.copy(), window_length=sg_window, polyorder=sg_polyorder)
            else:
                n_win= len(xi)
                if n_win%2==0: n_win-1
                xi_sg = filterDC_sGolay(xi.copy(), window_length=n_win, polyorder=sg_polyorder)


        pad_t0,pad_t1 = 0,0

        t0,t1 = pi-int(fs*dur_from_loc/1000), pi+int(fs*dur_from_loc/1000)

        if verbose: print('loc: ',pi,'dur: ',(t0,t1))
        if t0<0:
            pad_t0 = abs(t0)
            t0=0
        if t1>len(xi):
            pad_t1 = abs(t1-len(xi))
            t1 = len(xi)
        if verbose: print('loc: ',pi,'dur: ',(t0,t1),'pad: ',(pad_t0,pad_t1))

        xi_dur = xi_sg[t0:t1]
        t_offset = t0

        xi_dur_sg = xi_dur.copy()
        if remove_drift and apply_after:
            if sg_window<len(xi_dur):
                xi_dur_sg = filterDC_sGolay(xi_dur.copy(), window_length=sg_window, polyorder=sg_polyorder)
            else:
                n_win= len(xi_dur)
                if n_win%2==0: n_win-1
                xi_dur_sg = filterDC_sGolay(xi_dur.copy(), window_length=n_win, polyorder=sg_polyorder)

        xi_pad = xi_dur_sg.copy()
        if pad_t0: xi_pad = np.r_[np.ones(pad_t0)*pad,xi_dur_sg]
        if pad_t1: xi_pad = np.r_[xi_dur_sg,np.ones(pad_t1)*pad]

        if verbose:print('egm_size: ',xi_pad.shape)

        XE.append(xi_pad)

        #(t1-t0)/2
        pj = pi - t0 + pad_t0
        if verbose: print('-',k,(pi,pj),(t0,t1),(pad_t0,pad_t1))
        ATloc.append(pj)

        if plot:
            #if pad_x: print(pi,pad_x)
            #plt.plot(xi[t0:t1])
            tx = 1000*(t_offset+np.arange(len(xi_dur)))/fs if plot_tOffset else 1000*(np.arange(len(xi_dur)))/fs
            plt.figure(figsize=figsize)

            plt.subplot(121)
            plt.plot(tx,xi_dur,label='raw')
            plt.plot(tx,xi_dur-xi_dur_sg,label='drift')
            plt.plot(tx,xi_dur_sg,label='corrected')
            plt.axvline(tx[pi-t0],ls='-',color='C3',lw=1)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', borderaxespad=0.,ncol=3,)
            plt.xlabel('time (ms)')

            plt.subplot(122)
            tx = 1000*(pad_t0+np.arange(len(xi_pad)))/fs if plot_tOffset else 1000*np.arange(len(xi_pad))/fs
            plt.plot(tx,xi_pad)
            plt.axvline(tx[pj],ls='-',color='C3',lw=1, label='loc (at/rt)')
            plt.title(f'#{k}')
            plt.xlabel('time (ms)')
            plt.legend()
            plt.tight_layout()
            plt.show()

    return np.array(XE),np.array(ATloc)

def egm_features(x,act_loc,fs=25000,width_rel_height=0.75,findex_rel_dur=1,findex_rel_height=0.75,findex_npeak=False,verbose=1,plot=1,figsize=(8,3),title='',**kwargs):
    r"""


    Feature Extraction from a single EGM
    -------------------------------------

    Following features are extracted from given EGM x

    1) Peak to Peak voltage
    2) Duration of EGM
    3) Fractional Index
    4) Refined Duration of EGM based on fractionating peaks
    5) Energy of EGM
    6) Voltage Dispersion
    7) Noise variance

    1) Peak to Peak voltage
    -----------------------
    As activation time loc is a location of maximum negative deflection, that entails a high positive peak (p1) just before
    activation loc followed by very low negative peak (p2) after loc. So Peak to Peak voltage is computed as volatege
    difference between p1 and p2. As shows pictorially


                      +ve peak = v1
                       |
                       |  -
                       |    -          p2
                    ---|-------|-------|
                       p1     loc      |
                                  -    |
                                     - |
                                     -ve peak = v2


          Peak to Peak voltage = |v1-v2|


    2) Duration of EGM
    -------------------

    Each peak has a width, which computed by 'width_rel_height'


    width_rel_height  : Relative hight of peaks to estimate the width of peaks, default=0.75
                        Lower it is smaller the width of peak be, which leads to smaller duration of EGM


    In this diagram, peak p1 has a width of p12-p11, where p11 is the left end of positive peak, Similarly, peak p2
    has right end point of width as p22. So duration of EGM is computed as difference start of +ve peak width to
    end of -ve peak width, which is

            duration = p22-p11


                        ---!---p1---|-- | ----|---p2----!-
                          p11       p12 |    p21       p22
                                        |
                                        |
                                       loc


    3. Fractionation Index:
    --------------------
    Fractionation Index is defined as number of peaks after loc (activation time), within a search region, that exceeds the threshold of height.
    Threshold of height defined by findex_rel_height (= findex_rel_height*positive_peak_height,  findex_rel_height*negative_peak_height).

    Fractionating peaks can be positive or/negative. In ideal situation, every positive peak is followed by negative one, so they come in pair.
    Therefore:

       Fractionation Index = 1 + max(positive_fr_peaks, negative_fr_peaks)

    So;
    Fractionation Index = 1, means, no extra peaks within duration, other than main peaks
    Fractionation Index = 2, means, there is one peak after loc, that exceeds the threshold.

    However, postive frationating peaks are more reliable and negative peaks are often due to noise.
    Therefore, setting findex_npeak=False, avoid consideration of negative peaks


    Search Region of fractionation index is set by 'findex_rel_dur'

    findex_rel_dur    : Relative duration of search region, to find fractionating peaks. default = 1
                        As explained above for duration,

                        duration = p22-p11

                        ---|----------| -----------|-
                           p11       loc          p22


                        Search region  = p22 +  findex_rel_dur*duration

                        if findex_rel_dur = 0, means searching for fractionating peaks between activation time (loc) to end point of EGM's duration (p22)

    Threshold of height defined by findex_rel_height

    findex_rel_height : Relative height threshold for defining the fractionation, default=0.75
                        Any positive peak which exceeds the hight of 0.75*positive peak of EGM at loc, within  Search region is considered fractionating peak
                        Similarly, any negative peak which goes below 0.75*negative peak of EGM at loc, within  Search region is considered fractionating peak


    findex_npeak : if true, negative peaks are considered for fractionation, default=False



    4) Refined Duration of EGM based on fractionating peaks
       This is defined by new end point (right) of negative peak, which is the end point of last fractionating peak


    5) Energy of EGM      : Mean(x**2)          (mean squared voltage)
    6) Voltage Dispersion : SD(x**2)            (SD of squared voltage)
    7) Noise variance     : Median(|x|)/0.6745  (Estimated noise variance)



    Parameters
    ----------

    x  : 1d-array, input signal (EGM)
    fs : sampling frequency,

    act_loc : int, activation time location of EGM


    width_rel_height  : Relative hight of peaks to estimate the width of peaks, default=0.75
                        Lower it is smaller the width of peak be, which leads to smaller duration of EGM

    findex_rel_dur    : Relative duration of search region, to find fractionating peaks. default = 1
                        As explained above for duration,


    findex_rel_height : Relative height threshold for defining the fractionation, default=0.75
                        Any positive peak which exceeds the hight of 0.75*positive peak of EGM at loc, within  Search region is considered fractionating peak
                        Similarly, any negative peak which goes below 0.75*negative peak of EGM at loc, within  Search region is considered fractionating peak


    findex_npeak : if true, negative peaks are considered for fractionation, default=False


    plot: if True,
          plot EGM with all the features shown

    figsize=(8,3): size of figure


    verbose: 0 Silent
             1 a few details
             2 all the computations


    Returns
    -------

    features:  list of 7 featur values

        peak2peak : float,  peak to peak voltage, with same unite as input x

        duration  : float,  duration of EGM in samples, output would be a float, as width of peak is computed by interpolated.
                    Divide it by fs to compute in seconds
                    duration in (s) = duration/fs

        findex      : int, Fractionation Index, number of peaks after activation that crosses the 'findex_rel_height' of the main peak

        new_duration: float, New duration, refined by fractionating peaks, same in samples.
                      Divide it by fs to compute in seconds

        energy_mean : float,  Energy of EGM (mean squared voltage)
        energy_sd   : Voltage Dispersion (SD of squared voltage)
        noise_var   : Estimated noise variance (= median(|x|)/0.6745)


    names :  list of names of each feature


    Example
    -------
    #TODO


    See Also
    --------
    align_cycles, extract_egm

    """
    names = ['peak_to_peak','egm_duration','f_index','new_duration','energy_mean','energy_sd','noise_var']
    features_nan = [np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan]

    energy_mean, energy_sd, noise_var = np.nanmean(x**2), np.nanstd(x**2),np.nanmedian(np.abs(x))/0.6745
    if verbose>1:
        print('Energy SD :',energy_sd,np.log(energy_sd))
        print('Noise Var :',noise_var,np.log(noise_var))

    peaks_p, _ = find_peaks(x)
    peaks_n, _ = find_peaks(-x)

    if (len(peaks_p)==0) or (len(peaks_n)==0):
        return features_nan, names

    pprominences = peak_prominences(x, peaks_p)[0]
    nprominences = peak_prominences(-x, peaks_n)[0]


    idx_p = np.argmin(np.abs(peaks_p-act_loc))
    idx_n = np.argmin(np.abs(peaks_n-act_loc))

    p1,x1 = peaks_p[idx_p], x[peaks_p[idx_p]]
    p2,x2 = peaks_n[idx_n], x[peaks_n[idx_n]]

    peak2peak = np.abs(x1-x2)

    p1_h = peak_widths(x,[p1],rel_height=width_rel_height)
    p2_h = peak_widths(-x,[p2],rel_height=width_rel_height)

    #print(p1_h)
    #print(p2_h)

    # this works only if positive peak is followed by negative peak
    p3 = p1_h[2][0]
    p4 = p2_h[3][0]

    peak_width_idx = np.r_[p1_h[2][0], p1_h[3][0], p2_h[2][0], p2_h[3][0]]

    p3 = np.min(peak_width_idx)
    p4 = np.max(peak_width_idx)


    duration = abs(p4-p3)
    duration_s  = duration/fs
    duration_ms = 1000*duration/fs


    peaks_after_spikes_p = peaks_p[peaks_p>act_loc]
    peaks_after_spikes_n = peaks_n[peaks_n>p2]

    if verbose>1:
        print('peaks after spikes (p)',peaks_after_spikes_p)
        print('peaks after spikes (n)',peaks_after_spikes_n)
        print('peak p',p3, p1_h)
        print('peak n',p4, p2_h)
        print('duration',duration)



    search_duration_samples = p4+findex_rel_dur*duration

    if verbose>1:
        print('search_duration_samples for fractionation',search_duration_samples, 1000*search_duration_samples/fs)

    peaks_after_spikes_p = peaks_after_spikes_p[peaks_after_spikes_p<=search_duration_samples]
    peaks_after_spikes_n = peaks_after_spikes_n[peaks_after_spikes_n<=search_duration_samples]

    fr_p = peaks_after_spikes_p[x[peaks_after_spikes_p]>findex_rel_height*x1]

    fr_n =[]
    if findex_npeak:
        fr_n = peaks_after_spikes_n[x[peaks_after_spikes_n]<findex_rel_height*x2]

    findex =  1 + max(len(fr_p),len(fr_n))


    if verbose>1:
        print('peaks after spikes (p)',peaks_after_spikes_p)
        print('peaks after spikes (n)',peaks_after_spikes_n)
        print('height of Peak (p)',x1)
        print('height of Peak (n)',x2)
        print('heights (p)',x[peaks_after_spikes_p])
        print('heights (n)',x[peaks_after_spikes_n])

        print('heights>0.75H', x[peaks_after_spikes_p]>findex_rel_height*x1)
        print('heights>0.75H', x[peaks_after_spikes_n]<findex_rel_height*x2)
        print('fr_p', fr_p)
        print('fr_n', fr_n)
        print('findex', findex)

    new_duration = duration
    p4_i = p4
    if findex>1:
        fr_all = list(fr_p)+list(fr_n)
        fr_last = max(fr_all)
        if fr_last in fr_p:
            p4_i = peak_widths(x,[fr_last],rel_height=width_rel_height)[3][0]
        else:
            p4_i = peak_widths(-x,[fr_last],rel_height=width_rel_height)[3][0]

        new_duration = abs(p4_i-p3)
        if verbose>1:
            print('New Duration',new_duration, (p3,p4_i))

    #duration
    if verbose:
        print('------------------------------')
        print(f'peak-to-peak (mV)  :\t {peak2peak}')
        print(f'duration  (samples):\t {duration}')
        print(f'duration  (s)      :\t {duration_s}')
        print(f'duration  (ms)     :\t {duration_ms}')
        print(f'new duration (ms)  :\t {1000*new_duration/fs}')
        print(f'f-index     :\t {findex}')
        print('------------------------------')


    if plot:

        tx = 1000*np.arange(len(x))/fs

        plt.figure(figsize=figsize)
        plt.plot(tx,x)
        plt.plot(tx[peaks_p],x[peaks_p],'.C3',alpha=0.5,ms=5)
        plt.plot(tx[peaks_n],x[peaks_n],'.C2',alpha=0.5,ms=5)
        plt.plot(tx[p1],x1,'oC3',alpha=0.5)
        plt.plot(tx[p2],x2,'oC2',alpha=0.5)

        plt.axvline(tx[act_loc],color='k',ls='--',lw=1)

        plt.axhline(x1,color='C1',ls='--',lw=1)
        plt.axhline(x2,color='C1',ls='--',lw=1)



        plt.axvline(1000*p3/fs,color='C2',ls='--',lw=1)
        plt.axvline(1000*p4/fs,color='C2',ls='--',lw=1)
        plt.axvline(1000*p4_i/fs,color='C2',ls=':',lw=1)

        plt.fill_between(tx,x,0,(tx >=1000*p3/fs) & (tx<=1000*p4/fs), color='C0',alpha=0.5,interpolate=True)

        plt.axvline(1000*search_duration_samples/fs,color='C4',ls='--',lw=1)
        plt.axhline(findex_rel_height*x1,color='C4',ls='--',lw=1)
        plt.axhline(findex_rel_height*x2,color='C4',ls='--',lw=1)
        if findex>1:
            for pi in fr_all:
                plt.plot(tx[pi],x[pi],'+k')
        #plt.grid()
        plt.xlim(tx[0],tx[-1])
        plt.xlabel('time (ms)')
        if title!='': plt.title(title)
        plt.show()

    features = (peak2peak,duration,findex,new_duration, energy_mean, energy_sd, noise_var)
    return features, names

def plot_mea_grid(X,ch_labels, fs=25000, bad_channels=[],act_spikes=[],rep_spikes=[],fill_apd=False,fill_color='C0',verbose=1,
            xlim=(None,None),limy=False,ylim=(None,None),figsize=(12,9),title_style=1,show=True, title=''):

    r"""


    Plot MEA Grid 8x8:  Visualisation of EGM/Signal in MEA Grid Form
    ----------------------------------------------------------------
    MEA 8x8 GRID

               | 21 | 31 | 41 | 51 | 61  | 71 |
           |12 | 22 | 32 | 42 | 52 | 62  | 72 | 82 |
           |13 | 23 | 33 | 43 | 53 | 63  | 73 | 83 |
           |14 | 24 | 34 | 44 | 54 | 64  | 74 | 84 |
           |15 | 25 | 35 | 45 | 55 | 65  | 75 | 85 |
           |16 | 26 | 36 | 46 | 56 | 66  | 76 | 86 |
           |17 | 27 | 37 | 47 | 57 | 67  | 77 | 87 |
               | 28 | 38 | 48 | 58 | 68  | 78 |



    Parameters
    ----------

    X  : (nch,n) - nch = number of channels, n=number of samples of a SINGLE CYLCLE
     axis=0 is a channel.

    fs : sampling frequency,

    ch_labels: list of label of each channel.
               It is used to arrange signals in MEA grid

    bad_channels: list of bad channels, should be inclusive of ch_labels
                  Bad channels are plotted with red color
                  if passed empty list, all the channels are considered good and plotted in Blue


    act_spikes : list of activation spike location for each channel
                 Same length as number of channels. If passed, it is used to display activation time as
                 a verticle line with black color, if passed as empty, no line is plotted.

    rep_spikes : list of repolarisation spike location for each channel.
                 Same length as number of channels. If passed, it is used to display repolarisation time as
                 a verticle line with green color, if passed as empty, no line is plotted.

    fill_apd : bool, if True, a region between activation time and repolarisation time is shaded with 'fill_color'

    fill_color: str, color to fill for APD

    xlim : tuple (t0,t1), or default= (None,None)
           x-axis limits in ms. To zoom in or plot specific duration

    ylim : tuple (y0,y1), or default= (None,None)
           y-axis limits. To zoom in or plot specific height
           only used if limy=True

    limy : bool, default=False. if True, y-axis of all the channels are fixed to same limit
          If True, and ylim = (None, None), min and max of all the channels are computed as used same for all
          channels


    figsize=(12,9): Figure size

    title_style: int, if 1 : each channels has its title,
                      if 2 : only plots at boundaries has index number

    title: str, default =''
           title of whole figure

    show: bool, default = True, if True, plt.show() is executed, if false, not.
        useful if to edit some properties of figure

    verbose: 0 Silent
             1 a few details
             2 all the computations

    Returns
    -------

    None


    Example
    -------
    #TODO

    See Also
    --------
    mea_feature_map, mat_list_show, mat_1_show, arrange_mea_grid, feature_mat


    """

    assert X.shape[0]==len(ch_labels)
    ch_loc = [(int(str(ch)[0]),int(str(ch)[1])) for ch in ch_labels]

    rd = np.abs(np.nanmax(X)-np.nanmin(X))

    y1 = ylim[0] if ylim[0] is not None else np.nanmin(X)-(rd*0.05)
    y2 = ylim[1] if ylim[1] is not None else np.nanmax(X)+(rd*0.05)
    #y1,y2 = np.min(X), np.max(X)

    if verbose:print(y1,y2)
    fig = plt.figure(figsize=figsize)
    for k in range(X.shape[0]):
        #r,c = ch_loc[k]
        c,r = ch_loc[k]
        idx = 8*(r-1)+c
        if verbose>1:print(k, r,c,idx)
        tx = 1000*np.arange(X[k].shape[0])/fs
        plt.subplot(8,8,idx)
        #if k in bad_channels:
        if ch_labels[k] in bad_channels:
            plt.plot(tx,X[k],'r')
        else:
            plt.plot(tx,X[k])
        if limy:
            plt.ylim(y1,y2)
        if len(act_spikes):
            plt.axvline(1000*act_spikes[k]/fs,color='k')
        if len(rep_spikes):
            plt.axvline(1000*rep_spikes[k]/fs,color='C2')
            if fill_apd:
                plt.fill_between(tx,X[k]*0 + y2,y1,where=(tx>=1000*act_spikes[k]/fs) & (tx<=1000*rep_spikes[k]/fs),alpha=0.2,interpolate=True,color=fill_color)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        #if xlim[0] is not None:
        plt.xlim([xlim[0], xlim[1]])
        if title_style==1:
            plt.title(f'{ch_labels[k]}')
        elif title_style==2:
            if r==1:
                plt.title(f'{c}')
            if c==1:
                plt.ylabel(f'{r}')
            if r==8:
                plt.xlabel(f'{c}')
    plt.tight_layout()
    if title!='': fig.suptitle(title)
    fig.subplots_adjust(top=0.9)
    if show: plt.show()

def mea_feature_map(features,ch_labels,bad_channels=[],fmt='.1f',vmin=None, vmax=None,cmap='jet',label='unit',title='',figsize=None,show=False):

    r"""




    Displaying Feature values of MEA channels in a MEA Grid
    -------------------------------------------------------

    MEA 8x8 GRID

           | 21 | 31 | 41 | 51 | 61  | 71 |
       |12 | 22 | 32 | 42 | 52 | 62  | 72 | 82 |
       |13 | 23 | 33 | 43 | 53 | 63  | 73 | 83 |
       |14 | 24 | 34 | 44 | 54 | 64  | 74 | 84 |
       |15 | 25 | 35 | 45 | 55 | 65  | 75 | 85 |
       |16 | 26 | 36 | 46 | 56 | 66  | 76 | 86 |
       |17 | 27 | 37 | 47 | 57 | 67  | 77 | 87 |
           | 28 | 38 | 48 | 58 | 68  | 78 |



    Parameters
    ----------

    features  : list/array of a feature for all the channels
               e.g len(features) = 60

    ch_labels : list of channel labels corresponding to features
                same length as number of features

    bad_channels: list of Bad channels, values should be inclusive of ch_labels
                  if passed, two matrix plots are showm, one without excluding bad channels
                  one by excluding bad channels

    fmt: str, precision format to show value, default = '.1f', at one decimal point

    (vmin, vmax):  colormap range, default set to (None, None)

    cmap: colormap, default='jet'

    label   : str, label for colorbar default='unit'
    title   : str, title of figure default='',
    figsize : figure size

    show: bool, deafult=False
        : if true, plt.show() is executed, with show=False, properties of figure can be changed.



    Returns
    -------
    Fx : Feature Matrix of feature of each channel arranged in MEA grid form

    Fx_bad : Matrix of ones and NaN each channel arranged in MEA grid form
             1 if channel is good
             np.nan if channels in BAD, from given list of bad_channels

             Fx_bad can be used as by simply multiplying to Fx,
             Fxb = Fx*Fx_bad


    Example
    -------

    See Also
    --------
    plot_mea_grid, mat_list_show, mat_1_show, arrange_mea_grid, feature_mat

    """

    Fx, Fx_bad = feature_mat(features,ch_labels,bad_channels=bad_channels)

    if figsize is not None: plt.figure(figsize=figsize)
    if len(bad_channels):
        plt.subplot(121)
    else:
        plt.subplot(111)
    sns.heatmap(data=Fx, annot=True,square=True,fmt=fmt,vmin=vmin,vmax=vmax,cmap=cmap,cbar_kws={'label': label})
    plt.xticks(np.arange(Fx.shape[1])+0.5,np.arange(Fx.shape[1])+1)
    plt.yticks(np.arange(Fx.shape[0])+0.5,np.arange(Fx.shape[0])+1)
    plt.title(title)
    if len(bad_channels):
        plt.subplot(122)
        sns.heatmap(data=Fx*Fx_bad, annot=True,square=True,fmt=fmt,vmin=vmin,vmax=vmax, cmap = cmap,cbar_kws={'label': label})
        plt.xticks(np.arange(Fx.shape[1])+0.5,np.arange(Fx.shape[1])+1)
        plt.yticks(np.arange(Fx.shape[0])+0.5,np.arange(Fx.shape[0])+1)
        plt.title(title +' (- bad channels)')
    plt.tight_layout()
    if show: plt.show()
    return Fx, Fx_bad

def mat_list_show(A,fmt='.1f',vmin=None, vmax=None,cmap='jet',figsize=None,labels=[],titles=[],grid=(2,2)):
    r"""


    Display multiple Feature Matrix of MEA
    --------------------------------------


    Parameters
    ----------

    A:  list of feature matrix Fx, arranged in MEA-grid form

    fmt: str, precision format to show value, default = '.1f', at one decimal point, same applies to all the matrix plot

    (vmin, vmax):  colormap range, default set to (None, None), same applies to all the matrix plot

    cmap: colormap, default='jet'

    labels  : list of str, label for each colorbar default=[], an empty list
    titles  : list of str, title for each figure default= [], an empty list
    figsize : figure size
    grid    : tuple, (row,col), default = (2,2)
             grid as to arraged these feature matrixs


    Returns
    -------
    None

    Example
    -------
    #TODO

    See Also
    --------
    plot_mea_grid, mea_feature_map, mat_1_show, arrange_mea_grid, feature_mat

    """

    if len(labels)!=len(A): labels = ['unit' for _ in range(len(A))]
    if len(titles)!=len(A): titles = ['' for _ in range(len(A))]

    if figsize is not None: plt.figure(figsize=figsize)
    for k, Ai in enumerate(A):
        plt.subplot(grid[0],grid[1],k+1)
        sns.heatmap(data=Ai, annot=True,square=True,fmt=fmt,vmin=vmin,vmax=vmax,cmap=cmap,cbar_kws={'label': labels[k]})
        plt.xticks(np.arange(Ai.shape[1])+0.5,np.arange(Ai.shape[1])+1)
        plt.yticks(np.arange(Ai.shape[0])+0.5,np.arange(Ai.shape[0])+1)
        plt.title(titles[k])
    plt.tight_layout()
    plt.show()

def mat_1_show(A,fmt='.1f',vmin=None, vmax=None,cmap='jet',label='',title=''):
    r"""


    Display a single Feature Matrix of MEA
    --------------------------------------

    Useful to plot multiple matrix with different settings.
    As no figure is initiated, 'mat_1_show' can be followed by subplots

    Parameters
    ----------
    A:  single feature matrix

    fmt: str, precision format to show value, default = '.1f', at one decimal point

    (vmin, vmax):  colormap range, default set to (None, None)

    cmap: colormap, default='jet'

    label   : str, label for colorbar default='unit'
    title   : str, title of figure default='',


    """

    plot_mea_grid, mea_feature_map, mat_list_show, arrange_mea_grid, feature_mat

    sns.heatmap(data=A, annot=True,square=True,fmt=fmt,vmin=vmin,vmax=vmax,cmap=cmap,cbar_kws={'label': label})
    plt.xticks(np.arange(A.shape[1])+0.5,np.arange(A.shape[1])+1)
    plt.yticks(np.arange(A.shape[0])+0.5,np.arange(A.shape[0])+1)
    plt.title(title)

def arrange_mea_grid(features,ch_labels,grid=(8,8),default=np.nan):
    """
    Arranging features into MEA-Grid Matrix: Feature Matrix
    -------------------------------------------------------

    Arranging a list of features for each channel in a MEA-Grid form


    Parameters
    ----------
    features : list/array of feature values corresponding to channels as listed by ch_labels
    ch_labels: list of channel labels corresponding to features
    grid: (8,8)
    default: default values to fill matrix,  =np.nan

    Returns
    -------
    M: Matrix, 2d array of shape = grid

    Example
    -------
    >>> import spkit as sp
    #TODO



    See Also
    --------
    plot_mea_grid, mea_feature_map, mat_list_show, mat_1_show, feature_mat

    """
    assert len(features)==len(ch_labels)
    ch_loc = [(int(str(ch)[0]),int(str(ch)[1])) for ch in ch_labels]
    M = np.zeros(grid)*default
    for k in range(len(ch_labels)):
        #r,c = ch_loc[k]
        c,r = ch_loc[k]
        M[r-1,c-1] = features[k]
    return M

def unarrange_mea_grid(M,ch_labels):
    r"""


     Reverse the operation of 'arrange_mea_grid'
     That is given Feature Matrix of MEA grid, arrange it in order of ch_labels
    """
    ch_loc = [(int(str(ch)[0]),int(str(ch)[1])) for ch in ch_labels]

    features = [[]]*len(ch_labels)
    for k in range(len(ch_labels)):
        c,r = ch_loc[k]
        features[k] = M[r-1,c-1]
    return np.array(features)

def channel_mask(mask_ch,ch_labels,mask_val=np.nan,default=1,grid=(8,8)):
    r"""


    Create a Mask for MEA Matrix
    ------------------------


    Parameters
    ----------
    mask_ch: channels to be masked
    ch_labels: list of channel labels
    mask_val: value to subtitute for macsked channel default=np.nan
    default: default value for all the channels default=1
    grid = (8,8) 8x8 grid

    Returns
    -------
    M : matrix, 2d-array of shape = grid = (8,8)

    Example
    -------

    See Also
    --------
    plot_mea_grid, mea_feature_map, mat_list_show, mat_1_show, arrange_mea_grid, feature_mat

    """
    ch_loc = [(int(str(ch)[0]),int(str(ch)[1])) for ch in ch_labels]
    mask_ch_idx = [list(ch_labels).index(ch) for ch in mask_ch]
    M = np.zeros(grid)+default
    for k in mask_ch_idx:
        c,r = ch_loc[k]
        M[r-1,c-1] = mask_val
    return M

def feature_mat(features,ch_labels,bad_channels=[]):
    """
    Create Feature Matrix and Mask matrix for bad channels
    ------------------------


    Parameters
    ----------

    features     : list/array of feature values corresponding to channels as listed by ch_labels
    ch_labels    : list of channel labels corresponding to features
    bad_channels : list of bad channels, should be inclusive of ch_labels

    Returns
    -------

    Fx     : Feature matrix, feature arranged in MEA-grid-like matrix
    Fx_bad : Matrix of 1 and NaNs, same shape as Fx,
             value =1 for Good Channel
             value =np.nan for Bad Channele

    Example
    -------
    #TODO

    See Also
    --------
    plot_mea_grid, mea_feature_map, mat_list_show, mat_1_show, arrange_mea_grid


    """
    Fx = arrange_mea_grid(features,ch_labels)
    bad_channels_bin = np.array([0 if ch in bad_channels else 1 for ch in ch_labels])
    Fx_bad = arrange_mea_grid(bad_channels_bin, ch_labels=ch_labels)
    Fx_bad[Fx_bad==0] = np.nan
    return Fx, Fx_bad

def compute_cv(Ax,Ax_bad,eD=700,esp=1e-10,cv_pad=np.nan,cv_thr=100,arr_agg='mean',plots=1,verbose=True,flip=False,silent_mode=False,**kwargs):
    """
    Compute Conduction Velocity
    --------------------------

    Given Activation Time Matrix as MEA -Grid form, computing Conduction Velocity


    Parameters
    ----------

    Ax:  Interpolated Activation time matrix as MEA grid form.
         Ax should not have any NaN values. Use 'fill_nans_2d' to fill NaN values
         or replace Bad channel values


    Ax_bad: Matrix of 1 and np.nan values indicating bad channels.
            np.nan value corresponds to channel location indicate bad channel

    eD: scaler, inter-node distance (default=700 mm)
        distance between two nodes in horizontal and vertical axis on MEA electrode-plate
        distance is given in mm

    esp: scalar, epsilon, default =1e-10,
         to avoid diving by zero, esp is used

    cv_thr: scalar, default=100 cm/s
            threshold on conduction velocity to exclude
            any electrodes shows cv>=cv_thr is replaced by 'cv_pad' (np.nan)

    cv_pad: replacement value default=np.nan,
           any cv value above cv_thr is replaced by cv_pad, to avoid including in computation


    arr_agg: str, {'mean','median'} default='mean'
            method to aggregate the directional arrows, mean or median


    plots: int, default=1
          if 1, plot two figures for CV maps, two figures for directional compass
          if 2, also plot activation matrix and conduction velocity matrix with more details


    verbose: boot, default=True
             print information

    flip=False:  KEEP IT FALSE, DEV. MODE

    Returns:
    -------
    CV_df:  pandas Dataframe, including 'CV_mean','CV_median','CV_sd','inv_speed','avg_angle' computations
            for 'Interpolated values' and 'Original Values'. Original values exclude the Bad channels

    Ax_theta: Angle of direction for cv at each elctrodes, as a Featire Matix Form

    CV0: Conducntion Velocity matrix, that includes inpterpolated values

    CV : Conducntion Velocity matrix, excluding Bad Channels values

    Example
    -------

    See Also
    --------

    """

    if silent_mode:
        verbose=0
        plots=0


    # Time-gradient
    if flip:
        Ax_dx = np.gradient(Ax[::-1],axis=1)
        Ax_dy  = np.gradient(Ax[::-1],axis=0)
    else:
        Ax_dx = np.gradient(Ax,axis=1)
        Ax_dy = -np.gradient(Ax,axis=0)

    Ax_dxD,Ax_dyD = Ax_dx/eD, Ax_dy/eD

    #inv speed
    Ax_speed_inv = np.sqrt(Ax_dx**2 + Ax_dy**2)
    Ax_speed_inv_eD = np.sqrt(Ax_dxD**2 + Ax_dyD**2)


    #Conduction Velocity

    #Ax_CV0 = eD/(Ax_speed+esp)
    CV0 = (1/(Ax_speed_inv_eD+esp))/10
    CV = CV0.copy()
    CV[CV>=cv_thr] = cv_pad


    cv_mean = np.nanmean(CV)
    cv_mean_bd = np.nanmean(Ax_bad*CV)

    cv_median = np.nanmedian(CV)
    cv_median_bd = np.nanmedian(Ax_bad*CV)

    cv_std = np.nanstd(CV)
    cv_std_bd = np.nanstd(Ax_bad*CV)

    speed_avg = np.nanmean(Ax_speed_inv_eD)
    speed_avg_bd = np.nanmean(Ax_speed_inv_eD*Ax_bad)


    # Direction
    Ax_theta     = np.arctan2(Ax_dy,Ax_dx)
    #Ax_theta_eD = np.arctan2(Ax_dyD, Ax_dxD)

    theta_f = (Ax_theta).reshape(-1)
    theta_bd_f = (Ax_theta*Ax_bad).reshape(-1)

    Ax_theta_avg = agg_angles(theta_f, agg=arr_agg)
    Ax_theta_avg_bd = agg_angles(theta_bd_f, agg=arr_agg)


    CV_df= pd.DataFrame({'Interpolated': [cv_mean, cv_median,cv_std, speed_avg, Ax_theta_avg],
                         'Original': [cv_mean_bd, cv_median_bd,cv_std_bd,speed_avg_bd,Ax_theta_avg_bd]},
                        index=['CV_mean','CV_median','CV_sd','inv_speed','avg_angle'])

    if plots:

        if plots>1:
            mat_list_show([Ax, Ax_dx,Ax_dy],figsize=(15,4),vmax=None,grid=(1,3),titles=['AT','AT: dx', 'AT: dy'],labels=['cm/s','cm/s','cm/s'])
            mat_list_show([CV0, CV,CV*Ax_bad],figsize=(15,4),vmax=None,grid=(1,3),titles=['CV-raw','CV', 'CV (-bad channels)'],labels=['cm/s','cm/s','cm/s'])



        plt.figure(figsize=(12,4))
        plt.subplot(121)
        sns.histplot(CV.reshape(-1))
        plt.title(f'CV | Interpolated: mean ={np.nanmean(cv_mean).round(2)}')
        plt.xlabel('cm/s')
        plt.axvline(np.nanmean(CV),color='k',ls='-',lw=3,alpha=0.6, label='mean')
        plt.axvline(np.nanmedian(CV),color='C3',ls='--',lw=3,alpha=0.6, label='median')
        plt.legend()
        plt.subplot(122)
        sns.histplot((Ax_bad*CV).reshape(-1))
        plt.title(f'CV | Original: mean ={np.nanmean(cv_mean_bd).round(2)}')
        plt.xlabel('cm/s')
        plt.axvline(np.nanmean(Ax_bad*CV),color='k',ls='-',lw=3,alpha=0.6, label='mean')
        plt.axvline(np.nanmedian(Ax_bad*CV),color='C3',ls='--',lw=3,alpha=0.6, label='median')
        plt.legend()
        plt.tight_layout()
        plt.show()


        _ = show_compass(Ax_theta, Ax_bad,arr_agg='mean',figsize=(10,6),
                                        all_arrow_prop =dict(facecolor='C0',lw=1,zorder=10,alpha=0.2,edgecolor='C0',width=0.05),
                                        avg_arrow_prop =dict(facecolor='C3',lw=4,zorder=100,edgecolor='C3',width=0.045))


    return CV_df, Ax_theta, CV0, CV

def analyse_mea_file(file_name,stim_fhz,fs=25000,egm_number=-1,
                     dur_after_spike=None,exclude_first_dur=2,exclude_last_cycle=True,repol_computation=False,
                     bad_ch_stim_thr=2,bad_ch_mnmx=[None, None], p2p_thr=5,range_act_thr=[0,50],
                     bad_channels=[], good_channels=[], max_volt=1000,verbose=1,
                     stim_id_param = dict(method='max_dvdt',gradient_method='fdiff',ch=0,N=None,plot=0,figsize=(12,3)),
                     at_id_param = dict(at_range=[0,None],rt_range= [0.5, None],method='min_dvdt',gradient_method='fdiff',sg_window= 11,sg_polyorder=3,
                                       gauss_window=0,gauss_itr=1,plot=False,plot_dur=2),
                     egm_id_param = dict(dur_from_loc=5,remove_drift=True,apply_after=True,sg_window=201,sg_polyorder=1,pad=np.nan,verbose=0,plot=False),
                     egm_feat_param = dict(width_rel_height=0.75,findex_rel_dur=0.5,findex_rel_height=0.75,findex_npeak=False,plot=0,verbose=0),
                     intp_param = dict(pkind='linear',filter_size=3,method='conv'),
                     cv_param = dict(eD=700,esp=1e-10,cv_pad=np.nan,cv_thr=100,arr_agg='mean',plots=2,verbose=True),
                     map_prop = dict(at_range=[0,20],p2p_range=[0,100],dur_range=[0,10],f_range=[1,5],cv_range=[0,None],rt_range=[0,100],apd_range=[0,50],
                                    at_cmap='jet',interpolation='bilinear',countour_n=25,countour_clr='k',cv_arr_prop = dict(color='w',scale=None))):

    r"""


    A Complete Analysis of a MEA recording
    ======================================

    Given a MEA recoring file in HDF formate ('.h5'), a complete analysis is provided, step-wise.
    This function can be used to analyse 100s of files with same paramter settings. Function returns all the
    feature computed as a DataFrame, with can be exported as csv file


                           MEA 8x8 GRID

                   | 21 | 31 | 41 | 51 | 61  | 71 |
               |12 | 22 | 32 | 42 | 52 | 62  | 72 | 82 |
               |13 | 23 | 33 | 43 | 53 | 63  | 73 | 83 |
               |14 | 24 | 34 | 44 | 54 | 64  | 74 | 84 |
               |15 | 25 | 35 | 45 | 55 | 65  | 75 | 85 |
               |16 | 26 | 36 | 46 | 56 | 66  | 76 | 86 |
               |17 | 27 | 37 | 47 | 57 | 67  | 77 | 87 |
                   | 28 | 38 | 48 | 58 | 68  | 78 |


    There are many different parameters to set at each step, some of them can be configured via input argumets.
    The deafult setting of all the parameters at each steps are choosen as to work well for observed cases. For full analysis, try each step seprately
    to optimise parameters for your own file, once, they are determined, 100s of files can be analysed and same save all the computed metrics in a csv/excel file.



    Parameters
    ----------

    *Esential input argumets*


    file_name: str, file name with full accessible path. File should be in .h5 format. If you have bdf file, use Multichannel Data Manager to cover file type.

                 check here, how to convert bdf to hdf file:
                 Multichannel Data Manager: https://www.multichannelsystems.com/software/multi-channel-datamanager

    stim_fhz : (float, int)
               Frequency of Stimulus in Hz (cycles per seconds): eg. 1, 2 etc
               for 1Hz, stim_fhz =1

               Setting stim_fhz=None
                 if stim_fhz=None, it tries to extract it from file name, if exist, else error will be raised
                 for example, if file_name='MEA_North_1000_1Hz.h5' function will extract '1Hz' and set stim_fhz=1
                 This is usefull, if analysing mutiple files, with different Frequency of stimulus


    *Optional argumets*

    fs : int, default=25000 (= 25Khz)
         Sampling Frequency of MEA recording

    max_volt: float, int, default=1000
              Maximum voltage of signal or stimuli recorded
              Default datatype of signal recoridng in HDF file is usually 'int32', which leades to range from -32768 to 32767
              if max_volt=None, then unnormalised signal values are returned.
              if max_volt is not None, then return signal values will be float32 and from -max_volt to max_volt

        NOTE: If max_volt is changed, make sure to adjust all the other parameters.


   verbose: 0 Silent
          if >0, that is level of verbosity, heigher it is more details computations are printed on consol
            1 a few details
            2 a little more computations




    *Stim loc*
    ~~~~~~~~~~

     stim_id_param: dict, default = dict(method='max_dvdt',gradient_method='fdiff',ch=0,N=None,plot=0,figsize=(12,3)),
                   Parameter setting to identify stimuli locations

                       method: str, default='max_dvdt'
                               stim location as maximum deflection, which occures when stim shift from -ve to +ve voltage
                       gradient_method='fdiff': finite differentiation
                       ch=0: fist chennel to id stim loc
                       plot=0, if to plot, with figsize=(12,3)

                   Default setting:
                   stim_id_param = dict(method='max_dvdt',gradient_method='fdiff',ch=0,N=None,plot=0,figsize=(12,3)),
                   check help(get_stim_loc) for detail


    *Aligning Cycles*
    ~~~~~~~~~~~~~~~~~

        exclude_first_dur : float, default=2 (ms)
                           Exclude the duration (in ms) of signal after stimulus spike loc, default 1 ms
                           It depends on the method of finding stim loc, since stim is 1ms -ve, 1ms +v, and loc is
                           detected as middle transaction (max_dvdt), atleast 1ms of duration should be excluded.
                           Default 2ms is excluded, to be safe side


        dur_after_spike : float, int, default=None
                          Extract the duration (in ms) after stimulus spike loc to search for EGM
                          If set to None (default), dur_after_spike is computed based on stimuli frequency (stim_fhz)
                          stim_fhz = 1 --> dur_after_spike = 500
                          stim_fhz = 2 --> dur_after_spike = 300
                          stim_fhz = 3 --> dur_after_spike = 250
                          stim_fhz > 3 --> dur_after_spike = 200

        exclude_last_cycle: bool, default=True,
                            If True, last cycle of stimuli is excluded while aligning cycles. It is recommonded to exclude last egm,
                            as sometimes, last egm might shorter than other, which produces artifact while averaging.


    *Averaging or Selecting EGM/Cycle*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        egm_number : int, (>=-1), default=-1
                 EGM number (cycle number of stimuli) to be analysed. Number start from 0, to number of cycles of stimuli
                 egm_number = 0 for analysing 1st EGM of 1st cycle, 1, 2 for 2nd and 3rd and so on,

                 if egm_number=-1 is passed, all the EGMs are averaged first, before analysing, which is often a good idea to
                 cancel out noise


    *Activation & Repolarisation Time Localisation*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       repol_computation: bool, default=False,
                       if True, repolarisation time is computed, which leads to computation of APD = RT-AT
                       if True, parameter setting from 'at_id_param' is used



       at_id_param: dict, parameters to compute activation and repolarisation time loc

                at_range: list of two [t0,t1] default=[0,None], in ms
                         time limitation in which AT is to be searched, t0 ms to t1 ms
                         limiting it can improve the results

                for more detail check: help(get_activation_time)

                rt_range: list of two [t0,t1] default=[0.5, None], in ms
                         Only used if repol_computation =True
                         time limitation for repolarisation, AFTER activation time
                         Recommended to use t0 = 0.5, after 0.5 ms of activation time, search for RT

                for more detail check: help(get_repolarisation_time)


                parameters to identify location
                    - method='min_dvdt' : For activation and repolarisation minimum deflection, which is maximum -ve deflection is used,
                    - gradient_method='fdiff': finite differentiation of signal
                    - (sg_window= 11,sg_polyorder=3,gauss_window=0,gauss_itr=1), used if gradient_method other than 'fdiff'
                    - plot=False, if True, plot AT & RT
                    - plot_dur=2, used plot is True



                Default setting
                at_id_param= dict(at_range=[0,None],rt_range= [0.5, None],method='min_dvdt',gradient_method='fdiff',sg_window= 11,sg_polyorder=3,
                                       gauss_window=0,gauss_itr=1,plot=False,plot_dur=2),


                for more detail check: help(activation_time_loc), help(activation_repol_time_loc)


    *Exctracting EGM*
    ~~~~~~~~~~~~~~~~

       egm_id_param: dict, Parameters used while exctracting EGM from each channel.

          Given activation location for EGM, two type of parameters are required
          (1) duration of EGM to be extracted, and (2) preprocessing EGM


          dur_from_loc: float, default=5 ms
                        From given loc, dur_from_loc ms from both side of signal is extract, as EGM
                        if dur_from_loc=5, then 10ms of eletrogram is extracted
          remove_drift: bool, default=True,
                     If True, Savitzky-Golay filter is applied to remove the drift

          apply_after: bool, default=True,
                     If True, Savitzky-Golay filter to remove drift is applied after extracting EGM
                     Else drift is removed from entire signal and EGM is extracted

          Parameters for Savitzky-Golay filter
                : sg_window=91,sg_polyorder=1
                : keep the window size large enough and polyorder low enough to remove only drift

          NOTE: Parameters of Savitzky-Golay filter should be choosen appropriately, which depends if
                   it is applied before or after extracting EGM. In case of after EGM extraction,
                   signal is a small length, so need to adjust sg_window accordingly

          pad: default=np.nan
             To pad values to EGM, in case is to shorter than others.
             Padding is done to make all EGMs of same shape-nd-array

          plot: bool, default False
                  If True, two figures per channel are plotted
                  Figure 1 shows, a raw EGM, computed drift and corrected EGM
                  Figure 2 shows, Only corrected EGM with loc

          verbose: bool, default=False
                If True, intermediate computations are printed,


          Default setting
          egm_id_param = dict(dur_from_loc=5,remove_drift=True,apply_after=True,sg_window=201,sg_polyorder=1,pad=np.nan,verbose=0,plot=False)
          for more detail check: help(extract_egm)


    *Feature Exctracting from EGM*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       egm_feat_param: dict, Parameters to extract Features from EGM
            There are 7 features extracted from each EGM
             1) Peak to Peak voltage
             2) Duration of EGM
             3) Fractional Index
             4) Refined Duration of EGM based on fractionating peaks
             5) Energy of EGM
             6) Voltage Dispersion
             7) Noise variance

            For detailed overview of all the feature check: help(egm_features)


            width_rel_height  : Relative hight of peaks to estimate the width of peaks, default=0.75
                Lower it is smaller the width of peak be, which leads to smaller duration of EGM

            findex_rel_dur    : Relative duration of search region, to find fractionating peaks. default = 1
                                As explained above for duration,


            findex_rel_height : Relative height threshold for defining the fractionation, default=0.75
                                Any positive peak which exceeds the hight of 0.75*positive peak of EGM at loc, within  Search region is considered fractionating peak
                                Similarly, any negative peak which goes below 0.75*negative peak of EGM at loc, within  Search region is considered fractionating peak


            findex_npeak : if true, negative peaks are considered for fractionation, default=False


            plot: if True,
                  plot EGM with all the features shown

            figsize=(8,3): size of figure


            verbose: 0 Silent
                     1 a few details
                     2 all the computations

          Default setting
          egm_feat_param = dict(width_rel_height=0.75,findex_rel_dur=0.5,findex_rel_height=0.75,findex_npeak=False,plot=0,verbose=0)
          for more detail check: help(egm_features)


    *Interpolation of Feature Matrix - AT*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        intp_param: dict,  default = dict(pkind='linear',filter_size=3,method='conv')
                Parameters used for interpolating Activation Map
                pkind='linear': kind of interpolation
                filter_size=3: filter_size to smooth the map,
                method='conv': method of convolution


               NOTE: Increase 'filter_size' for more smoother MAP

               Default Setting
               intp_param = dict(pkind='linear',filter_size=3,method='conv')
               for more details check help(fill_nans_2d)


    *Conduction Velocity computation*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       cv_param: dict, Parameter setting to compute Conduction Velocity

             eD: scaler, inter-node distance (default=700 mm)
                distance between two nodes in horizontal and vertical axis on MEA electrode-plate
                distance is given in mm

             esp: scalar, epsilon, default =1e-10,
                 to avoid diving by zero, esp is used

             cv_thr: scalar, default=100 cm/s
                    threshold on conduction velocity to exclude
                    any electrodes shows cv>=cv_thr is replaced by 'cv_pad' (np.nan)

             cv_pad: replacement value default=np.nan,
                   any cv value above cv_thr is replaced by cv_pad, to avoid including in computation


             arr_agg: str, {'mean','median'} default='mean'
                    method to aggregate the directional arrows, mean or median


             plots: int, default=1
                  if 1, plot two figures for CV maps, two figures for directional compass
                  if 2, also plot activation matrix and conduction velocity matrix with more details


             verbose: boot, default=True
                     print information

            Default setting
            cv_param = dict(eD=700,esp=1e-10,cv_pad=np.nan,cv_thr=100,arr_agg='mean',plots=2,verbose=True)

            For more details check help(compute_cv)


    *Criteria for Identifying bad channels or bad EGMs*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        (1) Based on Stimulus duration
            If stimulus last longer than threshold duration, given by 'bad_ch_stim_thr' in ms

            bad_ch_thr: float, default = 2 (ms)
                        Duration, if average duration of stimuli per cycle exceed the threshold on either side +ve/-ve, it is flagged as bad channel
                        NOTE: Lower the threshold, more channels will be flagged, higher the threshold less channels will be flagged

            bad_ch_mnmx: list of two [None, None]
                         minimum and maximum valtage of stimulus, e.g. -1000, 1000.
                         Deault is None, in which case, it is computed from min and max of signal itself.
                         Increase this threhold to avoid flagging channels as bad.

            for more details, check help(find_bad_channels_idx)

        (2) Based on EGM voltage
            If EGM is very small, it is usually a noise

            p2p_thr : float, default=5,
                      Threhold on peak-to-peak voltage of EGM, if EGM has less the threshold peak-to-peak, it is flagged as bad
                      Descrese this threhold to avoid flagging channels as bad.

                      NOTE: Lower the threshold, less channels will be flagged, higher the threshold more channels will be flagged

        (3) Based on Activation Time
            If activation time is out of given range

            range_act_thr: list of two, deafult = [0, 50],
                           Minimum and maximum activation time range. If activation time of EGM is out of given limits, then it is flagged as bad.
                           By dfault, if activation time is greater than zero, it is considered okay.
                           Example, range_act_thr = [5,30] means, if activation time is less than 5 ms or grater than 30 ms, is it flagged as bad
                           range_act_thr = [0,30] works well too


        Manually passing list of channels as BAD and GOOD
        Passing list of good and bad channels overrides the criteria mentioned above, and enforces the list of Good and Bad

        bad_channels: list, deafault = []
                      Passing list of bad channels ensures that channel is in list of bad channels, regardless of criteria
                      For example, channel 15 is alwasy a bad so bad_channels=[15]
                      List should have same name of channels as in channel labels

        good_channels: list, deafault = []
                      Passing list of good channels overrides the criteria mentioned above, and exclude them from list of bad channels
                      List should have same name of channels as in channel labels




    *Settings for All the plots*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      map_prop: dict, Customising ranges for heatmap, colormap, style, type of objects to plot
                 Ranges of plot colormaps
                     at_range  =[0,20]  : activation time
                     p2p_range =[0,100] : peak-to-peak voltage
                     dur_range =[0,10]  : duration
                     f_range   =[1,5]   : F-index
                     cv_range  =[0,None]: Conduction velocity
                     rt_range  =[0,100] : repolarisation time
                     apd_range =[0,50]  : APD range

                 Final Activation Map with countours:
                     at_cmap='jet'   : colormap
                     countour_n=25   : levels for countour lines
                     countour_clr='k': color of countour lines
                     interpolation='bilinear' : interpolation

                 Conducntion Velocity Arrows
                    cv_arr_prop = dict(color='w',scale=None)

                    for more details check help(direction_flow_map)
            Default setting
            map_prop = dict(at_range=[0,20],p2p_range=[0,100],dur_range=[0,10],f_range=[1,5],cv_range=[0,None],rt_range=[0,100],apd_range=[0,50],
                                    at_cmap='jet',interpolation='bilinear',countour_n=25,countour_clr='k',cv_arr_prop = dict(color='w',scale=None))




    Returns
    -------
    Features_df  : Statistics of all the features (mean, median, sd across all the channels)
    Features_ch  : Features of all the channels
    Features_mat : Feature Matrices of Activation Time, Conduction Velocity, Bad Channel etc
    Data         : dict of signal, channel labes and fs
                   i.e. {'X':X, 'ch_labels':ch_labels, 'fs':fs}


    Example
    -------
    >>> import numpy as np
    >>> import spkit as sp
    >>> Fdf, Fch,Fmat, DataX = sp.mea.analyse_mea_file(files_name,stim_fhz=1)


    See Also
    --------
    #TODO

    """


    print('-'*100)
    print('file name')
    print(file_name)
    print('-'*100)



    map_prop_default = dict(at_range=[0,20],p2p_range=[0,100],dur_range=[0,10],f_range=[1,5],cv_range=[0,None],rt_range=[0,100],apd_range=[0,50],
                     at_cmap='jet',interpolation='bilinear',countour_n=25,countour_clr='k',cv_arr_prop =dict(color='w',scale=None,pivot='mid'))

    for prop in map_prop_default:
        if prop not in map_prop:
            map_prop[prop] = map_prop_default[prop]

    #at_range=[0,None]
    if 'at_range' not in at_id_param:
        at_id_param['at_range']=[0,None]


    if stim_fhz is None:
        stim_fhz = int(file_name[file_name.find('Hz')-1])
        print(f"Stimulus Frequency extracted from file name FHz = {stim_fhz} Hz, to change, set 'stim_fhz'")


    if dur_after_spike is None:
        if stim_fhz==1:
            dur_after_spike = 500
        elif stim_fhz==2:
            dur_after_spike = 300
        elif stim_fhz==3:
            dur_after_spike = 250
        else:
            dur_after_spike = 200
        print(f"{dur_after_spike} ms of duration after stimulus is selected, to change it set 'dur_after_spike'")


    Features_ch  = {}



    #=============STEP 1: Read file ================================
    print('')
    print('Reading File...')
    X,fs,ch_labels = read_hdf(file_name,max_volt=max_volt,fs=fs,
             base_key='Data',
             signal_hkeys=['Recording_0','AnalogStream','Stream_0'],
             signal_key = 'ChannelData',
             info_key ='InfoChannel',
             verbose=verbose)

    Data = {'X':X, 'ch_labels':ch_labels, 'fs':fs}

    #=============STEP 2: stim loc ================================

    stim_loc,_  = get_stim_loc(X,fs=fs,fhz=stim_fhz,**stim_id_param)
    stim_loc_s  = np.array(stim_loc)/fs
    stim_loc_ms = 1000*np.array(stim_loc)/fs


    #===PLOTs 1 =======
    t = 1000*np.arange(X.shape[1])/fs
    sep = 1000
    tix = np.arange(len(X))*sep
    tiy = np.arange(0,len(X)+1,5)
    tiy[0] =1
    plt.figure(figsize=(12,7))
    plt.plot(t,X.T - tix)
    plt.xlim([t[0],t[-1]])
    plt.yticks(-(tiy-1)*sep,tiy,fontsize=10)
    plt.ylabel('Channel #')
    plt.xlabel('time (ms)')
    plt.title('Full recording of channels')
    plt.xticks(stim_loc_ms)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,3))
    plt.subplot(211)
    plt.vlines(stim_loc,0,1)
    plt.xlim([0,X.shape[1]])
    plt.xticks(stim_loc)
    plt.xlabel('loc (index)')
    plt.ylim([0,1])
    plt.ylabel(' ')
    plt.title(f'Stimulus spikes loc., average duration = {np.diff(stim_loc).mean().round(2)} samples')

    plt.subplot(212)
    plt.vlines(stim_loc_ms,0,1)
    plt.xlim([0,X.shape[1]/fs])
    plt.xticks(stim_loc_ms)
    plt.ylim([0,1])
    plt.xlabel('time (ms)')
    plt.title(f'Stimulus spikes time (ms), average duration = {np.diff(stim_loc_ms).mean().round(2)} ms')
    plt.tight_layout()
    plt.show()



    #=============STEP 3.1: BAD Channel (1) ================================
    bad_channels_list =[]
    bad_channels_idx_1 = []
    if bad_ch_stim_thr is not None:
        bad_channels_idx_1 =find_bad_channels_idx(X,thr=bad_ch_stim_thr,stim_fhz=stim_fhz,fs=fs,mnmx=bad_ch_mnmx,plot=False,plot_dur=2,verbose=np.clip(verbose-1,0,None))
        bad_channels_ch_1 = list(ch_labels[bad_channels_idx_1])
        bad_channels_list = bad_channels_list + bad_channels_ch_1
        bad_channels_list = list(set(bad_channels_list))
        bad_channels_list.sort()


    #=============STEP 4: Align Cycles & Average or pick one ================================

    XB = align_cycles(X,stim_loc,fs=fs,exclude_first_dur=exclude_first_dur,dur_after_spike=dur_after_spike,exclude_last_cycle=exclude_last_cycle,pad=np.nan,verbose=True)

    print('Number of EGMs/Cycles per channel =',XB.shape[2])

    if egm_number<0:
        X1B = np.nanmean(XB,axis=2)
        print(' -- Averaged All EGM')
    else:
        # egm_number should be between from 0 to 'Number of EGMs/Cycles per channel '
        assert egm_number in list(range(XB.shape[2]))
        X1B = XB[:,:,egm_number]
        print(' -- Selected EGM ->',egm_number)

    print('EGM Shape : ',X1B.shape)


    #=============STEP 5: Activation Time (Repo. Time) ================================

    at_range = at_id_param['at_range']

    apd_ms = None
    rt_loc_ms = None
    if repol_computation:
        at_loc, rt_loc = activation_repol_time_loc(X1B,fs=fs,**at_id_param)
        rt_loc_ms = 1000*rt_loc/fs
        apd_ms = 1000*(rt_loc-at_loc)/fs
    else:
        at_loc = activation_time_loc(X1B,fs=fs,t_range=at_range,**at_id_param)

    at_loc_ms = 1000*at_loc/fs

    Features_ch['at_ms'] = at_loc_ms


    #===PLOTs 2 =======
    plot_mea_grid(X1B,act_spikes=at_loc,ch_labels=ch_labels,bad_channels=bad_channels_list,xlim=at_range,verbose=verbose>2,figsize=(10,7),title_style=1, title='Time Trace + AT')

    if repol_computation:
        Features_ch['rt_ms']  = rt_loc_ms
        Features_ch['apd_ms'] = apd_ms

        plot_mea_grid(X1B,act_spikes=at_loc,rep_spikes=rt_loc,fill_apd=True,ch_labels=ch_labels,bad_channels=bad_channels_list,xlim=(0,None),verbose=verbose>2,
                figsize=(10,7),title_style=1, title='AT & RT')


    #=============STEP 6: Extract EGM ================================

    XE,ATloc = extract_egm(X1B,fs=fs,act_loc=at_loc,**egm_id_param)


    #=============STEP 7: Feature Extraction from EGM ================

    EGM_feat = []
    for i in range(len(XE)):
        egmf, feat_names = egm_features(XE[i].copy(),act_loc=ATloc[i],fs=fs,title=f'#{i} #ch {ch_labels[i]}',**egm_feat_param)
        EGM_feat.append(egmf)

    #XE = np.array(XE)
    #ATloc = np.array(ATloc)
    EGM_feat = np.array(EGM_feat)

    for k in range(len(feat_names)):
        Features_ch[feat_names[k]] = EGM_feat[:,k]


    print('-'*100)
    print('Following EGM Features are etracted: ', feat_names)
    print('EGM_Feat shape :',EGM_feat.shape)
    print('Shapes: XE =',XE.shape, ', AT =',ATloc.shape,', EGM_F=' ,EGM_feat.shape)


    #=============STEP 3.2: BAD Channel (2) ================================

    bad_channels_idx_2 = []
    if p2p_thr is not None:
        bad_channels_idx_2 = np.where(EGM_feat[:,0]<p2p_thr)[0]
        bad_channels_ch_2 = list(ch_labels[bad_channels_idx_2])
        bad_channels_list = bad_channels_list + bad_channels_ch_2


    #=============STEP 3.3: BAD Channel (3) ================================

    bad_channels_idx_3 =[]
    if range_act_thr[0] is not None:
        bad_channels_idx_3 = bad_channels_idx_3 + list(np.where(at_loc_ms<range_act_thr[0])[0])

    if range_act_thr[1] is not None:
        bad_channels_idx_3 = bad_channels_idx_3 + list(np.where(at_loc_ms>range_act_thr[1])[0])

    bad_channels_ch_3 = list(ch_labels[bad_channels_idx_3])

    bad_channels_list = bad_channels_list + bad_channels_ch_3


    bad_channels_list = bad_channels_list + bad_channels
    bad_channels_list = list(set(bad_channels_list))
    bad_channels_list.sort()

    if len(good_channels):
        bad_channels_list = list(set(bad_channels_list) - set(good_channels))
        bad_channels_list.sort()


    good_channels_list = np.array([ch for ch in ch_labels if ch not in bad_channels_list])
    good_channels_list_idx = np.array([list(ch_labels).index(ch) for ch in ch_labels if ch not in bad_channels_list])

    print('-'*100)
    print('BAD CHANNELS')
    print('- BASED ON STIM thr =', bad_ch_stim_thr)
    print('  - ch:', bad_channels_ch_1)
    print('- BASED ON Peak-to-Peak volt thr:',p2p_thr)
    print('  - ch:', bad_channels_ch_2)
    print('- BASED ON Activation time (ms) range thr:', range_act_thr)
    print('  - ch:', bad_channels_ch_3)
    print('- Manually passed:')
    print('  - ch:', bad_channels)
    print('GOOD CHANNELS passed:')
    print('  - ch:',good_channels)
    print('-'*50)
    print('Final list of Bad Channels:')
    print(' - ch:',bad_channels_list)
    print('Final list of Good Channels:')
    print(' - ch:',good_channels_list)
    print('-'*100)


    #===PLOTs 3=======
    plot_mea_grid(XE,act_spikes=ATloc,ch_labels=ch_labels,bad_channels=bad_channels_list,xlim=(0,None),limy=False,verbose=0,figsize=(10,7),title_style=1,title='EGM')

    Ax,Mxbad = mea_feature_map(at_loc_ms,ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],label='ms',title='Activation Time')

    if repol_computation:
        #rt_range=[0,100],apd_range=[0,50],
        _ = mea_feature_map(rt_loc_ms,ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['rt_range'][0],vmax=map_prop['rt_range'][1],label='ms',title='Repolarisation Time')
        _ = mea_feature_map(apd_ms,ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['apd_range'][0],vmax=map_prop['apd_range'][1],label='ms',title='ARI--APD')

    _ = mea_feature_map(EGM_feat[:,0],ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['p2p_range'][0],vmax=map_prop['p2p_range'][1],label='mV',title='Peak-to-Peak')
    _ = mea_feature_map(1000*EGM_feat[:,1]/fs,ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['dur_range'][0],vmax=map_prop['dur_range'][1],label='ms',title='EGM Duration')
    _ = mea_feature_map(EGM_feat[:,2].astype(int),ch_labels=ch_labels,bad_channels=bad_channels_list,figsize=(10,4),
                               vmin=map_prop['f_range'][0],vmax=map_prop['f_range'][1],label='F-index',title='Fractionation Index',fmt='.0f')


    #=============STEP 8: Interpolating Activation Map ================

    AxI, AxIx = fill_nans_2d(Ax*Mxbad,clip_range=range_act_thr,**intp_param)
    #------------------------------------------------------------------------




    mat_list_show([Ax, Ax*Mxbad, AxI],figsize=(15,4),vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],grid=(1,3),
                  titles=['Activation Time', '- Bad Channels', 'Interpolated Activation Time'],labels=['ms','ms','ms'])

    mat_list_show([AxI, AxIx],figsize=(15,4),vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],grid=(1,3),
                  titles=['Interpolated (preserving original values)', 'Interpolation-Smooth'],labels=['ms','ms'])

    #=============STEP 9: Computing Conduction Velocity ================

    CV_df, CV_thetas, CV0, CV = compute_cv(AxI,Mxbad,flip=False,**cv_param)

    _, CV_thetas_smooth, _, _ = compute_cv(AxIx,Mxbad,flip=False,silent_mode=True,**cv_param)
    #------------------------------------------------------------------------



    #============= Features saving-formulation-stats ================
    Features_df  = {}
    Features_mat = {}


    Features_ch['cv_cm_s'] = unarrange_mea_grid(CV.copy(), ch_labels=ch_labels)


    Features_mat['AT']      = Ax
    Features_mat['AT_intp'] = AxI
    Features_mat['AT_smmoth'] = AxIx
    Features_mat['CV']      = CV
    Features_mat['CV_intp'] = CV0
    Features_mat['Bad_ch']  = Mxbad
    Features_mat['Angle']   = CV_thetas
    Features_mat['Angle_smooth']   = CV_thetas_smooth

    Features_df['Interpolated']= {}
    Features_df['Original']= {}

    for feat in Features_ch:
        fi = Features_ch[feat]
        Features_df['Interpolated'][feat+'_mean'] = np.nanmean(fi)
        Features_df['Interpolated'][feat+'_median'] = np.nanmedian(fi)
        Features_df['Interpolated'][feat+'_sd'] = np.nanstd(fi)

        fi = fi[good_channels_list_idx]
        Features_df['Original'][feat+'_mean'] = np.nanmean(fi)
        Features_df['Original'][feat+'_median'] = np.nanmedian(fi)
        Features_df['Original'][feat+'_sd'] = np.nanstd(fi)



    Features_ch['CH_Label'] = ch_labels
    Features_ch['Bad_CH'] = np.array([True if ch in bad_channels_list else False for ch in ch_labels])
    Features_ch = pd.DataFrame(Features_ch)

    for feat in CV_df['Interpolated'].index:
        if feat.lower().count('cv')==0:
            Features_df['Interpolated'][feat] = eval(f"CV_df['Interpolated'].{feat}")

    for feat in CV_df['Original'].index:
        if feat.lower().count('cv')==0:
            Features_df['Original'][feat] = eval(f"CV_df['Original'].{feat}")

    Features_df['Original']['n_bad_ch'] = len(bad_channels_list)
    Features_df['Original']['n_good_ch'] = len(ch_labels) - len(bad_channels_list)
    Features_df['Original']['file_name'] = file_name
    Features_df['Interpolated']['n_bad_ch'] = 0
    Features_df['Interpolated']['n_good_ch'] = len(ch_labels)
    Features_df['Interpolated']['file_name'] = file_name

    Features_df = pd.DataFrame([Features_df['Original'],Features_df['Interpolated']], index=['Original','Interpolated']).T
    if verbose:
        try:
            from IPython import display
            display(CV_df.round(3))
        except:
            print(CV_df.round(3))
    #------------------------------------------------------------------------



    #=== PLOTs 4=======
    #at_cmap='jet',interpolation='bilinear',countour_n=25,countour_clr='k',
    plt.figure(figsize=(15,6))
    plt.subplot(121)
    im = plt.imshow(AxI,cmap=map_prop['at_cmap'],interpolation=map_prop['interpolation'],vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],extent=[0,7,7,0])
    plt.contour(AxI,levels=map_prop['countour_n'],colors=map_prop['countour_clr'])
    plt.colorbar(im, label='ms')
    plt.title('Activation Map')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    mat_1_show(CV*Mxbad,fmt='.1f',vmin=map_prop['cv_range'][0],vmax=map_prop['cv_range'][1],cmap='jet',label='cm/s',title='Cond. Velocity')
    plt.tight_layout()
    plt.show()


    #pivot='mid'map_prop['pivot']
    pivot = 'mid'
    if 'pivot' in map_prop['cv_arr_prop']:
        pivot = map_prop['cv_arr_prop']['pivot']
        del map_prop['cv_arr_prop']['pivot']

    _  = direction_flow_map(X_theta=CV_thetas,X=AxI,upsample=1,
                            figsize=(15,7),square=True,cbar=False,arr_pivot=pivot,
                           stream_plot=True,title='CV',show=True,
                           heatmap_prop =dict(vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],cmap=map_prop['at_cmap']),
                           arr_prop =map_prop['cv_arr_prop'],
                           stream_prop =dict(density=1,color='k',linewidth=2))

    print('Smooth Plot')
    
    _  = direction_flow_map(X_theta=CV_thetas_smooth,X=AxIx,upsample=1,
                        figsize=(15,7),square=True,cbar=False,arr_pivot=pivot,
                        stream_plot=True,title='CV',show=True,
                        heatmap_prop =dict(vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],cmap=map_prop['at_cmap']),
                        arr_prop =map_prop['cv_arr_prop'],
                        stream_prop =dict(density=1,color='k',linewidth=2))


    print('With upsampling by 2')

    _  = direction_flow_map(X_theta=CV_thetas,X=AxI,upsample=2,
                        figsize=(15,7),square=True,cbar=False,arr_pivot=pivot,
                       stream_plot=True,title='CV',show=True,
                       heatmap_prop =dict(vmin=map_prop['at_range'][0],vmax=map_prop['at_range'][1],cmap=map_prop['at_cmap']),
                       arr_prop =map_prop['cv_arr_prop'],
                       stream_prop =dict(density=1,color='C0',linewidth=2))

    return Features_df,Features_ch,Features_mat,Data
