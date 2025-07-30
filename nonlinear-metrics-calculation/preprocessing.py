# Author:      Natasha Kovacheva <natasha.kovacheva@stud.hslu.ch>
# Created:     SS 2025
# Description: Defines all functions for preprocessing the data, 
#              including filtering and detrending, as described in the thesis;
#              it uses a butterworth filter for high-pass, low-pass, or band-pass filtering,
#              and applies polynomial detrending. It was customed to handle both
#              xarray DataArray and 1D numpy array inputs.

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from typing import Union


def _butter_filter(ts, fs, hp_cutoff=None, lp_cutoff=None, order=4):
    """
    Apply a Butterworth highpass, lowpass, or bandpass filter to a 1D signal."""
    
    nyq = 0.5 * fs
    if hp_cutoff is None and lp_cutoff is None:
        return ts

    if hp_cutoff and lp_cutoff:
        btype = 'band'
        freqs = [hp_cutoff/nyq, lp_cutoff/nyq]
    elif hp_cutoff:
        btype = 'high'; freqs = hp_cutoff/nyq
    else:
        btype = 'low';  freqs = lp_cutoff/nyq

    b, a = butter(order, freqs, btype=btype)
    return filtfilt(b, a, ts)


def preprocess_signal(data: Union[xr.DataArray, np.ndarray],
                      fs: float,
                      poly_order: int = 1,
                      hp_cutoff: float = None,
                      lp_cutoff: float = None,
                      filter_order: int = 4) -> np.ndarray:
    """
    Detrend and filter signal data, returning a NumPy array."""
   
    # handle a 1D numpy array input
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("Numpy array input must be 1D")
        
        ts = data.astype(float)
        N = len(ts)
        tidx = np.arange(N)
    
        tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)

        coeffs = np.polyfit(tidx, tsf, poly_order)
        return tsf - np.polyval(coeffs, tidx)
    
    # handle xarray DataArray input
    axes = data.coords['axis'].values
    chans = data.coords['channel'].values
    full = data.copy(deep=True).astype(float)
    N = full.sizes['time']
    tidx = np.arange(N)
    
    # apply filter and detrend on each axis
    for ax in axes:
        for ch in chans:
            ts = full.sel(axis=ax, channel=ch).values
            tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
            coeffs = np.polyfit(tidx, tsf, poly_order)
            full.loc[dict(axis=ax, channel=ch)] = tsf - np.polyval(coeffs, tidx)
    
    return full.values


