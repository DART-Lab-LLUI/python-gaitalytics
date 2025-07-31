
# Description: This script is built identically on the preprocessing.py file,
#              but it is used to preprocess the data when there is downsampling
#              needed as in the case of sLE, such that the script is distiguishable
#              The downsampling is done by resampling the data to a target frequency.
#              It uses a butterworth filter for high-pass, low-pass, or band-pass filtering
#              and applies polynomial detrending. Handles both xarray DataArray and 1D numpy
#              array inputs.    

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt, resample
from typing import Union, Optional

def _butter_filter(ts: np.ndarray,
                   fs: float,
                   hp_cutoff: Optional[float] = None,
                   lp_cutoff: Optional[float] = None,
                   order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth highpass, lowpass, or bandpass filter to a 1D signal."""
    
    nyq = 0.5 * fs
    if hp_cutoff is None and lp_cutoff is None:
        return ts

    if hp_cutoff and lp_cutoff:
        btype, freqs = 'band', [hp_cutoff/nyq, lp_cutoff/nyq]
    elif hp_cutoff:
        btype, freqs = 'high', hp_cutoff/nyq
    else:
        btype, freqs = 'low', lp_cutoff/nyq

    b, a = butter(order, freqs, btype=btype)
    return filtfilt(b, a, ts)

def preprocess_signal(data: Union[xr.DataArray, np.ndarray],
                      fs: float,
                      poly_order: int = 1,
                      hp_cutoff: float = None,
                      lp_cutoff: float = None,
                      filter_order: int = 4,
                      target_fs: Optional[float] = None) -> Union[np.ndarray, xr.DataArray]:
    """
    Detrend, filter, and optionally downsample signal data.
    """

    def _downsample(ts: np.ndarray, orig_fs: float, tgt_fs: float) -> np.ndarray:
        # return original if no downsampling
        if tgt_fs is None or tgt_fs >= orig_fs:
            return ts
        # compute output length and resample
        num_out = int(len(ts) * tgt_fs / orig_fs)
        return resample(ts, num_out)

    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("Numpy array input must be 1D")
        ts = data.astype(float)
        tidx = np.arange(len(ts))

        # filter
        tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)  
        # fit trend
        coeffs = np.polyfit(tidx, tsf, poly_order)                       
        # detrend
        tsd = tsf - np.polyval(coeffs, tidx)                              
        return _downsample(tsd, fs, target_fs)                             

    # xarray.DataArray case
    da = data.copy(deep=True).astype(float)
    axes = da.coords['axis'].values
    chans = da.coords['channel'].values
    N = da.sizes['time']
    tidx = np.arange(N)

    processed = []
    for ax in axes:
        ax_block = []
        for ch in chans:
            ts = da.sel(axis=ax, channel=ch).values
            tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
            coeffs = np.polyfit(tidx, tsf, poly_order)
            tsd = tsf - np.polyval(coeffs, tidx)
            ax_block.append(_downsample(tsd, fs, target_fs))
        processed.append(ax_block)

    # if downsampled, rebuild DataArray
    if target_fs is not None and target_fs < fs:
        num_new = int(N * target_fs / fs)
        t0, t1 = da.coords['time'][[0, -1]].values
        time_new = np.linspace(t0, t1, num_new)
        arr = np.array(processed).transpose(1, 0, 2)  # shape (channel, axis, time)
        return xr.DataArray(arr,
                            coords={'channel': chans, 'axis': axes, 'time': time_new},
                            dims=['channel', 'axis', 'time'])

    return da
