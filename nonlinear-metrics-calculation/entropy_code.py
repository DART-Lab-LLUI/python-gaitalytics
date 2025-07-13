
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import detrend, butter, filtfilt
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt

import numpy as np
from scipy.signal import butter, filtfilt

def _butter_filter(ts, fs, hp_cutoff=None, lp_cutoff=None, order=4):
    """High/low/band pass filter, same as before."""
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

# 1) Preprocessing: detrend, band‐pass, and z‐score
def preprocess_signal(data: Union[xr.DataArray, np.ndarray],
                      fs: float,
                      poly_order: int = 1,
                      hp_cutoff: float = None,
                      lp_cutoff: float = None,
                      filter_order: int = 4) -> np.ndarray:
    """
    Globally filter and detrend the data, then return as numpy array.
    Handles both xarray DataArrays and 1D numpy arrays.
    """
    # Handle 1D numpy array input
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("Numpy array input must be 1D")
        
        ts = data.astype(float)
        N = len(ts)
        tidx = np.arange(N)
        
        # Filter
        tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
        
        # Detrend
        coeffs = np.polyfit(tidx, tsf, poly_order)
        return tsf - np.polyval(coeffs, tidx)
    
    # Handle xarray DataArray (original code)
    axes = data.coords['axis'].values
    chans = data.coords['channel'].values
    full = data.copy(deep=True).astype(float)
    N = full.sizes['time']
    tidx = np.arange(N)
    
    for ax in axes:
        for ch in chans:
            ts = full.sel(axis=ax, channel=ch).values
            tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
            coeffs = np.polyfit(tidx, tsf, poly_order)
            full.loc[dict(axis=ax, channel=ch)] = tsf - np.polyval(coeffs, tidx)
    
    return full.values



def approximate_entropy(U: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Approximate Entropy (ApEn) with m=2, r=0.2*SD default.
    """
    U = np.asarray(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)
    
    def phi(mv):
        X = np.array([U[i:i+mv] for i in range(N-mv+1)])
        C = []
        for xi in X:
            # Count patterns within tolerance r
            matches = np.sum(np.max(np.abs(X - xi), axis=1) <= r)
            C.append(matches / (N - mv + 1))
        # Avoid log(0) by adding small epsilon
        C = np.array(C)
        C[C == 0] = np.finfo(float).eps
        return np.mean(np.log(C))
    
    return phi(m) - phi(m+1)

def sample_entropy(U: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Sample Entropy (SampEn) with m=2, r=0.2*SD default.
    """
    U = np.asarray(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)
    
    # Templates of length m
    X = np.array([U[i:i+m] for i in range(N-m+1)])
    # Templates of length m+1
    X1 = np.array([U[i:i+m+1] for i in range(N-m)])
    
    def count_matches(Xarr):
        cnt = 0
        for i in range(len(Xarr)):
            for j in range(i+1, len(Xarr)):
                if np.max(np.abs(Xarr[i] - Xarr[j])) <= r:
                    cnt += 1
        return cnt
    
    B = count_matches(X)
    A = count_matches(X1)
    
    if B == 0 or A == 0:
        return np.inf
    else:
        return -np.log(A / B)

def coarse_grain(ts: np.ndarray, scale: int) -> np.ndarray:
    """Non-overlapping average for MSE coarse-graining."""
    N = len(ts)
    num = N // scale
    return np.mean(ts[:num*scale].reshape(num, scale), axis=1)

def multiscale_entropy(ts: np.ndarray,
                      m: int = 2,
                      r: float = None,
                      max_scale: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Multiscale Entropy (MSE): SampEn on each coarse-grained series.
    """
    if r is None:
        r = 0.2 * np.std(ts)
    
    scales = np.arange(1, max_scale + 1)
    mse = []
    
    for scale in scales:
        # Coarse-grain the time series
        coarse_ts = coarse_grain(ts, scale)
        # Compute sample entropy on coarse-grained series
        se = sample_entropy(coarse_ts, m=m, r=r)
        mse.append(se)
    
    return scales, np.array(mse)

def compute_entropy_on_displacement(ts_disp, fs=100.0, r_factor=0.2):
    """
    Compute ApEn and SampEn directly on displacement signal.
    """
    # Normalize displacement signal
    ts_norm = (ts_disp - np.mean(ts_disp)) / np.std(ts_disp)
    
    # Set tolerance
    r = r_factor * np.std(ts_norm)  # This will be r_factor since ts_norm has std=1
    
    # Compute entropies
    apen = approximate_entropy(ts_norm, m=2, r=r)
    sampen = sample_entropy(ts_norm, m=2, r=r)
    
    return apen, sampen

def compute_entropy_on_acceleration(ts_disp, fs=100.0, r_factor=0.2):
    """
    Compute ApEn and SampEn on acceleration derived from displacement.
    """
    # 1) Compute velocity and acceleration
    vel = np.gradient(ts_disp, 1/fs)
    acc = np.gradient(vel, 1/fs)
    
    # 2) Band-pass filter acceleration (0.5-10 Hz)
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 10/nyq], btype='bandpass')
    acc_f = filtfilt(b, a, acc)
    
    # 3) Z-score normalization
    acc_z = (acc_f - np.mean(acc_f)) / np.std(acc_f)
    
    # 4) Compute entropies with tolerance
    r = r_factor * np.std(acc_z)  # This will be r_factor since acc_z has std=1
    apen = approximate_entropy(acc_z, m=2, r=r)
    sampen = sample_entropy(acc_z, m=2, r=r)
    
    return apen, sampen

def compute_MSE_on_acceleration(ts_disp, r_factor=0.2, fs=100.0, max_scale=20):
    """
    Compute Multiscale Entropy on acceleration derived from displacement.
    """
    # 1) Compute velocity and acceleration
    vel = np.gradient(ts_disp, 1/fs)
    acc = np.gradient(vel, 1/fs)
    
    # 2) Band-pass filter acceleration (0.5-10 Hz)
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 10/nyq], btype='bandpass')
    acc_f = filtfilt(b, a, acc)
    
    # 3) Z-score normalization
    acc_z = (acc_f - np.mean(acc_f)) / np.std(acc_f)
    
    # 4) Compute MSE
    r = r_factor * np.std(acc_z)  # This will be r_factor since acc_z has std=1
    scales, mse = multiscale_entropy(acc_z, m=2, r=r, max_scale=max_scale)
    
    return scales, mse