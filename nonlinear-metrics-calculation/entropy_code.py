
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from preprocessing import preprocess_signal, _butter_filter

def approximate_entropy(U: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Calculates Approximate Entropy (ApEn) with m=2, r=0.2*SD default.
    On displacement signal.
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
    Calcautes Sample Entropy (SampEn) with m=2, r=0.2*SD default.
    On displacement signal.
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
    r = r_factor * np.std(ts_norm) 
    
    # Compute entropies
    apen = approximate_entropy(ts_norm, m=2, r=r)
    sampen = sample_entropy(ts_norm, m=2, r=r)
    
    return apen, sampen

def compute_entropy_on_acceleration(ts_disp, fs=100.0, r_factor=0.2):
    """
    Compute ApEn and SampEn on acceleration derived from displacement.
    """
    # Compute velocity and acceleration
    vel = np.gradient(ts_disp, 1/fs)
    acc = np.gradient(vel, 1/fs)
    
    # Band-pass filter acceleration
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 10/nyq], btype='bandpass')
    acc_f = filtfilt(b, a, acc)
    
    # Z-score normalization
    acc_z = (acc_f - np.mean(acc_f)) / np.std(acc_f)
    
    # Compute entropies with tolerance
    r = r_factor * np.std(acc_z)  # This will be r_factor since acc_z has std=1
    apen = approximate_entropy(acc_z, m=2, r=r)
    sampen = sample_entropy(acc_z, m=2, r=r)
    
    return apen, sampen

def compute_MSE_on_acceleration(ts_disp, r_factor=0.2, fs=100.0, max_scale=20):
    """
    Compute Multiscale Entropy on acceleration derived from displacement.
    """
    # Compute velocity and acceleration
    vel = np.gradient(ts_disp, 1/fs)
    acc = np.gradient(vel, 1/fs)
    
    # Band-pass filter acceleration
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 10/nyq], btype='bandpass')
    acc_f = filtfilt(b, a, acc)
    
    # Z-score normalization
    acc_z = (acc_f - np.mean(acc_f)) / np.std(acc_f)
    
    # Compute MSE on acc
    r = r_factor * np.std(acc_z) 
    scales, mse = multiscale_entropy(acc_z, m=2, r=r, max_scale=max_scale)
    
    return scales, mse