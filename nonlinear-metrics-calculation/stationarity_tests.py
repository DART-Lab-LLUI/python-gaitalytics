#!/usr/bin/env python3
"""
analysis_metrics_pipeline.py

Batch pipeline to process all 'markers.nc' files under a data root:
  - Recursively finds every markers.nc (excluding any under 'segmented' folders)
  - Computes mean/variance, rolling stats, and stationarity tests (ADF & KPSS)
  - Saves outputs (plots and ADF/KPSS results) under a 'results/' subfolder next to each markers.nc

Usage:
  python analysis_metrics_pipeline.py [-d path/to/data] [-ch CHANNEL] [-w WINDOW]
"""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def mean_variance(data: xr.DataArray, dataset_name: str, channel: str = "COM"):
    print(f"\n--- Mean & Variance for {dataset_name} ({channel}) ---")
    for axis_val in data.coords['axis'].values:
        if channel in data.sel(axis=axis_val).coords['channel'].values:
            subset = data.sel(axis=axis_val, channel=channel)
            mean_val = float(subset.mean(dim='time', skipna=True))
            var_val = float(subset.var(dim='time', skipna=True))
            print(f"Axis={axis_val}: Mean={mean_val:.4f}, Variance={var_val:.4f}")
        else:
            print(f"Axis={axis_val}: channel '{channel}' not found")


def rolling_mean_variance(data: xr.DataArray,
                          dataset_name: str,
                          channel: str = "COM",
                          window_size: int = 100,
                          output_dir: Path = None):
    ensure_dir(output_dir)
    time = data.coords['time'].values
    for axis_val in data.coords['axis'].values:
        if channel not in data.sel(axis=axis_val).coords['channel'].values:
            continue
        series = pd.Series(data.sel(axis=axis_val, channel=channel).values, index=time)
        roll_mean = series.rolling(window=window_size).mean()
        roll_var = series.rolling(window=window_size).var()

        fig, ax = plt.subplots(figsize=(10, 6))
        # Raw data in gray
        ax.plot(series.index, series, label='Raw', color='gray', alpha=0.5)
        # Rolling mean in blue
        ax.plot(roll_mean.index, roll_mean, label='Rolling Mean', color='blue')
        # Mean line in black dashed
        ax.axhline(series.mean(), linestyle='--', color='black', label=f"Mean={series.mean():.4f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax2 = ax.twinx()
        # Rolling variance in red
        ax2.plot(roll_var.index, roll_var, label='Rolling Var', color='red', alpha=0.7)
        # Variance line in dark red dashed
        ax2.axhline(series.var(), linestyle='--', color='darkred', label=f"Var={series.var():.4f}")
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f"{dataset_name} Axis={axis_val}")
        plt.tight_layout()

        out_file = output_dir / f"{dataset_name}_axis{axis_val}_rolling.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved rolling plot: {out_file}")


def adf_kpss_test(data: xr.DataArray,
                  dataset_name: str,
                  channel: str = "COM",
                  output_dir: Path = None):
    """
    Runs both ADF and KPSS stationarity tests and saves the results to text files.
    """
    ensure_dir(output_dir)
    for axis_val in data.coords['axis'].values:
        out_path = output_dir / f"{dataset_name}_axis{axis_val}_stationarity.txt"
        try:
            series = data.sel(axis=axis_val, channel=channel).values.squeeze()
            # ADF test
            adf_stat, adf_p = adfuller(series)[:2]
            # KPSS test (default regression='c', nlags='auto')
            kpss_stat, kpss_p, _, _ = kpss(series, nlags='auto')
        except Exception as e:
            print(f"Stationarity test failed for axis {axis_val}: {e}")
            adf_stat = adf_p = kpss_stat = kpss_p = np.nan

        with open(out_path, 'w') as f:
            f.write(f"ADF: stat={adf_stat:.4f}, p={adf_p:.4f}\n")
            f.write(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f}\n")
        print(f"Saved stationarity results: {out_path}")



def plot_time_series(data: xr.DataArray, dataset_name: str, channel: str):
    """
    Plot raw time series for each axis separately.
    """
    time = data.coords['time'].values
    for axis_val in data.coords['axis'].values:
        if channel not in data.sel(axis=axis_val).coords['channel'].values:
            continue
        series = data.sel(axis=axis_val, channel=channel).values
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, series)
        ax.set_title(f"{dataset_name} - Axis={axis_val} Raw Time Series")
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        plt.tight_layout()
        plt.show()

    
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt, detrend


def detrend_only(data: xr.DataArray, fs: float = 100.0) :
    def detrend_signal(ts: np.ndarray) -> np.ndarray:
        # 1) Detrend
        ts_dt = detrend(ts)
        return ts_dt
    # Prepare an output array
    detrended = xr.full_like(data, np.nan)

    for axis_val in data.coords['axis'].values:
        for chan in data.coords['channel'].values:
            # extract 1D timeseries
            ts = data.sel(axis=axis_val, channel=chan).values.astype(float)

            # if NaNs/Infs, interpolate & fill
            if not np.isfinite(ts).all():
                s = pd.Series(ts)
                s = s.interpolate(limit_direction='both')
                ts = s.fillna(method='bfill').fillna(method='ffill').values

            # run the single-step preprocess
            try:
                processed_ts = detrend_signal(ts)
            except Exception as e:
                print(f"Preprocessing failed for axis={axis_val}, channel={chan}: {e}")
                continue

            # assign back
            detrended.loc[dict(axis=axis_val, channel=chan)] = processed_ts

    # preserve the time coordinate
    detrended = detrended.assign_coords(time=data.coords['time'])
    return detrended
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import kpss

def is_stationary(ts, alpha=0.05, regression='c'):
    stat, p_value, _, _ = kpss(ts, regression=regression, nlags='auto')
    return p_value > alpha

def _butter_filter(ts, fs, hp_cutoff=None, lp_cutoff=None, order=4):
    """
    Apply a Butterworth high-pass, low-pass, or band-pass filter to 1D array ts.
    - hp_cutoff: high-pass cutoff in Hz (None to skip)
    - lp_cutoff: low-pass cutoff in Hz (None to skip)
    - order: filter order
    """
    nyq = 0.5 * fs
    if (hp_cutoff is None) and (lp_cutoff is None):
        return ts

    if (hp_cutoff is not None) and (lp_cutoff is not None):
        low = hp_cutoff / nyq
        high = lp_cutoff / nyq
        b, a = butter(order, [low, high], btype='band')
    elif hp_cutoff is not None:
        freq = hp_cutoff / nyq
        b, a = butter(order, freq, btype='high')
    else:  # lp_cutoff only
        freq = lp_cutoff / nyq
        b, a = butter(order, freq, btype='low')

    return filtfilt(b, a, ts)

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import kpss

def is_stationary(ts, alpha=0.05, regression='c'):
    """
    Returns True if ts (1D) passes KPSS at level alpha.
    - Constant series → True
    - Too-short series (< 10 samples) → False
    - Any exception → False
    """
    ts = np.asarray(ts, dtype=float)
    n = ts.size

    # 1) require minimum length for KPSS
    if n < 10:
        return False

    # 2) constant series are stationary
    if np.allclose(ts, ts[0]):
        return True

    # 3) choose a sensible nlags (e.g. one-third of samples)
    nlags = max(1, n // 3)

    # 4) run KPSS, catching any errors
    try:
        stat, p_value, _, _ = kpss(ts, regression=regression, nlags=nlags)
    except Exception:
        return False

    return (p_value > alpha)


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


def preprocess_epochs(data: xr.DataArray,
                      fs: float,
                      epoch_length_s: float = 30.0,
                      poly_order: int = 1,
                      hp_cutoff: float = None,
                      lp_cutoff: float = None,
                      filter_order: int = 4,
                      require_stationarity: bool = True,
                      fallback_to_full: bool = True) -> xr.DataArray:
    """
    1) Globally filter + detrend.
    2) Chop into epochs.
    3) If require_stationarity:
         keep only epochs where ALL channels pass KPSS.
       If none pass & fallback_to_full=True:
         return the full cleaned signal.
    """
    # GLOBAL FILTER + DETREND
    axes    = data.coords['axis'].values
    chans   = data.coords['channel'].values
    full = data.copy(deep=True).astype(float)
    N    = full.sizes['time']
    tidx = np.arange(N)
    for ax in axes:
        for ch in chans:
            ts = full.sel(axis=ax, channel=ch).values
            tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
            coeffs = np.polyfit(tidx, tsf, poly_order)
            full.loc[dict(axis=ax, channel=ch)] = tsf - np.polyval(coeffs, tidx)

    
    n_samp = int(epoch_length_s * fs)
    kept   = []
    for start in range(0, N, n_samp):
        end = start + n_samp
        if end > N:
            break
        epoch = full.isel(time=slice(start, end))
        if not require_stationarity:
            kept.append(epoch)
            continue

        # check ALL channels
        ok = True
        for ax in axes:
            for ch in chans:
                if not is_stationary(epoch.sel(axis=ax, channel=ch).values):
                    ok = False
                    break
            if not ok:
                break

        if ok:
            kept.append(epoch)

    # if nothing passed and we allow fallback, return the full series
    if require_stationarity and not kept and fallback_to_full:
        warnings.warn("No epochs passed stationarity; returning full cleaned signal instead")
        return full

    if not kept:
        raise RuntimeError("No epochs passed stationarity and fallback disabled!")

    return xr.concat(kept, dim='time')

import numpy as np
import xarray as xr
from pathlib import Path
from scipy.signal import detrend, butter, filtfilt
import matplotlib.pyplot as plt
from typing import Union


# 1) Preprocessing: detrend, band‐pass, and z‐score
def process_data(data: Union[xr.DataArray, np.ndarray],
                 fs: float,
                 poly_order: int = 1,
                 hp_cutoff: float = None,
                 lp_cutoff: float = None,
                 filter_order: int = 4) -> Union[xr.DataArray, np.ndarray]:
    """
    Globally filter and detrend the data.
    Returns the same type as input (xarray DataArray or numpy array).
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
    
    # Handle xarray DataArray
    axes = data.coords['axis'].values
    chans = data.coords['channel'].values
    full = data.copy(deep=True).astype(float)
    N = full.sizes['time']
    # tidx = np.arange(N)
    
    if 'time' in data.coords and len(data.coords['time']) == N:
        time_values = data.coords['time'].values
        # Normalize time to start at 0 for polynomial fitting
        tidx = time_values - time_values[0]
    else:
        tidx = np.arange(N)
    

    for ax in axes:
        for ch in chans:
            ts = full.sel(axis=ax, channel=ch).values
            tsf = _butter_filter(ts, fs, hp_cutoff, lp_cutoff, filter_order)
            coeffs = np.polyfit(tidx, tsf, poly_order)
            full.loc[dict(axis=ax, channel=ch)] = tsf - np.polyval(coeffs, tidx)
    
    return full 
