# Author:      Natasha Kovacheva <natasha.kovacheva@stud.hslu.ch>
# Created:     SS 2025
# Description: This script is built for calculating the short term Lyapunov Exponent,
#              as it is described in the literature, following Rosenstein's Algorithm
#              and using Taken's Theorem. All steps ans functions needed 
#              for the computation are included in this script.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


def get_foot_strike(events_path, foot='Right', start_time=20.0, end_time=None):
   """ Get first and last foot strike times within a specified time window """
   
   # load events from CSV or NetCDF file
   if events_path.suffix == '.csv':
       df_events = pd.read_csv(events_path)
   elif events_path.suffix == '.nc':
       events_ds = xr.open_dataset(events_path)
       events_da = events_ds["events"] if "events" in events_ds else events_ds
       df_events = events_da.to_dataframe().reset_index()
   else:
       raise ValueError(f"Unsupported file type: {events_path.suffix}")
   
   # filter for foot strikes within time window
   foot_strikes = df_events[
       (df_events['label'] == 'Foot Strike') & 
       (df_events['context'] == foot) &
       (df_events['time'] >= start_time) &
       (df_events['time'] <= end_time)
   ]
   
   # get first and last strike times
   first_strike = foot_strikes['time'].min() if len(foot_strikes) > 0 else None
   last_strike = foot_strikes['time'].max() if len(foot_strikes) > 0 else None
   
   return {
       'first_strike': first_strike,
       'last_strike': last_strike
   }


def get_average_stride_duration(events_path, foot='Right', start_time=20.0, end_time=None):
   """" Get average stride duration for a specified foot within a time window """

   # load events from file
   if events_path.suffix == '.csv':
       df_events = pd.read_csv(events_path)
   elif events_path.suffix == '.nc':
       events_ds = xr.open_dataset(events_path)
       events_da = events_ds["events"] if "events" in events_ds else events_ds
       df_events = events_da.to_dataframe().reset_index()
   else:
       raise ValueError(f"Unsupported file type: {events_path.suffix}")
   
   # filter and sort foot strikes
   foot_strikes = df_events[
       (df_events['label'] == 'Foot Strike') & 
       (df_events['context'] == foot) &
       (df_events['time'] >= start_time) &
       (df_events['time'] <= end_time)
   ].sort_values('time')
   
   # calculate stride durations
   strike_times = foot_strikes['time'].values
   # compute the time difference between consecutive strikes
   durations = np.diff(strike_times) 
   
   if len(durations) > 0:
       return {
           'mean_duration': np.mean(durations),
           'std_duration': np.std(durations),
           'num_strides': len(durations),
           'durations': durations
       }
   else:
       return {
           'mean_duration': np.nan,
           'std_duration': np.nan,
           'num_strides': 0,
           'durations': np.array([])
       }

def average_mutual_information(ts: np.ndarray, lag: int, bins: int = 64) -> float:
   """ Compute Average Mutual Information (AMI) for a time series to search for a given lag """
   # create equal-occupancy bin edges
   edges = np.quantile(ts, np.linspace(0, 1, bins+1))
   x, y = ts[:-lag], ts[lag:]
   # compute 2D histogram
   hist2d, _, _ = np.histogram2d(x, y, bins=[edges, edges])
   # normalize to get joint probability
   pxy = hist2d / np.sum(hist2d)
   # marginal probabilities
   px = np.sum(pxy, axis=1)
   py = np.sum(pxy, axis=0)
   # mutual information calculation
   mask = pxy > 0
   return np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))


def optimal_lag(ts: np.ndarray,
               fs: float = 100.0,
               max_lag: int = 50,
               bins: int = 64) -> int: 
   """ Find optimal lag for time series using AMI method """

   # compute AMI for all lags
   ami_vals = [average_mutual_information(ts, lag, bins)
               for lag in range(1, max_lag + 1)]
   lags = np.arange(1, max_lag + 1)
   ami1 = ami_vals[0]

   # plot AMI curve - for visualization
   plt.figure(figsize=(5, 3))
   plt.plot(lags, ami_vals, '-o', label='AMI(τ)')
   plt.axhline(ami1 / np.e, color='r', linestyle='--', label='AMI(1)/e')
   plt.xlabel('Lag (samples)')
   plt.ylabel('AMI')
   plt.title('AMI vs. Lag')
   plt.grid(True)
   plt.legend()
   plt.tight_layout()
   plt.show()

   τ_max = min(max_lag, int(0.20 * fs)) 

   # can't form a "local minimum" search, 
   # if lag too low, so just skip to fallback
   if τ_max < 3:
       return 1

   # search for local minimum with constraints
   for i in range(1, τ_max - 1):  
       if ami_vals[i] <= ami_vals[i - 1] and ami_vals[i] <= ami_vals[i + 1]:
           τ_candidate = i + 1  
           # only accept it if AMI has fallen by ≥80%
           if τ_candidate < 5:
               if ami_vals[i] <= 0.80 * ami1:
                   return τ_candidate
               else:
                   continue
           return τ_candidate

   # fallback #1: first lag where AMI(τ) smaller than AMI(1)/e
   threshold = ami1 / np.e
   for i in range(τ_max): 
       if ami_vals[i] <= threshold:
           return i + 1

   # fallback #2: global minimum up to max_lag
   return int(np.argmin(ami_vals) + 1)

def compute_fnn_percentage(ts: np.ndarray, lag: int,
                           max_dim: int = 15, rtol: float = 0.15):
    """ Compute False Nearest Neighbors (FNN) percentage for a time series """

    N = len(ts)
    dims = np.arange(1, max_dim + 1)
    fnn_perc = np.zeros_like(dims, dtype=float)

    for idx, m in enumerate(dims):
        M = N - m * lag
        if M <= 0:
            fnn_perc[idx:] = np.nan
            break
        # build m-dimensional delay vectors
        embedded = np.vstack([ts[i * lag : i * lag + M] for i in range(m)]).T
        false = total = 0
        for i in range(M):
            # compute distances to all other points
            dists = np.linalg.norm(embedded - embedded[i], axis=1)
            dists[i] = np.inf
            # exclude temporally too-close points (Theiler window)
            dists[max(0, i - m * lag) : i + m * lag] = np.inf
            j = np.argmin(dists)
            if dists[j] == np.inf:
                continue
            d_extra = abs(ts[i + m * lag] - ts[j + m * lag])
            if d_extra / dists[j] > rtol:
                false += 1
            total += 1
        fnn_perc[idx] = false / total if total else np.nan

    return dims, fnn_perc

def optimal_embedding_dimension(ts: np.ndarray,
                               lag: int,
                               max_dim: int = 15,
                               rtol: float = 0.15) -> int:
   """ Find optimal embedding dimension using 
   False Nearest Neighbors (FNN) method """
   
   # compute FNN percentages
   dims, fnn = compute_fnn_percentage(ts, lag, max_dim, rtol)

   if len(dims) < 3 or np.all(np.isnan(fnn)):
       return max_dim
   
   # find elbow using second difference
   second_diff = np.diff(fnn, n=2)

   if np.all(np.isnan(second_diff)) or np.allclose(second_diff, 0, equal_nan=True):
       return max_dim

   # elbow at maximum curvature
   i_elbow = int(np.nanargmax(np.abs(second_diff)))
   m_elbow = i_elbow + 2

   # bounds checking
   if m_elbow < 1:
       return 1
   if m_elbow > max_dim:
       return max_dim
   return m_elbow


def compute_lyapunov_exponent(ts: np.ndarray, emb_dim: int, lag: int,
                              fs: float = 100.0, max_time: float = 5.0,
                              fit_start: float = 0.1, fit_end: float = 0.5,
                              plot_residuals: bool = True) -> float:
    """" Compute Lyapunov exponent from time series data, 
    using proper phase space reconstruction"""
    N = len(ts)
    M = N - (emb_dim - 1) * lag
    if M <= 0:
        return np.nan

    embedded = np.vstack([ts[i * lag : i * lag + M] for i in range(emb_dim)]).T
    min_sep = emb_dim * lag
    
    # calculate distance threshold for neighbor selection
    min_dist_threshold = np.std(embedded) * 0.01 
    
    divergence = []
    skipped_points = 0  # points skipped because no valid neighbors

    for i in range(M):
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        dists[i] = np.inf
        dists[max(0, i - min_sep) : i + min_sep] = np.inf
        
        # improved neighbor selection with distance threshold
        valid_neighbors = dists < min_dist_threshold
        if valid_neighbors.any():
            # find closest among valid neighbors
            valid_dists = dists.copy()
            valid_dists[~valid_neighbors] = np.inf
            j = np.argmin(valid_dists)
        else:
            # fallback to original approach if no neighbors within threshold
            j = np.argmin(dists)
            if dists[j] == np.inf:
                skipped_points += 1
                continue

        k_max = min(M - i, M - j, int(max_time * fs))
        for k in range(1, k_max):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 0:
                divergence.append((k / fs, np.log(d)))

    if not divergence:
        print(f"Warning: No valid divergence data found. Skipped {skipped_points} points.")
        return np.nan

    # print diagnostic info - for debugging/qualtiy check
    print(f"Distance threshold: {min_dist_threshold:.6f}")
    print(f"Skipped points due to no valid neighbors: {skipped_points}/{M}")
    print(f"Total divergence data points: {len(divergence)}")

    data = np.array(divergence)
    times, logs = data[:, 0], data[:, 1]

    # bin and average
    bins = np.linspace(0, max_time, 50)
    centers, means = [], []
    for b in range(len(bins) - 1):
        mask = (times >= bins[b]) & (times < bins[b + 1])
        if mask.any():
            centers.append((bins[b] + bins[b + 1]) / 2)
            means.append(logs[mask].mean())

    centers = np.array(centers)
    means = np.array(means)
    
    
    # plot divergence curve as visual inspection
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(centers, means, 'o-')
    plt.axvline(fit_start, color='r', linestyle='--', label=f'{fit_start} s')
    plt.axvline(fit_end, color='r', linestyle='--', label=f'{fit_end} s')
    plt.xlabel('Time (s)')
    plt.ylabel('Average ln(distance)')
    plt.title('Divergence curve')
    plt.grid(True)
    plt.legend()

    # fit only over fit_start–fit_end s (because is sLE)
    fit_mask = (centers >= fit_start) & (centers <= fit_end)
    if fit_mask.sum() < 2:
        print("Warning: Insufficient data points for reliable fit")
        plt.show()
        return np.nan

    slope, intercept = np.polyfit(centers[fit_mask], means[fit_mask], 1)
    
    # uncomment the following lines, if no R sq needed
    # calculate fit quality R sq
    fitted_values = slope * centers[fit_mask] + intercept
    r_squared = 1 - np.sum((means[fit_mask] - fitted_values)**2) / np.var(means[fit_mask]) if np.var(means[fit_mask]) > 0 else 0
    print(f"Fit quality (R²): {r_squared:.3f}")
    
    if r_squared < 0.5:
        print(f"Warning: Poor fit quality (R² = {r_squared:.3f})")
    
    # plot residuals if requested
    if plot_residuals:
        plt.subplot(1, 2, 2)
        # calculate residuals
        residuals = means[fit_mask] - fitted_values
        
        # plot residuals as quality check
        plt.scatter(centers[fit_mask], residuals, color='blue', alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', label='Zero line')
        plt.xlabel('Time (s)')
        plt.ylabel('Residuals')
        plt.title(f'Residuals of linear fit ({fit_start:.2f}-{fit_end:.2f} s)')
        plt.grid(True)
        plt.legend()
        
        # add fit line to divergence plot - for visualization
        plt.subplot(1, 2, 1)
        plt.plot(centers[fit_mask], fitted_values, 'r-', linewidth=2, 
                 label=f'Linear fit: slope={slope:.3f}, R²={r_squared:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

    return slope

