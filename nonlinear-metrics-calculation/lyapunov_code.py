import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


def get_foot_strike(events_path, foot='Right', start_time=20.0, end_time=None):
    """
    Get the first foot strike after defining a start_time and last foot strike 
    untill the defiend end_time.
    
    Returns:
    --------
    dict : Dictionary with 'first_strike' and 'last_strike' times in seconds
           Returns None for either if not found
    """
    
    if events_path.suffix == '.csv':
        df_events = pd.read_csv(events_path)
    elif events_path.suffix == '.nc':
        events_ds = xr.open_dataset(events_path)
        events_da = events_ds["events"] if "events" in events_ds else events_ds
        df_events = events_da.to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported file type: {events_path.suffix}")
    
    foot_strikes = df_events[
        (df_events['label'] == 'Foot Strike') & 
        (df_events['context'] == foot) &
        (df_events['time'] >= start_time) &
        (df_events['time'] <= end_time)
    ]
    

    first_strike = foot_strikes['time'].min() if len(foot_strikes) > 0 else None
    last_strike = foot_strikes['time'].max() if len(foot_strikes) > 0 else None
    
    return {
        'first_strike': first_strike,
        'last_strike': last_strike
    }


def get_average_stride_duration(events_path, foot='Right', start_time=20.0, end_time=None):
    """
    Calculate the mean duration between consecutive foot strikes for a specific foot.
   
    Returns:
    --------
    dict : Dictionary with mean_duration, std_duration, num_strides, and all durations
    """

    if events_path.suffix == '.csv':
        df_events = pd.read_csv(events_path)
    elif events_path.suffix == '.nc':
        events_ds = xr.open_dataset(events_path)
        events_da = events_ds["events"] if "events" in events_ds else events_ds
        df_events = events_da.to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported file type: {events_path.suffix}")
    
    foot_strikes = df_events[
        (df_events['label'] == 'Foot Strike') & 
        (df_events['context'] == foot) &
        (df_events['time'] >= start_time) &
        (df_events['time'] <= end_time)
    ].sort_values('time')
    

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
    """
    Compute AMI I(lag) between x(t) and x(t+lag) using equal‐occupancy bins.
    """
    edges = np.quantile(ts, np.linspace(0, 1, bins+1))
    x, y = ts[:-lag], ts[lag:]
    hist2d, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    pxy = hist2d / np.sum(hist2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    mask = pxy > 0
    return np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))


def optimal_lag(ts: np.ndarray,
                fs: float = 100.0,
                max_lag: int = 50,
                bins: int = 64) -> int:
    """
    Find a single, adaptive delay τ for embedding via Average Mutual Information (AMI),
    suitable for both healthy and Parkinson's gait.

    Steps:
      1) Compute AMI(τ) for τ = 1…max_lag.
      2) Plot AMI vs. τ with the AMI(1)/e threshold for visual check.
      3) Search for the first “filtered mild local minimum,”
           • AMI[i] ≤ AMI[i−1]  AND  AMI[i] ≤ AMI[i+1]
           • AND (AMI[i] ≤ 0.80·AMI(1)   OR   AMI[i] ≤ 0.90·AMI[i−1])
        (This prevents the algorithm from choosing a very small τ unless AMI has already
         dropped by ≥ 20% of AMI(1).)
      4) If no acceptable knee appears by τ = ⌊0.20·fs⌋, fallback to the first τ ≤ ⌊0.20·fs⌋
         where AMI(τ) ≤ AMI(1)/e.
      5) If nothing qualifies by τ = ⌊0.20·fs⌋, return the global‐minimum lag over 1…max_lag.
    """

    ami_vals = [average_mutual_information(ts, lag, bins)
                for lag in range(1, max_lag + 1)]
    lags = np.arange(1, max_lag + 1)
    ami1 = ami_vals[0]

 
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

    # If τ_max < 3 we can’t form a “local minimum” search, so just skip to fallback
    if τ_max < 3:
        return 1

    #    (Indices i = 1 … (τ_max - 2) correspond to τ = i+1 = 2 … (τ_max-1))
    for i in range(1, τ_max - 1):  # i+1 ranges from 2 to τ_max-1
        if ami_vals[i] <= ami_vals[i - 1] and ami_vals[i] <= ami_vals[i + 1]:
            τ_candidate = i + 1  # because ami_vals[0] is τ=1
            # 4a) If τ_candidate is small (<5), only accept it if AMI has fallen by ≥20%
            if τ_candidate < 5:
                if ami_vals[i] <= 0.80 * ami1:
                    return τ_candidate
                else:
                    # Too shallow a drop at a very small τ—ignore it
                    continue
            # 4b) If τ_candidate ≥ 5, accept it unconditionally
            return τ_candidate

    # 5) Fallback #1: first τ ≤ τ_max where AMI(τ) ≤ AMI(1)/e
    threshold = ami1 / np.e
    for i in range(τ_max):  # i corresponds to τ = i+1
        if ami_vals[i] <= threshold:
            return i + 1

    # 6) Fallback #2: global minimum over τ = 1…max_lag
    return int(np.argmin(ami_vals) + 1)


def compute_fnn_percentage(ts: np.ndarray, lag: int,
                           max_dim: int = 15, rtol: float = 0.15):
    """
    Compute % of false nearest neighbors for embedding dims from 1 to max_dim.
    """
    N = len(ts)
    dims = np.arange(1, max_dim + 1)
    fnn_perc = np.zeros_like(dims, dtype=float)

    for idx, m in enumerate(dims):
        M = N - m * lag
        if M <= 0:
            fnn_perc[idx:] = np.nan
            break
        # Build m-dimensional delay vectors
        embedded = np.vstack([ts[i * lag : i * lag + M] for i in range(m)]).T
        false = total = 0
        for i in range(M):
            # compute distances to all other points
            dists = np.linalg.norm(embedded - embedded[i], axis=1)
            dists[i] = np.inf
            # exclude temporally too-close points
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
    """
    Choose embedding dimension m by finding the 'elbow' in the FNN% curve:
      1) Compute FNN% for m = 1…max_dim
      2) Take the second discrete difference of the fnn% array
      3) Return the dimension m corresponding to the index of maximal |second_difference|

    This follows the theoretical idea that the largest curvature (elbow) in the FNN% vs. m
    plot is the best embedding dimension when no fixed threshold is applied.
    """
    dims, fnn = compute_fnn_percentage(ts, lag, max_dim, rtol)

    if len(dims) < 3 or np.all(np.isnan(fnn)):
        return max_dim
    
    second_diff = np.diff(fnn, n=2)

    if np.all(np.isnan(second_diff)) or np.allclose(second_diff, 0, equal_nan=True):
        return max_dim

    i_elbow = int(np.nanargmax(np.abs(second_diff)))
    m_elbow = i_elbow + 2

    if m_elbow < 1:
        return 1
    if m_elbow > max_dim:
        return max_dim
    return m_elbow


def compute_lyapunov_exponent(ts: np.ndarray, emb_dim: int, lag: int,
                              fs: float = 100.0, max_time: float = 5.0,
                              fit_start: float = 0.1, fit_end: float = 0.5,
                              plot_residuals: bool = True) -> float:
    """
    Compute short‐term Lyapunov exponent λ* by tracking divergence of nearest neighbors.
    Fit ⟨ln d(t)⟩ vs. t over fit_start–fit_end s.
    """
    N = len(ts)
    M = N - (emb_dim - 1) * lag
    if M <= 0:
        return np.nan

    embedded = np.vstack([ts[i * lag : i * lag + M] for i in range(emb_dim)]).T
    min_sep = emb_dim * lag
    divergence = []

    for i in range(M):
        dists = np.linalg.norm(embedded - embedded[i], axis=1)
        dists[i] = np.inf
        dists[max(0, i - min_sep) : i + min_sep] = np.inf
        j = np.argmin(dists)
        if dists[j] == np.inf:
            continue

        k_max = min(M - i, M - j, int(max_time * fs))
        for k in range(1, k_max):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 0:
                divergence.append((k / fs, np.log(d)))

    if not divergence:
        return np.nan

    data = np.array(divergence)
    times, logs = data[:, 0], data[:, 1]

    # Bin and average
    bins = np.linspace(0, max_time, 50)
    centers, means = [], []
    for b in range(len(bins) - 1):
        mask = (times >= bins[b]) & (times < bins[b + 1])
        if mask.any():
            centers.append((bins[b] + bins[b + 1]) / 2)
            means.append(logs[mask].mean())

    centers = np.array(centers)
    means = np.array(means)

    # Plot divergence curve for inspection
    plt.figure(figsize=(10, 4))
    
    # Subplot 1: Divergence curve
    plt.subplot(1, 2, 1)
    plt.plot(centers, means, 'o-')
    plt.axvline(fit_start, color='r', linestyle='--', label=f'{fit_start} s')
    plt.axvline(fit_end, color='r', linestyle='--', label=f'{fit_end} s')
    plt.xlabel('Time (s)')
    plt.ylabel('Average ln(distance)')
    plt.title('Divergence curve')
    plt.grid(True)
    plt.legend()

    # Fit only over fit_start–fit_end s
    fit_mask = (centers >= fit_start) & (centers <= fit_end)
    if fit_mask.sum() < 2:
        plt.show()
        return np.nan

    slope, intercept = np.polyfit(centers[fit_mask], means[fit_mask], 1)
    
    # Plot residuals if requested
    if plot_residuals:
        # Subplot 2: Residuals
        plt.subplot(1, 2, 2)
        
        # Calculate fitted values and residuals
        fitted_values = slope * centers[fit_mask] + intercept
        residuals = means[fit_mask] - fitted_values
        
        # Plot residuals
        plt.scatter(centers[fit_mask], residuals, color='blue', alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', label='Zero line')
        plt.xlabel('Time (s)')
        plt.ylabel('Residuals')
        plt.title(f'Residuals of linear fit ({fit_start:.2f}-{fit_end:.2f} s)')
        plt.grid(True)
        plt.legend()
        
        # Add fit line to divergence plot
        plt.subplot(1, 2, 1)
        plt.plot(centers[fit_mask], fitted_values, 'r-', linewidth=2, 
                 label=f'Linear fit: slope={slope:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

    return slope