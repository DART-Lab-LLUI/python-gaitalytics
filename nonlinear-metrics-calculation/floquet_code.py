
# Description: This script is built for calculating Floquet Multiplier.
#              The code is adpated to handle strides and steps, plots the Poincare Plot
#              by default only on the heel strike phase, but if needed can be adjusted to
#              plot other phases or as well full cycle. 
#              The code is implemented following the methodology outlined 
#              in Hurmuzlu & Basdogan (1994) and Dingwell & Kang (2007).


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from numpy.linalg import pinv, eig
from scipy import signal
from sklearn.utils import resample


def apply_lowpass_filter(data: np.ndarray, 
                        fs: float, 
                        cutoff: float = 8.0,
                        order: int = 4) -> np.ndarray:
    """Applies zero-lag Butterworth low-pass filter to data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = np.zeros_like(data)
    for ax in range(data.shape[1]):
        filtered_data[:, ax] = signal.filtfilt(b, a, data[:, ax])
    
    return filtered_data


def find_fixed_point(S: np.ndarray, max_iter: int = 20, tol: float = 1e-6) -> np.ndarray:
    """
    Find fixed point of Poincaré map where x_{k+1} = x_k.

    """
    X = S[:-1, :]  # States at stride k
    Y = S[1:, :]   # States at stride k+1
    
    # Initial guess: mean state
    x_star = S.mean(axis=0)
    
    for i in range(max_iter):
        # Find closest point in X to current guess
        distances = np.sum((X - x_star)**2, axis=1)
        idx = np.argmin(distances)
        
        # The fixed point should satisfy: F(x_star) = x_star
        # So Y[idx] should equal X[idx] at the fixed point
        x_star_new = 0.5 * (X[idx, :] + Y[idx, :])
        
        # Check convergence
        if np.linalg.norm(x_star_new - x_star) < tol:
            
            break
        x_star = x_star_new
    else:
        None
    
    return x_star


def compute_floquet_from_events(com_da: xr.DataArray,
                                events_df: pd.DataFrame,
                                channel: str = "COM",
                                axes: list = ["x", "y", "z"],
                                normalization_points: int = 101,
                                rcond: float = 1e-6,
                                phase_dependent: bool = False,
                                use_same_side: bool = True,
                                force_side: str = None,
                                trim_cycles: int = 5,
                                bootstrap_n: int = 100,
                                bootstrap_ci: float = 95.0,
                                lowpass_cutoff: float = 8.0,
                                return_all_eigenvalues: bool = False,
                                use_fixed_point: bool = True,
                                detrend_y: bool = True) -> dict:
    """
    Compute Floquet multipliers on events from a DataArray of center of mass (COM) data.
    Parametrs are dynamic.
    """
    # Detrend y-axis if requested - extracts the speed on treadmill
    com_da_processed = com_da.copy()
    if detrend_y and 'y' in axes:
        y_data = com_da.sel(channel=channel, axis='y').values
        t_secs = (com_da.time.values - com_da.time.values[0])  # array of seconds
        coeffs = np.polyfit(t_secs, y_data, 1)
        trend  = np.polyval(coeffs, t_secs)
        y_detrended = y_data - trend    
        time_points = np.arange(len(y_data))
        
        # Fit linear trend
        coeffs = np.polyfit(time_points, y_data, 1)
        trend = np.polyval(coeffs, time_points)
        y_detrended = y_data - trend
        
        com_da_processed.loc[dict(channel=channel, axis='y')] = y_detrended
        print(f"Removed drift from y-axis: {trend[-1] - trend[0]:.1f} units total")
    
    # Extract all Foot Strike times and side (context)
    df_fs = events_df[events_df['label'] == 'Foot Strike'].copy()
    df_fs = df_fs[['time','context']].sort_values('time')
    strike_times = df_fs['time'].values
    strike_context = df_fs['context'].values
    
    # Determine which strikes to use based on same-side setting
    if use_same_side:
        left_indices = np.where(strike_context == 'Left')[0]
        right_indices = np.where(strike_context == 'Right')[0]
        
        if force_side is not None:
            # Use the forced side
            side_used = force_side
            use_indices = left_indices if force_side == 'Left' else right_indices
        else:
            # Auto-select the side with more strikes, if no side defined 
            if len(left_indices) >= len(right_indices):
                use_indices = left_indices
                side_used = 'Left'
            else:
                use_indices = right_indices
                side_used = 'Right'
        
        # Create stride segments from same-side strikes
        stride_starts = use_indices[:-1]
        stride_ends = use_indices[1:]
        print(f"Using {side_used}-to-{side_used} strides: {len(stride_starts)} strides")
    else:
        # Use all consecutive strikes
        stride_starts = np.arange(len(strike_times) - 1)
        stride_ends = stride_starts + 1
        side_used = 'both'
        print(f"Using all consecutive strikes: {len(stride_starts)} strides")
    
    # Build raw segments and store stride durations
    raw_segs = []
    raw_ts = []
    raw_contexts = []
    stride_durations = []
    
    for i, (start_idx, end_idx) in enumerate(zip(stride_starts, stride_ends)):
        t0 = strike_times[start_idx]
        t1 = strike_times[end_idx]
        
        seg = (com_da_processed
               .sel(channel=channel, axis=axes)
               .sel(time=slice(t0, t1))
               .transpose('time', 'axis'))
        t = seg.time.values
        
        if t.size < 10:  # if segment very short, skip
            continue
            
        raw_segs.append(seg.values)
        raw_ts.append(t)
        raw_contexts.append(strike_context[start_idx])
        stride_durations.append(t1 - t0)  # Store stride duration
    
    n_cycles = len(raw_segs)
    n_axes = len(axes)
    state_dim = 2 * n_axes

    
    if n_cycles < state_dim + 1 + 2*trim_cycles:
        raise ValueError(f"Need ≥{state_dim + 1 + 2*trim_cycles} valid strides, got {n_cycles}")
    
    # Calculate mean stride duration BEFORE trimming 
    mean_duration = np.mean(stride_durations)
    print(f"Mean stride duration: {mean_duration:.3f} seconds")
    
    # Apply low-pass filter to each segment before normalization
    filtered_segs = []
    for i, (seg_vals, t) in enumerate(zip(raw_segs, raw_ts)):
        fs = 1.0 / np.mean(np.diff(t))
        filtered_seg = apply_lowpass_filter(seg_vals, fs, lowpass_cutoff)
        filtered_segs.append(filtered_seg)
    
    # Time-normalize each cycle while preserving actual time scale
    M = normalization_points
    norm_t = np.linspace(0, 1, M)
    com_norm = np.zeros((n_cycles, M, n_axes), dtype=float)
    
    for i, (seg_vals, t) in enumerate(zip(filtered_segs, raw_ts)):
        t_norm = (t - t[0]) / (t[-1] - t[0])
        for ax in range(n_axes):
            com_norm[i, :, ax] = np.interp(norm_t, t_norm, seg_vals[:, ax])
    
    # Calculated velocity
    dt_actual = mean_duration / (M - 1)
    
    # Trim first and last cycles AFTER calculating mean duration
    if trim_cycles > 0:
        com_norm = com_norm[trim_cycles:-trim_cycles]
        raw_contexts = raw_contexts[trim_cycles:-trim_cycles]
        stride_durations = stride_durations[trim_cycles:-trim_cycles]
        n_cycles_trimmed = n_cycles - 2*trim_cycles
        print(f"Trimmed {trim_cycles} cycles from start/end, using {n_cycles_trimmed} cycles")
    else:
        n_cycles_trimmed = n_cycles
    
    # Central-difference velocity
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :]) / (2 * dt_actual)
    vel_norm[:, 0, :]     = (com_norm[:, 1, :] - com_norm[:, 0, :]) / dt_actual
    vel_norm[:, -1, :]    = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt_actual
    
    def _estimate_jacobian_and_max(S_pos, S_vel, return_jacobian=False):
        """Estimate Jacobian and maximum Floquet multiplier."""
        S = np.hstack([S_pos, S_vel])
        X = S[:-1, :]
        Y = S[1:,  :]
        
        # Find linearization point
        if use_fixed_point:
            linearization_point = find_fixed_point(S)
        else:
            linearization_point = S.mean(axis=0)
        
        dX = (X - linearization_point).T
        dY = (Y - linearization_point).T
        
        try:
            pinv_dX = np.linalg.pinv(dX, rcond=rcond)
            J = dY @ pinv_dX
        except np.linalg.LinAlgError:
            lam = rcond * np.trace(dX @ dX.T)
            J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))
        
        eigs = np.linalg.eigvals(J)
        mags = np.abs(eigs)
        
        # Check for neutral mode (should be close to 1)
        neutral_idx = np.argmin(np.abs(mags - 1.0))
        if np.abs(mags[neutral_idx] - 1.0) > 0.1:
            None
        
        
        if return_all_eigenvalues:
            sorted_idx = np.argsort(mags)[::-1]
            return eigs[sorted_idx], mags[sorted_idx], J if return_jacobian else None
        elif return_jacobian:
            return eigs, np.max(mags), J
        else:
            return eigs, np.max(mags)
    
    def _bootstrap_floquet(S_pos, S_vel, n_boot=100, ci=95.0):
        """Compute bootstrap confidence interval for max Floquet multiplier."""
        n_samples = S_pos.shape[0]
        max_lams = []
        
        for _ in range(n_boot):
            indices = resample(np.arange(n_samples), n_samples=n_samples)
            S_pos_boot = S_pos[indices]
            S_vel_boot = S_vel[indices]
            
            try:
                _, max_lam = _estimate_jacobian_and_max(S_pos_boot, S_vel_boot)
                max_lams.append(max_lam)
            except:
                continue
        
        if len(max_lams) > 0:
            max_lams = np.array(max_lams)
            ci_low = np.percentile(max_lams, (100 - ci) / 2)
            ci_high = np.percentile(max_lams, 100 - (100 - ci) / 2)
            return np.mean(max_lams), (ci_low, ci_high)
        else:
            return None, (None, None)
    
    # Compute results
    if not phase_dependent:
        S_pos = com_norm[:, 0, :]
        S_vel = vel_norm[:, 0, :]
        
        # Compute main result
        if return_all_eigenvalues:
            eigs, mags = _estimate_jacobian_and_max(S_pos, S_vel, return_jacobian=False)[:2]
            max_lam = mags[0]  # sorted by magnitude
        else:
            _, max_lam = _estimate_jacobian_and_max(S_pos, S_vel)
        
        # Bootstrap confidence interval
        mean_lam, (ci_low, ci_high) = _bootstrap_floquet(S_pos, S_vel, bootstrap_n, bootstrap_ci)
        
        # Poincaré plot
        if n_axes == 1:
            # For single axis analysis, plot in position-velocity space
            S = np.hstack([S_pos, S_vel])
            xk = S[:-1, 0]
            xk1 = S[1:, 0]
            
            axis_labels = {
                'x': ('Medio-lateral', 'ML'),
                'y': ('Anterior-posterior', 'AP'),
                'z': ('Vertical', 'Vert')
            }
            full_name, short_name = axis_labels[axes[0]]
            
            plt.figure(figsize=(6, 6))
            color = 'blue' if side_used == 'Right' else 'red'
            plt.scatter(xk, xk1, s=30, alpha=0.6, c=color, label=f'{side_used}→{side_used}')
            
            mn, mx = min(xk.min(), xk1.min()), max(xk.max(), xk1.max())
            plt.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5)
            plt.xlabel(f'{short_name} position at strike k (m)')
            plt.ylabel(f'{short_name} position at strike k+1 (m)')
            plt.title(f'Poincaré Map ({full_name}, {side_used} side)\nMax |λ| = {max_lam:.3f} [{ci_low:.3f}, {ci_high:.3f}]')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.show()
            plt.close()
        else:
            # For multi-axis analysis, plot using first axis
            S = np.hstack([S_pos, S_vel])
            axis_idx = 0  # Use first axis for visualization
            xk = S[:-1, axis_idx]
            xk1 = S[1:, axis_idx]
            
            plt.figure(figsize=(6, 6))
            color = 'blue' if side_used == 'Right' else 'red'
            plt.scatter(xk, xk1, s=30, alpha=0.6, c=color, label=f'{side_used}→{side_used}')
            
            mn, mx = min(xk.min(), xk1.min()), max(xk.max(), xk1.max())
            plt.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5)
            
            state_dim_str = f"{2*n_axes}D"  # Dynamic dimension string
            plt.xlabel(f'{axes[0].upper()} position at strike k (m)')
            plt.ylabel(f'{axes[0].upper()} position at strike k+1 (m)')
            plt.title(f'Poincaré Map (Combined {state_dim_str}, {side_used} side)\nMax |λ| = {max_lam:.3f} [{ci_low:.3f}, {ci_high:.3f}]')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.show()
            plt.close()
        
        result = {
            'max_floquet': float(max_lam),
            'confidence_interval': (ci_low, ci_high),
            'n_cycles_used': n_cycles_trimmed,
            'trimmed_cycles': trim_cycles,
            'side_used': side_used if use_same_side else 'both',
            'axes_analyzed': axes,
            'mean_stride_duration': mean_duration,
            'used_fixed_point': use_fixed_point
        }
        
        if return_all_eigenvalues:
            result['all_eigenvalues'] = eigs
            result['all_magnitudes'] = mags
        
        return result
    
    else:
        # Phase-dependent analysis
        max_vs_phi = np.zeros(M)
        ci_low_vs_phi = np.zeros(M)
        ci_high_vs_phi = np.zeros(M)
        
        for phi in range(M):
            S_pos = com_norm[:, phi, :]
            S_vel = vel_norm[:, phi, :]
            
            _, max_vs_phi[phi] = _estimate_jacobian_and_max(S_pos, S_vel)
            
            # Bootstrap for this phase
            _, (ci_low, ci_high) = _bootstrap_floquet(S_pos, S_vel, bootstrap_n, bootstrap_ci)
            ci_low_vs_phi[phi] = ci_low if ci_low is not None else max_vs_phi[phi]
            ci_high_vs_phi[phi] = ci_high if ci_high is not None else max_vs_phi[phi]
        
        phases = np.linspace(0, 100, M)
        
        plt.figure(figsize=(10, 6))
        plt.plot(phases, max_vs_phi, '-', linewidth=2, label=f'Mean ({side_used} side)')
        plt.fill_between(phases, ci_low_vs_phi, ci_high_vs_phi, alpha=0.3, label=f'{bootstrap_ci}% CI')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Stability boundary')
        plt.xlabel('Gait phase (%)')
        plt.ylabel('Max |λ|')
        
        if n_axes == 1:
            axis_name = axes[0]
            axis_label = {'x': 'Medio-lateral', 'y': 'Anterior-posterior', 'z': 'Vertical'}[axis_name]
            plt.title(f'Phase-dependent Floquet ({axis_label}, {side_used} side, n={n_cycles_trimmed})')
        else:
            state_dim_str = f"{2*n_axes}D"
            plt.title(f'Phase-dependent Floquet (Combined {state_dim_str}, {side_used} side, n={n_cycles_trimmed})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.close()
        
        return {
            'max_floquet': max_vs_phi,
            'confidence_interval': (ci_low_vs_phi, ci_high_vs_phi),
            'n_cycles_used': n_cycles_trimmed,
            'trimmed_cycles': trim_cycles,
            'side_used': side_used if use_same_side else 'both',
            'axes_analyzed': axes,
            'mean_stride_duration': mean_duration,
            'used_fixed_point': use_fixed_point
        }


def compute_floquet_per_axis(com_da: xr.DataArray,
                            events_df: pd.DataFrame,
                            channel: str = "COM",
                            detrend_y: bool = True,
                            **kwargs):
    """Compute Floquet multipliers for x, y, and z axes individually."""
    results = {}
    
    print("\n" + "="*60)
    print("COMPUTING PER-AXIS FLOQUET MULTIPLIERS")
    print("="*60)
    
    for axis, name in [('x', 'Medio-lateral'), ('y', 'Anterior-posterior'), ('z', 'Vertical')]:
        print(f"\nAnalyzing {name} ({axis}) axis...")
        
        # Only detrend if it's the y-axis (due to treadmill speed)
        should_detrend = detrend_y and (axis == 'y')
        
        axis_result = compute_floquet_from_events(
            com_da=com_da,
            events_df=events_df,
            channel=channel,
            axes=[axis],
            detrend_y=should_detrend,
            **kwargs
        )
        results[axis] = axis_result
    
    return results


def extract_normalized_data_with_filter(com_window, df_window, 
                                       use_same_side=True, 
                                       side_used='Right',
                                       trim_cycles=5,
                                       lowpass_cutoff=8.0,
                                       normalization_points=101,
                                       detrend_y=True):
    """Extract normalized and filtered COM data matching the Floquet analysis preprocessing."""
    # Detrend y-axis (due to treadmill speed)
    com_window_processed = com_window.copy()
    if detrend_y:
        if 'y' in com_window.axis.values:
            y_data = com_window.sel(channel="COM", axis='y').values
            time_points = np.arange(len(y_data))
            coeffs = np.polyfit(time_points, y_data, 1)
            trend = np.polyval(coeffs, time_points)
            y_detrended = y_data - trend
            com_window_processed.loc[dict(channel="COM", axis='y')] = y_detrended
    
    # Get foot strike events
    df_fs = df_window.query("label=='Foot Strike'")[['time','context']].sort_values('time')
    strike_times = df_fs['time'].values
    strike_context = df_fs['context'].values
    
    # Determine which strikes to use
    if use_same_side:
        side_indices = np.where(strike_context == side_used)[0]
        stride_starts = side_indices[:-1]
        stride_ends = side_indices[1:]
        
    else:
        stride_starts = np.arange(len(strike_times) - 1)
        stride_ends = stride_starts + 1
    
    # Build raw segments
    raw_segs = []
    raw_ts = []
    stride_durations = []
    com_data = com_window_processed.sel(channel="COM", axis=["x","y","z"])  # All three axes
    
    for i, (start_idx, end_idx) in enumerate(zip(stride_starts, stride_ends)):
        t0 = strike_times[start_idx]
        t1 = strike_times[end_idx]
        
        seg = com_data.sel(time=slice(t0, t1)).transpose('time','axis')
        t = seg.time.values
        
        if t.size < 10:
            continue
            
        raw_segs.append(seg.values)
        raw_ts.append(t)
        stride_durations.append(t1 - t0)
    
    # Calculate mean duration BEFORE trimming
    mean_duration = np.mean(stride_durations)
    
    # Apply low-pass filter to each segment
    filtered_segs = []
    for seg_vals, t in zip(raw_segs, raw_ts):
        fs = 1.0 / np.mean(np.diff(t))
        filtered_seg = apply_lowpass_filter(seg_vals, fs, lowpass_cutoff)
        filtered_segs.append(filtered_seg)
    
    # Time-normalize each cycle
    n_cycles = len(filtered_segs)
    M = normalization_points
    norm_t = np.linspace(0, 1, M)
    com_norm = np.zeros((n_cycles, M, 3))  
    
    for idx, (vals, t) in enumerate(zip(filtered_segs, raw_ts)):
        t_norm = (t - t[0])/(t[-1] - t[0])
        for ax in range(3):  # All 3 axes
            com_norm[idx, :, ax] = np.interp(norm_t, t_norm, vals[:, ax])
    
    # Trim cycles
    if trim_cycles > 0 and n_cycles > 2*trim_cycles:
        com_norm = com_norm[trim_cycles:-trim_cycles]
        print(f"Trimmed {trim_cycles} cycles from start/end")
    
    # Calculate velocities
    dt_actual = mean_duration / (M - 1)
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :]) / (2 * dt_actual)
    vel_norm[:, 0, :]     = (com_norm[:, 1, :] - com_norm[:, 0, :]) / dt_actual
    vel_norm[:, -1, :]    = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt_actual
    
    return com_norm, vel_norm, mean_duration