import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def compute_spatial_nsi(markers: xr.DataArray,
                       window_samples: int = 500,
                       fs: float = 100.0) -> dict:
   """  Compute Spatial Non-Stationarity Index (NSI) 
   for the Center of Mass (COM) marker"""
   
   # extract COM marker and reorganize dimensions
   com = markers.sel(channel='COM').transpose('time', 'axis')
   data = com.values.copy()  

   com_dt = xr.DataArray(data, coords=com.coords, dims=com.dims)

   # plot filtered COM traces for all axes - ok for nsi
   fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
   axis_names = ['x', 'y', 'z']
   axis_labels = ['ML (x)', 'AP (y)', 'VT (z)']
   
   for idx, (ax_name, ax_label) in enumerate(zip(axis_names, axis_labels)):
       if ax_name in com_dt.axis.values:
           axes[idx].plot(com_dt.sel(axis=ax_name).values, label=f"Filtered {ax_label}")
           axes[idx].set_ylabel(f"{ax_label} Position\n(detrended)")
           axes[idx].legend()
           axes[idx].grid(True)
   
   axes[0].set_title("High-Pass Filtered COM Traces (All Axes)")
   axes[-1].set_xlabel("Sample")
   plt.tight_layout()
   plt.show()

   # calculate nsi with detailed window statistics
   def nsi_detailed(ts: np.ndarray, w: int) -> tuple:
       n = len(ts)
       if n < w:
           return np.nan, []
       
       windows = []
       window_means = []
       window_stds = []
       
       # process non-overlapping windows
       for i in range(0, n - w + 1, w):
           window_data = ts[i:i+w]
           windows.append((i, i+w))
           window_means.append(np.mean(window_data))
           window_stds.append(np.std(window_data, ddof=1))
       
       # actual calculation
       overall_std = np.std(ts, ddof=1)
       if overall_std == 0:
           return np.nan, []
       
       nsi_value = np.std(window_means, ddof=1) / overall_std
       
       return nsi_value, list(zip(windows, window_means, window_stds))

   # prints
   N = com_dt.sizes['time']
   M = (N - window_samples) // window_samples + 1
   print(f"COM series length: {N} samples ({N/fs:.2f} s)")
   print(f"Window size:       {window_samples} samples ({window_samples/fs:.2f} s)")
   print(f"Number of windows: {M}")
   print("-" * 80)

   # compute nsi for each axis
   results = {}
   detailed_results = {}
   
   for ax in com_dt.axis.values:
       arr = com_dt.sel(axis=ax).values
       nsi_val, window_details = nsi_detailed(arr, window_samples)
       results[ax] = nsi_val
       detailed_results[ax] = window_details

       # print window statistics
       print(f"\nAxis: {ax}")
       print(f"Overall NSI: {nsi_val:.4f}")
       print(f"{'Window':<15} {'Time Range (s)':<20} {'Mean':<12} {'Std':<12}")
       print("-" * 60)
       
       for idx, ((start, end), mean, std) in enumerate(window_details):
           time_start = start / fs
           time_end = end / fs
           print(f"Window {idx+1:<8} {time_start:6.2f} - {time_end:6.2f} s    {mean:8.4f}    {std:8.4f}")
  
   # create detailed visualization
   fig = plt.figure(figsize=(16, 12))
   gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
   
   axis_names = ['x', 'y', 'z']
   axis_labels = ['ML (x)', 'AP (y)', 'VT (z)']
   colors = ['blue', 'green', 'red']
   
   for idx, (ax_name, ax_label, color) in enumerate(zip(axis_names, axis_labels, colors)):
       if ax_name in com_dt.axis.values and ax_name in detailed_results:
           # plot time series with windows
           ax1 = fig.add_subplot(gs[idx, 0])
           time_array = np.arange(len(com_dt.sel(axis=ax_name))) / fs
           ax1.plot(time_array, com_dt.sel(axis=ax_name).values, 'k-', alpha=0.5, linewidth=0.5)
           
           window_data = detailed_results[ax_name]
           for win_idx, ((start, end), mean, std) in enumerate(window_data):
               t_start = start / fs
               t_end = end / fs
               # shade windows
               ax1.axvspan(t_start, t_end, alpha=0.2, color=color if win_idx % 2 == 0 else 'gray')
               # show window means
               ax1.hlines(mean, t_start, t_end, colors=color, linewidth=2)
               # label windows
               ax1.text((t_start + t_end) / 2, ax1.get_ylim()[1] * 0.9, f'W{win_idx+1}', 
                       ha='center', va='top', fontsize=8)
           
           ax1.set_ylabel(f'{ax_label} Position')
           ax1.set_title(f'{ax_label} Time Series with Windows')
           ax1.grid(True, alpha=0.3)
           if idx == 2:
               ax1.set_xlabel('Time (s)')
          
           # plot window statistics
           ax2 = fig.add_subplot(gs[idx, 1])
           window_indices = list(range(1, len(window_data) + 1))
           window_means = [w[1] for w in window_data]
           window_stds = [w[2] for w in window_data]
           
           ax2.errorbar(window_indices, window_means, yerr=window_stds, 
                       fmt='o-', color=color, capsize=5, capthick=2,
                       label='Mean ± Std')
           ax2.axhline(y=np.mean(window_means), color='black', linestyle='--', 
                      alpha=0.5, label='Overall mean')
           
           ax2.set_ylabel('Value')
           ax2.set_title(f'{ax_label} Window Statistics')
           ax2.grid(True, alpha=0.3)
           ax2.legend()
           if idx == 2:
               ax2.set_xlabel('Window Number')
           
           # plot deviations from overall mean
           ax3 = fig.add_subplot(gs[idx, 2])
           overall_mean = np.mean(com_dt.sel(axis=ax_name).values)
           deviations = [w[1] - overall_mean for w in window_data]
           
           ax3.bar(window_indices, deviations, color=color, alpha=0.7)
           ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
           ax3.set_ylabel('Deviation from Overall Mean')
           ax3.set_title(f'{ax_label} Window Mean Deviations')
           ax3.grid(True, alpha=0.3)
           
           # add NSI value
           ax3.text(0.98, 0.95, f'NSI = {results.get(ax_name, np.nan):.4f}', 
                   transform=ax3.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
           
           if idx == 2:
               ax3.set_xlabel('Window Number')
   
   fig.suptitle(f'NSI Window Analysis (Window size: {window_samples/fs:.1f}s, Total: {N/fs:.1f}s)', 
                fontsize=14)
   plt.tight_layout()
   plt.show()
   
   # summary plots just for window - just check
   fig2, ax = plt.subplots(figsize=(12, 6))
   
   for ax_name, color in zip(axis_names, colors):
       if ax_name in detailed_results:
           window_data = detailed_results[ax_name]
           if window_data:
               window_centers = [(w[0][0] + w[0][1]) / 2 / fs for w in window_data]
               window_means = [w[1] for w in window_data]
               
               ax.plot(window_centers, window_means, 'o-', color=color, 
                      label=f'{ax_name.upper()} (NSI={results.get(ax_name, np.nan):.4f})',
                      markersize=8, linewidth=2)
   
   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Window Mean Position')
   ax.set_title('Window Means Comparison Across All Axes')
   ax.grid(True, alpha=0.3)
   ax.legend()
   plt.tight_layout()
   plt.show()

   return results, detailed_results


def compute_temporal_nsi_from_events(df_events: pd.DataFrame,
                                    strides_per_window: int = 5,
                                    trim_strides: int = 0,
                                    outlier_thresh: float = 3.0) -> float:
   """ Compute Temporal Non-Stationarity Index (NSI)"""
   
   # filter for Foot Strike events and sort by time
   df_fs = (df_events[df_events['label'] == 'Foot Strike']
            .sort_values('time'))

   # calculate time intervals between consecutive foot strikes
   intervals = df_fs['time'].diff().dropna().values

   # remove startup/shutdown strides from both ends - this is needed (adjustable)
   if len(intervals) > 2 * trim_strides:
       intervals = intervals[trim_strides:-trim_strides]

   # remove outliers beyond threshold * standard deviations
   mu, sigma = intervals.mean(), intervals.std(ddof=1)
   mask = np.abs(intervals - mu) <= outlier_thresh * sigma
   intervals = intervals[mask]

   # calculate temp nsi using block averaging
   def nsi(ts, k):
       n = len(ts)
       if n < k or ts.std(ddof=1) == 0:
           return np.nan
       # create non-overlapping blocks of k strides and compute their means
       block_means = [ts[i:i+k].mean() for i in range(0, n-k+1, k)]
       # nsi = variability of block means / overall variability
       return np.std(block_means, ddof=1) / ts.std(ddof=1)

   return nsi(intervals, strides_per_window)