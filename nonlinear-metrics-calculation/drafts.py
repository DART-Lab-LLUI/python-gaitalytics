# def compute_spatial_nsi(markers: xr.DataArray,
#                         window_samples: int = 500,
#                         fs: float = 100.0) -> dict:
#     """
    
#      1) Extract 'COM' channel, transpose to dims ('time','axis')
#      2) Detrend all axes (x, y, z) with a 0.1 Hz high-pass filter
#      3) Anchor each filtered axis so it starts at zero
#      4) Compute NSI separately for x, y, z using non-overlapping windows of size `window_samples`

#     Parameters
#     ----------
#     markers : xr.DataArray
#         DataArray with dims ('axis','channel','time') and a channel named 'COM'.
#     window_samples : int
#         Number of time-samples in each non-overlapping window for NSI.
#     fs : float
#         Sampling rate (Hz) of the COM time series.

#     Returns
#     -------
#     dict
#         Spatial NSI for each axis: {'x': val, 'y': val, 'z': val}
#     """
#     # 1) Extract COM channel and reorder dims → (time, axis)
#     com = markers.sel(channel='COM').transpose('time', 'axis')
#     data = com.values.copy()  # shape: (time, axis)

#     # 2) High-pass filter all axes at 0.1 Hz (cutting out drifts > 10 s)
#     hp_cutoff = 0.1  # Hz
#     b, a = butter(4, hp_cutoff / (fs / 2), btype='highpass')
    
#     # Apply filter to each axis
#     for i in range(data.shape[1]):
#         data[:, i] = filtfilt(b, a, data[:, i])
#         # 3) Anchor each axis to start at zero
#         data[:, i] -= data[0, i]

#     # 4) Rebuild the detrended/filtered COM DataArray
#     com_dt = xr.DataArray(data, coords=com.coords, dims=com.dims)

#     # Optional: visualize detrended traces for all axes
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#     axis_names = ['x', 'y', 'z']
#     axis_labels = ['ML (x)', 'AP (y)', 'VT (z)']
    
#     for idx, (ax_name, ax_label) in enumerate(zip(axis_names, axis_labels)):
#         if ax_name in com_dt.axis.values:
#             axes[idx].plot(com_dt.sel(axis=ax_name).values, label=f"Filtered {ax_label}")
#             axes[idx].set_ylabel(f"{ax_label} Position\n(detrended)")
#             axes[idx].legend()
#             axes[idx].grid(True)
    
#     axes[0].set_title("High-Pass Filtered COM Traces (All Axes)")
#     axes[-1].set_xlabel("Sample")
#     plt.tight_layout()
#     plt.show()

#     # 5) NSI helper on a 1D array
#     def nsi(ts: np.ndarray, w: int) -> float:
#         n = len(ts)
#         if n < w or np.std(ts, ddof=1) == 0:
#             return np.nan
#         block_means = [ts[i:i+w].mean() for i in range(0, n - w + 1, w)]
#         return np.std(block_means, ddof=1) / np.std(ts, ddof=1)

#     # 6) Print window info for the user
#     N = com_dt.sizes['time']
#     M = (N - window_samples) // window_samples + 1
#     print(f"COM series length: {N} samples ({N/fs:.2f} s)")
#     print(f"Window size:       {window_samples} samples ({window_samples/fs:.2f} s)")
#     print(f"Number of windows: {M}")

#     # 7) Compute NSI per axis
#     results = {}
#     for ax in com_dt.axis.values:
#         arr = com_dt.sel(axis=ax).values
#         results[ax] = nsi(arr, window_samples)

#     return results

# def compute_temporal_nsi_from_events(events_df: pd.DataFrame,
#                                      strides_per_window: int = 10,
#                                      trim_strides: int = 5,
#                                      outlier_thresh: float = 3.0) -> float:
#     """
#     Compute temporal NSI over the full trial from a CSV of events.

#     Expects columns:
#       - time: float (event timestamp, in seconds)
#       - leg:  str   ('Left' or 'Right')
#       - event: str  (e.g. 'Foot Strike')

#     Steps:
#       1) Keep only rows where event == 'Foot Strike'
#       2) Extract timestamps for Left vs. Right
#       3) Compute per-side stride intervals (diff of successive times)
#       4) Pool Left+Right intervals into one array
#       5) Trim the first/last `trim_strides` intervals (belt ramp artifacts)
#       6) Discard intervals beyond ± outlier_thresh × SD
#       7) NSI = SD(non-overlapping block-means) / SD(full interval series)

#     Parameters
#     ----------
#     events_df : pd.DataFrame
#         Must contain columns ['time','leg','event'].
#         'leg' should be 'Left' or 'Right'. 'event' should include 'Foot Strike'.
#     strides_per_window : int
#         Number of strides in each non-overlapping window for NSI.
#     trim_strides : int
#         Number of strides to drop at start and end to avoid transients.
#     outlier_thresh : float
#         Discard any interval that is farther than ± outlier_thresh × SD from the mean.

#     Returns
#     -------
#     float
#         Temporal NSI for the pooled stride-time series.
#     """
#     # 1) Keep only Foot Strike rows
#     df_fs = events_df[events_df['label'] == 'Foot Strike'].copy()

#     # 2) Sorted timestamp arrays by leg
#     tL = np.sort(df_fs.loc[df_fs['context'] == 'Left',  'time'].values)
#     tR = np.sort(df_fs.loc[df_fs['context'] == 'Right', 'time'].values)

#     # 3) Compute stride intervals per side
#     strides_L = np.diff(tL)
#     strides_R = np.diff(tR)

#     # 4) Pool into one array
#     all_strides = np.concatenate([strides_L, strides_R])

#     # 5) Trim first/last trim_strides to drop belt ramp-up/down
#     if len(all_strides) > 2 * trim_strides:
#         all_strides = all_strides[trim_strides:-trim_strides]

#     # 6) Remove outliers beyond ± outlier_thresh × SD
#     mu, sigma = all_strides.mean(), all_strides.std(ddof=1)
#     mask = np.abs(all_strides - mu) <= outlier_thresh * sigma
#     all_strides = all_strides[mask]

#     # 7) NSI helper: std of non-overlapping window-means / std of full series
#     def nsi(ts: np.ndarray, k: int) -> float:
#         n = len(ts)
#         if n < k or np.std(ts, ddof=1) == 0:
#             return np.nan
#         block_means = [ts[i:i + k].mean() for i in range(0, n - k + 1, k)]
#         return np.std(block_means, ddof=1) / np.std(ts, ddof=1)

#     return nsi(all_strides, strides_per_window)









# def process_data(data: xr.DataArray,
#                  fs: float = 100.0,
#                  low_lp: float | None = 10.0,
#                  low_bp: float | None= 0.1,
#                  high_bp: float | None = 5.0,
#                  do_zscore: bool = False) -> xr.DataArray:
#     """
#     Full preprocessing for an xarray DataArray with dims ('axis','channel','time'):
#       1) Remove linear trend.
#       2) Gentle low-pass at `low_lp` Hz.
#       3) Band-pass between `low_bp`–`high_bp` Hz.
#       4) Optionally z-score.
#     Returns a new DataArray with the same coords/dims as `data`.
#     """

#     def preprocess_signal(ts: np.ndarray) -> np.ndarray:
#         # 1) Detrend
#         ts_dt = detrend(ts)

#         # 2) Gentle low-pass
#         nyq = fs / 2.0
#         b_lp, a_lp = butter(N=4, Wn=(low_lp/nyq), btype='low')
#         ts_lp = filtfilt(b_lp, a_lp, ts_dt)

#         # 3) Band-pass
#         b_bp, a_bp = butter(N=4, Wn=[low_bp/nyq, high_bp/nyq], btype='bandpass')
#         ts_bp = filtfilt(b_bp, a_bp, ts_lp)

#         # 4) Optional z-score
#         if do_zscore:
#             return (ts_bp - np.mean(ts_bp)) / np.std(ts_bp)
#         return ts_bp
#         # return ts_dt  # For now, just detrend

#     # Prepare an output array
#     processed = xr.full_like(data, np.nan)

#     for axis_val in data.coords['axis'].values:
#         for chan in data.coords['channel'].values:
#             # extract 1D timeseries
#             ts = data.sel(axis=axis_val, channel=chan).values.astype(float)

#             # if NaNs/Infs, interpolate & fill
#             if not np.isfinite(ts).all():
#                 s = pd.Series(ts)
#                 s = s.interpolate(limit_direction='both')
#                 ts = s.fillna(method='bfill').fillna(method='ffill').values

#             # run the single-step preprocess
#             try:
#                 processed_ts = preprocess_signal(ts)
#             except Exception as e:
#                 print(f"Preprocessing failed for axis={axis_val}, channel={chan}: {e}")
#                 continue

#             # assign back
#             processed.loc[dict(axis=axis_val, channel=chan)] = processed_ts

#     # preserve the time coordinate
#     processed = processed.assign_coords(time=data.coords['time'])
#     return processed

# # def is_stationary(ts, alpha=0.05, regression='c'):
# #     # KPSS null is "stationary" → p > alpha means we do NOT reject stationarity
# #     stat, p_value, _, _ = kpss(ts, regression=regression, nlags='auto')
# #     return p_value > alpha
# # import numpy as np
# # import xarray as xr
# # from scipy.signal import detrend
# # from statsmodels.tsa.stattools import kpss
# # from sklearn.preprocessing import StandardScaler

# # def preprocess_epochs(data: xr.DataArray,
# #                       fs: float,
# #                       epoch_length_s: float = 30.0,
# #                       poly_order: int = 1) -> xr.DataArray:
# #     """
# #     Splits each axis/channel timeseries into fixed-length epochs,
# #     detrends (poly of order `poly_order`), z-scores, and only keeps
# #     epochs that pass a KPSS stationarity test.
# #     Returns one concatenated DataArray with the same coords/dims.
# #     """
# #     n_samples_epoch = int(epoch_length_s * fs)
# #     processed = []

# #     for start in range(0, data.sizes['time'], n_samples_epoch):
# #         end = start + n_samples_epoch
# #         if end > data.sizes['time']:
# #             break  #—or pad if you prefer

# #         epoch = data.isel(time=slice(start, end))
# #         # Detrend each timeseries in the epoch
# #         epoch_dt = epoch.copy(deep=True).astype(float)
# #         for ax in epoch.coords['axis'].values:
# #             for ch in epoch.coords['channel'].values:
# #                 ts = epoch.sel(axis=ax, channel=ch).values
# #                 # polynomial detrend:
# #                 x = np.arange(len(ts))
# #                 coeffs = np.polyfit(x, ts, poly_order)
# #                 trend = np.polyval(coeffs, x)
# #                 ts_dt = ts - trend
# #                 epoch_dt.loc[dict(axis=ax, channel=ch)] = ts_dt

# #         # Z-score each channel across the epoch
# #         # scaler = StandardScaler()
# #         # flat = epoch_dt.values.reshape(-1, epoch_dt.sizes['time']).T
# #         # flat_z = scaler.fit_transform(flat).T
# #         # epoch_z = xr.DataArray(
# #         #     flat_z.reshape(epoch_dt.shape),
# #         #     coords=epoch_dt.coords,
# #         #     dims=epoch_dt.dims
# #         # )

# #         # Stationarity test: flattened per-channel or aggregate?
# #         # Here we test the first channel; you can extend to all.
# #         ts0 = epoch_dt.sel(axis=epoch_dt.coords['axis'][0],
# #                            channel=epoch_dt.coords['channel'][0]).values
# #         # ts0 = epoch_z.sel(axis=epoch_dt.coords['axis'][0],
# #                         #    channel=epoch_dt.coords['channel'][0]).values
# #         if is_stationary(ts0):
# #             # processed.append(epoch_z)
# #             processed.append(epoch_dt)
# #             # Optionally adjust poly_order or split further; for simplicity we drop it
# #             print(f"Epoch {start}-{end} failed stationarity; dropping it.")

# #     # Concatenate all kept epochs along time
# #     if not processed:
# #         raise RuntimeError("No epochs passed stationarity!")
# #     return xr.concat(processed, dim='time')