from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

import gaitalytics.events as events
import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.utils.linalg as linalg
import gaitalytics.utils.mocap as mocap


class FeatureCalculation(ABC):
    """Base class for feature calculations.

    This class provides a common interface for calculating features.
    """

    def __init__(self, config: mapping.MappingConfigs):
        """Initializes a new instance of the BaseFeatureCalculation class.

        Args:
            config: The mapping configuration to use for the feature calculation.
        """
        self._config = config

    @abstractmethod
    def calculate(self, trial: model.TrialCycles) -> xr.DataArray:
        """Calculate the features for a trial.

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        raise NotImplementedError


class _CycleFeaturesCalculation(FeatureCalculation, ABC):
    def calculate(self, trial: model.TrialCycles) -> xr.DataArray:
        """Calculate the features for a trial.

        Calls the _calculate method for each cycle in the trial and combines
        results into a single DataArray.

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        results: list = []

        context_dim: list[str] = []
        for context, context_cycles in trial.get_all_cycles().items():
            context_results: list = []
            cycle_dim: list[int] = []
            for cycle_id, cycle in context_cycles.items():
                feature = self._calculate(cycle)
                context_results.append(feature)
                cycle_dim.append(cycle_id)

            context_dim.append(context)
            context_results = xr.concat(
                context_results, pd.Index(cycle_dim, name="cycle")
            )
            results.append(context_results)

        result = xr.concat(results, pd.Index(context_dim, name="context"))
        return result

    @abstractmethod
    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the features for a trial.

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        raise NotImplementedError

    @staticmethod
    def get_event_times(
        trial_events: pd.DataFrame | None,
    ) -> tuple[float, float, float, float, float]:
        """Checks the sequence of events in the trial and returns the times.

        Args:
            trial_events: The events to be checked and extracted.

        Returns:
            The times of the events. in following order
            [contra_fo, contra_fs, ipsi_fo, end_time]

        Raises:
            ValueError: If the sequence of events is incorrect.
        """
        if trial_events is None:
            raise ValueError("Trial does not have events.")
        curren_context = trial_events.attrs["context"]
        cycle_id = trial_events.attrs["cycle_id"]

        if len(trial_events) < 3:
            raise ValueError(
                f"Missing events in segment {curren_context} nr. {cycle_id}"
            )
        ipsi = trial_events[
            trial_events[io._EventInputFileReader.COLUMN_CONTEXT] == curren_context
        ]
        contra = trial_events[
            trial_events[io._EventInputFileReader.COLUMN_CONTEXT] != curren_context
        ]
        if len(ipsi) != 3:
            raise ValueError(f"Error events sequence {curren_context} nr. {cycle_id}")
        if len(contra) != 2:
            raise ValueError(f"Error events sequence {curren_context} nr. {cycle_id}")

        contra_fs = contra[
            contra[io._EventInputFileReader.COLUMN_LABEL] == events.FOOT_STRIKE
        ]
        contra_fo = contra[
            contra[io._EventInputFileReader.COLUMN_LABEL] == events.FOOT_OFF
        ]
        ipsi_fs = ipsi[
            ipsi[io._EventInputFileReader.COLUMN_LABEL] == events.FOOT_STRIKE
        ]
        ipsi_fo = ipsi[ipsi[io._EventInputFileReader.COLUMN_LABEL] == events.FOOT_OFF]

        ipsi_fs_time_start = ipsi_fs[io._EventInputFileReader.COLUMN_TIME].values[0]
        ipsi_fs_time_end = ipsi_fs[io._EventInputFileReader.COLUMN_TIME].values[1]
        ipsi_fo_time = ipsi_fo[io._EventInputFileReader.COLUMN_TIME].values[0]
        contra_fs_time = contra_fs[io._EventInputFileReader.COLUMN_TIME].values[0]
        contra_fo_time = contra_fo[io._EventInputFileReader.COLUMN_TIME].values[0]

        return (
            ipsi_fs_time_start,
            contra_fo_time,
            contra_fs_time,
            ipsi_fo_time,
            ipsi_fs_time_end,
        )

    @staticmethod
    def _create_result_from_dict(result_dict: dict) -> xr.DataArray:
        """Create a xarray DataArray from a dictionary.

        The dictionary should follow the format {feature: value}.
        In example:
        {
            "min": 1.0,
            "max": 2.0,
            "mean": 1.5,
            "median": 1.5,
            "std": 0.5,
        }

        Args:
            result_dict: The dictionary to create the DataArray from.

        Returns:
            An xarray DataArray containing the data from the dictionary.
        """
        xr_dict = {
            "coords": {
                "feature": {"dims": "feature", "data": list(result_dict.keys())}
            },
            "data": list(result_dict.values()),
            "dims": "feature",
        }
        return xr.DataArray.from_dict(xr_dict)

    @staticmethod
    def _flatten_features(features: xr.DataArray) -> xr.DataArray:
        """Flatten the features into a single dimension.

        The dimension channel will be integrated into the features dim

        Args:
            features: The features to be flattened.

        Returns:
            The reshaped features.
        """

        np_data = features.to_numpy()
        rs_data = np_data.reshape((np_data.shape[1] * np_data.shape[0]), order="C")

        feature = features.coords["feature"].values
        channel = features.coords["channel"].values
        new_labels = []
        for f in feature:
            for c in channel:
                new_labels.append(f"{c}_{f}")

        new_format = xr.DataArray(
            rs_data,
            coords={"feature": new_labels},
            dims=["feature"],
        )
        return new_format


class _PointDependentFeature(_CycleFeaturesCalculation, ABC):
    def _get_marker_data(
        self, trial: model.Trial, marker: mapping.MappedMarkers
    ) -> xr.DataArray:
        """Get the marker data for a trial.

        Args:
            trial: The trial for which to get the marker data.
            marker: The marker to get the data for.

        Returns:
            An xarray DataArray containing the marker data.
        """
        return mocap.get_marker_data(trial, self._config, marker)

    def _get_sacrum_marker(self, trial: model.Trial) -> xr.DataArray:
        """Get the sacrum marker data for a trial.

        Try to get the sacrum marker data from the trial.
        If the sacrum marker not found calculate from posterior hip markers

        Args:
            trial: The trial for which to get the marker data.

        Returns:
            An xarray DataArray containing the sacrum marker data.
        """
        return mocap.get_sacrum_marker(trial, self._config)

    def _get_progression_vector(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the progression vector for a trial.

        The progression vector is the vector from the sacrum to the anterior hip marker.

        Args:
            trial: The trial for which to calculate the progression vector.

        Returns:
            An xarray DataArray containing the calculated progression vector.
        """
        return mocap.get_progression_vector(trial, self._config)


class TimeSeriesFeatures(_CycleFeaturesCalculation):
    """Calculate time series features for a trial.

    This class calculates following time series features for a trial.
    - min
    - max
    - mean
    - median
    - std
    """

    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the time series features for a trial.

        Following features are calculated:
            - min
            - max
            - mean
            - median
            - std
            - amplitude

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        features = self._calculate_features(trial)
        return self._flatten_features(features)

    @staticmethod
    def _calculate_features(trial: model.Trial) -> xr.DataArray:
        """Calculate the time series features for a trial.

        Following features are calculated:
            - min
            - max
            - mean
            - median
            - std
            - amplitude


        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        markers = trial.get_data(model.DataCategory.ANALYSIS)
        min_feat = markers.min(dim="time", skipna=True)
        max_feat = markers.max(dim="time", skipna=True)
        mean_feat = markers.mean(dim="time", skipna=True)
        median_feat = markers.median(dim="time", skipna=True)
        std_feat = markers.std(dim="time", skipna=True)
        amplitude_feat = max_feat - min_feat
        features = xr.concat(
            [min_feat, max_feat, mean_feat, median_feat, std_feat, amplitude_feat],
            pd.Index(
                ["min", "max", "mean", "median", "std", "amplitude"], name="feature"
            ),
        )

        return features


class PhaseTimeSeriesFeatures(TimeSeriesFeatures):
    """Calculate phase time series features for a trial.

    This class calculates following phase time series features for a trial.
        - stand_min
        - stand_max
        - stand_mean
        - stand_median
        - stand_std
        - stand_amplitude
        - swing_min
        - swing_max
        - swing_mean
        - swing_median
        - swing_std
        - swing_amplitude
    """

    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the time series features for a trial by phase.

        Following features are calculated:
            - stand_min
            - stand_max
            - stand_mean
            - stand_median
            - stand_std
            - stand_amplitude
            - swing_min
            - swing_max
            - swing_mean
            - swing_median
            - swing_std
            - swing_amplitude

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        event_table = trial.events
        analysis_data = trial.get_data(model.DataCategory.ANALYSIS)

        context = analysis_data.attrs["context"]

        event_ipsi = event_table.loc[
            event_table[io.C3dEventInputFileReader.COLUMN_CONTEXT] == context  # type: ignore
        ]
        event_ipsi_fo = event_ipsi.loc[
            event_table[io.C3dEventInputFileReader.COLUMN_LABEL] == events.FOOT_OFF  # type: ignore
        ]
        fo_time = round(event_ipsi_fo.time.values[0], 4)
        start_time = analysis_data.coords["time"].values[0]
        end_time = analysis_data.coords["time"].values[-1]

        stand_trial = self._create_phase_trial(trial, slice(start_time, fo_time))
        swing_trial = self._create_phase_trial(trial, slice(fo_time, end_time))
        stand_features = super()._calculate(stand_trial)
        stand_features.assign_coords(
            feature=[f"{f}_swing" for f in stand_features.feature.values]
        )
        swing_features = super()._calculate(swing_trial)
        swing_features.assign_coords(
            feature=[f"{f}_swing" for f in swing_features.feature.values]
        )

        return xr.concat([stand_features, swing_features], dim="feature")

    @staticmethod
    def _create_phase_trial(trial: model.Trial, time_slice: slice):
        """Create a trial containing only the data for a specific phase.

        Args:
            trial: The trial to create the phase trial from.
            time_slice: The time slice to extract the phase data from.
        """
        phase_trial = model.Trial()
        for data_category, data in trial.get_all_data().items():
            phase_trial.add_data(data_category, data.sel(time=time_slice))
        return phase_trial


class TemporalFeatures(_CycleFeaturesCalculation):
    """Calculate temporal features for a trial.

    This class calculates following temporal features for a trial.
        - double_support
        - single_support
        - foot_off
        - opposite_foot_off
        - opposite_foot_contact
        - stride_time
        - step_time
        - cadence
    """

    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the support times for a trial.

        Definitions of the temporal features
        Hollmann et al. 2011 (doi: 10.1016/j.gaitpost.2011.03.024)

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.

        Raises:
            ValueError: If the sequence of events is incorrect.
        """
        trial_events = trial.events
        if trial_events is None:
            raise ValueError("Trial does not have events.")

        rel_event_times = self.get_event_times(trial_events)

        result_dict = self._calculate_supports(
            rel_event_times[1],
            rel_event_times[2],
            rel_event_times[3],
            rel_event_times[4],
        )
        result_dict["foot_off"] = rel_event_times[3] / rel_event_times[4]
        result_dict["opposite_foot_off"] = rel_event_times[1] / rel_event_times[4]
        result_dict["opposite_foot_contact"] = rel_event_times[2] / rel_event_times[4]
        result_dict["stride_time"] = rel_event_times[4]
        result_dict["step_time"] = rel_event_times[4] - rel_event_times[2]
        result_dict["cadence"] = 60 / (rel_event_times[4] / 2)

        return self._create_result_from_dict(result_dict)

    @staticmethod
    def _calculate_supports(
        contra_fo_time: float,
        contra_fs_time: float,
        ipsi_fo_time: float,
        end_time: float,
    ) -> dict[str, float]:
        """Calculate the support times for a trial.

        Args:
            contra_fo_time: The time of the contra foot off event.
            contra_fs_time: The time of the contra foot strike event.
            ipsi_fo_time: The time of the ipsi foot off event.
            end_time: The end time of the trial.

        Returns:
            The calculated support times.
        """
        double_support = (contra_fo_time + (ipsi_fo_time - contra_fs_time)) / end_time
        single_support = (contra_fs_time - contra_fo_time) / end_time
        return {"double_support": double_support, "single_support": single_support}


class SpatialFeatures(_PointDependentFeature):
    """Calculate spatial features for a trial.

    This class calculates following spatial features for a trial.
    - step_length
    - step_width
    """

    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the spatial features for a trial.

        Definitions of the spatial features
        Hollmann et al. 2011 (doi: 10.1016/j.gaitpost.2011.03.024)

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.

        Raises:
            ValueError: If the trial does not have events.
        """

        if trial.events is None:
            raise ValueError("Trial does not have events.")

        if trial.events.attrs["context"] == "Right":
            ipsi_marker = mapping.MappedMarkers.R_TOE
            contra_marker = mapping.MappedMarkers.L_TOE
        else:
            ipsi_marker = mapping.MappedMarkers.R_TOE
            contra_marker = mapping.MappedMarkers.L_TOE

        results_dict = self._calculate_step_length(trial, ipsi_marker, contra_marker)
        results_dict.update(
            self._calculate_step_width(trial, ipsi_marker, contra_marker)
        )
        return self._create_result_from_dict(results_dict)

    def _calculate_step_length(
        self,
        trial: model.Trial,
        ipsi_marker: mapping.MappedMarkers,
        contra_marker: mapping.MappedMarkers,
    ) -> dict[str, np.ndarray]:
        """Calculate the step length for a trial.

        Args:
            trial: The trial for which to calculate the step length.
            ipsi_marker: The ipsi-lateral heel marker.
            contra_marker: The contra-lateral heel marker.


        Returns:
            The calculated step length.
        """

        event_times = self.get_event_times(trial.events)

        ipsi_heel = self._get_marker_data(trial, ipsi_marker).sel(
            time=event_times[-1], method="nearest"
        )
        contra_heel = self._get_marker_data(trial, contra_marker).sel(
            time=event_times[-1], method="nearest"
        )
        progress_axis = self._get_progression_vector(trial)
        progress_axis = linalg.normalize_vector(progress_axis)
        projected_ipsi = linalg.project_point_on_vector(ipsi_heel, progress_axis)
        projected_contra = linalg.project_point_on_vector(contra_heel, progress_axis)
        distance = linalg.calculate_distance(projected_ipsi, projected_contra).values
        return {"step_length": distance}

    def _calculate_step_width(
        self,
        trial: model.Trial,
        ipsi_marker: mapping.MappedMarkers,
        contra_marker: mapping.MappedMarkers,
    ) -> dict[str, np.ndarray]:
        """Calculate the step width for a trial.

        Args:
            trial: The trial for which to calculate the step width.
            ipsi_marker: The ipsi-lateral heel marker.
            contra_marker: The contra-lateral heel marker.

        Returns:
            The calculated step width in a dict.
        """

        event_times = self.get_event_times(trial.events)
        contra_heel = self._get_marker_data(trial, contra_marker)
        contra_vector = contra_heel.sel(
            time=event_times[2], method="nearest"
        ) - contra_heel.sel(time=event_times[0], method="nearest")

        ipsi_heel = self._get_marker_data(trial, ipsi_marker).sel(
            time=event_times[-1], method="nearest"
        )

        norm_vector = linalg.normalize_vector(contra_vector)
        projected_ipsi = linalg.project_point_on_vector(ipsi_heel, norm_vector)
        distance = linalg.calculate_distance(ipsi_heel, projected_ipsi).values

        return {"step_width": distance}
