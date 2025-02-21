"""This module contains classes for checking and detecting events in a trial."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

## to remove

import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.utils.mocap as mocap

FOOT_STRIKE = "Foot Strike"
FOOT_OFF = "Foot Off"
LEFT = "Left"
RIGHT = "Right"
SIDES = [LEFT, RIGHT]
EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]
TIME_COLUMN = io._EventInputFileReader.COLUMN_TIME
LABEL_COLUMN = io._EventInputFileReader.COLUMN_LABEL
CONTEXT_COLUMN = io._EventInputFileReader.COLUMN_CONTEXT
ICON_COLUMN = io._EventInputFileReader.COLUMN_ICON
key_init = "init"
key_ev_type = "event_type"


class MappedMethods(str, Enum):
    """
    Map each method to a string code¨
    """

    ZENI = "Zen"
    DESAILLY = "Des"
    AC1 = "AC1"
    AC2 = "AC2"
    AC3 = "AC3"
    AC4 = "AC4"
    AC5 = "AC5"
    AC6 = "AC6"
    GRF = "GRF"


AC_METHODS = [
    MappedMethods.AC1,
    MappedMethods.AC2,
    MappedMethods.AC3,
    MappedMethods.AC4,
    MappedMethods.AC5,
    MappedMethods.AC6,
]


class _BaseEventChecker(ABC):
    """Abstract class for event checkers.

    This class provides a common interface for checking events in a trial,
    which makes them interchangeable.
    """

    @abstractmethod
    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the events in the trial.

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the events are correct, False otherwise.
            list | None: A list of incorrect time slices,
                or None if the events are correct.
        """
        raise NotImplementedError


class SequenceEventChecker(_BaseEventChecker):
    """A class for checking the sequence of events in a trial.

    This class provides a method to check the sequence of events in a trial.
    It checks the sequence of event labels and contexts.
    """

    _TIME_COLUMN = TIME_COLUMN
    _LABEL_COLUMN = LABEL_COLUMN
    _CONTEXT_COLUMN = CONTEXT_COLUMN
    _SEQUENCE = pd.DataFrame(
        {
            "current": [
                "Foot Strike Right",
                "Foot Off Left",
                "Foot Strike Left",
                "Foot Off Right",
            ],
            "next": [
                "Foot Off Left",
                "Foot Strike Left",
                "Foot Off Right",
                "Foot Strike Right",
            ],
            "previous": [
                "Foot Off Right",
                "Foot Strike Right",
                "Foot Off Left",
                "Foot Strike Left",
            ],
        }
    ).set_index("current")

    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the sequence of events in the trial.

        In a normal gait cycle, the sequence of events is as follows:
        1. Foot Strike (right)
        2. Foot Off (left)
        3. Foot Strike (left)
        4. Foot Off (right)

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the sequence is correct, False otherwise.
            list | None: A list time slice of incorrect sequence,
            or None if the sequence is correct.

        """
        if events is None:
            raise ValueError("Trial does not have events.")

        incorrect_times = []

        incorrect_labels = self._check_labels(events)
        incorrect_contexts = self._check_contexts(events)

        if incorrect_labels:
            incorrect_times.append(incorrect_labels)
        if incorrect_contexts:
            incorrect_times.append(incorrect_contexts)

        return not bool(incorrect_times), incorrect_times if incorrect_times else None

    def _check_labels(self, events: pd.DataFrame) -> list[tuple]:
        """Check alternating sequence of event labels.

        Expected sequence of event labels:
        1. Foot Strike
        2. Foot Off
        3. Foot Strike
        4. Foot Off

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        last_label = None
        last_time = None
        for i, label in enumerate(events[self._LABEL_COLUMN]):
            time = events[self._TIME_COLUMN].iloc[i]
            if label == last_label:
                incorrect_times.append((last_time, time))

            last_time = time
            last_label = label
        return incorrect_times

    def _check_contexts(self, events: pd.DataFrame) -> list[tuple]:
        """Check sequence of contexts of events.

        Expected sequence of event contexts:
        1. Right
        2. Right
        3. Left
        4. Left

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        # Check the occurrence of the context in windows of 3 events.
        for i in range(len(events) - 3):
            max_occurance = (
                events[self._CONTEXT_COLUMN].iloc[i : i + 3].value_counts().max()
            )

            # If the context occurs more than twice in the window, it is incorrect.
            if max_occurance > 2:
                incorrect_times.append(
                    (
                        events[self._TIME_COLUMN].iloc[i],
                        events[self._TIME_COLUMN].iloc[i + 3],
                    )
                )

        return incorrect_times


class BaseEventDetection(ABC):
    _TIME_COLUMN = TIME_COLUMN
    _LABEL_COLUMN = LABEL_COLUMN
    _CONTEXT_COLUMN = CONTEXT_COLUMN
    _ICON_COLUMN = ICON_COLUMN

    """Abstract class for event detectors.

    This class provides a common interface for detecting events of a specific type (Foot Strike vs Foot Off) for one side in a trial,
    which makes them interchangeable.
    """

    def __init__(
        self,
        configs: mapping.MappingConfigs,
        context: str,
        label: str,
        offset: float = 0.0,
    ):
        """Initializes a new instance of the BaseEventDetection class for an event type on a single side.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            offset: offset by which all the events are shifted

        """
        self._configs = configs
        self._context = context
        self._label = label
        self._offset = offset

    def _create_data_frame(self, times: np.ndarray) -> pd.DataFrame:
        """Creates a DataFrame from the detected events.

        Args:
            times: The detected event times.
            context: The context of the detected events.
            label: The label of the detected events.

        Returns:
            pd.DataFrame: A DataFrame containing the detected events.
        """
        contexts = [self._context] * len(times)
        labels = [self._label] * len(times)
        icons = [1 if self._label == FOOT_STRIKE else 2] * len(times)

        table = {
            self._TIME_COLUMN: times,
            self._LABEL_COLUMN: labels,
            self._CONTEXT_COLUMN: contexts,
            self._ICON_COLUMN: icons,
        }
        events = pd.DataFrame.from_dict(table)
        return events

    def detect_events(
        self, trial: model.Trial, parameters: dict | None = None
    ) -> np.ndarray:
        """Detects the events in the trial and adds an offset if there is any

        Args:
            trial: The trial for which to detect the events.
            parameters: dictionary of event detection parameters. Default None

        Returns:
            np.ndarray: An array containing the timings of the specific event in seconds
        """
        self.set_parameters(parameters) if parameters is not None else None
        events = self._detect_events(trial)
        return self._add_offset(events, self._offset)

    def _add_offset(self, events: np.ndarray, offset: float) -> np.ndarray:
        """Add an offset to all predicted events

        Args:
            offset: offset by which all the events are shifted
        """
        return events - offset

    def set_parameters(self, parameters: dict):
        """
        Adds a dictionary parameters as object's attribute
        """
        self._parameters = parameters if not hasattr(self, "_parameters") else None

    @abstractmethod
    def _detect_events(self, trial: model.Trial) -> np.ndarray:
        """Detects the events in the trial.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the detected events.
        """
        raise NotImplementedError

    def _rotate_markers(self, points: list[xr.DataArray], trial) -> list[xr.DataArray]:
        """Rotates the 3D coordinates of the marker space, such that the progression axis is the x axis

        Args:
            points: list of xr.DataArrays containing all the marker trajectories to be rotated
            trial: The trial for which to detect the events.

        Returns:
            list[xr.DataArrays] : the same list of markers, into the rotated coordinate system
        """
        if not points:
            raise ValueError("points [list[xr.DataArray]] cannot be empty")
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        l_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.L_ANT_HIP
        )
        r_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.R_ANT_HIP
        )
        ant_hip = (l_ant_hip + r_ant_hip) / 2
        progress_axis = ant_hip - sacrum
        x_axis = xr.DataArray(
            [1, 0, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]}
        )
        angles = self._calculate_angle(progress_axis, x_axis)
        ant_hip = self._rotate_point(sacrum, sacrum, angles)
        scale = self._get_flip_scale(ant_hip - sacrum)
        for i in range(len(points)):
            point = self._rotate_point(points[i], sacrum, angles)
            points[i] = (point.T * scale).T
        return points

    @staticmethod
    def _calculate_angle(progress: xr.DataArray, axis: xr.DataArray) -> float:
        """Calculate the angle between two vectors.

        Args:
            progress: The first vector.
            axis: The second vector.

        Returns:
            float: The angle between the two vectors.
        """
        progress = progress.drop_sel(axis="z")
        axis = axis.drop_sel(axis="z")
        theta = np.arccos(
            progress.dot(axis, dim="axis")
            / (progress.meca.norm(dim="axis") * axis.meca.norm(dim="axis"))
        )
        return theta.values

    @staticmethod
    def _rotate_point(
        point: xr.DataArray, fix_point: xr.DataArray, angle: float
    ) -> xr.DataArray:
        """Rotate a point around a fixed point.

        Args:
            point: The point to rotate.
            fix_point: The fixed point.
            angle: The angle to rotate by.

        Returns:
            xr.DataArray: The rotated point.
        """
        fix = fix_point.drop_sel(axis="z")
        rel_point = point.drop_sel(axis="z")
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rel_fix = fix.to_numpy()
        np_rel_point = rel_point.to_numpy()
        two_d_point = np.empty(np_rel_point.shape)
        for i in range(np_rel_point.shape[1]):
            two_d_point[:, i] = np_rel_point[:, i] - rel_fix[:, i]
            two_d_point[:, i] = two_d_point[:, i] @ rot[:, :, i].T
            two_d_point[:, i] = two_d_point[:, i] + rel_fix[:, i]
        point = point.copy(deep=True)
        point.loc["x"] = two_d_point[0]
        point.loc["y"] = two_d_point[1]
        return point

    @staticmethod
    def _get_flip_scale(rot_progress: xr.DataArray) -> list:
        if (rot_progress.loc["x"] < 0).sum() > rot_progress.loc["x"].shape[0] / 2:
            return [-1, 1, 1]
        else:
            return [1, 1, 1]


class GrfEventDetection(BaseEventDetection):
    """
    Class for Ground Reaction Forces based event detection
    """

    _CODE = MappedMethods.GRF

    def __init__(self, configs, context, label, offset=0):
        """Initializes a new instance of the AC class.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            offset: offset by which all the events are shifted
        """
        super().__init__(configs, context, label, offset)
        self.frate = 100  # TODO Hz --> take it from c3d file.

    def processing_masked_signal(self, grf_signal):
        """
        Processes GRF signal according to these 2 processes:
            1. remove GRF activations that are too short
            2. if GRF signal lingers near 0 for too long, it is cut short
        Note: Assumes that GRF is set to Nan if not activated
        """
        nan_mask = np.isnan(grf_signal.data).astype(int)
        non_nan_groups = np.split(
            np.arange(len(grf_signal)), np.where(nan_mask[:-1] & ~nan_mask[1:])[0] + 1
        )

        # Process 1
        min_duration = 60
        for group in non_nan_groups:
            if len(group) < min_duration:
                grf_signal[group] = np.nan

        # Process 2
        nan_mask_ = np.isnan(grf_signal.data).astype(int)
        non_nan_groups_ = np.split(
            np.arange(len(grf_signal)), np.where(nan_mask_[:-1] & ~nan_mask_[1:])[0] + 1
        )

        zero_threshold = 5
        for group_ in non_nan_groups_:
            close_to_zero = np.abs(grf_signal[group_].data) < zero_threshold
            for i, j in enumerate(self._get_range(group_)):
                if close_to_zero[j] and close_to_zero[self._get_range(group_)[i + 1]]:
                    grf_signal[group_[j]] = np.nan
                elif np.isnan(grf_signal[group_].data[j]):
                    continue
                else:
                    break
        return grf_signal

    def _detect_events(self, trial):
        """Detects the events in the trial according to GRF

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the detected events.
        """
        GRF_3d = mocap.get_marker_data(
            trial,
            self._configs,
            mapping.MappedMarkers.L_GRF
            if self._context == LEFT
            else mapping.MappedMarkers.R_GRF,
        )
        GRF = GRF_3d.loc["z"]
        GRF_processed = self.processing_masked_signal(GRF)
        nan_mask = np.isnan(GRF_processed.data).astype(int)
        if self._label == FOOT_STRIKE:
            index = np.where((~nan_mask[1:]) & (nan_mask[:-1]))[0] + 1
        else:
            index = np.where((nan_mask[1:]) & (~nan_mask[:-1]))[0]
        time_ = GRF.time.data
        events = time_[index]
        return events

    def _get_range(self, group):
        """
        Utility function tahta gets a range according to the event type
        """
        if self._label == FOOT_STRIKE:
            return range(len(group) - 1)
        else:
            return range(len(group) - 1, 0, -1)


class BaseOptimisedEventDetection(BaseEventDetection, ABC):
    """Abstract class for event detectors used with a reference for optimisation.

    This class provides a common interface for detecting events in a trial with a reference,
    which makes them interchangeable.
    """

    def __init__(
        self,
        configs: mapping.MappingConfigs,
        context: str,
        label: str,
        trial_ref: None | model.Trial = None,
        offset: float = 0,
    ):
        """Initializes a new instance of the BaseOptimisedEventDetection class for an event type on a single side.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            offset: offset by which all the events are shifted
            trial_ref: Trial to be used as reference, if any is required. Otherwise None
        """
        super().__init__(configs, context, label, offset)
        self.trial_ref = trial_ref
        self.ref_events = (
            None
            if (
                trial_ref is None
                or trial_ref.events is None
                or trial_ref.events.size == 0
            )
            else self.get_event_times(trial_ref.events)
        )
        self.min_dist = self.get_min_dist()
        self.frate = 100  # TODO Hz --> take it from c3d file. How?
        # trial.get_data(model.DataCategory.MARKERS).attrs["rate"]

    def get_event_times(self, events_df: pd.DataFrame) -> np.ndarray:
        """Gets the relevant events for this instance form the complete events table of the trial

        Args:
            events_df: complete event table of a trial

        Returns:
            np.ndarray: array containing the event timings for the event and side of the instance
        """
        events = events_df.loc[
            (events_df[self._CONTEXT_COLUMN] == self._context)
            & (events_df[self._LABEL_COLUMN] == self._label),
            self._TIME_COLUMN,
        ]
        return events.to_numpy()

    def get_min_dist(self):
        """
        Gets the minimum distance in terms of time duration between two reference events of this instance
        """
        if self.ref_events is not None:
            return np.min(self.ref_events[1:] - self.ref_events[:-1])
        else:
            return 0.7  # TODO: very arbitrary value

    def _get_accuracy(self, times: np.ndarray, true_events: np.ndarray | None = None):
        """Computes the accuracy of detected events with the true reference events

        Args:
            times: array containing the detected events timings

        Returns:
            np.ndarray: array containing the errors of the detected events with the reference events
            float [0; 1]: fraction of events that have not been detected (missed events)
            float [0; 1]: fraction of events detected that do not take place in reality (excess events)
        """
        events_ref = self.ref_events if true_events is None else true_events
        rad_ = 0.5 * self.min_dist
        rad = 0.2
        in_ = np.array([])
        out_ = np.copy(times)
        missed = 0
        diff_list = np.array([])
        if events_ref is None:
            raise ValueError("Reference trial must be provided")
        for ev_ref in events_ref:
            tmp = out_[(out_ <= ev_ref + rad_) & (out_ >= ev_ref - rad_)]
            if not len(tmp):
                missed += 1
            else:
                if len(tmp) == 1:
                    ev = tmp[0]
                else:
                    ev = tmp[np.argmin(np.abs(tmp - ev_ref))]
                out_ = out_[out_ != ev]
                in_ = np.append(in_, ev)
                diff_list = np.append(diff_list, ev_ref - ev)
        if len(diff_list) == 0:
            return np.zeros(len(events_ref)), 1.0, 0
        else:
            offset = self._compute_offset(
                np.mean(diff_list), self._compute_quantiles(diff_list)
            )
            idx = np.argwhere(np.abs(diff_list - offset) <= rad)
            idx_ = np.argwhere(np.abs(diff_list - offset) > rad)
            diff_list = diff_list[idx]
            out_ = np.append(out_, in_[idx_])
            missed += len(in_[idx_])
            return (
                np.squeeze(diff_list),
                missed / len(events_ref),
                len(out_) / len(times),
            )

    def _save_performance(self, errors, missed, excess):
        """
        stores performance metrics after detecting events in reference
        """
        self._errors = errors  # array of all errors
        self._mean_error = np.mean(errors)
        self._missed = missed
        self._excess = excess
        self._quantiles = self._compute_quantiles(errors)
        self._offset = self._compute_offset(self._mean_error, self._quantiles)

    def _save_parameters(self, parameters: dict):
        """
        Saves the optimisation parameters for event detection
        """
        self._parameters = parameters

    @staticmethod
    def _compute_quantiles(errors: np.ndarray):
        """Computes the 2.5th and 97.5th percentile of an array of errors

        Args:
            errors: array of errors

        Returns:
            np.ndarray: Size 2 array containing the 2.5th and 97.5th percentile in this order
        """
        if errors.size == 0:
            return np.array([np.nan, np.nan])
        else:
            return np.array([np.percentile(errors, 2.5), np.percentile(errors, 97.5)])

    @staticmethod
    def _compute_offset(mean_error: float, quantiles: np.ndarray) -> float:
        """
        Computes the offset of this instance's performance
        """
        if quantiles[0] * quantiles[1] >= 0:
            return mean_error
        else:
            return 0

    @abstractmethod
    def optimise(self, signal: np.ndarray):
        """
        Optimizes the method parameters accroding to the reference

        Args:
            signal: signal according to time where events are detected
        """
        raise NotImplementedError


class PeakEventDetection(BaseOptimisedEventDetection, ABC):
    """Abstract class for event detectors that use peak detection of marker-based methods.

    This class provides a common interface for detecting events by finding peaks,
    which makes them interchangeable."""

    def optimise(self, signal):
        """
        Optimizes the method parameters accroding to the reference

        Args:
            signal: signal according to time where events are detected

        Returns:
            np.ndarray: indeces of each detected event after the optimization
            dict: optimization parameters for the method
        """

        def return_best(
            min_acc, acc, min_missed, missed, min_excess, excess, current_best, test
        ):
            acc_test = np.mean(acc)
            if missed < min_missed:
                return acc_test, missed, excess, test
            elif missed == min_missed:
                if excess < min_excess:
                    return acc_test, missed, excess, test
                elif excess == min_excess:
                    if np.abs(acc_test) < np.abs(min_acc):
                        return acc_test, missed, excess, test
                    else:
                        return min_acc, min_missed, min_excess, current_best
                else:
                    return min_acc, min_missed, min_excess, current_best
            else:
                return min_acc, min_missed, min_excess, current_best

        distances = (self.min_dist - np.arange(-0.2, 0.55, 0.05)) * self.frate
        prominences = np.arange(0.05, 0.75, 0.05)
        min_missed = 1.0
        min_accuracy = 100
        min_excess = 100
        best_params = {"distance": distances[0], "prominence": prominences[0]}
        i = 1
        for d in distances:
            i += 1
            for p in prominences:
                if d >= 1:
                    index, _ = sp.signal.find_peaks(-signal, distance=d, prominence=p)
                    if self.trial_ref is not None and self.trial_ref.events is not None:
                        times = (
                            self.trial_ref.get_data(model.DataCategory.MARKERS)[
                                :, :, index
                            ]
                            .coords["time"]
                            .values
                        )
                        times = times[
                            (times < self.trial_ref.events[self._TIME_COLUMN].max())
                            & (times > self.trial_ref.events[self._TIME_COLUMN].min())
                        ]
                    else:
                        raise ValueError(
                            "Reference trial must be provided, or privide a value for the parameter 'distance'."
                        )
                    acc, missed, excess = self._get_accuracy(times)
                    min_accuracy, min_missed, min_excess, best_params = return_best(
                        min_accuracy,
                        acc,
                        min_missed,
                        missed,
                        min_excess,
                        excess,
                        best_params,
                        {"distance": d, "prominence": p},
                    )
        idx, _ = sp.signal.find_peaks(
            -signal,
            distance=best_params["distance"],
            prominence=best_params["prominence"],
        )
        return idx, best_params

    def _detect_events(self, trial: model.Trial) -> np.ndarray:
        """Detects the events in the trial using peak detection.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            np.ndarray: An array containing the timings of the detected events.
        """

        if hasattr(self, "_parameters") and self._parameters is not None:
            parameters = self._parameters
            distance = parameters.get("distance")
            prominence = parameters.get("prominence")
            height = parameters.get("height")
        else:
            parameters = None
        points = self._get_relevant_channels(trial)
        method = self._get_output(points)
        method = self.normalize(method)
        if parameters is None and self.trial_ref is not None:
            index, parameters = self.optimise(method)
            self._save_parameters(parameters)
        elif parameters is None and self.trial_ref is None:
            index, _ = sp.signal.find_peaks(-method, distance=90)
        else:
            index, _ = sp.signal.find_peaks(
                -method,
                distance=distance,
                prominence=prominence,
                height=height,
            )
        time = trial.get_data(model.DataCategory.MARKERS).coords["time"].values
        times = time[index[index < time.shape]]
        return times

    def _get_heel_or_toe(self, trial: model.Trial) -> xr.DataArray:
        """
        Returns the heel or toe marker trjectory, either left or right, according to the instance
        """
        if self._context == LEFT and self._label == FOOT_STRIKE:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_HEEL
            )
        elif self._context == LEFT and self._label == FOOT_OFF:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_TOE
            )
        elif self._context == RIGHT and self._label == FOOT_STRIKE:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_HEEL
            )
        else:
            point = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_TOE
            )
        return point

    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        """
        Performs and returns min-max normalisation of an array
        """
        min_pos = np.min(signal)
        max_pos = np.max(signal)
        norm_traj = (signal - min_pos) / (max_pos - min_pos)
        return norm_traj

    @abstractmethod
    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detecion
        """
        raise NotImplementedError

    @abstractmethod
    def _get_output(self, points: dict) -> np.ndarray:
        """
        Computes the ouput marker-based method on which peak detection is performed

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: output method that allows event detection through its peaks
        """
        raise NotImplementedError


class Zeni(PeakEventDetection):
    """
    Class for marker-based event detector based on Zeni et al. (2008)
    """

    _CODE = MappedMethods.ZENI

    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel or Toe, left or right according to the type and side of this instance, and sacrum
        """
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        point = self._get_heel_or_toe(trial)
        [sacrum, point] = self._rotate_markers([sacrum, point], trial)
        points = {}
        points["sacrum"] = sacrum
        points["point"] = point
        return points

    def _get_output(self, points: dict) -> np.ndarray:
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. the distance between the heel (or toe) and the sacrum

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: distance between the heel (or toe) and the sacrum
        """
        if "point" in points.keys() and "sacrum" in points.keys():
            point = points["point"]
            sacrum = points["sacrum"]
        else:
            raise ValueError(
                f"Sacrum or {'heel' if self._label == FOOT_STRIKE else 'toe'} trajectories have not been provided"
            )
        if self._label == FOOT_OFF:
            distance = sacrum - point
        else:
            distance = point - sacrum
        distance = self.normalize(distance.sel(axis="x"))
        return -distance.to_numpy()


class Desailly(PeakEventDetection):
    """
    Class for marker-based event detector based on Desailly et al. (2009)
    """

    _CODE = MappedMethods.DESAILLY

    def _get_relevant_channels(self, trial: model.Trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel or Toe, left or right according to the type and side of this instance
        """
        point = self._get_heel_or_toe(trial)
        [point] = self._rotate_markers([point], trial)
        points = {}
        points["point"] = point
        return points

    def _get_output(self, points: dict):
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. output of the High Pass Algorithm

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: output of the High Pass Algorithm
        """
        if self.ref_events is None and not hasattr(self, "gait_freq"):
            self.gait_freq = 1  # approximation of gait frequency
        elif self.ref_events is not None and not hasattr(self, "gait_freq"):
            self.gait_freq = 1 / np.mean(self.ref_events[1:] - self.ref_events[:-1])
        if "point" in points.keys():
            point = points["point"]
        else:
            raise ValueError(
                f"{'Heel' if self._label == FOOT_STRIKE else 'Toe'} trajectory has not been provided"
            )
        point_norm = point.meca.normalize()
        filt_point = self.filt_signal(point_norm, 4, 7)
        point_hpfilt = self.sos_filt_signal(filt_point, 4, 0.5 * self.gait_freq, "high")
        if self._label == FOOT_OFF:
            return self.sos_filt_signal(point_hpfilt, 4, 1.1 * self.gait_freq, "high")[
                0, :
            ]
        else:
            return -point_hpfilt[0, :]

    def filt_signal(self, signal, order, Freq, type="low"):
        """
        Filters signals using a zero-lag butterworth filter

        Args:
            signal: array to be filtered
            order: order of the Butterworth filter
            Freq: cut-off frequency in Herz
            type: "low" for a low-pass filter and "high" for high-pass filter

        Returns:
            np.ndarray: filtered signal
        """
        b, a = sp.signal.butter(order, Freq, btype=type, fs=self.frate)
        return sp.signal.filtfilt(b, a, signal)

    def sos_filt_signal(self, signal, order, Freq, type):
        """
        Filters signals using a second-order section digital filter sos butterworth filter

        Args:
            signal: array to be filtered
            order: order of the Butterworth filter
            Freq: cut-off frequency in Herz
            type: "low" for a low-pass filter and "high" for high-pass filter

        Returns:
            np.ndarray: filtered signal
        """
        sos = sp.signal.butter(order, Freq, btype=type, output="sos", fs=self.frate)
        return sp.signal.sosfilt(sos, signal)


class AC(PeakEventDetection):
    """
    Factory class for all marker-based Auto-Correlation methods for event detection described in Fonseca et al. (2022)
    """

    _CODE = "AC"

    def __init__(self, configs, context, label, functions, trial_ref, offset=0):
        """Initializes a new instance of the AC class.

        Args:
            configs: The mapping configurations.
            context: The context of the detected events.
            label: The label of the detected events.
            functions: list of functions that compute the target values
            trial_ref: Trial to be used as reference
            offset: offset by which all the events are shifted
        """
        if trial_ref is None:
            raise TypeError("Method 'AC' works only with a reference trial")
        super().__init__(configs, context, label, trial_ref, offset)
        self.functions = functions

    @classmethod
    def get_AC1(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker and the horizontal distance between the sacrum and the heel to detect heel strikes
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC1
        return instance

    @classmethod
    def get_AC2(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker, the horizontal distance between the sacrum and the heel, and the angle of the foot to detect heel strikes
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x, cls.Foot_alpha],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC2
        return instance

    @classmethod
    def get_AC3(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the hortizontal distance between the anterior hips and the horizontal distance between the sacrum and the heel to detect heel strikes
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Hip_x, cls.Sacr_Heel_x],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC3
        return instance

    @classmethod
    def get_AC4(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the vertical componenent of the heel marker, the horizontal distance between the sacrum and the heel, the angle of the foot and the horizontal distance between the anterior hips to detect heel strikes
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Heel_z, cls.Sacr_Heel_x, cls.Foot_alpha, cls.Hip_x],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC4
        return instance

    @classmethod
    def get_AC5(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the vertical componenent of the toe marker and the horizontal distance between the sacrum and the toe to detect toe offs
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Toe_z, cls.Sacr_Toe_x],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC5
        return instance

    @classmethod
    def get_AC6(cls, configs, context, label, trial_ref, offset=0):
        """
        Initializes an instance of class AC using the vertical componenent of the toe marker, the horizontal distance between the sacrum and the toe, and the angle of the foot to detect toe offs
        """
        instance = cls(
            configs,
            context,
            label,
            functions=[cls.Toe_z, cls.Sacr_Toe_x, cls.Foot_alpha],
            trial_ref=trial_ref,
            offset=offset,
        )
        instance._CODE = MappedMethods.AC6
        return instance

    @staticmethod
    def p_function(param: np.ndarray, param_ref: np.ndarray) -> np.ndarray:
        """Parameter function estimation: for any target value, the mean of the target value of the reference trial at the times of events is substracted

        Args:
            param: any target value (eg: heel trajectory in z direction)
            param_ref: same target value as param in the reference trial at times of true events

        Returns:
            np.ndarray: Parameter function estimation
        """
        return np.abs(param - np.mean(param_ref))

    def _get_relevant_channels(self, trial) -> dict:
        """
        Returns a dictionary containing the markers (xr.DataArrays) relevant for the event detection,
        i.e. Heel, toe (left or right according to the instance), left and right anterior hips and sacrum
        """
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        l_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.L_ANT_HIP
        )
        r_ant_hip = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.R_ANT_HIP
        )
        if self._context == LEFT:
            toe = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_TOE
            )
            heel = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.L_HEEL
            )
        else:
            toe = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_TOE
            )
            heel = mocap.get_marker_data(
                trial, self._configs, mapping.MappedMarkers.R_HEEL
            )
        points = {}
        points["sacrum"] = sacrum
        points["l_ant_hip"] = l_ant_hip
        points["r_ant_hip"] = r_ant_hip
        points["toe"] = toe
        points["heel"] = heel
        return points

    def _get_output(self, points: dict):
        """Computes the ouput marker-based method on which peak detection is performed
        i.e. output of the Autocorrelation method using the instances's parameters

        Args:
            points: Dictionary containing the markers (xr.DataArray) (returned by _get_relevant_channels)

        Returns:
            np.ndarray: Autocorrelation output
        """
        points_ref = self._get_relevant_channels(self.trial_ref)
        gait_params = self._get_params(points)
        gait_params_ref = self._get_params(points_ref)
        p_ = np.zeros(len(gait_params[0]))
        for i, parameter in enumerate(gait_params_ref):
            params_ref_at_events = self._get_param_at_ref_events(parameter)
            p_ += self.p_function(gait_params[i], params_ref_at_events)
        return p_

    def _get_param_at_ref_events(self, param_ref: np.ndarray) -> np.ndarray:
        """Computes the mean of the target value obtained at the true events

        Args:
            param_ref : target value/parameter (eg. vetical component of the heel) of the reference trial

        Returns:
            np.ndarray : target values only at the times of the true events
        """
        if self.trial_ref is None or self.ref_events is None:
            raise ValueError("Reference trial should be provided")
        else:
            idx = np.isin(
                self.trial_ref.get_data(model.DataCategory.MARKERS).time.data,
                self.ref_events,
            )
            return param_ref[idx]

    def _get_params(self, points: dict) -> list[np.ndarray]:
        """Gets the target values/parameters for the autocorrelation calculation

        Args:
            points: dictionary containing the relevant markers trajectories

        Returns:
        list[np.ndarray] : list of target values/parameters
        """
        params = []
        for function in self.functions:
            params.append(function(points))
        return params

    @classmethod
    def Heel_z(cls, points: dict) -> np.ndarray:
        """
        Extracts the vertical position of the heel marker
        """
        if "heel" in points.keys():
            heel = points["heel"]
            return cls.normalize(heel.loc["z"])
        else:
            raise ValueError("Heel trajectory has not been provided")

    @classmethod
    def Toe_z(cls, points: dict) -> np.ndarray:
        """
        Extracts the vertical position of the toe marker
        """
        if "toe" in points.keys():
            toe = points["toe"]
            return cls.normalize(toe.loc["z"])
        else:
            raise ValueError("Toe trajectory has not been provided")

    @classmethod
    def Sacr_Heel_x(cls, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the heel and the sacrum
        """
        if "sacrum" in points.keys() and "heel" in points.keys():
            sacrum = points["sacrum"]
            heel = points["heel"]
            diff = np.abs(sacrum.loc["x"] - heel.loc["x"])
            return cls.normalize(diff)
        else:
            raise ValueError("Sacrum or heel trajectories have not been provided")

    @classmethod
    def Sacr_Toe_x(self, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the toe and the sacrum
        """
        if "sacrum" in points.keys() and "toe" in points.keys():
            sacrum = points["sacrum"]
            toe = points["toe"]
            diff = np.abs(sacrum.loc["x"] - toe.loc["x"])
            return self.normalize(diff)
        else:
            raise ValueError("Sacrum or toe trajectories have not been provided")

    @classmethod
    def Foot_alpha(cls, points: dict) -> np.ndarray:
        """
        Extracts the foot (defined by the heel and the toe) angle relative to the floor
        """
        if "heel" in points.keys() and "toe" in points.keys():
            heel = points["heel"]
            toe = points["toe"]
            angle = np.arctan2(
                toe.loc["z"] - heel.loc["z"],
                toe.loc["x"] - heel.loc["x"],
            )
            return cls.normalize(angle)
        else:
            raise ValueError("Heel or toe trajectories have not been provided")

    @classmethod
    def Hip_x(cls, points: dict) -> np.ndarray:
        """
        Extracts the horizontal distance between the anterior iliac spine markers
        """
        if "l_ant_hip" in points.keys() and "r_ant_hip" in points.keys():
            l_ant_hip = points["l_ant_hip"]
            r_ant_hip = points["r_ant_hip"]
            diff = np.abs(l_ant_hip.loc["x"] - r_ant_hip.loc["x"])
            return cls.normalize(diff)
        else:
            raise ValueError("Anterior hips trajectories have not been provided")


class EventDetector:
    """
    Container of Event detection methods to use for all sides and event types
    """

    def __init__(
        self,
        hs_left: BaseEventDetection,
        hs_right: BaseEventDetection,
        to_left: BaseEventDetection,
        to_right: BaseEventDetection,
    ):
        """Initializes a new instance of the EventDetector class.

        Args:
            hs_left: Event detection object class that predicts left Foot Strike
            hs_right: Event detection object class that predicts right Foot Strike
            to_left: Event detection object class that predicts left Foot Off
            to_right: Event detection object class that predicts right Foot Off
        """
        self.hs_left = hs_left
        self.hs_right = hs_right
        self.to_left = to_left
        self.to_right = to_right

    def detect_events(
        self, trial: model.Trial, parameters: dict | None = None
    ) -> pd.DataFrame:
        """Detects events for all event types and sides

        Args:
            trial: The trial for which to detect the events.
            parameters: dictionary of event detection parameters. Default None

        Returns:
            pd.DataFrame: Table containing all the detected events
        """
        hs_l_events = self.hs_left.detect_events(trial, parameters)
        hs_r_events = self.hs_right.detect_events(trial, parameters)
        to_l_events = self.to_left.detect_events(trial, parameters)
        to_r_events = self.to_right.detect_events(trial, parameters)

        events = self._create_data_frame(
            hs_l_events, hs_r_events, to_l_events, to_r_events
        )

        return events

    def _create_data_frame(
        self, hs_l_events, hs_r_events, to_l_events, to_r_events
    ) -> pd.DataFrame:
        """Creates a dataframe from the detected events.

        Args:
            hs_l_events: array of detected left foot strike timings
            hs_r_events: array of detected right foot strike timings
            to_l_events: array of detected left foot off timings
            to_r_events: array of detected right foot off timings

        Returns:
            pd.DataFrame: Complete table of detected events
        """
        hs_l_events_df = self.hs_left._create_data_frame(hs_l_events)
        hs_r_events_df = self.hs_right._create_data_frame(hs_r_events)
        to_l_events_df = self.to_left._create_data_frame(to_l_events)
        to_r_events_df = self.to_right._create_data_frame(to_r_events)

        events = pd.concat(
            [hs_l_events_df, hs_r_events_df, to_l_events_df, to_r_events_df]
        )
        events = events.sort_values(
            by=self.hs_left._TIME_COLUMN, ascending=True
        ).reset_index(drop=True)
        return events


class EventDetectorBuilder:
    """
    Mapping class to easily access Event detector classes
    """

    MAPPING: dict = {
        MappedMethods.ZENI: {
            key_init: Zeni,
            key_ev_type: [FOOT_STRIKE, FOOT_OFF],
        },
        MappedMethods.DESAILLY: {
            key_init: Desailly,
            key_ev_type: [FOOT_STRIKE, FOOT_OFF],
        },
        MappedMethods.AC1: {key_init: AC.get_AC1, key_ev_type: [FOOT_STRIKE]},
        MappedMethods.AC2: {key_init: AC.get_AC2, key_ev_type: [FOOT_STRIKE]},
        MappedMethods.AC3: {key_init: AC.get_AC3, key_ev_type: [FOOT_STRIKE]},
        MappedMethods.AC4: {key_init: AC.get_AC4, key_ev_type: [FOOT_STRIKE]},
        MappedMethods.AC5: {key_init: AC.get_AC5, key_ev_type: [FOOT_OFF]},
        MappedMethods.AC6: {key_init: AC.get_AC6, key_ev_type: [FOOT_OFF]},
        MappedMethods.GRF: {
            key_init: GrfEventDetection,
            key_ev_type: [FOOT_STRIKE, FOOT_OFF],
        },
    }

    @classmethod
    def get_method(cls, name: str):
        """
        Gets event detection method from a string code
        """
        if name not in cls.MAPPING.keys():
            raise ValueError(f"Unknown method: {name}")
        return cls.MAPPING[name][key_init]

    @classmethod
    def get_event_types(cls, name: str):
        """
        Gets detection method's list of event types from a string code
        """
        if name not in cls.MAPPING.keys():
            raise ValueError(f"Unknown method: {name}")
        return cls.MAPPING[name][key_ev_type]

    @classmethod
    def check_event_type(cls, name: str, type: str):
        """Raises ValueError if a method is used for an event type detection it cannot compute for"""
        event_types = cls.get_event_types(name)
        if type not in event_types:
            raise ValueError(f"Method '{name}' cannot be used for {type} detection")

    @classmethod
    def get_event_detector_with_ref(
        cls,
        configs: mapping.MappingConfigs,
        name_hs: str,
        name_to: str,
        trial_ref: model.Trial,
        offset: float = 0,
    ) -> EventDetector:
        """Builds an EventDetector instance with the same method predicting events of same type
           The detection of events will be performed with a reference trial
        Args:
            configs: The mapping configurations
            name_hs: Code of the method for heel strike
            name_to: Code of the method for toe off
            trial_ref: trial to be used as reference, if necessary. Otherwise None
            offset: offset by which all the events are shifted
        Returns:
            EventDetector instance initialized"""
        # Check if given methods are valid for assigned event types (eg. AC1 cannot compute for Foot Off)
        cls.check_event_type(name_hs, FOOT_STRIKE)
        cls.check_event_type(name_to, FOOT_OFF)
        method_hs = cls.get_method(name_hs)
        method_to = cls.get_method(name_to)
        # raise error if you try to use GRF method with a reference
        if name_hs == MappedMethods.GRF:
            raise TypeError(
                f"Method '{name_hs}' cannot be used with a reference to detect events"
            )
        elif name_to == MappedMethods.GRF:
            raise TypeError(
                f"Method '{name_to}' cannot be used with a reference to detect events"
            )

        return EventDetector(
            method_hs(configs, LEFT, FOOT_STRIKE, trial_ref=trial_ref, offset=offset),
            method_hs(configs, RIGHT, FOOT_STRIKE, trial_ref=trial_ref, offset=offset),
            method_to(configs, LEFT, FOOT_OFF, trial_ref=trial_ref, offset=offset),
            method_to(configs, RIGHT, FOOT_OFF, trial_ref=trial_ref, offset=offset),
        )

    @classmethod
    def get_event_detector_no_ref(
        cls,
        configs: mapping.MappingConfigs,
        name_hs: str,
        name_to: str,
        offset: float = 0,
    ) -> EventDetector:
        """Builds an EventDetector instance with the same method predicting events of same type
           The detection of events will be performed without a reference trial

        Args:
            configs: The mapping configurations
            name_hs: Code of the method for heel strike
            name_to: Code of the method for toe off
            offset: offset by which all the events are shifted
        Returns:
            EventDetector instance initialized"""
        # Check if given methods are valid for assigned event types (eg. AC1 cannot compute for Foot Off)
        cls.check_event_type(name_hs, FOOT_STRIKE)
        cls.check_event_type(name_to, FOOT_OFF)
        method_hs = cls.get_method(name_hs)
        method_to = cls.get_method(name_to)
        # Raise error if you try to use AC method without reference
        if name_hs in AC_METHODS or name_to in AC_METHODS:
            raise TypeError("Method 'AC' works only with a reference trial")
        return EventDetector(
            method_hs(configs, LEFT, FOOT_STRIKE, offset=offset),
            method_hs(configs, RIGHT, FOOT_STRIKE, offset=offset),
            method_to(configs, LEFT, FOOT_OFF, offset=offset),
            method_to(configs, RIGHT, FOOT_OFF, offset=offset),
        )

    @classmethod
    def get_mixed_event_detector(
        cls,
        configs: mapping.MappingConfigs,
        name_hs_l: str,
        name_hs_r: str,
        name_to_l: str,
        name_to_r: str,
        offset: float = 0,
        trial_ref: model.Trial | None = None,
    ) -> EventDetector:
        """Builds an EventDetector instance with different methods predicting event types

        Args:
            configs: The mapping configurations
            name_to_l: code of the method to predict left Toe Off
            name_to_r: code of the method to predict right Toe Off
            name_hs_l: code of the method to predict left Heel Strike
            name_hs_r: code of the method to predict right Heel Strike
            offset: offset by which all the events are shifted
            trial_ref: trial to be used as reference, if necessary. Otherwise None

        Returns:
            EventDetector instance initialized"""
        # Check if given methods are valid for assigned event types (eg. AC1 cannot compute for Foot Off)
        cls.check_event_type(name_hs_l, FOOT_STRIKE)
        cls.check_event_type(name_hs_r, FOOT_STRIKE)
        cls.check_event_type(name_to_l, FOOT_OFF)
        cls.check_event_type(name_to_r, FOOT_OFF)
        # Raise error if you try to use AC method without reference
        if (
            name_hs_l in AC_METHODS
            or name_hs_r in AC_METHODS
            or name_to_l in AC_METHODS
            or name_to_r in AC_METHODS
        ) and trial_ref is None:
            raise ValueError(
                "Method 'AC' works only with a reference trial. Please provide a reference trial"
            )

        method_to_l = cls.get_method(name_to_l)
        method_to_r = cls.get_method(name_to_r)
        method_hs_l = cls.get_method(name_hs_l)
        method_hs_r = cls.get_method(name_hs_r)
        return EventDetector(
            method_hs_l(configs, LEFT, FOOT_STRIKE, offset=offset, trial_ref=trial_ref)
            if name_hs_l != MappedMethods.GRF
            else method_hs_l(configs, LEFT, FOOT_STRIKE, offset=offset),
            method_hs_r(configs, RIGHT, FOOT_STRIKE, offset=offset, trial_ref=trial_ref)
            if name_hs_r != MappedMethods.GRF
            else method_hs_r(configs, RIGHT, FOOT_STRIKE, offset=offset),
            method_to_l(configs, LEFT, FOOT_OFF, offset=offset, trial_ref=trial_ref)
            if name_to_l != MappedMethods.GRF
            else method_to_l(configs, LEFT, FOOT_OFF, offset=offset),
            method_to_r(configs, RIGHT, FOOT_OFF, offset=offset, trial_ref=trial_ref)
            if name_to_r != MappedMethods.GRF
            else method_hs_l(configs, RIGHT, FOOT_OFF, offset=offset),
        )


class ReferenceFromGrf:
    """
    Class for creation of reference events using GRF data on a given trial.
    To create such reference, we select 15 gait cycles that to meet the following empirical conditions:
    1. Events are regularly spaced (not too close and not too far apart)
    2. Detected events should have a GRF value close to 0
    3. Event should be followed/preceded by a large slope
    4. Order of events is correct
    """

    _TIME_COLUMN = TIME_COLUMN
    _LABEL_COLUMN = LABEL_COLUMN
    _CONTEXT_COLUMN = CONTEXT_COLUMN
    _ICON_COLUMN = ICON_COLUMN
    _COND_1 = "cond_1"
    _COND_2 = "cond_2"
    _COND_3 = "cond_3"
    _COND_4 = "cond_4"
    _COND_TOTAL = "total"
    _REF_EVENTS = "ref"

    def __init__(
        self,
        grf_events: pd.DataFrame,
        trial: model.Trial,
        config: mapping.MappingConfigs,
        gait_cycles_ref: int = 15,
    ):
        """Initialization of an instance of the ReferenceFromGrf class

        Args:
            grf_events: events detected with the GRF-based algorithm
            trial: trial whose detcted events with GRF-algorithm will be used as reference
            gait_cycles_ref: number of gait cycles to include in reference
        """
        self.grf_events = grf_events
        self.trial = trial
        self.l_GRF = mocap.get_marker_data(trial, config, mapping.MappedMarkers.L_GRF)
        self.r_GRF = mocap.get_marker_data(trial, config, mapping.MappedMarkers.R_GRF)
        self.gait_cycles_ref = gait_cycles_ref
        self.frate = 100  # TODO: read from c3d file

    def _condition_1(self, grf_events: pd.DataFrame) -> pd.DataFrame:
        """Tests if events are regularly spaced. If an event is too close to (or too far from) another events (1st condition)

        Args:
            grf_events: table of events detected with GRF

        Returns:
            pd.DataFrame: same event table with added column specifying which event meets the condition
        """
        events_l_hs = grf_events[
            (grf_events[self._LABEL_COLUMN] == FOOT_STRIKE)
            & (grf_events[self._CONTEXT_COLUMN] == LEFT)
        ][self._TIME_COLUMN].to_numpy()
        gait_freq = np.mean(events_l_hs[1:] - events_l_hs[:-1])
        if (gait_freq > 2) or (gait_freq < 0.5):
            gait_freq = 1
        max_dist = 0.5 * gait_freq
        min_dist = 0.01 * gait_freq
        grf_events[self._COND_1] = [True] * len(grf_events)
        distances = (
            grf_events[self._TIME_COLUMN][1:].to_numpy()
            - grf_events[self._TIME_COLUMN][:-1].to_numpy()
        )
        grf_events.loc[1 : len(grf_events), self._COND_1] = (distances > min_dist) & (
            distances < max_dist
        )
        return grf_events

    def _condition_2(
        self, grf_events: pd.DataFrame, GRF_l: xr.DataArray, GRF_r: xr.DataArray
    ) -> pd.DataFrame:
        """Tests if GRF values at the time of the event is not too high (2nd condition)

        Args:
            grf_events: table of events detected with GRF
            GRF_l: Left Ground Reaction Forces
            GRF_r: Right Ground Reaction Forces
        Returns:
            pd.DataFrame: same event table with added column specifying which event meets the condition
        """
        grf_events[self._COND_2] = [False] * len(grf_events)
        for context, GRF in zip(SIDES, [GRF_l, GRF_r]):
            threshold = 0.125 * np.nanmax(GRF.loc["z"].data)  # arbitrary threshold
            events = grf_events[(grf_events[self._CONTEXT_COLUMN] == context)][
                self._TIME_COLUMN
            ].to_numpy()
            values = GRF.loc["z"][np.isin(GRF.time.data, events)].data
            idx = np.squeeze(np.argwhere(values < threshold))
            grf_events.loc[idx, self._COND_2] = [True] * len(idx)
        return grf_events

    def _condition_3(self, grf_events, GRF_l, GRF_r, frate):
        """Tests if GRF values at the time of the event is preceded or followed by a large slope (3rd condition)

        Args:
            grf_events: table of events detected with GRF

        Returns:
            pd.DataFrame: same event table with added column specifying which event meets the condition
        """
        threshold = 0.125  # arbitrary threshold
        grf_events[self._COND_3] = [True] * len(grf_events)
        window_size = 10
        for context, GRF in zip(SIDES, [GRF_l, GRF_r]):
            events_all = grf_events[(grf_events[self._CONTEXT_COLUMN] == context)]
            events = events_all[self._TIME_COLUMN]
            labels = events_all[self._LABEL_COLUMN]
            grf_windows = np.zeros((events.shape[0], window_size))
            for i, event in enumerate(events):
                edge_TO = event - (window_size - 1) / frate
                edge_HS = event + (window_size - 1) / frate
                if labels.iloc[i] == FOOT_OFF and edge_TO >= GRF.time.data[0]:
                    rge = np.linspace(
                        edge_TO,
                        event,
                        window_size,
                        endpoint=True,
                    )
                elif labels.iloc[i] == FOOT_STRIKE and edge_HS <= GRF.time.data[-1]:
                    rge = np.linspace(
                        event,
                        edge_HS,
                        window_size,
                        endpoint=True,
                    )
                else:
                    continue
                grf_windows[i] = GRF.loc["z"][
                    np.isin(GRF.time.data.astype("float32"), rge.astype("float32"))
                ].data
            derivatives = (grf_windows[:, 1:] - grf_windows[:, :-1]) * frate
            max_der = np.nanmax(derivatives, axis=0)
            mean_derivatives = np.nanmean(derivatives, axis=0)
            idx = np.squeeze(
                np.argwhere(np.abs(mean_derivatives) < threshold * max_der)
            )
            grf_events.loc[idx, self._COND_3] = [False] * len(idx)
        return grf_events

    def _condition_4(self, grf_events: pd.DataFrame) -> pd.DataFrame:
        """Tests which events follow the correct order of events (4th condition)

        Args:
            grf_events: table of events detected with GRF

        Returns:
            pd.DataFrame: same event table with added column specifying which event meets the condition
        """
        grf_events[self._COND_4] = [True] * len(grf_events)
        correct_sequence, incorrect_times = SequenceEventChecker().check_events(
            grf_events
        )
        if not correct_sequence:
            grf_events[self._COND_4] = ~grf_events[self._TIME_COLUMN].isin(
                incorrect_times
            )
        return grf_events

    def get_reference(self) -> model.Trial:
        """Checks the conditions for a trial and assigns the resulting events to the trial

        Returns:
            model.Trial: : trial with reference events from GRF data as attribute
        """
        ref_events = self._check_conditions(self.grf_events)
        trial = self._write_ref_events(self.trial, ref_events)
        return trial

    def _check_conditions(self, grf_events: pd.DataFrame) -> pd.DataFrame:
        """Check all the conditions and selects 15 consecutive gait cycles where are all conditions are met

        Args:
            grf_events: table of events detected with GRF

        Returns:
            pd.DataFrame: subset of the event table of 15 gaitcycles where conditions are met
        """
        correct_size = self.gait_cycles_ref * 4
        out_events = grf_events.copy(deep=True)
        out_events = self._condition_1(out_events)
        out_events = self._condition_2(out_events, self.l_GRF, self.r_GRF)
        out_events = self._condition_3(out_events, self.l_GRF, self.r_GRF, self.frate)
        out_events = self._condition_4(out_events)
        out_events[self._COND_TOTAL] = [True] * len(grf_events)
        out_events[self._REF_EVENTS] = [False] * len(grf_events)
        # For each event, see if every condition is met
        for cond in [self._COND_1, self._COND_2, self._COND_3, self._COND_4]:
            if cond in out_events.columns:
                out_events[self._COND_TOTAL] = out_events.apply(
                    lambda x: x[self._COND_TOTAL] * x[cond], axis=1
                )
        # select the events that will be taken as reference (number of gait_cycles defined by self.gait_cycles_ref)
        for i in range(correct_size, len(grf_events)):
            ref = (
                out_events.loc[i - correct_size : (i - 1), self._COND_TOTAL].sum()
                == correct_size
            )
            if ref:
                out_events.loc[i - correct_size : (i - 1), self._REF_EVENTS] = [
                    ref
                ] * correct_size
                break
        return grf_events[out_events[self._REF_EVENTS]]

    def _write_ref_events(self, trial: model.Trial, grf_events: pd.DataFrame):
        """Assigns an event table to a model.Trial object"""
        trial.events = grf_events
        return trial


class AutoEventDetection:
    """
    Class for the automatisation of marker-based event detection
    """

    def __init__(
        self,
        configs,
        trial_ref,
        method_list=[
            Zeni,
            Desailly,
            AC.get_AC1,
            AC.get_AC2,
            AC.get_AC3,
            AC.get_AC4,
            AC.get_AC5,
            AC.get_AC6,
        ],
    ):
        """Initializes a new instance of the AutoEventDetection class.

        Args:
            configs: The mapping configurations.
            trial_ref: Trial to be used as reference
            method_list: list of methods (classes) to attempt detection of events on the reference
        """
        self._configs = configs
        self.trial_ref = trial_ref
        self.method_list = method_list

    def get_optimised_event_detectors(self) -> tuple[EventDetector, dict]:
        """Performs event detection using all the specified methods on the reference trial, and selects the best performing one
        Also returns feedback for user

        Returns:
            EventDetector: instance of EventDetector class containing the optimized event detectors for each event type and side
            user_show : dict containing the performance of all selected methods, as well as the parameters used to find the events
        """
        opt_detectors: dict[str, dict] = {FOOT_STRIKE: {}, FOOT_OFF: {}}
        for label in EVENT_TYPES:
            for context in SIDES:
                opt = np.zeros(len(self.method_list))
                event_detectors = []
                for idx, method in enumerate(self.method_list):
                    event_detector = method(
                        self._configs, context, label, trial_ref=self.trial_ref
                    )  ##an instance for each side
                    if label in EventDetectorBuilder.get_event_types(
                        event_detector._CODE
                    ):
                        times = event_detector._detect_events(self.trial_ref)
                        times = times[
                            (
                                times
                                < self.trial_ref.events[
                                    event_detector._TIME_COLUMN
                                ].max()
                            )
                            & (
                                times
                                > self.trial_ref.events[
                                    event_detector._TIME_COLUMN
                                ].min()
                            )
                        ]
                        errors, missed, excess = event_detector._get_accuracy(times)
                        event_detector._save_performance(errors, missed, excess)
                        opt[idx] = self._optim_function(event_detector)
                    event_detectors.append(event_detector)
                index = np.argmax(opt)
                opt_detectors[label][context] = event_detectors[index]
        event_detector = EventDetector(
            opt_detectors[FOOT_STRIKE][LEFT],
            opt_detectors[FOOT_STRIKE][RIGHT],
            opt_detectors[FOOT_OFF][LEFT],
            opt_detectors[FOOT_OFF][RIGHT],
        )
        user_show: dict = self.create_user_indicatation(opt_detectors)
        return event_detector, user_show

    @staticmethod
    def create_user_indicatation(opt_detectors):
        """Creates a dictionary containing the performance and detection parameters of each selected methods
        for user feedback

        Args:
            opt_detectors: dictionary containing the BaseOptimisedEventDetection object slected for each event types and sides

        Returns:
            dictionary with performance and parameters information for each method selected
        """
        user_show: dict = {}
        for label in EVENT_TYPES:
            user_show[label] = {}
            for context in SIDES:
                detector = opt_detectors[label][context]
                user_show[label][context] = {
                    "method": detector._CODE,
                    "mean error": detector._mean_error,
                    "missed": detector._missed,
                    "excess": detector._excess,
                    "quantiles": detector._quantiles,
                    "parameters": detector._parameters,
                }
        return user_show

    @staticmethod
    def _optim_function(detector: BaseOptimisedEventDetection) -> float:
        """Computes the result of the optimisation function for a specific event detector

        Args:
            detector: Event detector whose performance on the reference is being evaluated

        Returns:
            float: result of the optimisation function
        """
        shifted_accuracy = detector._mean_error - detector._offset
        quantiles = detector._quantiles
        width = np.abs(quantiles[1] - quantiles[0])
        return (
            0.6 * (1 - np.abs(shifted_accuracy))
            + 0.4 * (1 - width)
            + 0.5 * (1 - detector._missed)
            + 0.5 * (1 - detector._excess)
        )
