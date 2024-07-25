"""This module contains classes for checking and detecting events in a trial."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.utils.mocap as mocap

FOOT_STRIKE = "Foot Strike"
FOOT_OFF = "Foot Off"


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

    _TIME_COLUMN = io._EventInputFileReader.COLUMN_TIME
    _LABEL_COLUMN = io._EventInputFileReader.COLUMN_LABEL
    _CONTEXT_COLUMN = io._EventInputFileReader.COLUMN_CONTEXT

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


class _BaseEventDetection(ABC):
    """Abstract class for event detectors.

    This class provides a common interface for detecting events in a trial,
    which makes them interchangeable.
    """

    def __init__(self, configs: mapping.MappingConfigs):
        """Initializes a new instance of the BaseEventDetection class.

        Args:
            configs: The mapping configurations.
        """
        self._configs = configs

    @abstractmethod
    def detect_events(self, trial: model.Trial) -> pd.DataFrame:
        """Detects the events in the trial.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            pd.DataFrame: A DataFrame containing the detected events.
        """
        raise NotImplementedError


class MarkerEventDetection(_BaseEventDetection):
    """A class for detecting events using marker data.

    This class provides a method to detect events using marker data in a trial.
    The algorithm is based on the paper by Zeni et al. (2008).
    """

    _TIME_COLUMN = io._EventInputFileReader.COLUMN_TIME
    _LABEL_COLUMN = io._EventInputFileReader.COLUMN_LABEL
    _CONTEXT_COLUMN = io._EventInputFileReader.COLUMN_CONTEXT
    _ICON_COLUMN = io._EventInputFileReader.COLUMN_ICON

    def __init__(self, configs: mapping.MappingConfigs, **kwargs):
        """Initializes a new instance of the MarkerEventDetection class.

        Args:
            configs: The mapping configurations.
            height: The height of peaks for events. Default = None
            threshold: The threshold for detecting events. Default = None
            distance: The min distance in frames between events. Default = None
            rel_height: The relative height of peak for events. Default = 0.5
        """
        self._height = kwargs.get("height", None)
        self._threshold = kwargs.get("threshold", None)
        self._distance = kwargs.get("distance", None)
        self._rel_height = kwargs.get("rel_height", 0.5)
        super().__init__(configs)

    def detect_events(self, trial: model.Trial) -> pd.DataFrame:
        """Detects the events in the trial using marker data.

        Args:
            trial: The trial for which to detect the events.

        Returns:
            pd.DataFrame: A DataFrame containing the detected events.
        """
        scarum = mocap.get_sacrum_marker(trial, self._configs)
        l_heel = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.L_HEEL
        )
        r_heel = mocap.get_marker_data(
            trial, self._configs, mapping.MappedMarkers.R_HEEL
        )
        l_toe = mocap.get_marker_data(trial, self._configs, mapping.MappedMarkers.L_TOE)
        r_toe = mocap.get_marker_data(trial, self._configs, mapping.MappedMarkers.R_TOE)
        sacrum = mocap.get_sacrum_marker(trial, self._configs)
        l_heel, l_toe, r_heel, r_toe = self._rotate_markers(
            l_heel, l_toe, r_heel, r_toe, sacrum, scarum, trial
        )

        l_hs_times = self._detect_events(scarum, l_heel, False)
        r_hs_times = self._detect_events(scarum, r_heel, False)
        l_to_times = self._detect_events(scarum, l_toe, True)
        r_to_times = self._detect_events(scarum, r_toe, True)

        l_hs_events = self._create_data_frame(l_hs_times, "Left", FOOT_STRIKE)
        r_hs_events = self._create_data_frame(r_hs_times, "Right", FOOT_STRIKE)
        l_to_events = self._create_data_frame(l_to_times, "Left", FOOT_OFF)
        r_to_events = self._create_data_frame(r_to_times, "Right", FOOT_OFF)

        events = pd.concat([l_hs_events, r_hs_events, l_to_events, r_to_events])
        events = events.sort_values(by=self._TIME_COLUMN, ascending=True).reset_index(
            drop=True
        )
        return events

    def _rotate_markers(self, l_heel, l_toe, r_heel, r_toe, sacrum, scarum, trial):
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
        l_heel = self._rotate_point(l_heel, scarum, angles)
        r_heel = self._rotate_point(r_heel, scarum, angles)
        l_toe = self._rotate_point(l_toe, scarum, angles)
        r_toe = self._rotate_point(r_toe, scarum, angles)
        ant_hip = self._rotate_point(ant_hip, scarum, angles)

        scale = self._get_flip_scale(ant_hip - sacrum)
        l_heel = (l_heel.T * scale).T
        r_heel = (r_heel.T * scale).T
        l_toe = (l_toe.T * scale).T
        r_toe = (r_toe.T * scale).T

        return l_heel, l_toe, r_heel, r_toe

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

    def _detect_events(
        self, sacrum: xr.DataArray, point: xr.DataArray, toe_off: bool
    ) -> np.ndarray:
        """Detects the events in the trial using projected points.
        Args:
            sacrum: The projected sacrum point.
            point: The projected point to detect the events for.
            toe_off: True if the event is toe off, False otherwise.

        Returns:
            np.ndarray: The detected event times.
        """
        if toe_off:
            distance = sacrum - point
        else:
            distance = point - sacrum
        distance = distance.sel(axis="x").meca.normalize().meca.center()
        index, heights = sp.signal.find_peaks(
            distance.to_numpy(),
            height=self._height,
            threshold=self._threshold,
            distance=self._distance,
            rel_height=self._rel_height,
        )

        # take smaller peaks if toe_off, larger peaks if heel_strike

        times = point[:, index].coords["time"].values

        return times

    def _create_data_frame(
        self, times: np.ndarray, context: str, label: str
    ) -> pd.DataFrame:
        """Creates a DataFrame from the detected events.


        Args:
            times: The detected event times.
            context: The context of the detected events.
            label: The label of the detected events.

        Returns:
            pd.DataFrame: A DataFrame containing the detected events.
        """
        contexts = [context] * len(times)
        labels = [label] * len(times)
        icons = [1 if label == FOOT_STRIKE else 2] * len(times)

        table = {
            self._TIME_COLUMN: times,
            self._LABEL_COLUMN: labels,
            self._CONTEXT_COLUMN: contexts,
            self._ICON_COLUMN: icons,
        }
        events = pd.DataFrame.from_dict(table)
        return events
