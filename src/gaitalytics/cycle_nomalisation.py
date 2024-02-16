from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

import gaitalytics.utils


class TimeNormalisationAlgorithm(ABC):

    def __init__(self, number_frames: int = 100):
        self._number_frames = number_frames
        self._data_type_fiter = {
            gaitalytics.utils.PointDataType.MARKERS,
            gaitalytics.utils.PointDataType.ANGLES,
            gaitalytics.utils.PointDataType.FORCES,
            gaitalytics.utils.PointDataType.MOMENTS,
            gaitalytics.utils.PointDataType.POWERS,
            gaitalytics.utils.PointDataType.SCALARS,
            gaitalytics.utils.PointDataType.REACTIONS,
        }

    def normalise(self, r_data_list: dict[str, gaitalytics.utils.BasicCyclePoint]) -> dict[str, gaitalytics.utils.BasicCyclePoint]:
        n_data_list = {}
        for data_key in r_data_list:
            r_cycle_point = r_data_list[data_key]
            if r_cycle_point.data_type in self._data_type_fiter:
                n_cycle_point = gaitalytics.utils.TestCyclePoint(
                    len(r_cycle_point.data_table), self._number_frames, gaitalytics.utils.BasicCyclePoint.TYPE_NORM
                )
                n_cycle_point.cycle_point_type = gaitalytics.utils.BasicCyclePoint.TYPE_NORM
                n_cycle_point.translated_label = r_cycle_point.translated_label
                n_cycle_point.direction = r_cycle_point.direction
                n_cycle_point.data_type = r_cycle_point.data_type
                n_cycle_point.context = r_cycle_point.context
                n_cycle_point.subject = r_cycle_point.subject
                n_cycle_point.frames = r_cycle_point.frames

                for cycle_key in r_cycle_point.data_table.index.to_list():
                    cycle_data = r_cycle_point.data_table.loc[cycle_key].to_list()

                    interpolated_data = self._run_algorithm(cycle_data, self._number_frames)
                    n_cycle_point.data_table.loc[cycle_key] = interpolated_data
                    events = self._define_event_frame(
                        np.array(r_cycle_point.event_frames.loc[cycle_key].to_list()), len(cycle_data), self._number_frames
                    )
                    n_cycle_point.event_frames.loc[cycle_key] = events
                n_data_list[data_key] = n_cycle_point
        return n_data_list

    @abstractmethod
    def _run_algorithm(self, data: np.array, number_frames: int = 100) -> np.array:
        pass

    @abstractmethod
    def _define_event_frame(self, event_frames: np.array, frame_number_cycle: int, number_frames: int = 100) -> int:
        pass


class LinearTimeNormalisation(TimeNormalisationAlgorithm):

    def _define_event_frame(self, event_frames: DataFrame, frame_number_cycle: int, number_frames: int = 100) -> int:
        events = event_frames / frame_number_cycle * number_frames
        return events.round()

    def _run_algorithm(self, data: np.array, number_frames: int = 100) -> np.array:
        data = np.array(data)
        data = data[np.logical_not(np.isnan(data))]
        times = np.arange(0, len(data), 1)
        times_new = np.linspace(0, len(data), num=number_frames)
        return np.interp(times_new, times, data)
