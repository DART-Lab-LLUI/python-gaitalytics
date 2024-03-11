from __future__ import annotations

import copy
from abc import ABC
from abc import abstractmethod

import numpy as np

import gaitalytics.model as model


class TimeNormalisationAlgorithm(ABC):

    def __init__(self, number_frames: int = 100):
        self._number_frames = number_frames
        self._data_type_fiter = {
            model.PointDataType.MARKERS,
            model.PointDataType.ANGLES,
            model.PointDataType.FORCES,
            model.PointDataType.MOMENTS,
            model.PointDataType.POWERS,
            model.PointDataType.SCALARS,
            model.PointDataType.REACTIONS,
        }

    def normalise(self, raw_data: model.ExtractedCycles) -> model.ExtractedCycles:
        norm_data = copy.deepcopy(raw_data)
        norm_data.data_condition = model.ExtractedCycleDataCondition.NORM_DATA
        norm_data.cycle_points = self._create_context_points(raw_data.cycle_points)
        return norm_data

    def _create_context_points(
        self, raw_cycles: dict[model.GaitEventContext, model.ExtractedContextCycles]
    ) -> dict[model.GaitEventContext, model.ExtractedContextCycles]:
        norm_cycles = copy.deepcopy(raw_cycles)
        for key in norm_cycles:
            raw_cycle_list = raw_cycles[key]
            norm_cycle_list = copy.deepcopy(raw_cycle_list)

            points: list[model.ExtractedCyclePoint] = []
            for point in raw_cycle_list.points:
                points.append(self._norm_point(point))
            norm_cycle_list.points = points
            norm_cycle_list.meta_data = self._norm_meta_data(raw_cycle_list.meta_data)
            norm_cycles[key] = norm_cycle_list

        return norm_cycles

    def _norm_meta_data(self, raw_meta_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        norm_meta_data = {}
        cycle_length = raw_meta_data["length"]
        for meta_key in raw_meta_data:
            if meta_key == "end_frame" or meta_key == "start_frame":
                norm_meta_data[meta_key] = raw_meta_data[meta_key]
            elif not meta_key == "length":
                norm_meta_data[meta_key] = self._define_event_frame(raw_meta_data[meta_key], cycle_length, self._number_frames)
        return norm_meta_data

    def _norm_point(self, point: model.ExtractedCyclePoint) -> model.ExtractedCyclePoint:
        shape = point.data_table.shape
        data_table = np.full((shape[0], shape[1], self._number_frames), np.nan)
        new_point = copy.deepcopy(point)
        for direction_index in range(point.data_table.shape[0]):
            for cycle_number in range(point.data_table.shape[1]):
                data_table[direction_index, cycle_number, :] = self._run_algorithm(
                    point.data_table[direction_index, cycle_number, :], self._number_frames
                )
        new_point.data_table = data_table
        return new_point

    @abstractmethod
    def _run_algorithm(self, data: np.array, number_frames: int = 100) -> np.array:
        pass

    @abstractmethod
    def _define_event_frame(self, event_frames: np.array, frame_number_cycle: np.array, number_frames: int = 100) -> int:
        pass


class LinearTimeNormalisation(TimeNormalisationAlgorithm):

    def _define_event_frame(self, event_frames: np.array, frame_number_cycle: np.array, number_frames: int = 100) -> int:
        events = event_frames / frame_number_cycle * number_frames
        return events.round()

    def _run_algorithm(self, data: np.array, number_frames: int = 100) -> np.array:
        data = np.array(data)
        data = data[np.logical_not(np.isnan(data))]
        times = np.arange(0, len(data), 1)
        times_new = np.linspace(0, len(data), num=number_frames)
        return np.interp(times_new, times, data)
