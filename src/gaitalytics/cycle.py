from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

import gaitalytics.c3d_reader as c3d_reader
import gaitalytics.events as events
import gaitalytics.model as model
import gaitalytics.utils as utils


# Cycle Builder
class CycleBuilder(ABC):

    def __init__(self, event_anomaly_checker: events.AbstractEventAnomalyChecker):
        self.eventAnomalyChecker = event_anomaly_checker

    def build_cycles(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        if file_handler.get_events_size() < 1:
            raise AttributeError("No Events in File")
        else:
            [detected, detail_tuple] = self.eventAnomalyChecker.check_events(file_handler)
            if detected:
                raise RuntimeError(detail_tuple)

        return self._build(file_handler)

    @abstractmethod
    def _build(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        pass


class EventCycleBuilder(CycleBuilder):
    def __init__(self, event_anomaly_checker: events.AbstractEventAnomalyChecker,
                 event: model.GaitEventLabel):
        super().__init__(event_anomaly_checker)
        self.event_label = event.value

    def _build(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        gait_cycles = model.GaitCycleList()
        numbers = {model.GaitEventContext.LEFT.value: 0, model.GaitEventContext.RIGHT.value: 0}
        for event_index in range(file_handler.get_events_size()):
            start_event = file_handler.get_event(event_index)
            context = start_event.context

            label = start_event.label
            if label == self.event_label:
                try:
                    [end_event, unused_events] = events.find_next_event(file_handler,
                                                                        label,
                                                                        context,
                                                                        event_index)
                    if end_event is not None:
                        numbers[context] = numbers[context] + 1
                        cycle = model.GaitCycle(
                            numbers[context], model.GaitEventContext(context), start_event.frame,
                            end_event.frame, unused_events
                        )
                        gait_cycles.add_cycle(cycle)
                except IndexError:
                    pass  # If events do not match in the end this will be raised
        return gait_cycles


class HeelStrikeToHeelStrikeCycleBuilder(EventCycleBuilder):
    def __init__(self, event_anomaly_checker: events.AbstractEventAnomalyChecker):
        super().__init__(event_anomaly_checker, model.GaitEventLabel.FOOT_STRIKE)


class ToeOffToToeOffCycleBuilder(EventCycleBuilder):
    def __init__(self, event_anomaly_checker: events.AbstractEventAnomalyChecker):
        super().__init__(event_anomaly_checker, model.GaitEventLabel.FOOT_OFF)


def _create_empty_table(n_axis: int, n_cycles: int, n_frames) -> np.ndarray:
    return np.full((n_axis, n_cycles, n_frames), np.nan)


# Cycle Extractor
def extract_point_cycles(configs: utils.ConfigProvider, cycles: model.GaitCycleList,
                         file_handler: c3d_reader.FileHandler) -> model.ExtractedCycles:
    points_left = model.ExtractedContextCycles(model.GaitEventContext.LEFT)
    points_right = model.ExtractedContextCycles(model.GaitEventContext.RIGHT)

    # loop though each point in c3d to extract each cycle
    for point_index in range(file_handler.get_points_size()):

        # init nan numpy arrays
        cycle_data_left = _create_empty_table(3, cycles.get_number_of_cycles(model.GaitEventContext.LEFT),
                                              cycles.get_longest_cycle_length(model.GaitEventContext.LEFT))

        cycle_data_right = _create_empty_table(3, cycles.get_number_of_cycles(model.GaitEventContext.RIGHT),
                                               cycles.get_longest_cycle_length(model.GaitEventContext.RIGHT))

        point = file_handler.get_point(point_index)
        for cycle in cycles.cycles:

            # extract values in cycle
            cycle_data = point.values[cycle.start_frame: cycle.end_frame, :]
            cycle_data = cycle_data.T

            # store it in the array of the context
            if cycle.context == model.GaitEventContext.LEFT:
                cycle_data_left[:, cycle.number - 1, :cycle_data.shape[1]] = cycle_data
            else:
                cycle_data_right[:, cycle.number - 1, :cycle_data.shape[1]] = cycle_data
        # split right and left cycles
        cycle_point_left = _create_cycle_point(configs,
                                               point,
                                               cycle_data_left)

        cycle_point_right = _create_cycle_point(configs,
                                                point,
                                                cycle_data_right)

        # store markers in overarching model
        points_left.add_cycle_points(cycle_point_left)
        points_right.add_cycle_points(cycle_point_right)

    meta_data = _extract_general_cycle_data(cycles, model.GaitEventContext.LEFT)
    points_left.meta_data = meta_data

    meta_data = _extract_general_cycle_data(cycles, model.GaitEventContext.RIGHT)
    points_right.meta_data = meta_data

    return model.ExtractedCycles(model.ExtractedCycleDataCondition.RAW_DATA, file_handler.get_subject_measures(),
                                 points_left,
                                 points_right)


def _extract_general_cycle_data(cycles: model.GaitCycleList, context: model.GaitEventContext) -> dict[str, np.ndarray]:
    def add_to_dict(key: str, value: int, cycle_number: int, dictionary: dict[str, np.ndarray],
                    max_cycle_length: int) -> dict[str: np.ndarray]:

        if key not in meta_data:
            dictionary[key] = np.full(max_cycle_length, np.nan)
        dictionary[key][cycle_number] = value
        return dictionary

    meta_data: dict[str, np.ndarray] = {}
    length = cycles.get_number_of_cycles(context)
    for cycle in cycles.cycles:
        if cycle.context == context:
            index = cycle.number - 1
            add_to_dict("start_frame", cycle.start_frame, index, meta_data, length)
            add_to_dict("end_frame", cycle.end_frame, index, meta_data, length)
            add_to_dict("length", cycle.length, index, meta_data, length)

            for unused_event_key in cycle.unused_events:
                add_to_dict(unused_event_key, cycle.unused_events[unused_event_key], index, meta_data, length)

    return meta_data


def _create_cycle_point(configs: utils.ConfigProvider,
                        point: model.Point,
                        cycle_data: np.ndarray) -> model.ExtractedCyclePoint:
    translated_label = configs.get_translated_label(point.label, point.type)
    cycle_point = model.ExtractedCyclePoint(translated_label, point.type)
    cycle_point.data_table = cycle_data
    return cycle_point
