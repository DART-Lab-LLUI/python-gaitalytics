from __future__ import annotations

import os
import re
from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np

import gaitalytics.events
import gaitalytics.c3d_reader
import gaitalytics.utils


# Cycle Builder
class CycleBuilder(ABC):

    def __init__(self, event_anomaly_checker: gaitalytics.events.AbstractEventAnomalyChecker):
        self.eventAnomalyChecker = event_anomaly_checker

    def build_cycles(self, file_handler: gaitalytics.c3d_reader.FileHandler) -> gaitalytics.utils.GaitCycleList:
        if file_handler.get_events_size() < 1:
            raise AttributeError("No Events in File")
        else:
            [detected, detail_tuple] = self.eventAnomalyChecker.check_events(file_handler)
            if detected:
                raise RuntimeError(detail_tuple)

        return self._build(file_handler)

    @abstractmethod
    def _build(self, file_handler: gaitalytics.c3d_reader.FileHandler) -> gaitalytics.utils.GaitCycleList:
        pass


class EventCycleBuilder(CycleBuilder):
    def __init__(self, event_anomaly_checker: gaitalytics.events.AbstractEventAnomalyChecker, event: gaitalytics.utils.GaitEventLabel):
        super().__init__(event_anomaly_checker)
        self.event_label = event.value

    def _build(self, file_handler: gaitalytics.c3d_reader.FileHandler) -> gaitalytics.utils.GaitCycleList:
        gait_cycles = gaitalytics.utils.GaitCycleList()
        numbers = {gaitalytics.utils.GaitEventContext.LEFT.value: 0, gaitalytics.utils.GaitEventContext.RIGHT.value: 0}
        for event_index in range(file_handler.get_events_size()):
            start_event = file_handler.get_event(event_index)
            context = start_event.context

            label = start_event.label
            if label == self.event_label:
                try:
                    [end_event, unused_events] = gaitalytics.events.find_next_event(file_handler, label, context, event_index)
                    if end_event is not None:
                        numbers[context] = numbers[context] + 1
                        cycle = gaitalytics.utils.GaitCycle(
                            numbers[context], gaitalytics.utils.GaitEventContext(context), start_event.frame, end_event.frame, unused_events
                        )
                        gait_cycles.add_cycle(cycle)
                except IndexError:
                    pass  # If events do not match in the end this will be raised
        return gait_cycles


class HeelStrikeToHeelStrikeCycleBuilder(EventCycleBuilder):
    def __init__(self, event_anomaly_checker: gaitalytics.events.AbstractEventAnomalyChecker):
        super().__init__(event_anomaly_checker, gaitalytics.utils.GaitEventLabel.FOOT_STRIKE)


class ToeOffToToeOffCycleBuilder(EventCycleBuilder):
    def __init__(self, event_anomaly_checker: gaitalytics.events.AbstractEventAnomalyChecker):
        super().__init__(event_anomaly_checker, gaitalytics.utils.GaitEventLabel.FOOT_OFF)


# Cycle Extractor
class CycleDataExtractor:
    def __init__(self, configs: gaitalytics.utils.ConfigProvider):
        self._configs = configs

    def extract_data(
        self, cycles: gaitalytics.utils.GaitCycleList, file_handler: gaitalytics.c3d_reader.FileHandler
    ) -> dict[str, gaitalytics.utils.BasicCyclePoint]:
        subject = file_handler.get_subject_measures()
        data_list: dict[str, gaitalytics.utils.BasicCyclePoint] = {}
        for point_index in range(file_handler.get_points_size()):
            cycle_counts_left = len(cycles.left_cycles.values())
            cycle_counts_right = len(cycles.right_cycles.values())
            cycle_counts = 0
            if cycle_counts_left > cycle_counts_right:
                cycle_counts = cycle_counts_right
            else:
                cycle_counts = cycle_counts_left

            longest_cycle_left = cycles.get_longest_cycle_length(gaitalytics.utils.GaitEventContext.LEFT)
            longest_cycle_right = cycles.get_longest_cycle_length(gaitalytics.utils.GaitEventContext.RIGHT)

            point = file_handler.get_point(point_index)

            for direction_index in range(len(point.values[0])):
                label = point.label
                data_type = point.type
                translated_label = self._configs.get_translated_label(label, data_type)
                if translated_label is not None:
                    direction = gaitalytics.utils.AxesNames(direction_index)
                    left = self._create_point_cycle(
                        cycle_counts,
                        longest_cycle_left,
                        translated_label,
                        direction,
                        data_type,
                        gaitalytics.utils.GaitEventContext.LEFT,
                        subject,
                    )
                    right = self._create_point_cycle(
                        cycle_counts,
                        longest_cycle_right,
                        translated_label,
                        direction,
                        data_type,
                        gaitalytics.utils.GaitEventContext.RIGHT,
                        subject,
                    )

                    for cycle_number in range(1, cycles.get_number_of_cycles() + 1):
                        if len(cycles.right_cycles) + 1 > cycle_number:
                            self._extract_cycle(right, cycles.right_cycles[cycle_number], point.values[:, direction_index])
                        if len(cycles.left_cycles) + 1 > cycle_number:
                            self._extract_cycle(left, cycles.left_cycles[cycle_number], point.values[:, direction_index])

                    key_left = gaitalytics.utils.ConfigProvider.define_key(
                        translated_label, data_type, direction, gaitalytics.utils.GaitEventContext.LEFT
                    )
                    key_right = gaitalytics.utils.ConfigProvider.define_key(
                        translated_label, data_type, direction, gaitalytics.utils.GaitEventContext.RIGHT
                    )
                    data_list[key_left] = left
                    data_list[key_right] = right

        return data_list

    @staticmethod
    def _create_point_cycle(
        cycle_counts: int,
        longest_cycle,
        label: Enum,
        direction: gaitalytics.utils.AxesNames,
        data_type: gaitalytics.utils.PointDataType,
        context: gaitalytics.utils.GaitEventContext,
        subject: gaitalytics.utils.SubjectMeasures,
    ) -> gaitalytics.utils.TestCyclePoint:
        cycle = gaitalytics.utils.TestCyclePoint(cycle_counts, longest_cycle, gaitalytics.utils.BasicCyclePoint.TYPE_RAW)
        cycle.direction = direction
        cycle.context = context
        cycle.translated_label = label
        cycle.data_type = data_type
        cycle.subject = subject
        return cycle

    @staticmethod
    def _extract_cycle(cycle_point: gaitalytics.utils.TestCyclePoint, cycle: gaitalytics.utils.GaitCycle, values: np.array):
        cycle_point.data_table.loc[cycle.number][0 : cycle.length] = values[cycle.start_frame : cycle.end_frame]
        events = np.array(list(cycle.unused_events.values()))
        events = events - cycle.start_frame
        cycle_point.event_frames.loc[cycle.number] = events
        cycle_point.frames.loc[cycle.number] = [cycle.start_frame, cycle.end_frame]


# Normalisation


class CyclePointLoader:

    def __init__(self, configs: gaitalytics.utils.ConfigProvider, dir_path: Path):
        self._raw_cycle_data = {}
        self._norm_cycle_data = {}
        file_names = os.listdir(dir_path)
        postfix = gaitalytics.utils.BasicCyclePoint.TYPE_RAW
        raw_file_names = self._filter_filenames(file_names, postfix)
        subject = gaitalytics.utils.SubjectMeasures.from_file(dir_path / "subject.yml")
        self._raw_cycle_data = self._init_buffered_points(configs, dir_path, raw_file_names, subject)

        postfix = gaitalytics.utils.BasicCyclePoint.TYPE_NORM
        norm_file_names = self._filter_filenames(file_names, postfix)
        self._norm_cycle_data = self._init_buffered_points(configs, dir_path, norm_file_names, subject)

    @staticmethod
    def _init_buffered_points(
        configs: gaitalytics.utils.ConfigProvider, dir_path: Path, file_names: list[str], subject: gaitalytics.utils.SubjectMeasures
    ) -> dict[str, gaitalytics.utils.BasicCyclePoint]:
        cycle_data: dict[str, gaitalytics.utils.BasicCyclePoint] = {}
        for file_name in file_names:
            point = gaitalytics.utils.BufferedCyclePoint(configs, dir_path, file_name, subject)
            foo, key, foo = gaitalytics.utils.get_key_from_filename(file_name)
            cycle_data[key] = point
        return cycle_data

    @classmethod
    def _filter_filenames(cls, file_names, postfix) -> list[str]:
        r = re.compile(rf".*{gaitalytics.utils.FILENAME_DELIMITER}{postfix}.*\.csv")
        return list(filter(r.match, file_names))

    def get_raw_cycle_points(self) -> dict[str, gaitalytics.utils.BasicCyclePoint]:
        return self._raw_cycle_data

    def get_norm_cycle_points(self) -> dict[str, gaitalytics.utils.BasicCyclePoint]:
        return self._norm_cycle_data
