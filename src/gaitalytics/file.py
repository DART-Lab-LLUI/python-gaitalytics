from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np

import gaitalytics.model as model
import gaitalytics.utils as utils

logger = logging.getLogger(__name__)


class FileStore(ABC):

    def __init__(self, path: Path, config: utils.ConfigProvider):
        self.out_path: Path = path
        self.config: utils.ConfigProvider = config

    @abstractmethod
    def save_to_file(self, cycle_data: model.ExtractedCycles):
        pass

    @abstractmethod
    def read_from_file(self) -> model.ExtractedCycles:
        pass


class Hdf5FileStore(FileStore):
    POINT_TYPE_ATTR = "point_type"

    def save_to_file(self, cycle_data: model.ExtractedCycles):
        h5_file = h5py.File(self.out_path, 'a')

        subject = cycle_data.subject
        data_con_group = h5_file.create_group(cycle_data.data_condition.value)
        data_con_group.attrs.update(subject.__dict__)
        self._save_datasets(cycle_data.left_cycle_points, data_con_group)
        self._save_datasets(cycle_data.right_cycle_points, data_con_group)
        h5_file.close()

    def _save_datasets(self, cycle_data: model.ExtractedContextCycles, h5_group: h5py.Group):
        side_group = h5_group.create_group(cycle_data.context.name)
        side_group.attrs.update(cycle_data.meta_data)
        for point in cycle_data.points:
            if point.translated_label:
                data_set = side_group.create_dataset(point.translated_label.name, point.data_table.shape, dtype=float,
                                                     data=point.data_table)
                data_set.attrs[self.POINT_TYPE_ATTR] = point.point_type.name

    def read_from_file(self) -> dict[model.ExtractedCycleDataCondition, model.ExtractedCycles]:
        h5_file = h5py.File(self.out_path, 'r')
        extracted_cycles = {}
        for h5_group_key in h5_file:
            subject = self.create_subject(h5_file[h5_group_key])

            left_points = self.create_context_points(h5_file[h5_group_key], model.GaitEventContext.LEFT)
            right_points = self.create_context_points(h5_file[h5_group_key], model.GaitEventContext.LEFT)
            condition = model.ExtractedCycleDataCondition(h5_group_key)
            extracted_cycles[condition] = model.ExtractedCycles(condition,
                                                                subject,
                                                                left_points,
                                                                right_points)
        return extracted_cycles

    def create_context_points(self, h5_group: h5py.Group,
                              context: model.GaitEventContext) -> model.ExtractedContextCycles:
        h5_context_group = h5_group[context.name]
        cycle_points = model.ExtractedContextCycles(context)
        meta_data: dict[str, np.ndarray] = {}
        for meta_key in h5_context_group.attrs:
            meta_data[meta_key] = h5_context_group.attrs[meta_key]
        cycle_points.meta_data = meta_data
        cycle_points.context = context
        points = []
        for h5_data_key in h5_context_group:
            points.append(self.create_point(h5_context_group[h5_data_key]))
        cycle_points.points = points
        return cycle_points

    def create_point(self, h5_point: h5py.Dataset) -> model.ExtractedCyclePoint:
        point_type = model.PointDataType[h5_point.attrs[self.POINT_TYPE_ATTR]]
        label = self.config.get_translated_label(h5_point.name.split('/')[3], point_type)
        point = model.ExtractedCyclePoint(label, point_type)
        point.data_table = h5_point[:]
        return point

    @staticmethod
    def create_subject(h5_group: h5py.Group) -> model.SubjectMeasures:
        body_height = h5_group.attrs['body_height']
        body_mass = h5_group.attrs['body_mass']
        left_leg_length = h5_group.attrs['left_leg_length']
        right_leg_length = h5_group.attrs['right_leg_length']
        start_frame = h5_group.attrs['start_frame']
        subject = h5_group.attrs['subject']
        subject_measure = model.SubjectMeasures(body_mass, body_height, left_leg_length, right_leg_length, subject,
                                                start_frame)
        return subject_measure


class CSVFileStore(FileStore):

    def save_to_file(self, cycle_data: model.ExtractedContextCycles):
        pass

    def read_from_file(self, key: str) -> model.ExtractedContextCycles:
        pass
