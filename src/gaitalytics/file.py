from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
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
    def save_extracted_cycles(self, cycle_data: model.ExtractedCycles):
        pass

    @abstractmethod
    def read_extracted_cycles(self) -> model.ExtractedCycles:
        pass

    @abstractmethod
    def save_analysis(self, analysis: dict[str, np.ndarray]) -> None:
        pass


class Hdf5FileStore(FileStore):
    POINT_TYPE_ATTR = "point_type"

    def save_extracted_cycles(self, cycle_data: model.ExtractedCycles):
        with h5py.File(self.out_path, "a") as h5_file:
            data_con_group = h5_file.create_group(cycle_data.data_condition.value)
            # data_con_group.attrs.update(subject.__dict__)
            for cycle_points in cycle_data.cycle_points.values():
                self._save_datatables(cycle_points, data_con_group)

    def read_extracted_cycles(self) -> dict[model.ExtractedCycleDataCondition, model.ExtractedCycles]:
        with h5py.File(self.out_path, "r") as h5_file:
            extracted_cycles = {}
            for h5_group_key in h5_file:
                points = {
                    model.GaitEventContext.LEFT: self._create_context_points(h5_file[h5_group_key], model.GaitEventContext.LEFT),
                    model.GaitEventContext.RIGHT: self._create_context_points(h5_file[h5_group_key], model.GaitEventContext.RIGHT),
                }
                condition = model.ExtractedCycleDataCondition(h5_group_key)
                extracted_cycles[condition] = model.ExtractedCycles(condition, points)
        return extracted_cycles

    def save_analysis(self, analysis: dict[str, np.ndarray]) -> None:
        with h5py.File(self.out_path, "a") as h5_file:
            analysis_group = h5_file.create_group("analysis")
            for key, value in analysis.items():
                analysis_group.create_dataset(key, data=value)

    def _create_context_points(self, h5_group: h5py.Group, context: model.GaitEventContext) -> model.ExtractedContextCycles:
        h5_context_group = h5_group[context.name]
        cycle_points = model.ExtractedContextCycles(context)
        meta_data: dict[str, np.ndarray] = {}
        for meta_key in h5_context_group.attrs:
            meta_data[meta_key] = h5_context_group.attrs[meta_key]
        cycle_points.meta_data = meta_data
        cycle_points.context = context
        points = []
        for h5_data_key in h5_context_group:
            points.append(self._create_point(h5_context_group[h5_data_key]))
        cycle_points.points = points
        return cycle_points

    def _save_datatables(self, cycle_data: model.ExtractedContextCycles, h5_group: h5py.Group):
        side_group = h5_group.create_group(cycle_data.context.name)
        side_group.attrs.update(cycle_data.meta_data)
        for point in cycle_data.points:
            if point.translated_label:
                data_set = side_group.create_dataset(
                    point.translated_label.name, point.data_table.shape, dtype=float, data=point.data_table
                )
                data_set.attrs[self.POINT_TYPE_ATTR] = point.point_type.name

    def _create_point(self, h5_point: h5py.Dataset) -> model.ExtractedCyclePoint:
        point_type = model.PointDataType[h5_point.attrs[self.POINT_TYPE_ATTR]]
        label = self.config.get_translated_label(h5_point.name.split("/")[3], point_type)
        point = model.ExtractedCyclePoint(label, point_type)
        point.data_table = h5_point[:]
        return point
