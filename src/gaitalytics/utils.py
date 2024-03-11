from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml

import gaitalytics.model as model

FILENAME_DELIMITER = "-"


def min_max_norm(data):
    scale_min = -1
    scale_max = 1
    max_data = max(data)
    min_data = min(data)
    diff = max_data - min_data
    return [((entry - min_data) * (scale_max - scale_min) / diff) + scale_min for entry in data]


class ConfigProvider:
    _MARKER_MAPPING = "marker_set_mapping"
    _MODEL_MAPPING = "model_mapping"

    def __init__(self, file_path: str):
        file_path_obj = Path(file_path)
        self._read_configs(file_path_obj)
        self.MARKER_MAPPING = Enum("MarkerMapping", self._config[self._MARKER_MAPPING])
        self.MODEL_MAPPING = Enum("ModelMapping", self._config[self._MODEL_MAPPING])

    def get_translated_label(self, label: str, point_type: model.PointDataType) -> Enum | None:
        try:
            if point_type.value == model.PointDataType.MARKERS.value:
                return self.MARKER_MAPPING(label)
            else:
                return self.MODEL_MAPPING(label)
        except ValueError:
            try:
                if point_type.value == model.PointDataType.MARKERS.value:
                    return self.MARKER_MAPPING[label]
                else:
                    return self.MODEL_MAPPING[label]
            except KeyError:
                return None

    def _read_configs(self, file_path: Path):
        with file_path.open("r") as f:
            self._config = yaml.safe_load(f)

    @staticmethod
    def define_key(
        translated_label: Enum, point_type: model.PointDataType, direction: model.AxesNames, side: model.GaitEventContext
    ) -> str:
        if translated_label is not None:
            return f"{translated_label.name}.{point_type.name}.{direction.name}.{side.value}"


def get_key_from_filename(filename: str) -> [str, str, str]:
    return filename.split(FILENAME_DELIMITER)


def get_meta_data_filename(filename: str) -> [str, model.PointDataType, model.AxesNames, model.GaitEventContext, str, str]:
    prefix, key, postfix = get_key_from_filename(filename)
    meta_data = key.split(".")
    label = meta_data[0]
    data_type = model.PointDataType[meta_data[1]]
    direction = model.AxesNames[meta_data[2]]
    context = model.GaitEventContext(meta_data[3])
    postfix = postfix.split(".")[0]
    return [label, data_type, direction, context, postfix, prefix]
