from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import gaitalytics.utils

logger = logging.getLogger(__name__)


class FileStore(ABC):

    def __init__(self, subject: gaitalytics.utils.SubjectMeasures, out_path: Path):
        self.subject = subject
        self.out_path = out_path

    @abstractmethod
    def save_to_file(self, cycle_data: dict[str, gaitalytics.utils.BasicCyclePoint]):
        pass

    @abstractmethod
    def read_from_file(self, key: str) -> gaitalytics.utils.BasicCyclePoint:
        pass


class CSVFileStore(FileStore):

    def save_to_file(self, cycle_data: dict[str, gaitalytics.utils.BasicCyclePoint]):
        pass

    def read_from_file(self, key: str) -> gaitalytics.utils.BasicCyclePoint:
        pass


def _cycle_points_to_csv(cycle_data: dict[str, gaitalytics.utils.BasicCyclePoint], dir_path: str | Path, prefix: str):
    logger.info("_cycle_points_to_csv")
    subject_saved = False
    for key in cycle_data:
        cycle_data[key].to_csv(dir_path, prefix)
        if not subject_saved:
            cycle_data[key].subject.to_file(dir_path)
