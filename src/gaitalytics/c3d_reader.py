import importlib
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from statistics import mean

import numpy as np

import gaitalytics.model as model

ANALOG_VOLTAGE_PREFIX_LABEL = "Voltage."

logger = logging.getLogger(__name__)


def is_progression_axes_flip(left_heel, left_toe):
    """
    Determine if the progression axes need to be flipped based on the positions of the left heel and toe.

    Args:
        left_heel: The position of the left heel.
        left_toe: The position of the left toe.

    Returns:
        bool: True if the progression axes need to be flipped, False otherwise.
    """
    return 0 < mean(left_toe[model.AxesNames.y.value] - left_heel[model.AxesNames.y.value])


class FileHandler(ABC):
    """
    Abstract base class for handling files.

    This class provides an interface for reading and writing files, as well as manipulating events within the files.
    """

    def __init__(self, file_path: Path):
        """
        Initialize the FileHandler with the given file path.

        Args:
            file_path (Path): The path to the file to handle.
        """
        self._file_path = str(file_path)
        self.read_file()

    def sort_events(self):
        """
        Sort the events in the acquisition.
        """
        events = self.get_events()

        value_frame: dict[int, model.GaitEvent] = {}

        for index in range(len(events)):
            if events[index].frame not in value_frame:
                value_frame[events[index].frame] = events[index]

        sorted_keys: dict[int, model.GaitEvent] = dict(sorted(value_frame.items()))

        self.clear_events()
        self.set_events(list(sorted_keys.values()))

    @abstractmethod
    def read_file(self):
        """
        Abstract method to read the file.

        This method must be implemented by any class that inherits from FileHandler.
        """
        pass

    def write_file(self, out_file_path: str | Path | None = None):
        """
        Write the file to the given path.

        Args:
            out_file_path (str | Path | None): The path to write the file to. If None, the original file path is used.
        """
        if out_file_path is None:
            out_file_path = self._file_path
        self._write_file(str(out_file_path))

    @abstractmethod
    def _write_file(self, out_file_path: str):
        """
        Abstract method to write the file to the given path.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            out_file_path (str): The path to write the file to.
        """
        pass

    @abstractmethod
    def get_events_size(self) -> int:
        """
        Abstract method to get the number of events in the file.

        This method must be implemented by any class that inherits from FileHandler.

        Returns:
            int: The number of events in the file.
        """
        pass

    @abstractmethod
    def get_events(self) -> list[model.GaitEvent]:
        """
        Abstract method to get the events in the file.

        This method must be implemented by any class that inherits from FileHandler.

        Returns:
            list[model.GaitEvent]: The events in the file.
        """
        pass

    @abstractmethod
    def set_events(self, events: list[model.GaitEvent]):
        """
        Abstract method to set the events in the file.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            events (list[model.GaitEvent]): The events to set in the file.
        """
        pass

    @abstractmethod
    def get_event(self, index: int) -> model.GaitEvent:
        """
        Abstract method to get the event at the given index.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            index (int): The index of the event to get.

        Returns:
            model.GaitEvent: The event at the given index.
        """
        pass

    @abstractmethod
    def add_event(self, event: model.GaitEvent):
        """
        Abstract method to add an event to the file.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            event (model.GaitEvent): The event to add to the file.
        """
        pass

    @abstractmethod
    def clear_events(self):
        """
        Abstract method to clear all events from the file.

        This method must be implemented by any class that inherits from FileHandler.
        """
        pass

    @abstractmethod
    def get_point_frequency(self) -> int:
        """
        Abstract method to get the point frequency of the file.

        This method must be implemented by any class that inherits from FileHandler.

        Returns:
            int: The point frequency of the file.
        """
        pass

    @abstractmethod
    def get_actual_start_frame(self) -> int:
        """
        Abstract method to get the actual start frame of the file.

        This method must be implemented by any class that inherits from FileHandler.

        Returns:
            int: The actual start frame of the file.
        """
        pass

    @abstractmethod
    def get_points_size(self) -> int:
        """
        Abstract method to get the number of points in the file.

        This method must be implemented by any class that inherits from FileHandler.

        Returns:
            int: The number of points in the file.
        """
        pass

    @abstractmethod
    def get_point(self, marker_index: int | str) -> model.Point:
        """
        Abstract method to get the point at the given marker index.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            marker_index (int | str): The marker index of the point to get.

        Returns:
            model.Point: The point at the given marker index.
        """
        pass

    @abstractmethod
    def add_point(self, new_point: model.Point):
        """
        Abstract method to add a point to the file.

        This method must be implemented by any class that inherits from FileHandler.

        Args:
            new_point (model.Point): The point to add to the file.
        """
        pass

# The documentation for the BtkFileHandler and EzC3dFileHandler classes is omitted for brevity.
# Please add the appropriate docstrings following the same style as above.
