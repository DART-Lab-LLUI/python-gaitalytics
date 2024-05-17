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
    """
    Abstract base class for building gait cycles from a file handler.

    A gait cycle is a sequence of events that occur from the time one foot contacts the ground to the time the same foot contacts the ground again.
    """

    def build_cycles(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        """
        Build gait cycles from the given file handler.

        This method checks if there are any events in the file. If there are no events, it raises an AttributeError.
        If there are events, it calls the abstract method _build to construct the gait cycles.

        Args:
            file_handler (c3d_reader.FileHandler): The file handler to read the events from.

        Returns:
            model.GaitCycleList: A list of gait cycles.

        Raises:
            AttributeError: If there are no events in the file.
        """
        if file_handler.get_events_size() < 1:
            raise AttributeError("No Events in File")

        return self._build(file_handler)

    @abstractmethod
    def _build(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        """
        Abstract method to build gait cycles from the given file handler.

        This method must be implemented by any class that inherits from CycleBuilder.

        Args:
            file_handler (c3d_reader.FileHandler): The file handler to read the events from.

        Returns:
            model.GaitCycleList: A list of gait cycles.
        """
        pass


class EventCycleBuilder(CycleBuilder):
    """
    A class used to build gait cycles based on specific events.

    This class inherits from the CycleBuilder abstract base class and implements the _build method.

    Attributes:
        event_label (str): The label of the event to build the gait cycles from.
    """

    def __init__(self, event: model.GaitEventLabel):
        """
        Initialize the EventCycleBuilder with the given event.

        Args:
            event (model.GaitEventLabel): The event to build the gait cycles from.
        """
        super().__init__()
        self.event_label = event.value

    def _build(self, file_handler: c3d_reader.FileHandler) -> model.GaitCycleList:
        """
        Build gait cycles from the given file handler based on the event label.

        This method iterates over all events in the file handler. If an event matches the event label, it tries to find the next event of the same type and context.
        If such an event is found, a new gait cycle is created and added to the list of gait cycles.

        Args:
            file_handler (c3d_reader.FileHandler): The file handler to read the events from.

        Returns:
            model.GaitCycleList: A list of gait cycles.
        """
        gait_cycles = model.GaitCycleList()
        numbers = {model.GaitEventContext.LEFT.value: 0, model.GaitEventContext.RIGHT.value: 0}
        for event_index in range(file_handler.get_events_size()):
            start_event = file_handler.get_event(event_index)
            context = start_event.context

            label = start_event.label
            if label == self.event_label:
                try:
                    [end_event, unused_events] = events.find_next_event(file_handler, label, context, event_index)
                    if end_event is not None:
                        numbers[context] = numbers[context] + 1
                        cycle = model.GaitCycle(
                            numbers[context], model.GaitEventContext(context), start_event.frame, end_event.frame,
                            unused_events
                        )
                        gait_cycles.add_cycle(cycle)
                except IndexError:
                    pass  # If events do not match in the end this will be raised
        return gait_cycles


class HeelStrikeToHeelStrikeCycleBuilder(EventCycleBuilder):
    """
    A class used to build gait cycles based on heel strike events.

    This class inherits from the EventCycleBuilder class and sets the event label to FOOT_STRIKE.
    """

    def __init__(self):
        """
        Initialize the HeelStrikeToHeelStrikeCycleBuilder.
        """
        super().__init__(model.GaitEventLabel.FOOT_STRIKE)


class ToeOffToToeOffCycleBuilder(EventCycleBuilder):
    """
    A class used to build gait cycles based on toe off events.

    This class inherits from the EventCycleBuilder class and sets the event label to FOOT_OFF.
    """

    def __init__(self):
        """
        Initialize the ToeOffToToeOffCycleBuilder.
        """
        super().__init__(model.GaitEventLabel.FOOT_OFF)


def _create_empty_table(n_axis: int, n_cycles: int, n_frames) -> np.ndarray:
    """
    Create an empty table with the given dimensions filled with NaN values.

    Args:
        n_axis (int): The number of axes.
        n_cycles (int): The number of cycles.
        n_frames: The number of frames.

    Returns:
        np.ndarray: A 3D numpy array filled with NaN values.
    """
    return np.full((n_axis, n_cycles, n_frames), np.nan)


def extract_point_cycles(
    configs: utils.ConfigProvider, cycles: model.GaitCycleList, file_handler: c3d_reader.FileHandler
) -> model.ExtractedCycles:
    """
    Extract point cycles from the given file handler based on the provided configurations and cycles.

    Args:
        configs (utils.ConfigProvider): The configurations provider.
        cycles (model.GaitCycleList): The list of gait cycles.
        file_handler (c3d_reader.FileHandler): The file handler to read the points from.

    Returns:
        model.ExtractedCycles: The extracted cycles.
    """
    points_left = model.ExtractedContextCycles(model.GaitEventContext.LEFT)
    points_right = model.ExtractedContextCycles(model.GaitEventContext.RIGHT)

    # loop though each point in c3d to extract each cycle
    for point_index in range(file_handler.get_points_size()):

        # init nan numpy arrays
        cycle_data_left = _create_empty_table(
            3, cycles.get_number_of_cycles(model.GaitEventContext.LEFT),
            cycles.get_longest_cycle_length(model.GaitEventContext.LEFT)
        )

        cycle_data_right = _create_empty_table(
            3, cycles.get_number_of_cycles(model.GaitEventContext.RIGHT),
            cycles.get_longest_cycle_length(model.GaitEventContext.RIGHT)
        )

        point = file_handler.get_point(point_index)
        for cycle in cycles.cycles:

            # extract values in cycle
            cycle_data = point.values[cycle.start_frame: cycle.end_frame, :]
            cycle_data = cycle_data.T

            # store it in the array of the context
            if cycle.context == model.GaitEventContext.LEFT:
                cycle_data_left[:, cycle.number - 1, : cycle_data.shape[1]] = cycle_data
            else:
                cycle_data_right[:, cycle.number - 1, : cycle_data.shape[1]] = cycle_data
        # split right and left cycles
        cycle_point_left = _create_cycle_point(configs, point, cycle_data_left)

        cycle_point_right = _create_cycle_point(configs, point, cycle_data_right)

        # store markers in overarching model
        points_left.add_cycle_points(cycle_point_left)
        points_right.add_cycle_points(cycle_point_right)

    meta_data = _extract_general_cycle_data(cycles, model.GaitEventContext.LEFT)
    points_left.meta_data = meta_data

    meta_data = _extract_general_cycle_data(cycles, model.GaitEventContext.RIGHT)
    points_right.meta_data = meta_data
    points = {model.GaitEventContext.LEFT.value: points_left, model.GaitEventContext.RIGHT.value: points_right}

    return model.ExtractedCycles(model.ExtractedCycleDataCondition.RAW_DATA, points)


def _extract_general_cycle_data(cycles: model.GaitCycleList, context: model.GaitEventContext) -> dict[str, np.ndarray]:
    """
    Extract general cycle data from the given cycles based on the provided context.

    Args:
        cycles (model.GaitCycleList): The list of gait cycles.
        context (model.GaitEventContext): The context of the gait event.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the extracted cycle data.
    """

    def add_to_dict(
        key: str, value: int, cycle_number: int, dictionary: dict[str, np.ndarray], max_cycle_length: int
    ) -> dict[str: np.ndarray]:

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


def _create_cycle_point(configs: utils.ConfigProvider, point: model.Point,
                        cycle_data: np.ndarray) -> model.ExtractedCyclePoint:
    """
    Create a cycle point from the given configurations, point, and cycle data.

    Args:
        configs (utils.ConfigProvider): The configurations provider.
        point (model.Point): The point to create the cycle point from.
        cycle_data (np.ndarray): The cycle data.

    Returns:
        model.ExtractedCyclePoint: The created cycle point.
    """
    translated_label = configs.get_translated_label(point.label, point.type)
    cycle_point = model.ExtractedCyclePoint(translated_label, point.type)
    cycle_point.data_table = cycle_data
    return cycle_point
