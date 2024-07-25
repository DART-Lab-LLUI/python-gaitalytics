"""This module contains classes for segmenting the trial data with different methods.

The module provides classes for segmenting the trial data based on
gait events as well as a base class to implement additional methods.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

import gaitalytics.events as ga_events
import gaitalytics.io as io
import gaitalytics.model as model
import gaitalytics.utils.math as ga_math


class _BaseSegmentation(ABC):
    @abstractmethod
    def segment(self, trial: model.Trial) -> model.TrialCycles:
        """Segments the trial data based on the segmentation method.

        Args:
            trial: The trial to be segmented.
        """
        raise NotImplementedError


class GaitEventsSegmentation(_BaseSegmentation):
    """A class for segmenting the trial data based on gait events.

    This class provides a method to segment the trial data based on gait events.
    It splits the trial data based on the event label and context.
    """

    def __init__(self, event_label: str = ga_events.FOOT_STRIKE):
        """Initializes a new instance of the GaitEventsSegmentation class.

        Args:
            event_label: The label of the event to be used for segmentation.
        """
        self.event_label = event_label

    def segment(self, trial: model.Trial) -> model.TrialCycles:
        """Segments the trial data based on gait events and contexts.

        Args:
            trial: The trial to be segmented.

        Returns:
            A new trial containing the all the cycles.

        Raises:
            ValueError: If the trial does not have events.
        """
        events = trial.events
        if events is None:
            raise ValueError("Trial does not have events.")

        events_times = self._get_times_of_events(events)

        trial_cycles = model.TrialCycles()
        for context, times in events_times.items():
            for cycle_id in range(len(times) - 1):
                start_time = times[cycle_id]
                end_time = times[cycle_id + 1]
                trial_cycles.add_cycle(
                    context,
                    cycle_id,
                    self._get_segment(trial, start_time, end_time, cycle_id, context),
                )

        return trial_cycles

    def _get_times_of_events(self, events: pd.DataFrame) -> dict[str, list]:
        """Gets the times of the events in the trial.

        This method splits the trial data based on the event label and context.

        Args:
            events: The events in the trial.

        Returns:
            A dictionary containing the contexts
            as keys and the event times as values.
        """
        splits = {}
        interesting_events = events[
            events[io._EventInputFileReader.COLUMN_LABEL] == self.event_label
        ]
        contexts = events[io._EventInputFileReader.COLUMN_CONTEXT].unique()
        for context in contexts:
            context_events = interesting_events[
                interesting_events[io._EventInputFileReader.COLUMN_CONTEXT] == context
            ]
            splits[context] = context_events[
                io._EventInputFileReader.COLUMN_TIME
            ].values
        return splits

    def _get_segment(
        self,
        trial: model.Trial,
        start_time: float,
        end_time: float,
        cycle_id: int,
        context: str,
    ) -> model.Trial:
        """Segments the trial data based on the start and end times.

        Args:
            trial: The trial to be segmented.
            start_time: The start time of the segment.
            end_time: The end time of the segment.
            cycle_id: The cycle id of the segment.
            context: The context of the segment.

        Returns:
            A new trial containing the segmented data.
        """

        trial_segment = model.Trial()
        # segment the data
        for category, data in trial.get_all_data().items():
            rate = int(data.attrs["rate"])
            dec_places = ga_math.get_decimal_places(1 / rate)

            segment = data.sel(time=slice(start_time, end_time))
            segment = segment.reset_index("time")

            times = (segment.time - start_time).to_numpy()
            times = np.round(times, dec_places)
            times = np.absolute(times)

            segment = segment.assign_coords(time=times)
            self._update_attrs(segment, start_time, end_time, cycle_id, context)
            trial_segment.add_data(category, segment)
        # segment the events
        trial_segment.events = self._segment_events(
            context, cycle_id, trial.events, start_time, end_time
        )
        return trial_segment

    @staticmethod
    def _segment_events(
        context: str,
        cycle_id: int,
        events: pd.DataFrame | None,
        start_time: float,
        end_time: float,
    ) -> pd.DataFrame:
        """Segments the events based on the start and end times.

        Args:
            context: The context of the segment.
            cycle_id: The cycle id of the segment.
            events: The events to be segmented.
            start_time: The start time of the segment.
            end_time: The end time of the segment.

        Returns:
            A DataFrame containing the segmented events.
        """
        if events is None:
            raise ValueError("Events are not set.")
        new_events = events[
            (events[io._EventInputFileReader.COLUMN_TIME] >= start_time)
            & (events[io._EventInputFileReader.COLUMN_TIME] <= end_time)
        ]
        new_events.loc[:, "time"] -= start_time
        new_events.attrs = {
            "end_time": end_time,
            "start_time": start_time,
            "context": context,
            "cycle_id": cycle_id,
            "used": 1,
        }
        return new_events

    @staticmethod
    def _update_attrs(
        segment: xr.DataArray, start_time, end_time, cycle_id: int, context: str
    ):
        """Updates the attributes of the segment based on the data.

        Updates time, and frames to relative values. Add additional information
        such as context, cycles_id and used. Based on the "used"-Flag cycles can be
        included or excluded in the analysis

        Args:
            segment: The segment to be updated.
            cycle_id: The cycle id of the segment.
            context: The context of the segment.
        """

        segment.attrs["start_time"] = start_time
        segment.attrs["end_time"] = end_time
        segment.attrs["cycle_id"] = cycle_id
        segment.attrs["context"] = context
        # netcdf can not handle booleans :(
        segment.attrs["used"] = 1
