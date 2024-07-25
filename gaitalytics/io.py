"""This module provides classes for reading biomechanical file-types."""

import math
from abc import abstractmethod
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
import pyomeca
import xarray as xr

import gaitalytics.mapping as mapping

_MAX_EVENTS_PER_SECTION = 255


# Input Section
class _BaseFileHandler:
    """Base class for file handler.

    This class provides a common interface for file handlers.

    Attributes:
        file_path: The path to the  file.
    """

    def __init__(self, file_path: Path):
        """Initialize a new instance of the _BaseFileHandler class.

        Args:
            file_path: The path to the input file.
        """
        self.file_path = file_path


class _EventFileWriter(_BaseFileHandler):
    @abstractmethod
    def write_events(self, events: pd.DataFrame, file_path: Path | None = None):
        """Write the events to the output file.

        Args:
            events: The events to write to the output file.
            file_path: The path to the output file if deviating from the input file.
        """
        raise NotImplementedError


class C3dEventFileWriter(_EventFileWriter):
    """A class for handling C3D files in an easy and convenient way."""

    _CONTEXT_SECTION = "CONTEXTS"
    _ICON_SECTION = "ICON_IDS"
    _LABEL_SECTION = "LABELS"
    _TIME_SECTION = "TIMES"

    def write_events(self, events: pd.DataFrame, file_path: Path | None = None):
        """Write the events to the output file.

        Args:
            events: The events to write to the output file.
            file_path: The path to the output file if deviating from the input file.
        """
        c3d = ezc3d.c3d(str(self.file_path))
        n_sections = math.ceil(len(events) / _MAX_EVENTS_PER_SECTION)
        for i in range(n_sections):
            start = i * _MAX_EVENTS_PER_SECTION
            end = (i + 1) * _MAX_EVENTS_PER_SECTION
            if end > len(events):
                end = len(events)

            context_label = self._CONTEXT_SECTION
            icon_label = self._ICON_SECTION
            label_label = self._LABEL_SECTION
            time_label = self._TIME_SECTION

            if i > 0:
                context_label += f"{i + 1}"
                icon_label += f"{i + 1}"
                label_label += f"{i + 1}"
                time_label += f"{i + 1}"
            subset_events = events.iloc[start:end]
            c3d.add_parameter("EVENT", context_label, subset_events["context"].tolist())
            c3d.add_parameter("EVENT", icon_label, subset_events["icon_id"].tolist())
            c3d.add_parameter("EVENT", label_label, subset_events["label"].tolist())

            # times = [[time // 60, time % 60] for time in subset_events["time"].tolist()]
            raw_times = subset_events["time"].to_numpy()
            minutes = raw_times // 60
            seconds = raw_times % 60

            times = np.vstack((minutes, seconds))

            c3d.add_parameter("EVENT", time_label, times)
        c3d.add_parameter("EVENT", "USED", len(events))
        path = file_path if file_path else self.file_path
        c3d.write(str(path))


class _EventInputFileReader(_BaseFileHandler):
    """Abstract base class for reading event input files.

    This class defines the interface for reading event input files
    in gait analysis.
    Subclasses must implement the abstract methods to provide specific
    implementations.

    """

    COLUMN_TIME = "time"
    COLUMN_LABEL = "label"
    COLUMN_CONTEXT = "context"
    COLUMN_ICON = "icon_id"

    @abstractmethod
    def get_events(self) -> pd.DataFrame:
        """Get the events from the input file sorted by time.

        Returns:
            A DataFrame containing the events.
        """
        raise NotImplementedError


class C3dEventInputFileReader(_EventInputFileReader):
    """A class for handling C3D files in an easy and convenient way.

    Implements the EventInputFileReader interface to read events from C3D files.
    """

    def __init__(self, file_path: Path):
        """Initializes a new instance of the EzC3dFileHandler class.

        Args:
            file_path: The path to the C3D file.

        """
        self._c3d = ezc3d.c3d(str(file_path))
        super().__init__(file_path)

    def get_events(self) -> pd.DataFrame:
        """Gets the events from the input file sorted by time.

        Returns:
            A DataFrame containing the events.

        Raises:
            ValueError: If no events are found in the C3D file.
        """
        labels = self._get_event_labels()
        times = self._get_event_times()
        contexts = self._get_event_contexts()
        icons = self._get_event_icons()

        # Check if there are any events in the C3D file
        if not (labels and times and contexts and icons):
            raise ValueError("No events found in the C3D file.")

        table = pd.DataFrame(
            {
                self.COLUMN_TIME: times,
                self.COLUMN_LABEL: labels,
                self.COLUMN_CONTEXT: contexts,
                self.COLUMN_ICON: icons,
            }
        )
        table = table.sort_values(by=self.COLUMN_TIME, ascending=True).reset_index(
            drop=True
        )
        return table

    def _get_event_labels(self) -> list[str]:
        """Gets the labels of the events in the C3D file.

        Returns:
            The labels of the events.

        """
        section_base = "LABELS"
        labels = self._concat_sections(section_base)
        return labels

    def _get_event_times(self) -> list:
        """Returns the event times.

        Returns:
            An array containing the event times.
        """
        section_base = "TIMES"
        return self._concat_sections(section_base)

    def _get_event_contexts(self) -> list[str]:
        """Gets the contexts of the events in the C3D file.

        Returns:
            The contexts of the events.

        """
        section_base = "CONTEXTS"
        return self._concat_sections(section_base)

    def _get_event_icons(self) -> list:
        """Gets the icons of the events in the C3D file.

        Returns:
            The icons of the events.
        """
        section_base = "ICON_IDS"
        return self._concat_sections(section_base)

    def _get_sections(self, section_base):
        """Gets the sections of the specified type in the C3D file.

        Args:
            The base name of the sections to get.

        Returns:
            A list containing the sections of the specified type.
        """
        sections = []
        for section in self._c3d["parameters"]["EVENT"].keys():
            if section.startswith(section_base):
                sections.append(section)
        return sections

    def _concat_sections(self, section_base: str) -> list:
        """Concatenates the values of the specified sections in the C3D file.

        Args:
            section_base: The base name of the sections to concatenate.

        Returns:
            A list containing the concatenated values of the sections.
        """
        values: list = []
        for section in self._get_sections(section_base):
            current_values = self._c3d["parameters"]["EVENT"][section]["value"]
            if len(current_values) == 2:
                # convert TIMES: c3d specifics values[0] as
                # minutes and values[1] as seconds
                current_values = current_values[1] + (current_values[0] * 60)
                current_values = np.round(current_values, 3)
            if type(current_values) is np.ndarray:
                current_values = current_values.tolist()
            values += current_values
        return values


class _PyomecaInputFileReader(_BaseFileHandler):
    """Base class for handling input files using pyomeca.

    This class provides a common interface for reading input files with pyomeca.
    """

    def __init__(
        self, file_path: Path, pyomeca_class: type[pyomeca.Markers | pyomeca.Analogs]
    ):
        """Initializes a new instance of the MarkersInputFileReader class.

        Determines the file format and uses the appropriate pyomeca class
        to read the data. Further it converts the data to absolute time if needed.

        Args:
            file_path: The path to the marker data file.
            pyomeca_class:
                The pyomeca class to use for reading the data.

        """
        file_ext = file_path.suffix
        if file_ext == ".c3d" and (
            pyomeca_class == pyomeca.Analogs or pyomeca_class == pyomeca.Markers
        ):
            data = pyomeca_class.from_c3d(file_path)
        elif file_ext == ".trc" and pyomeca_class == pyomeca.Markers:
            raise NotImplementedError("TRC file format is not supported for markers")
        elif file_ext == ".mot" and pyomeca_class == pyomeca.Analogs:
            data = pyomeca_class.from_mot(
                file_path, pandas_kwargs={"sep": "\t", "index_col": False}
            )
        elif file_ext == ".sto" and pyomeca_class == pyomeca.Analogs:
            raise NotImplementedError("STO file format is not supported for analogs")
        else:
            raise ValueError(
                f"Unsupported file extension: {file_ext} for class {pyomeca_class}"
            )

        if "first_frame" in data.attrs and "rate" in data.attrs:
            first_frame = data.attrs["first_frame"]
            frame_rate = data.attrs["rate"]
            data = self._to_absolute_time(data, first_frame, frame_rate)

        self._data = data
        super().__init__(file_path)

    @staticmethod
    def _to_absolute_time(
        data: xr.DataArray, first_frame: int, rate: float
    ) -> xr.DataArray:
        """Converts the time to absolute time.

        Args:
            data: The data to convert.
            first_frame: The first frame of the data.
            rate: The rate of the data.
        """
        first_frame = first_frame
        data.coords["time"] = data.coords["time"] + (first_frame * 1 / rate)
        return data


class MarkersInputFileReader(_PyomecaInputFileReader):
    """A class for handling marker data in an easy and convenient way.

    Uses the pyomeca.Markers class to read marker data from a file.
    """

    def __init__(self, file_path: Path):
        """Initializes a new instance of the MarkersInputFileReader class.

        Args:
            file_path: The path to the marker data file.

        """
        super().__init__(file_path, pyomeca.Markers)
        self.data = self._data.drop_sel(axis="ones")

    def get_markers(self) -> xr.DataArray:
        """Gets the markers from the input file.

        Returns:
            An xarray DataArray containing the markers.
        """
        return self.data


class AnalogsInputFileReader(_PyomecaInputFileReader):
    """A class for handling analog data in an easy and convenient way.

    Uses the pyomeca.Analogs class to read analog data from a file.
    """

    def __init__(self, file_path: Path):
        """Initializes a new instance of the AnalogsInputFileReader class.

        Args:
            file_path: The path to the analog data file.

        """
        super().__init__(file_path, pyomeca.Analogs)

    def get_analogs(self) -> xr.DataArray:
        """Gets the analog data from the input file.

        Returns:
            An xarray DataArray containing the analog data.
        """
        return self._data


class AnalysisInputReader(_PyomecaInputFileReader):
    """Read out data from modelled data form different input format."""

    def __init__(self, file_path: Path, configs: mapping.MappingConfigs):
        """Initializes a new instance of the AnalysisInputReader class.

        Args:
            file_path: The path to the input file.
            configs: The mapping configurations.
        """
        extension = file_path.suffix
        pyomeca_class: type[pyomeca.Markers | pyomeca.Analogs]
        if extension == ".c3d":
            pyomeca_class = pyomeca.Markers
        elif extension == ".mot":
            pyomeca_class = pyomeca.Analogs
        elif extension == ".sto":
            raise NotImplementedError("STO file format is not supported for analogs")
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        super().__init__(file_path, pyomeca_class)
        self.configs = configs

        if pyomeca_class == pyomeca.Markers:
            self._data = self._data.drop_sel(axis="ones")
            self._filter_markers()
            self._flatten_array()
        else:
            self._filter_analogs()

    def _filter_analogs(self):
        """Filter the analogs based on the mapping configurations."""
        labels = self.configs.get_analogs_analysis()
        if labels:
            self._data = self._data.sel(channel=labels)

    def _filter_markers(self):
        """Filter the markers based on the mapping configurations."""
        labels = self.configs.get_markers_analysis()
        if labels:
            self._data = self._data.sel(channel=labels)

    def _flatten_array(self):
        """Flatten the markers array to a 2D array.

        The markers xarray is reshaped to a 2D array with the shape to generalize format
        from c3d and sto files.
        """
        np_data = self._data.to_numpy()
        rs_data = np_data.reshape(
            (np_data.shape[1] * np_data.shape[0], np_data.shape[2]), order="F"
        )

        time = self._data.coords["time"].values
        new_labels = self._create_labels()
        new_format = xr.DataArray(
            rs_data,
            coords={"channel": new_labels, "time": time},
            dims=["channel", "time"],
        )
        self._copy_attrs(new_format)
        self._data = new_format

    def _create_labels(self) -> list[str]:
        """Create new labels for the flattened array.

        Returns:
            List of labels including axis in the label name.
        """
        labels = self._data.coords["channel"].values
        axis = self._data.coords["axis"].values
        new_labels = []
        for label in labels:
            for axe in axis:
                new_labels.append(f"{label}_{axe}")
        return new_labels

    def _copy_attrs(self, new_data: xr.DataArray):
        """Copy the attributes from the old data to the new data.

        Args:
            new_data: The new data array.
        """
        new_data.attrs = self._data.attrs
        new_data.name = self._data.name

    def get_analysis(self) -> xr.DataArray:
        """Gets the analysis data from the input file.

        Returns:
            An xarray DataArray containing the analysis data.
        """
        return self._data
