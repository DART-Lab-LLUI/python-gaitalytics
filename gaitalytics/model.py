"""This module provides classes for structuring, storing and loading trial data."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import h5netcdf as netcdf
import pandas as pd
import xarray as xr


class DataCategory(Enum):
    """Enum class for the array categories.

    This class provides the categories for the data arrays.


    Attributes:
        MARKERS: The marker's category.
        ANALOGS: The analog's category.
        ANALYSIS: The analysis category.
    """

    MARKERS: str = "markers"
    ANALOGS: str = "analogs"
    ANALYSIS: str = "analysis"


class BaseTrial(ABC):
    """Abstract base class for trials.

    This class provides a common interface for trials to load and save data.
    """

    @abstractmethod
    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Local implementation of the to_hdf5 method.

        Args:
            file_path: The path to the HDF5 file.
            base_group: The base group to save the data.
            If None, the data will be saved in the root of the file.
            Default = None
        """
        raise NotImplementedError

    def to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Saves the trial data to an HDF5 file.

        Args:
            file_path: The path to the HDF5 file.
            base_group: The base group to save the data.
            If None, the data will be saved in the root of the file.
            Default = None

        Raises:
            FileExistsError: If the file already exists.
            ValueError: If the trial is a segmented trial and
            the file path is a single file.
            ValueError: If the trial is a trial and the file path is a folder.
        """
        if file_path.exists():
            raise FileExistsError(f"{file_path} already exists.")
        elif type(self) is TrialCycles and file_path.suffix:
            raise ValueError("Cannot save a segmented trial in a single file.")
        elif type(self) is Trial and not file_path.suffix:
            raise ValueError("Cannot save a trial in folder")

        paths, data, groups = self._to_hdf5(file_path, base_group)
        if len(data) > 0:
            if file_path.suffix:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path.mkdir(parents=True, exist_ok=True)
            xr.save_mfdataset(data, paths, groups=groups, mode="a", engine="h5netcdf")
        else:
            raise ValueError("No data to save.")


class Trial(BaseTrial):
    """Represents a trial.

    A trial is a collection of data arrays (typically markers & analogs) and events.
    """

    def __init__(self):
        """Initializes a new instance of the Trial class."""
        self._data: dict[DataCategory, xr.DataArray] = {}
        self._events: pd.DataFrame | None = None

    @property
    def events(self) -> pd.DataFrame | None:
        """Gets the events in the trial.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the events if present.
            None: If no events are present.
        """
        return self._events

    @events.setter
    def events(self, events: pd.DataFrame):
        """Sets the events in the trial.

        Args:
            events: The events to be set.
        """
        self._events = events

    def add_data(self, category: DataCategory, data: xr.DataArray):
        """Adds data to the trial.

        Args:
            category: The category of the data.
            data: The data array to be added.
        """
        if category in self._data:
            self._data[category] = xr.concat([self._data[category], data], dim="time")
        else:
            self._data[category] = data

    def get_data(self, category: DataCategory) -> xr.DataArray:
        """Gets the data from the trial.

        Args:
            category: The category of the data.

        Returns:
            The data array.
        """
        return self._data[category]

    def get_all_data(self) -> dict[DataCategory, xr.DataArray]:
        """Gets all data from the trial.

        Returns:
            A dictionary containing the data arrays.
        """
        return self._data

    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Saves trial into an HDF5 file.

        Structure:

        - root
            - markers
                - xarray.DataArray
            - analogs
                - xarray.DataArray
            - events
                - xarray.Dataset

        Args:
            file_path: The path to the HDF5 file.
            base_group: The base group to save the data.
                If None, the data will be saved in the root of the file.
                Default = ""
        """
        if base_group is None:
            base_group = ""
        else:
            base_group = f"{base_group}"

        groups = []
        data = []
        paths = []
        # Gather all data
        if self.get_all_data() is not None and len(self.get_all_data()) > 0:
            groups += [
                f"{base_group}{category.value}"
                for category in self.get_all_data().keys()
            ]
            data += [data.to_dataset() for data in self.get_all_data().values()]
            paths += [file_path for _ in groups]

        if self.events is not None:
            groups.append(f"{base_group}events")
            data.append(self.events.to_xarray())
            paths.append(file_path)

        return paths, data, groups


class TrialCycles(BaseTrial):
    """Represents a segmented trial."""

    def __init__(self):
        """Initializes a new instance of the SegmentedTrial class."""
        self._cycles: dict[str, dict[int, Trial]] = {}

    def add_cycle(self, context: str, cycle_id: int, segment: Trial):
        """Adds a Cycle to the segmented trial.

        Args:
            context: The context of the cycle.
            cycle_id: The id of the cycle.
            segment: The segment to be added.
        """
        if context not in self._cycles.keys():
            self._cycles[context] = {}

        self._cycles[context][cycle_id] = segment

    def get_cycle(self, context: str, cycle_id: int) -> Trial:
        """Gets a cycle from the segmented trial.

        Args:
            context: The context of the cycle.
            cycle_id: The id of the cycle.

        Raises:
            KeyError: If the cycle does not exist.
        """
        return self._cycles[context][cycle_id]

    def get_all_cycles(self) -> dict[str, dict[int, Trial]]:
        """Gets all cycles from the segmented trial.

        Returns:
            A nexted dictionary containing the cycles.
            Whereas the key of the first dictionary is the context and the second
            key is the cycle number.
        """
        return self._cycles

    def get_cycles_per_context(self, context: str) -> dict[int, Trial]:
        """Gets all cycles from the segmented trial for a specific context.

        Args:
            context: The context of the cycles.

        Returns:
            A dictionary containing the cycles
            for the specified context.
        """
        return self._cycles[context]

    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Recursively saves the segmented trial data to an HDF5 file.

        Unfortunately, writing a huge a mount of separate arrays in a single file
        is not efficient at the moment.
        So the data is saved in separate files.

        Structure example of GaitEventsSegmentation:
        /folder
            - 0.h5 (cycle_id)
                - Left
                    - markers
                        - xarray.DataArray
                    - analogs
                        - xarray.DataArray
                    - events
                        - xarray.DataSet
                - Right
                    - markers
                        - xarray.DataArray
                    - analogs
                        - xarray.DataArray
                    - events
                        - xarray.DataSet
            - ...

        Args:
            file_path: The path to the HDF5 file.
            base_group: The base group to save the data.
                If None, the data will be saved in the root of the file.
        """
        if base_group is None:
            base_group = "/"

        groups = []
        data = []
        paths = []

        for context, cycles in self.get_all_cycles().items():
            for cycle_id, cycle in cycles.items():
                new_file = (file_path / f"{cycle_id}").with_suffix(".h5")
                new_base_group = f"{base_group}{context}/"
                seg_path, seg_data, seg_groups = cycle._to_hdf5(
                    new_file, base_group=new_base_group
                )
                groups += seg_groups
                data += seg_data
                paths += seg_path

        return paths, data, groups


def trial_from_hdf5(file_path: Path) -> Trial | TrialCycles:
    """Loads trial data from an HDF5 file.

    Following structure is expected:
    Trial:
    - file_path (hdf5 file)
        - Left (context)
            - markers
                - xarray.DataArray
            - analogs
                - xarray.DataArray
            - events
                - xarray.Dataset
        - Right (context)
            ...

    TrialCycles:
    - file_path (folder)
        - 0.h5 (cycle_id)
            - Left (context)
                - markers
                    - xarray.DataArray
                - analogs
                    - xarray.DataArray
                - events
                    - xarray.Dataset
            - Right (context)
                ...
        - ...

    Args:
        file_path: The path to the HDF5 file or folder with the expected structure.

    Returns:
        Trial: A new instance of the Trial class if file_path is a single file.
        TrialCycles: A new instance of the TrialCycles class if file_path is a folder.
    """
    trial: Trial | TrialCycles
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    elif file_path.suffix:
        trial = _load_trial_file(file_path)
    else:
        trial = _load_segmented_trial_file(file_path)

    return trial


def _load_segmented_trial_file(file_path: Path) -> TrialCycles:
    """Loads a segmented trial from a folder containing HDF5 files.

    Following structure is expected:
    - folder
        - 0.h5 (cycle_id)
            - Left (context)
                - markers
                    - xarray.DataArray
                - analogs
                    - xarray.DataArray
                - events
                    - xarray.Dataset
            - Right (context)
                ...
        - ...

    Args:
        file_path: folder path containing the HDF5 files.

    Returns:
        A new instance of the TrialCycles class.

    """
    trial_cycles = TrialCycles()

    for file in file_path.glob("**/*.h5"):
        with netcdf.File(str(file), "r") as f:
            cycle_id = file.name.replace(".h5", "")
            for context in f.groups.keys():
                trial = _load_trial(f[context], file)
                trial_cycles.add_cycle(context, int(cycle_id), trial)
    return trial_cycles


def _load_trial_file(file_path: Path) -> Trial:
    """Loads a trial from an HDF5 file.

    Following structure is expected:
    - file_path (hdf5 file)
        - Left (context)
            - markers
                - xarray.DataArray
            - analogs
                - xarray.DataArray
            - events
                - xarray.Dataset
        - Right (context)
            ...

    Args:
        file_path: The path to the HDF5 file.
    """
    with netcdf.File(file_path, "r") as f:
        trial = _load_trial(f, file_path)

    return trial


def _load_trial(group: netcdf.File, file_path: Path) -> Trial:
    """Loads a trial from an HDF5 group.

    following structure is expected:
    - groupy (hdf5)
        - Left (context)
            - markers
                - xarray.DataArray
            - analogs
                - xarray.DataArray
            - events
                - xarray.Dataset
        - Right (context)
            ...
    Args:
        group: The group containing the trial data.
        file_path: The path to the HDF5 file.

    Returns:
        A new instance of the Trial class.
    """
    correct_file_format = False
    trial = Trial()
    for category in DataCategory:
        if category.value in group.groups.keys():
            with xr.load_dataarray(
                file_path, group=f"{group.name}/{category.value}"
            ) as data:
                trial.add_data(category, data)

            correct_file_format = True
    if "events" in group.groups.keys():
        with xr.load_dataset(
            file_path, group=f"{group.name}/events", engine="h5netcdf"
        ) as events:
            trial.events = events.to_dataframe()
        correct_file_format = True

    if not correct_file_format:
        raise ValueError(f"File {file_path} does not have the correct format.")

    return trial
