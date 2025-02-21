from functools import wraps
from inspect import signature, Parameter
from pathlib import Path

import pandas as pd
import xarray as xr

import gaitalytics.events as events
import gaitalytics.features as features
import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.normalisation as normalisation
import gaitalytics.segmentation as segmentation


class _PathConverter:
    """A decorator to convert Path | str annotations to Path objects.

    This decorator is used to convert parameters with Path | str annotations
    to Path objects.
    """

    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.sig = signature(func)

    def __call__(self, *args, **kwargs):
        bound_arguments = self.sig.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        new_args = []
        for name, value in bound_arguments.arguments.items():
            param: Parameter = self.sig.parameters[name]
            # Check if the annotation is Path | str
            if param.annotation == Path | str or param.annotation == Path | str | None:
                if isinstance(value, (Path, str)):
                    value = Path(value)
            if param.kind in [
                param.VAR_POSITIONAL,
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            ]:
                new_args.append(value)
            else:
                bound_arguments.arguments[name] = value

        return self.func(*new_args, **bound_arguments.kwargs)


@_PathConverter
def load_config(config_path: Path | str) -> mapping.MappingConfigs:
    """Loads the mapping configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A MappingConfigs object.
    """
    return mapping.MappingConfigs(config_path)  # type: ignore


@_PathConverter
def load_c3d_trial(
    c3d_file: Path | str, configs: mapping.MappingConfigs
) -> model.Trial:
    """Loads a Trial from a c3d file.

    Be aware that all the required data for the trial must be present in the c3d file.
    i.e. markers, analogs, events, etc.

    Args:
        c3d_file: The path to the c3d file.
        configs: The mapping configurations

    Returns:
        A Trial object.
    """
    markers = io.MarkersInputFileReader(c3d_file).get_markers()  # type: ignore
    analogs = io.AnalogsInputFileReader(c3d_file).get_analogs()  # type: ignore
    analysis = io.AnalysisInputReader(c3d_file, configs).get_analysis()  # type: ignore
    event_table = io.C3dEventInputFileReader(c3d_file).get_events()  # type: ignore

    trial = model.Trial()
    trial.add_data(model.DataCategory.MARKERS, markers)
    trial.add_data(model.DataCategory.ANALOGS, analogs)
    trial.add_data(model.DataCategory.ANALYSIS, analysis)
    trial.events = event_table

    return trial


def get_event_detector(
    method_hs: str,
    method_to: str,
    configs: mapping.MappingConfigs,
    offset: float = 0,
    trial_ref=None,
) -> events.EventDetector:
    """Builds an EventDetector object whose method is the same for all event types

    Args:
        method_hs: event detection method for heel strike
        method_to: event detection method for toe off
                - "Zen" will test the Zeni method
                - "Des" will test the Desailly method
                - "AC1" to "AC6" will test the Autocorrelation 1 to 6 methods
        config: The mapping configurations
        offset: offset to be applied to all event timings, default is 0
        trial_ref: object of model.Trial, reference trial if the event detection requires a reference

    Returns:
        EventDetector object
    """
    if trial_ref is None:
        return events.EventDetectorBuilder.get_event_detector_no_ref(
            configs, method_hs, method_to, offset
        )
    else:
        return events.EventDetectorBuilder.get_event_detector_with_ref(
            configs, method_hs, method_to, trial_ref, offset
        )


def get_GRF_event_detector(
    configs: mapping.MappingConfigs, offset: float = 0
) -> events.EventDetector:
    """Builds an EventDetector object whose method is the same for all event types

    Args:
        config: The mapping configurations
        offset: offset to be applied to all event timings, default is 0

    Returns:
        EventDetector object
    """
    return events.EventDetectorBuilder.get_event_detector_no_ref(
        configs, "GRF", "GRF", offset
    )


def get_mixed_event_detector(
    method_hs_l: str,
    method_hs_r: str,
    method_to_l: str,
    method_to_r: str,
    configs: mapping.MappingConfigs,
    offset: float = 0,
    trial=None,
) -> events.EventDetector:
    """Builds an EventDetector object whose method is the same for all event type

    Args:
        method_to_l: event detection method for left toe off
        method_to_r: event detection method for right toe off
        method_hs_l: event detection method for left heel strike
        method_hs_r: event detection method for right heel strike
                - "Zen" will test the Zeni method
                - "Des" will test the Desailly method
                - "AC1" to "AC6" will test the Autocorrelation 1 to 6 methods
        config: The mapping configurations
        offset: offset to be applied to all event timings, default is 0
        trial_ref: object of model.Trial, reference trial if the event detection requires a reference

    Returns:
        EventDetector object
    """
    return events.EventDetectorBuilder.get_mixed_event_detector(
        configs, method_hs_l, method_hs_r, method_to_l, method_to_r, offset, trial
    )


def detect_events(
    trial: model.Trial,
    event_detector: events.EventDetector,
    parameters: dict | None = None,
) -> pd.DataFrame:
    """Detects the events in the trial.

    Args:
        trial: The trial to detect the events for.
        event_detector: object containing detection methods (optimized or not) for each event type
        parameters: dictionary of event detection parameters. Default None

    Returns:
        A DataFrame containing the detected events.
    """
    event_table = event_detector.detect_events(trial, parameters)
    return event_table


def find_optimal_detectors(
    trial_ref: model.Trial,
    config: mapping.MappingConfigs,
    method_list: list[str] = ["Zen", "Des", "AC1", "AC2", "AC3", "AC4", "AC5", "AC6"],
) -> tuple[events.EventDetector, dict]:
    """Finds the set of best methods that best detect all Gait Event types on a short labeled reference trial
    Also returns feedback for user

    Args:
        trial_ref: The reference trial with some labeled gait events
        config: The mapping configurations
        method_list: list of methods it tests for
                        - "Zen" will test the Zeni method
                        - "Des" will test the Desailly method
                        - "AC1" to "AC6" will test the Autocorrelation 1 to 6 methods
    Returns:
        An EventDetector object with optimized detection methods for each gait event
        user_show : dict containing the performance of all selected methods, as well as the parameters used to find the events
    """
    method_list_mapping = [
        events.EventDetectorBuilder.get_method(name) for name in method_list
    ]
    auto_obj = events.AutoEventDetection(config, trial_ref, method_list_mapping)
    event_detector, user_show = auto_obj.get_optimised_event_detectors()
    return event_detector, user_show


def get_ref_from_GRF(
    trial: model.Trial, config: mapping.MappingConfigs, gait_cycles_ref: int = 15
) -> model.Trial:
    """Creates a reference set of events detected with Ground Reaction Forces (if available) for the given trial.
    The detected events meet the following set of conditions:
    1. Events are regularly spaced (not too close and not too far apart)
    2. Detected events should have a GRF value close to 0
    3. Event should be followed/preceded by a large slope
    4. Order of events is correct
    If the required number of events is not found, an error is raised¨

    Args:
        trial: The trial whose events to be detected and used as reference
        config: The mapping configurations
        gait_cycles_ref: number of gait cycles to use as reference. Default is 15

    Returns:
        Trial: the same trial with detected events as attributes

    Raises:
        ValueError if no events have been selected to use as reference
    """
    event_detector = get_GRF_event_detector(config)
    events_table = detect_events(trial, event_detector)

    obj = events.ReferenceFromGrf(events_table, trial, config, gait_cycles_ref)
    trial_ref = obj.get_reference()
    if trial_ref.events is not None and not trial_ref.events.empty:
        return trial_ref
    else:
        raise ValueError(
            "No valid events detected with GRF. Try manually labeling events to use as reference"
        )


def check_events(event_table: pd.DataFrame, method: str = "sequence"):
    """Checks the events in the trial.

    Args:
        event_table: The event table to check.
        method: The method to use for checking the events.
                Currently, only supports "sequence" which checks the sequence of events
                in terms of context and label. Default is "sequence".

    Returns:
        The trial with the checked events.

    Raises:
        ValueError: If the event sequence is not correct or the method is not supported.
    """
    match method:
        case "sequence":
            checker = events.SequenceEventChecker()
        case _:
            raise ValueError(f"Unsupported method: {method}")
    good, errors = checker.check_events(event_table)
    if not good:
        raise ValueError(f"Event sequence is not correct: {errors}")


@_PathConverter
def write_events_to_c3d(
    c3d_path: Path | str,
    event_table: pd.DataFrame,
    output_path: Path | str | None = None,
):
    """Writes the events to the c3d file.

    Args:
        c3d_path: The path to the original c3d file.
        event_table: The DataFrame containing the events.
        output_path: The path to write the c3d file with the events.
                     If None, the original file will be overwritten.
    """
    io.C3dEventFileWriter(c3d_path).write_events(event_table, output_path)  # type: ignore


def segment_trial(trial: model.Trial, method: str = "HS") -> model.TrialCycles:
    """Segments the trial into cycles

    Args:
        trial: The trial to segment.
        method: The method to use for segmenting the trial.
                Currently, only supports "HS" which segments the trial based on heel strikes.
                Default is "HS".

    Returns:
        The trial with the segmented data.
    """
    match method:
        case "HS":
            method_obj = segmentation.GaitEventsSegmentation()
        case "TO":
            method_obj = segmentation.GaitEventsSegmentation(events.FOOT_OFF)
        case _:
            raise ValueError(f"Unsupported method: {method}")

    trial_cycles = method_obj.segment(trial)
    return trial_cycles


def time_normalise_trial(
    trial: model.Trial | model.TrialCycles, method: str = "linear", **kwargs
) -> model.Trial | model.TrialCycles:
    """Normalises the time in the trial.

    Args:
        trial: The trial to normalise the time for.
        method: The method to use for normalising the time. Currently, only supports
                "linear" which normalises the time linearly. Default is "linear".
        **kwargs:
            - n_frames: The number of frames to normalise the data to.

    Returns:
        The trial with the normalised time.
    """
    match method:
        case "linear":
            normaliser = normalisation.LinearTimeNormaliser(**kwargs)
        case _:
            raise ValueError(f"Unsupported method: {method}")

    trial = normaliser.normalise(trial)
    return trial


def calculate_features(
    trial: model.TrialCycles,
    config: mapping.MappingConfigs,
    methods: list | tuple = (
        features.TimeSeriesFeatures,
        features.PhaseTimeSeriesFeatures,
        features.TemporalFeatures,
        features.SpatialFeatures,
    ),
    **kwargs,
) -> xr.DataArray:
    """Calculates the features of the trial.

    Args:
        trial: The trial to calculate the features for.
        config: The mapping configurations
        methods: Class objects of the feature calculation methods to use.
        **kwargs: Currently not used.

    Returns:
        The trial with the calculated features.
    """
    method_objs = _create_feature_methods(methods, config, **kwargs)
    feature_list = []
    for method in method_objs:
        feature_list.append(method.calculate(trial))
    feature_array = xr.concat(feature_list, dim="feature")
    return feature_array


def _create_feature_methods(
    methods: list[type] | tuple[type], config: mapping.MappingConfigs, **kwargs
) -> list[features.FeatureCalculation]:
    """Checks the feature calculation methods.

    Args:
        methods: The list of feature calculation methods to use.
        config: The mapping configurations
        **kwargs: Currently not used.

    Returns:
        A list of the feature calculation method objects.
    """
    method_objects = []
    for method in methods:
        if not issubclass(method, features.FeatureCalculation):
            raise ValueError(f"Unsupported method: {method}")
        method_objects.append(method(config, **kwargs))
    return method_objects


@_PathConverter
def export_trial(
    trial: model.Trial | model.TrialCycles,
    output_path: Path | str,
    method: str = "netcdf",
):
    """Exports the trial to a c3d file.

    This function will create a folder and save the trial as NetCDF files.
    Depending on the trial following files will be written:

    - markers.nc
    - analogs.nc
    - analysis.nc
    - events.nc

    Args:
        trial: The trial to export.
        output_path: The path to write the c3d file.
        method: The method to use for exporting the trial.
                Currently, only supports "netcdf".
    """
    match method:
        case "netcdf":
            io.NetCDFTrialExporter(output_path).export_trial(trial)  # type: ignore
        case _:
            raise ValueError(f"Unsupported method: {method}")
