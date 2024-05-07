from __future__ import annotations

import logging
import pathlib
import sys
from pathlib import Path

import numpy as np

import gaitalytics.analysis as analysis
import gaitalytics.c3d_reader as c3d_reader
import gaitalytics.cycle as cycle
import gaitalytics.cycle_normalisation as cycle_normalisation
import gaitalytics.events as events
import gaitalytics.file as file
import gaitalytics.model as model
import gaitalytics.modelling as modelling
import gaitalytics.utils as utils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# constants
CYCLE_METHOD_HEEL_STRIKE = "HS"
CYCLE_METHOD_TOE_OFF = "TO"

NORMALISE_METHODE_LINEAR = "linear"
NORMALISE_METHODE_LIST = NORMALISE_METHODE_LINEAR

GAIT_EVENT_METHODE_MARKER = "Marker"
GAIT_EVENT_METHODE_FP = "Forceplate"
GAIT_EVENT_METHODE_LIST = (GAIT_EVENT_METHODE_MARKER, GAIT_EVENT_METHODE_FP)

GAIT_EVENT_CHECKER_CONTEXT = "context"
GAIT_EVENT_CHECKER_SPACING = "spacing"
GAIT_EVENT_CHECKER_LIST = (GAIT_EVENT_CHECKER_CONTEXT, GAIT_EVENT_CHECKER_SPACING)

ANALYSIS_TIMESERIES = analysis.TimeseriesAnalysis
ANALYSIS_TOE_CLEARANCE = analysis.MinimalToeClearance
ANALYSIS_SPATIO_TEMP = analysis.SpatioTemporalAnalysis
ANALYSIS_CMOS = analysis.CMosAnalysis
ANALYSIS_MOS = analysis.MosAnalysis
ANALYSIS_LIST = (
    ANALYSIS_TIMESERIES,
    ANALYSIS_SPATIO_TEMP,
    ANALYSIS_TOE_CLEARANCE,
    ANALYSIS_CMOS,
    ANALYSIS_MOS,
)

MODELLING_COM = "com"
MODELLING_CMOS = "cmos"
MODELLING_XCOM = "xcom"
MODELLING_LIST = [MODELLING_COM, MODELLING_CMOS, MODELLING_XCOM]


def detect_gait_events(
    c3d_file_path: str | Path,
    output_path: str | Path,
    configs: utils.ConfigProvider,
    methode: str = GAIT_EVENT_METHODE_MARKER,
    anomaly_checker: list[str] = GAIT_EVENT_CHECKER_LIST,
    file_handler_class: type[c3d_reader.FileHandler] = c3d_reader.BtkFileHandler,
    **kwargs,
):
    """
    Adds gait events to c3d file and saves it in output_path with a '.4.c3d' extension. Checks events aditionally
    with given anomaly_checker method and saves it in output_path with '*_anomaly.txt' extension

    :param file_handler_class:
    :param c3d_file_path: path of c3d file with modelled filtered data '.3.c3d'
    :param output_path: path to dir to store c3d file with events
    :param configs: configs from marker and model mapping
    :param methode: methode to detect events 'Marker' api.GAIT_EVENT_METHODE_MARKER or
        'Force plate' api.GAIT_EVENT_METHODE_FP
    :param anomaly_checker: list of anomaly checkers, "context" api.GAIT_EVENT_CHECKER_CONTEXT,
        "spacing" api.GAIT_EVENT_CHECKER_SPACING
    """
    logger.info("detect_gait_events")
    c3d_file_path_obj = _create_path_object(c3d_file_path)
    out_path_obj = _create_path_object(output_path)

    if methode not in GAIT_EVENT_METHODE_LIST:
        raise KeyError(f"{methode} is not a valid methode")
    if not all(item in GAIT_EVENT_CHECKER_LIST for item in anomaly_checker):
        raise KeyError(f"{anomaly_checker} are not a valid anomaly checker")

    # read c3d
    motion_file = file_handler_class(c3d_file_path_obj)

    if methode == GAIT_EVENT_METHODE_FP:
        methode = events.ForcePlateEventDetection()
    elif methode == GAIT_EVENT_METHODE_MARKER:
        methode = events.ZenisGaitEventDetector(configs, **kwargs)

    methode.detect_events(motion_file, **kwargs)

    # define output name
    filename = c3d_file_path_obj.name.replace(".3.c3d", ".4.c3d")
    output_path_obj = out_path_obj / filename

    # write events c3d
    motion_file.write_file(output_path_obj)

    check_gait_event(output_path_obj, out_path_obj, file_handler_class=file_handler_class)


def check_gait_event(
    c3d_file_path: str | Path,
    output_path: str | Path,
    anomaly_checker: list[str] = GAIT_EVENT_CHECKER_LIST,
    file_handler_class: type[c3d_reader.FileHandler] = c3d_reader.BtkFileHandler,
):
    """
    Checks events additionally
    with given anomaly_checker method and saves it in output_path with '*_anomaly.txt' extension

    :param file_handler_class:
    :param c3d_file_path: path of c3d file with modelled filtered data '.3.c3d'
    :param output_path: path to dir to store c3d file with events
    :param anomaly_checker: list of anomaly checkers, "context" api.GAIT_EVENT_CHECKER_CONTEXT,
       "spacing" api.GAIT_EVENT_CHECKER_SPACING
    """
    logger.info("check_gait_event")

    c3d_file_path_obj = _create_path_object(c3d_file_path)
    output_path_obj = _create_path_object(output_path)

    if not all(item in GAIT_EVENT_CHECKER_LIST for item in anomaly_checker):
        raise KeyError(f"{anomaly_checker} are not a valid anomaly checker")
    # read c3d
    motion_file = file_handler_class(c3d_file_path_obj)
    # get anomaly detection
    checker = _get_anomaly_checker(anomaly_checker)
    detected, anomalies = checker.check_events(motion_file)

    # write anomalies to file
    if detected:
        filename = c3d_file_path_obj.name.replace(".4.c3d", "_anomalies.txt")
        out_path = output_path_obj / filename
        f = out_path.open("w")
        for anomaly in anomalies:
            print(anomaly, file=f)
        f.close()


def model_data(
    c3d_file_path: str | pathlib.Path,
    output_path: str,
    configs: utils.ConfigProvider,
    methode: str,
    file_handler_class: type[c3d_reader.FileHandler] = c3d_reader.BtkFileHandler,
    **kwargs,
):
    """
    Models data according to chosen method and saves new c3d file in output path with '.5.c3d extension'

    :param file_handler_class:
    :param c3d_file_path: path of c3d file with modelled filtered data '.3.c3d'
    :param output_path: path to dir to store c3d file with events
    :param configs: configs from marker and model mapping
    :param methode: methode to detect events
    :keyword belt_speed: belt speed
    """
    logger.info("model_data")

    output_path_obj = _create_path_object(output_path)
    c3d_file_path_obj = _create_path_object(c3d_file_path)
    motion_file = file_handler_class(c3d_file_path_obj)

    if methode not in MODELLING_LIST:
        raise KeyError(f"{methode} is not a valid modelling methode")

    methods: list[modelling.BaseOutputModeller] = []

    if methode == MODELLING_CMOS:
        methods.append(modelling.COMModeller(configs))
        methods.append(modelling.XCOMModeller(configs))
        methods.append(modelling.CMoSModeller(configs, **kwargs))

    elif methode == MODELLING_XCOM:
        methods.append(modelling.COMModeller(configs))
        methods.append(modelling.XCOMModeller(configs))
    elif methode == MODELLING_COM:
        methods.append(modelling.COMModeller(configs))

    for methode in methods:
        methode.create_point(motion_file, **kwargs)

    filename = c3d_file_path_obj.name.replace(".4.c3d", ".5.c3d")
    output_path_obj = output_path_obj / filename
    motion_file.write_file(output_path_obj)


def extract_cycles(
    c3d_file_path: str | Path,
    configs: utils.ConfigProvider,
    buffer_output_path: str | Path,
    methode: str = CYCLE_METHOD_HEEL_STRIKE,
    file_handler_class: type[c3d_reader.FileHandler] = c3d_reader.BtkFileHandler,
) -> model.ExtractedCycles:
    """
    extracts and returns cycles from c3d. If a buffered path is delivered data will be stored in the path in separated
    csv file. Do not edit files and structure.

    :param file_handler_class: Define class for handling c3d files
    :param c3d_file_path: path of c3d file with foot_off and foot_strike events '*.4.c3d'
    :param configs: configs from marker and model mapping
    :param methode: method to cut gait cycles either "HS" api.CYCLE_METHOD_HEEL_STRIKE or "TO" api.CYCLE_METHOD_TOE_OFF
    :param buffer_output_path: if buffering needed path to folder
    :param anomaly_checker: list of anomaly checkers, "context" api.GAIT_EVENT_CHECKER_CONTEXT,
        "spacing" api.GAIT_EVENT_CHECKER_SPACING
    :return: extracted gait cycles
    """
    logger.info("extract_cycles")
    # check params for validity
    c3d_file_path_obj = Path(c3d_file_path)
    buffer_output_path_obj = Path(buffer_output_path)

    if methode not in [CYCLE_METHOD_HEEL_STRIKE, CYCLE_METHOD_TOE_OFF]:
        raise KeyError(f"{methode} is not a valid methode")

    # read c3d
    motion_file = file_handler_class(c3d_file_path_obj)

    # get anomaly detection

    # choose cut method
    cycle_builder = None
    if methode == CYCLE_METHOD_TOE_OFF:
        cycle_builder = cycle.ToeOffToToeOffCycleBuilder()
    elif methode == CYCLE_METHOD_HEEL_STRIKE:
        cycle_builder = cycle.HeelStrikeToHeelStrikeCycleBuilder()

    # get cycles
    cycles = cycle_builder.build_cycles(motion_file)

    # extract cycles
    cycle_data = cycle.extract_point_cycles(configs, cycles, motion_file)

    # buffer cycles
    out_file = file.Hdf5FileStore(buffer_output_path_obj, configs)
    out_file.save_extracted_cycles(cycle_data)

    return cycle_data


def extract_cycles_buffered(
    buffer_output_path: str | Path, configs: utils.ConfigProvider
) -> dict[model.ExtractedCycleDataCondition, model.ExtractedCycles]:
    """
    gets normalised and full length data from buffered folder. It is needed to run api.extract_cycles as
    api.normalise_cycles with given buffer_output_path once to use this function

    :param buffer_output_path: path to folder
    :param configs: configs from marker and model mapping
    :return: object containing normalised and full length data lists
    """
    logger.info("extract_cycles_buffered")
    buffer_output_path_obj = _create_path_object(buffer_output_path)

    # load cycles
    file_handler = file.Hdf5FileStore(buffer_output_path_obj, configs)

    return file_handler.read_extracted_cycles()


def normalise_cycles(
    configs: utils.ConfigProvider,
    cycle_data: model.ExtractedCycles,
    method: str = NORMALISE_METHODE_LINEAR,
    buffer_output_path: str | Path | None = None,
) -> model.ExtractedCycles:
    """
    normalise and returns cycles

    :param configs: configs from marker and model mapping
    :param cycle_data: full length cycle data
    :param method: method normalise "linear" api.NORMALISE_METHODE_LINEAR
    :param buffer_output_path: if buffering needed path to folder
    :return: normalised gait cycles
    """
    logger.info("normalise_cycles")
    # check params for validity
    if method not in NORMALISE_METHODE_LIST:
        raise KeyError(f"{method} is not a valid methode")

    buffer_output_path_obj = _create_path_object(buffer_output_path)

    # get method
    if method == NORMALISE_METHODE_LINEAR:
        method = cycle_normalisation.LinearTimeNormalisation()

    # normalise
    normalised_data = method.normalise(cycle_data)

    # buffer cycles
    out_file = file.Hdf5FileStore(buffer_output_path_obj, configs)
    out_file.save_extracted_cycles(normalised_data)

    return normalised_data


def analyse_data(
    cycle_data: dict[model.ExtractedCycleDataCondition : model.ExtractedCycles],
    config: utils.ConfigProvider,
    methods: list[type[analysis.AbstractAnalysis]] = ANALYSIS_LIST,
    **kwargs: dict,
) -> dict[str, np.ndarray]:
    """
    Runs specified analysis and concatenates into one Dataframe

    :param cycle_data: full length cycle data
    :param config: configs from marker and model mapping
    :param methods: list of methods classes
    :return: results of analysis
    """
    logger.info("analyse_data")

    results = None
    for methode_cls in methods:
        methode = methode_cls(cycle_data, config)
        result = methode.analyse(**kwargs)
        if results is None:
            results = result
        else:
            results.update(result)

    return results


def _create_path_object(file_path: str | Path) -> Path:
    logger.debug("_create_path_object")
    return Path(file_path)


def _get_anomaly_checker(anomaly_checker: list[str]) -> events.AbstractEventAnomalyChecker:
    """
    defines checker by list of inputs

    :param anomaly_checker: list of checker name
    :return: checker object
    """
    logger.info("_get_anomaly_checker")
    checker = None
    if GAIT_EVENT_CHECKER_CONTEXT in anomaly_checker:
        checker = events.ContextPatternChecker()
    if GAIT_EVENT_CHECKER_SPACING in anomaly_checker:
        checker = events.EventSpacingChecker(checker)
    return checker
