import shutil
from pathlib import Path

import pandas as pd
import pytest

import gaitalytics.api as api
import gaitalytics.mapping as mapping
import gaitalytics.model as model


@pytest.fixture()
def out_path(request):
    out = Path('out/test_small_events.c3d')

    if out.exists():
        out.unlink()
    elif not out.parent.exists():
        out.parent.mkdir(parents=True)
    return out


@pytest.fixture()
def out_folder(request):
    out = Path('out/test_small')

    if out.exists():
        shutil.rmtree(out)
    elif not out.exists():
        out.mkdir(parents=True)
    return out


def test_load_config():
    config = api.load_config("./tests/pig_config.yaml")
    marker_name = config.get_marker_mapping(mapping.MappedMarkers.SACRUM)
    assert marker_name == "SACR"


def test_load_c3d_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    assert len(trial.get_all_data().keys()) == 3
    assert trial.events is not None


def test_load_c3d_trial_no_events():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_no_events.c3d", config)
    assert trial.events is None

def test_get_event_detector_no_ref():
    config = api.load_config("./tests/pig_config.yaml")
    event_detector = api.get_event_detector("Zen", "Des", config)
    assert event_detector.hs_left._CODE == "Zen"
    assert event_detector.hs_right._CODE == "Zen"
    assert event_detector.to_left._CODE == "Des"
    assert event_detector.to_right._CODE == "Des"

def test_get_event_detector_no_ref_AC():
    config = api.load_config("./tests/pig_config.yaml")
    with pytest.raises(TypeError):
        api.get_event_detector("Zen", "AC6", config)

def test_get_event_detector_ref_GRF():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    with pytest.raises(TypeError):
        api.get_event_detector("GRF", "Zen", config, trial_ref=trial)

def test_get_event_detector_ref_method():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    with pytest.raises(ValueError):
        api.get_event_detector("AC6", "AC5", config, trial_ref=trial)
    
def test_get_event_detector_ref():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    event_detector = api.get_event_detector("Zen", "Zen", config, trial_ref = trial)
    assert event_detector.hs_left._CODE == "Zen"
    assert event_detector.hs_right._CODE == "Zen"
    assert event_detector.to_left._CODE == "Zen"
    assert event_detector.to_right._CODE == "Zen"
    assert event_detector.hs_left.trial_ref is not None
    assert event_detector.to_left.ref_events is not None

def test_get_mixed_event_detector_no_ref():
    config = api.load_config("./tests/pig_config.yaml")
    event_detector = api.get_mixed_event_detector(
                                method_hs_l = "Des",
                                method_hs_r = "Zen",
                                method_to_l = "Des",
                                method_to_r = "Zen",
                                configs = config
                                )
    assert event_detector.hs_left._CODE == "Des"
    assert event_detector.hs_right._CODE == "Zen"
    assert event_detector.to_left._CODE == "Des"
    assert event_detector.to_right._CODE == "Zen"

def test_get_mixed_event_detector_ref():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    event_detector = api.get_mixed_event_detector(
                                method_hs_l = "Des",
                                method_hs_r = "AC1",
                                method_to_l = "AC6",
                                method_to_r = "Zen",
                                configs = config,
                                trial= trial
                                )
    assert event_detector.hs_left._CODE == "Des"
    assert event_detector.hs_right._CODE == "AC1"
    assert event_detector.to_left._CODE == "AC6"
    assert event_detector.to_right._CODE == "Zen"
    assert event_detector.hs_left.trial_ref is not None
    assert event_detector.to_left.ref_events is not None

def find_optimal_detectors():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    obj, _ = find_optimal_detectors(trial, config, method_list = ["AC1", "AC6"])
    assert obj.hs_left._CODE == "AC1"
    assert obj.hs_right._CODE == "AC1"
    assert obj.to_left._CODE == "AC6"
    assert obj.to_right._CODE == "AC6"
    assert hasattr(obj.hs_left, "_mean_error")

def test_detect_events():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    event_detector = api.get_event_detector("Zen", "Zen", config)
    event_table = api.detect_events(trial, event_detector, parameters={"distance": 1000})
    assert event_table is not None
    assert len(event_table) == 5

def test_get_ref_from_GRF():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_no_events.c3d", config)
    trial_ref = api.get_ref_from_GRF(trial, config, gait_cycles_ref = 1)
    events = trial_ref.events
    assert len(events) == 4
    api.check_events(events)
    assert True

def test_check_events():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    api.check_events(trial.events)
    assert True


def test_check_events_methode():
    with pytest.raises(ValueError):
        api.check_events(pd.DataFrame(), method="foo")


def test_write_events(out_path):
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    event_detector = api.get_event_detector("Zen", "Zen", config)
    event_table = api.detect_events(trial, event_detector, parameters={"distance": 1000})
    api.write_events_to_c3d("./tests/treadmill_no_events.c3d", event_table, out_path)
    assert True


def test_segment_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    segm_trial = api.segment_trial(trial)
    assert len(segm_trial.get_all_cycles().keys()) == 2


def test_segment_trial_no_events():
    trial = model.Trial()
    with pytest.raises(ValueError):
        api.segment_trial(trial)


def test_segment_trial_method():
    trial = model.Trial()
    with pytest.raises(ValueError):
        api.segment_trial(trial, method="foo")


def test_time_normalize_trail():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    norm_trial = api.time_normalise_trial(trial)
    assert norm_trial is not None


def test_time_normalize_cycle_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    trial_cycles = api.segment_trial(trial)
    norm_trial = api.time_normalise_trial(trial_cycles)
    assert norm_trial is not None


def test_time_normalize_trial_frames():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    trial_cycles = api.segment_trial(trial)
    norm_trial = api.time_normalise_trial(trial_cycles, n_frames=200)
    markers = norm_trial.get_cycle("Left", 0).get_data(model.DataCategory.MARKERS)
    assert markers.shape[2] == 200


def test_calculate_features():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    trial_cycles = api.segment_trial(trial)
    features = api.calculate_features(trial_cycles, config)
    assert features.shape == (2, 10, 2281)


def test_export_trial(out_folder):
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    api.export_trial(trial, out_folder)
    assert (out_folder / "markers.nc").exists()
    assert (out_folder / "analogs.nc").exists()
    assert (out_folder / "analysis.nc").exists()
    assert (out_folder / "events.nc").exists()

def test_export_segmented_trial(out_folder):
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/treadmill_events.c3d", config)
    trial_cycles = api.segment_trial(trial)
    api.export_trial(trial_cycles, out_folder)
    assert (out_folder / "markers.nc").exists()
    assert (out_folder / "analogs.nc").exists()
    assert (out_folder / "analysis.nc").exists()
