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


def test_load_config():
    config = api.load_config("./tests/pig_config.yaml")
    marker_name = config.get_marker_mapping(mapping.MappedMarkers.SACRUM)
    assert marker_name == "SACR"


def test_load_c3d_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    assert len(trial.get_all_data().keys()) == 3
    assert trial.events is not None


def test_detect_events():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    event_table = api.detect_events(trial, config, distance=1000)
    assert event_table is not None
    assert len(event_table) == 4


def test_detect_events_methode():
    config = api.load_config("./tests/pig_config.yaml")
    trial = model.Trial()
    with pytest.raises(ValueError):
        api.detect_events(trial, config, method="ForcePlate")


def test_check_events():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    api.check_events(trial.events)
    assert True


def test_check_events_methode():
    with pytest.raises(ValueError):
        api.check_events(pd.DataFrame(), method="foo")


def test_write_events(out_path):
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    event_table = api.detect_events(trial, config, distance=1000)
    api.write_events_to_c3d("./tests/test_small.c3d", event_table, out_path)
    assert True


def test_segment_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
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
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    norm_trial = api.time_normalise_trial(trial)
    assert norm_trial is not None


def test_time_normalize_cycle_trial():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    trial_cycles = api.segment_trial(trial)
    norm_trial = api.time_normalise_trial(trial_cycles)
    assert norm_trial is not None


def test_time_normalize_trial_frames():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    trial_cycles = api.segment_trial(trial)
    norm_trial = api.time_normalise_trial(trial_cycles, n_frames=200)
    markers = norm_trial.get_cycle("Left", 0).get_data(model.DataCategory.MARKERS)
    assert markers.shape[2] == 200


def test_calculate_features():
    config = api.load_config("./tests/pig_config.yaml")
    trial = api.load_c3d_trial("./tests/test_small.c3d", config)
    trial_cycles = api.segment_trial(trial)
    features = api.calculate_features(trial_cycles, config)
    assert features.shape == (2, 2, 2278)
