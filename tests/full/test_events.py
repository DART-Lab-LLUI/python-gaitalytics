from pathlib import Path

import pytest

from gaitalytics.events import SequenceEventChecker, MarkerEventDetection
from gaitalytics.io import C3dEventInputFileReader, MarkersInputFileReader
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import DataCategory, Trial

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')
CONFIG_FILE: Path = Path('./tests/full/config/pig_config.yaml')

INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')
OUTPUT_PATH_BIG: Path = Path('out/test_big')


@pytest.fixture()
def trial_small(request):
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()
    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.events = events
    return trial


@pytest.fixture()
def trial_big(request):
    markers = MarkersInputFileReader(INPUT_C3D_BIG).get_markers()
    events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()
    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.events = events
    return trial


@pytest.fixture()
def config(request):
    return MappingConfigs(CONFIG_FILE)


class TestEventSequenceChecker:

    def test_sequence_small(self):
        events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()
        checker = SequenceEventChecker()
        good, errors = checker.check_events(events)
        assert good, f"Event sequence is not correct but it should be. {errors}"

    def test_sequence_big(self):
        events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()
        checker = SequenceEventChecker()
        good, _ = checker.check_events(events)
        assert good, "Event sequence is not correct but it should be."

    def test_sequence_empty(self):
        checker = SequenceEventChecker()
        with pytest.raises(ValueError):
            checker.check_events(None)


class TestMarkerEventDetection:

    def test_small(self, trial_small, config):
        pred_events = MarkerEventDetection(config).detect_events(trial_small)
        events = trial_small.events
        events = events.drop(0).reset_index(drop=True)  # can't detect first event
        rec_value = len(pred_events)
        exp_value = len(events)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        pred_events = MarkerEventDetection(config).detect_events(trial_big)
        events = trial_big.events

        rec_value = len(pred_events)
        exp_value = len(events)
        assert rec_value == exp_value

        for i in range(len(pred_events)):
            rec_value = pred_events.iloc[i].loc['time'] - events.iloc[i].loc['time']
            exp_value = 2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events.iloc[i].loc['icon_id']
            assert rec_value == exp_value
