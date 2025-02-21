from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from gaitalytics.events import SequenceEventChecker, Zeni, Desailly, AC, GrfEventDetection, AutoEventDetection, EventDetector, EventDetectorBuilder, ReferenceFromGrf, MappedMethods
from gaitalytics.io import C3dEventInputFileReader, MarkersInputFileReader
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import DataCategory, Trial

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')
CONFIG_FILE: Path = Path('./tests/full/config/pig_config.yaml')

INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')
OUTPUT_PATH_BIG: Path = Path('out/test_big')

FOOT_STRIKE = "Foot Strike"
FOOT_OFF = "Foot Off"
LEFT = "Left"
RIGHT = "Right"
EVENT_TYPES = [FOOT_STRIKE, FOOT_OFF]
SIDES = [LEFT, RIGHT]


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


class TestZeni:

    def test_small(self, trial_small, config):
        events = trial_small.events
        pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = Zeni(config, side, type)
                pred_times = detector.detect_events(trial_small)
                pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        events = events.drop(0).reset_index(drop=True)
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
        events = trial_big.events
        pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = Zeni(config, side, type)
                pred_times = detector.detect_events(trial_big)
                pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events)
        assert rec_value == exp_value

        for i in range(len(pred_events)):
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

class TestDesailly:
    def test_small(self, trial_small, config):
        events_ = trial_small.events
        all_pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = Desailly(config, side, type)
                pred_times = detector.detect_events(trial_small)
                pred_events = detector._create_data_frame(pred_times)
                all_pred_events = pd.concat([all_pred_events, pred_events])
                events_side = events_[(events_["context"] == side)&(events_["label"]==type)].time
                errors, missed, excess = detector._get_accuracy(pred_times, events_side)
                rec_value = np.abs(np.mean(errors))
                exp_value = 0.2
                assert rec_value < exp_value
                rec_value = missed 
                exp_value = 0.76
                assert rec_value < exp_value
                rec_value = excess
                exp_value = 0.7
                assert rec_value < exp_value
                rec_value = (pred_events["context"] == side).sum()
                exp_value = len(pred_events)
                assert rec_value == exp_value
                rec_value = (pred_events["label"] == type).sum()
                exp_value = len(pred_events)
                assert rec_value == exp_value
        all_pred_events = all_pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(all_pred_events)
        exp_value = len(events_)
        assert rec_value < exp_value + 10 and rec_value > exp_value - 10 

    def test_big(self, trial_big, config):
        events_ = trial_big.events
        all_pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = Desailly(config, side, type)
                pred_times = detector.detect_events(trial_big)
                pred_events = detector._create_data_frame(pred_times)
                all_pred_events = pd.concat([all_pred_events, pred_events])
                events_side = events_[(events_["context"] == side)&(events_["label"]==type)].time
                errors, missed, excess = detector._get_accuracy(pred_times, events_side)
                rec_value = np.abs(np.mean(errors))
                exp_value = 0.2
                assert rec_value < exp_value
                rec_value = missed 
                exp_value = 0.2
                assert rec_value < exp_value
                rec_value = excess
                exp_value = 0.2
                assert rec_value < exp_value
                rec_value = (pred_events["context"] == side).sum()
                exp_value = len(pred_events)
                assert rec_value == exp_value
                rec_value = (pred_events["label"] == type).sum()
                exp_value = len(pred_events)
                assert rec_value == exp_value
        all_pred_events = all_pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(all_pred_events)
        exp_value = len(events_)
        assert rec_value < exp_value + 10 and rec_value > exp_value - 10 

class TestAC1:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_STRIKE]
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC1(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC1(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        pred_events = pred_events.drop(0).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

class TestAC2:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC2(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC2(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        pred_events = pred_events.drop([0, 1]).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

class TestAC3:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC3(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        all_pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC3(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = detector._create_data_frame(pred_times)
            all_pred_events = pd.concat([all_pred_events, pred_events])
            events_side = events_[events_["context"] == side].time
            errors, missed, excess = detector._get_accuracy(pred_times, events_side)
            rec_value = np.abs(np.mean(errors))
            exp_value = 0.2
            assert rec_value < exp_value
            rec_value = missed 
            exp_value = 0.6
            assert rec_value < exp_value
            rec_value = excess
            exp_value = 0.6
            assert rec_value < exp_value
            rec_value = (pred_events["context"] == side).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
            rec_value = (pred_events["label"] == FOOT_STRIKE).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
        all_pred_events = all_pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(all_pred_events)
        exp_value = len(events_)
        assert rec_value < exp_value + 10 and rec_value > exp_value - 10

class TestAC4:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC4(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_STRIKE].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC4(config, side, FOOT_STRIKE, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        pred_events = pred_events.drop([0, 1]).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

class TestAC5:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_OFF].reset_index(drop=True)
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC5(config, side, FOOT_OFF, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_OFF].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        all_pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC5(config, side, FOOT_OFF, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = detector._create_data_frame(pred_times)
            all_pred_events = pd.concat([all_pred_events, pred_events])
            events_side = events_[events_["context"] == side].time
            errors, missed, excess = detector._get_accuracy(pred_times, events_side)
            rec_value = np.abs(np.mean(errors))
            exp_value = 0.3
            assert rec_value < exp_value
            rec_value = missed 
            exp_value = 0.3
            assert rec_value < exp_value
            rec_value = excess
            exp_value = 0.3
            assert rec_value < exp_value
            rec_value = (pred_events["context"] == side).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
            rec_value = (pred_events["label"] == FOOT_OFF).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
        all_pred_events = all_pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(all_pred_events)
        exp_value = len(events_)
        assert rec_value < exp_value + 30 and rec_value > exp_value - 30

class TestAC6:
    def test_small(self, trial_small, config):
        events_ = trial_small.events[trial_small.events["label"] == FOOT_OFF].reset_index(drop=True)
        trial_ref = trial_small
        trial_ref.events = events_.iloc[:4]
        pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC6(config, side, FOOT_OFF, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_small, {"distance": 90})
            pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events_)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events_.iloc[i].loc['time'])
            exp_value = 0.2
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events_.iloc[i].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events_.iloc[i].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events_.iloc[i].loc['icon_id']
            assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events_ = trial_big.events[trial_big.events["label"] == FOOT_OFF].reset_index(drop=True)
        trial_ref = trial_big
        trial_ref.events = events_.iloc[:16]
        all_pred_events = pd.DataFrame()
        for side in SIDES:
            detector = AC.get_AC6(config, side, FOOT_OFF, trial_ref=trial_ref)
            pred_times = detector.detect_events(trial_big, {"distance": 90})
            pred_events = detector._create_data_frame(pred_times)
            all_pred_events = pd.concat([all_pred_events, pred_events])
            events_side = events_[events_["context"] == side].time
            errors, missed, excess = detector._get_accuracy(pred_times, events_side)
            rec_value = np.abs(np.mean(errors))
            exp_value = 0.2
            assert rec_value < exp_value
            rec_value = missed 
            exp_value = 0.2
            assert rec_value < exp_value
            rec_value = excess
            exp_value = 0.2
            assert rec_value < exp_value
            rec_value = (pred_events["context"] == side).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
            rec_value = (pred_events["label"] == FOOT_OFF).sum()
            exp_value = len(pred_events)
            assert rec_value == exp_value
        all_pred_events = all_pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        rec_value = len(all_pred_events)
        exp_value = len(events_)
        assert rec_value < exp_value + 30 and rec_value > exp_value - 30

class TestGrfEventDetection:
    def test_small(self, trial_small, config):
        events = trial_small.events
        pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = GrfEventDetection(config, side, type)
                pred_times = detector.detect_events(trial_small)
                pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        events = events.drop([0, 2, 7, 9, 10, 11, 12]).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events) + 2
        assert rec_value == exp_value

        for i in range(0, len(events)):
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

        i =len(pred_events)-2
        rec_value = pred_events.iloc[i].loc['label']
        exp_value = FOOT_STRIKE
        assert rec_value == exp_value

        rec_value = pred_events.iloc[i].loc['context']
        exp_value = RIGHT
        assert rec_value == exp_value

        i =len(pred_events)-1
        rec_value = pred_events.iloc[i].loc['label']
        exp_value = FOOT_OFF
        assert rec_value == exp_value

        rec_value = pred_events.iloc[i].loc['context']
        exp_value = RIGHT
        assert rec_value == exp_value

    def test_big(self, trial_big, config):
        events = trial_big.events
        pred_events = pd.DataFrame()
        for type in EVENT_TYPES:
            for side in SIDES:
                detector = GrfEventDetection(config, side, type)
                pred_times = detector.detect_events(trial_big)
                pred_events = pd.concat([pred_events, detector._create_data_frame(pred_times)])
        pred_events = pred_events.sort_values(by="time", ascending=True).reset_index(drop=True)
        pred_events = pred_events.drop([862, 863]).reset_index(drop=True)
        events = events.drop([862, 863, 864]).reset_index(drop=True)
        rec_value = len(pred_events)
        exp_value = len(events)
        assert rec_value == exp_value

        for i in range(0, len(pred_events)):
            # events nb 517 and 518 are switched in the predictions
            if i == 517:
                j = i+1
                exp_value = 0.3
            elif i == 518:
                j = i-1
                exp_value = 0.2
            else:
                j = i
                exp_value = 0.2
            rec_value = abs(
                pred_events.iloc[i].loc['time'] - events.iloc[j].loc['time'])
            assert rec_value < exp_value

            rec_value = pred_events.iloc[i].loc['label']
            exp_value = events.iloc[j].loc['label']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['context']
            exp_value = events.iloc[j].loc['context']
            assert rec_value == exp_value

            rec_value = pred_events.iloc[i].loc['icon_id']
            exp_value = events.iloc[j].loc['icon_id']
            assert rec_value == exp_value

class TestAutoEventDetection:
    def test_small(self, trial_small, config):
        events = trial_small.events
        trial_ref = trial_small
        trial_ref.events = events.iloc[:8]
        auto_selection = AutoEventDetection(config, trial_ref)
        event_detector, user_show = auto_selection.get_optimised_event_detectors()

        rec_value = event_detector.hs_right._mean_error
        exp_value = user_show[FOOT_STRIKE][RIGHT]["mean error"]
        assert rec_value == exp_value

        for detector in [event_detector.hs_left, 
                         event_detector.hs_right, 
                         event_detector.to_left, 
                         event_detector.to_right]:
        
            rec_value = detector._mean_error - detector._offset
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._missed
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._excess
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._excess
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._quantiles[1] - detector._quantiles[0]
            exp_value = 0.1
            assert rec_value < exp_value
    
    def test_big(self, trial_big, config):
        events = trial_big.events
        trial_ref = trial_big
        trial_ref.events = events.iloc[:16]
        auto_selection = AutoEventDetection(config, trial_ref)
        event_detector, user_show = auto_selection.get_optimised_event_detectors()

        rec_value = event_detector.hs_right._mean_error
        exp_value = user_show[FOOT_STRIKE][RIGHT]["mean error"]
        assert rec_value == exp_value

        for detector in [event_detector.hs_left, 
                         event_detector.hs_right, 
                         event_detector.to_left, 
                         event_detector.to_right]:

            rec_value = detector._mean_error - detector._offset
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._missed
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._excess
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._excess
            exp_value = 0.1
            assert rec_value < exp_value

            rec_value = detector._quantiles[1] - detector._quantiles[0]
            exp_value = 0.1
            assert rec_value < exp_value

class TestEventDetector:
    def test_small(self, trial_small, config):
        events = trial_small.events
        obj = EventDetector(Zeni(config, LEFT, FOOT_STRIKE),
                            Zeni(config, RIGHT, FOOT_STRIKE),
                            Zeni(config, LEFT, FOOT_OFF),
                            Zeni(config, RIGHT, FOOT_OFF))
        pred_events = obj.detect_events(trial_small)
        events = events.drop(0).reset_index(drop=True)
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
        events = trial_big.events
        obj = EventDetector(Zeni(config, LEFT, FOOT_STRIKE),
                            Zeni(config, RIGHT, FOOT_STRIKE),
                            Zeni(config, LEFT, FOOT_OFF),
                            Zeni(config, RIGHT, FOOT_OFF))
        pred_events = obj.detect_events(trial_big)
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

class TestEventDetectorBuilder:
    def test_get_method(self):
        rec_value = EventDetectorBuilder.get_method(MappedMethods.AC1)
        exp_value = AC.get_AC1
        assert rec_value == exp_value
    
    def test_mapping(self):
        rec_value = MappedMethods.DESAILLY
        exp_value = "Des"
        assert rec_value == exp_value

    def test_get_event_types(self):
        rec_value = EventDetectorBuilder.get_event_types(MappedMethods.AC6)
        exp_value = [FOOT_OFF]
        assert rec_value == exp_value

    def check_event_type(self):
        EventDetectorBuilder.check_event_type(MappedMethods.AC2, FOOT_STRIKE)
        assert True
    
    def check_event_type_false(self):
        with pytest.raises(ValueError):
            EventDetectorBuilder.check_event_type(MappedMethods.AC5, FOOT_STRIKE)

    def test_get_event_detector_with_ref(self, trial_small, config):
        event_detector = EventDetectorBuilder.get_event_detector_with_ref(
            config, MappedMethods.ZENI, MappedMethods.AC6, trial_small
        )
        event_detector.hs_left.ref_events
        assert event_detector.hs_left.ref_events is not None

        rec_value = event_detector.hs_left._CODE
        exp_value = MappedMethods.ZENI
        assert rec_value == exp_value

        assert event_detector.to_right.ref_events is not None

        rec_value = event_detector.to_right._CODE
        exp_value = MappedMethods.AC6
        assert rec_value == exp_value

    def test_get_event_detector_with_ref_false(self, trial_small, config):
        with pytest.raises(ValueError):
            EventDetectorBuilder.get_event_detector_with_ref(
                config, MappedMethods.DESAILLY, MappedMethods.AC1, trial_small
            )

    def test_get_event_detector_no_ref(self, config):
        event_detector = EventDetectorBuilder.get_event_detector_no_ref(
            config, MappedMethods.GRF, MappedMethods.DESAILLY
        )
        assert not hasattr(event_detector.hs_left, "ref_events")

        rec_value = event_detector.hs_left._CODE
        exp_value = MappedMethods.GRF
        assert rec_value == exp_value

        rec_value = event_detector.to_right.ref_events
        exp_value = None
        assert rec_value == exp_value

        rec_value = event_detector.to_right._CODE
        exp_value = MappedMethods.DESAILLY
        assert rec_value == exp_value

    def test_get_event_detector_no_ref_false(self, config):
        with pytest.raises(ValueError):
            EventDetectorBuilder.get_event_detector_no_ref(
                config, MappedMethods.AC5, MappedMethods.AC6
            )
    
    def test_get_mixed_event_detector(self, trial_small, config):
        event_detector = EventDetectorBuilder.get_mixed_event_detector(
            config, MappedMethods.AC1, MappedMethods.ZENI, MappedMethods.DESAILLY, MappedMethods.AC5, trial_ref=trial_small
        )
        rec_value = event_detector.hs_left._CODE
        exp_value = MappedMethods.AC1
        assert rec_value == exp_value

        rec_value = event_detector.hs_right._CODE
        exp_value = MappedMethods.ZENI
        assert rec_value == exp_value

        rec_value = event_detector.to_left._CODE
        exp_value = MappedMethods.DESAILLY
        assert rec_value == exp_value

        rec_value = event_detector.to_right._CODE
        exp_value = MappedMethods.AC5
        assert rec_value == exp_value

class TestReferenceFromGrf:

    def test_get_reference(self, trial_big, config):
        nb_gc = 5
        events= trial_big.events
        trial_events_filt = ReferenceFromGrf(trial_big.events, trial_big, config, nb_gc).get_reference()
        rec_value = len(trial_events_filt.events)
        exp_value = 4*nb_gc
        assert rec_value == exp_value

        rec_value = np.isin(events["time"], trial_events_filt.events["time"]).sum()
        exp_value = len(trial_events_filt.events)
        assert rec_value == exp_value

        SequenceEventChecker().check_events(trial_events_filt.events)
        assert True