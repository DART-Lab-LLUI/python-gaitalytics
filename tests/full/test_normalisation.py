import shutil
from pathlib import Path

import pytest

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, \
    C3dEventInputFileReader
from gaitalytics.model import DataCategory, Trial, trial_from_hdf5, TrialCycles
from gaitalytics.normalisation import LinearTimeNormaliser
from gaitalytics.segmentation import GaitEventsSegmentation

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')

INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')


@pytest.fixture()
def output_path_small(request):
    path = OUTPUT_PATH_SMALL

    def delete_file():
        if path.exists():
            try:
                shutil.rmtree(path)
            except PermissionError:
                pass

    delete_file()
    return path


@pytest.fixture()
def trial_small(request):
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


@pytest.fixture()
def trial_big(request):
    markers = MarkersInputFileReader(INPUT_C3D_BIG).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_BIG).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


class TestLinearTimeNormalisation:

    def test_normalisation_trial(self, trial_small):
        normaliser = LinearTimeNormaliser()
        normalised_trial = normaliser.normalise(trial_small)

        rec_value = len(
            normalised_trial.get_data(DataCategory.MARKERS).loc["x", "LHipAngles"])
        exp_value = 100
        assert rec_value == exp_value

        rec_value = len(
            normalised_trial.get_data(DataCategory.ANALOGS).loc["Force.Fx1"])
        exp_value = 100
        assert rec_value == exp_value

    def test_normalisation_segment_trial(self, trial_small):
        normaliser = LinearTimeNormaliser()
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)

        normalised_trial: TrialCycles = normaliser.normalise(segments)
        cycle = normalised_trial.get_cycle("Right", 0)
        hip = cycle.get_data(DataCategory.MARKERS).loc["x", "LHipAngles"]
        force = cycle.get_data(DataCategory.ANALOGS).loc["Force.Fx1"]

        rec_value = len(hip)
        exp_value = 100
        assert rec_value == exp_value

        rec_value = len(force)
        exp_value = 100
        assert rec_value == exp_value

    def test_normalisation_segment_trial_big(self, trial_big):
        normaliser = LinearTimeNormaliser()
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_big)

        normalised_trial = normaliser.normalise(segments)
        cycle = normalised_trial.get_cycle("Right", 0)
        hip = cycle.get_data(DataCategory.MARKERS).loc["x", "LHipAngles"]
        force = cycle.get_data(DataCategory.ANALOGS).loc["Force.Fx1"]

        rec_value = len(hip)
        exp_value = 100
        assert rec_value == exp_value

        rec_value = len(force)
        exp_value = 100
        assert rec_value == exp_value

    def test_normalisation_segment_trial_reload(self, trial_small, output_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        segments.to_hdf5(output_path_small)
        new_trial = trial_from_hdf5(output_path_small)

        normaliser = LinearTimeNormaliser()
        normalised_trial = normaliser.normalise(new_trial)

        cycle = normalised_trial.get_cycle("Right", 0)
        hip = cycle.get_data(DataCategory.MARKERS).loc["x", "LHipAngles"]
        force = cycle.get_data(DataCategory.ANALOGS).loc["Force.Fx1"]

        rec_value = len(hip)
        exp_value = 100
        assert rec_value == exp_value

        rec_value = len(force)
        exp_value = 100
        assert rec_value == exp_value
