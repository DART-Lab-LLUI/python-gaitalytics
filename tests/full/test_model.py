import shutil
from pathlib import Path

import h5netcdf as netcdf
import pytest
import xarray as xr

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, \
    C3dEventInputFileReader, AnalysisInputReader
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import DataCategory, Trial, TrialCycles, trial_from_hdf5
from gaitalytics.segmentation import GaitEventsSegmentation

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')

INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')
OUTPUT_PATH_BIG: Path = Path('out/test_big')

CONFIG_FILE: Path = Path('./tests/full/config/pig_config.yaml')


@pytest.fixture()
def trial_small(request):
    configs = MappingConfigs(CONFIG_FILE)
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    analysis = AnalysisInputReader(INPUT_C3D_SMALL, configs).get_analysis()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.add_data(DataCategory.ANALYSIS, analysis)
    trial.events = events
    return trial


@pytest.fixture()
def output_file_path_small(request):
    file_path = OUTPUT_PATH_SMALL.with_suffix('.hdf5')

    def delete_file():
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                pass

    delete_file()
    return file_path


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
def trial_big(request):
    configs = MappingConfigs(CONFIG_FILE)
    markers = MarkersInputFileReader(INPUT_C3D_BIG).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_BIG).get_analogs()
    analysis = AnalysisInputReader(INPUT_C3D_BIG, configs).get_analysis()
    events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.add_data(DataCategory.ANALYSIS, analysis)
    trial.events = events
    return trial


@pytest.fixture()
def output_file_path_big(request):
    file_path = OUTPUT_PATH_BIG.with_suffix('.hdf5')

    def delete_file():
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                pass

    delete_file()
    return file_path


@pytest.fixture()
def output_path_big(request):
    path = OUTPUT_PATH_BIG

    def delete_file():
        if path.exists():
            try:
                shutil.rmtree(path)
            except PermissionError:
                pass

    delete_file()
    return path


class TestTrial:

    def test_add(self):
        markers_input = MarkersInputFileReader(INPUT_C3D_SMALL)
        markers = markers_input.get_markers()
        new_markers = markers.copy(deep=True)
        labels = [f"{label}_new" for label in new_markers["channel"].values]
        new_markers = new_markers.assign_coords(channel=labels)

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.MARKERS, new_markers)

        rec_value = len(trial.get_data(DataCategory.MARKERS).coords["channel"])
        exp_value = 382

        assert rec_value == exp_value

    def test(self, trial_small):
        rec_value = len(trial_small.get_all_data())
        exp_value = 3
        assert rec_value == exp_value

    def test_save_empty_to_hdf5(self, output_file_path_small):
        trial = Trial()
        with pytest.raises(ValueError):
            trial.to_hdf5(output_file_path_small)

        assert not output_file_path_small.exists()

    def test_save_to_existing_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)
        with pytest.raises(FileExistsError):
            trial_small.to_hdf5(output_file_path_small)

    def test_save_to_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)

        assert output_file_path_small.exists()

        with netcdf.File(output_file_path_small, 'r') as f:
            rec_value = len(f.groups.keys())
            exp_value = 4
            assert rec_value == exp_value

    def test_load_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)

        loaded_trial = trial_from_hdf5(output_file_path_small)

        rec_value = len(loaded_trial.get_all_data())
        exp_value = 3
        assert rec_value == exp_value

        assert loaded_trial.events is not None
        del loaded_trial

    def test_save_to_hdf5_big(self, trial_big, output_file_path_big):
        trial_big.to_hdf5(output_file_path_big)

        assert output_file_path_big.exists()

        with netcdf.File(output_file_path_big, 'r') as f:
            rec_value = len(f.groups.keys())
            exp_value = 4
            assert rec_value == exp_value

    def test_load_hdf5_big(self, trial_big, output_file_path_big):
        trial_big.to_hdf5(output_file_path_big)

        loaded_trial = trial_from_hdf5(output_file_path_big)

        rec_value = len(loaded_trial.get_all_data())
        exp_value = 3
        assert rec_value == exp_value

        assert loaded_trial.events is not None
        del loaded_trial

    def test_save_to_folder(self, trial_small, output_path_small):
        with pytest.raises(ValueError):
            trial_small.to_hdf5(output_path_small)


class TestSegmentedTrial:
    def test_empy(self, output_path_small):
        trial = TrialCycles()
        with pytest.raises(ValueError):
            trial.to_hdf5(output_path_small)

        assert not output_path_small.exists()

    def test_to_hdf5_small(self, trial_small, output_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        segments.to_hdf5(output_path_small)

        assert output_path_small.exists()

    def test_to_write_cycle(self, trial_small, output_file_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        trial: xr.DataArray = segments.get_cycle("Left", 0).get_data(
            DataCategory.MARKERS)
        trial.to_netcdf(output_file_path_small, group="Left/0/markers", engine="h5netcdf")

        assert output_file_path_small.exists()
    def test_to_hdf5_big(self, trial_big, output_path_big):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_big)
        segments.to_hdf5(output_path_big)

        assert output_path_big.exists()

    def test_load_hdf5_small(self, trial_small, output_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        segments.to_hdf5(output_path_small)
        trial = trial_from_hdf5(output_path_small)
        assert output_path_small.exists()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            trial_from_hdf5(Path("foo.hdf5"))

    def test_save_segment_in_file(self, trial_small, output_file_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        with pytest.raises(ValueError):
            segments.to_hdf5(output_file_path_small)

    def test_segment_events(self, trial_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        for context in segments.get_all_cycles().keys():
            for id, cycle in segments.get_cycles_per_context(context).items():
                event_times = cycle.events["time"].values
                markers = cycle.get_data(DataCategory.MARKERS)

                for event_time in event_times:
                    exp_value = markers.coords["time"][0]
                    rec_value = round(event_time,2)
                    assert rec_value >= exp_value

                    exp_value = markers.coords["time"][-1]
                    rec_value = round(event_time, 2)
                    assert rec_value <= exp_value
