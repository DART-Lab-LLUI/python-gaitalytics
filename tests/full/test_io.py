import array as arr
import shutil
from pathlib import Path

import numpy as np
import pytest

from gaitalytics.events import MarkerEventDetection
from gaitalytics.io import C3dEventInputFileReader, MarkersInputFileReader, \
    AnalogsInputFileReader, AnalysisInputReader, C3dEventFileWriter
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import Trial, DataCategory

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
INPUT_TRC_SMALL: Path = Path('./tests/full/data/test_small.trc')
INPUT_MOT_SMALL: Path = Path('./tests/full/data/test_small.mot')
INPUT_STO_SMALL: Path = Path('./tests/full/data/test_small.sto')

INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')
INPUT_C3D_BIG_NO_EVENTS: Path = Path('./tests/full/data/test_big_no_events.c3d')


@pytest.fixture()
def out_path(request):
    input_file = INPUT_C3D_BIG_NO_EVENTS
    out = Path('out/test_small_events.c3d')

    if out.exists():
        out.unlink()
    elif not out.parent.exists():
        out.parent.mkdir(parents=True)
    shutil.copy(input_file, out)
    return out


class TestWriteEvents:

    def test_write_c3d_events_small(self, out_path):
        configs = MappingConfigs(Path('./tests/full/config/pig_config.yaml'))
        markers = MarkersInputFileReader(out_path).get_markers()
        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        events = MarkerEventDetection(configs).detect_events(trial)
        C3dEventFileWriter(out_path).write_events(events)

        rec_events = C3dEventInputFileReader(out_path).get_events()

        assert len(rec_events) == len(events)
        for i in range(len(events)):
            assert rec_events['time'].iloc[i] == pytest.approx(events['time'].iloc[ i])
            assert rec_events['label'].iloc[i] == events['label'].iloc[i]
            assert rec_events['context'].iloc[i] == events['context'].iloc[i]
            assert rec_events['icon_id'].iloc[i] == events['icon_id'].iloc[i]


class TestReadEvents:
    def test_c3d_events_small(self):
        c3d_events = C3dEventInputFileReader(INPUT_C3D_SMALL)
        events = c3d_events.get_events()
        assert len(events) == 13
        assert events["time"].iloc[0] == pytest.approx(2.5)
        assert events["label"].iloc[0] == "Foot Off"
        assert events["context"].iloc[0] == "Right"
        assert events["icon_id"].iloc[0] == 2

    def test_c3d_events_big(self):
        c3d_events = C3dEventInputFileReader(INPUT_C3D_BIG)
        events = c3d_events.get_events()
        assert len(events) == 865

        # Test first event
        assert events["time"].iloc[0] == pytest.approx(1.03)
        assert events["label"].iloc[0] == "Foot Off"
        assert events["context"].iloc[0] == "Left"
        assert events["icon_id"].iloc[0] == 2
        # Test last event
        assert events["time"].iloc[-1] == pytest.approx(249.91)
        assert events["label"].iloc[-1] == "Foot Off"
        assert events["context"].iloc[-1] == "Left"
        assert events["icon_id"].iloc[-1] == 2


class TestMarkers:
    def test_c3d_markers_small(self):
        c3d_markers = MarkersInputFileReader(INPUT_C3D_SMALL)
        markers = c3d_markers.get_markers()
        assert len(markers.coords['channel']) == 191
        exp_x_values = arr.array('f', [-1300.67626953, -1290.06420898, -1275.91687012,
                                       -1258.42175293, -1237.95727539])
        assert (markers.loc['x', 'RTOE'][0:5].data == exp_x_values).all()

        assert markers.coords['time'][0] == 2.48
    def test_c3d_markers_big(self):
        c3d_markers = MarkersInputFileReader(INPUT_C3D_BIG)
        markers = c3d_markers.get_markers()
        # Test markers length
        assert len(markers.coords['channel']) == 127

        # Test first 5 and last 5 x values of LASIS
        exp_x_values = arr.array('f',
                                 [152.31216431, 152.31573486, 152.33493042,
                                  152.36070251,
                                  152.38232422])
        rec_x_values = markers.loc['x', 'LASIS'][0:5].data
        assert (rec_x_values == exp_x_values).all()

        # Test first 5 and last 5 x values of LASIS
        exp_x_values = arr.array('f',
                                 [165.27760315, 165.07537842, 164.69329834,
                                  164.05558777,
                                  163.1633606])
        rec_x_values = markers.loc['x', 'LASIS'][-5:].data
        assert (rec_x_values == exp_x_values).all()

    def test_trc_markers_small(self):
        with pytest.raises(NotImplementedError):
            MarkersInputFileReader(INPUT_TRC_SMALL)

    def test_mot_markers_small(self):
        mot_analogs = AnalogsInputFileReader(INPUT_MOT_SMALL)
        analogs = mot_analogs.get_analogs()
        # Test analogs length
        exp_value = 36
        rec_value = len(analogs.coords['channel'])
        assert rec_value == exp_value

        # Test first 5 values of Voltage.RERS
        exp_values = [-2.061715, -2.824317, -3.442063, -3.703248, -4.052287]
        rec_values = list(analogs.loc['ground_force4_vx'][0:5].data)
        assert rec_values == exp_values

        exp_value = 2.48
        rec_value = analogs.coords['time'][0]
        assert rec_value == exp_value


class TestAnalogs:
    def test_c3d_analog_small(self):
        c3d_analogs = AnalogsInputFileReader(INPUT_C3D_SMALL)
        analogs = c3d_analogs.get_analogs()

        # Test analogs length
        exp_value = 42
        rec_value = len(analogs.coords['channel'])
        assert rec_value == exp_value

        # Test first 5 values of Voltage.RERS
        exp_values = [0.024871826171875, 0.01682281494140625, 0.00705718994140625,
                      -0.0067138671875, -0.01224517822265625]
        rec_values = list(analogs.loc['Voltage.RERS'][0:5].data)
        assert rec_values == exp_values

        exp_value = 2.48
        rec_value = analogs.coords['time'][0]
        assert rec_value == exp_value

    def test_sto_analogs_small(self):
        with pytest.raises(NotImplementedError):
            AnalogsInputFileReader(INPUT_STO_SMALL)

    def test_wrong_file_format(self):
        with pytest.raises(ValueError):
            MarkersInputFileReader(Path("foo.csv"))


class TestAnalysis:
    def test_c3d_analysis_small(self):
        configs = MappingConfigs(Path('./tests/full/config/pig_config.yaml'))
        analysis = AnalysisInputReader(INPUT_C3D_SMALL, configs).get_analysis()
        rec_value = analysis.shape[0]
        exp_value = 126
        assert rec_value == exp_value

        rec_value = analysis.shape[1]
        exp_value = 337
        assert rec_value == exp_value

    def test_reshape(self):
        configs = MappingConfigs(Path('./tests/full/config/pig_config.yaml'))
        analysis = AnalysisInputReader(INPUT_C3D_SMALL, configs).get_analysis()
        markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()

        new_labels = analysis.coords["channel"].values
        new_times = analysis.coords["time"].values
        for new_label in new_labels:
            old_label, old_axis = new_label.split("_")
            for time in new_times:
                if not np.isnan(markers.loc[old_axis, old_label, time]):
                    rec_value = analysis.loc[new_label, time]
                    exp_value = markers.loc[old_axis, old_label, time]
                    assert rec_value == exp_value
