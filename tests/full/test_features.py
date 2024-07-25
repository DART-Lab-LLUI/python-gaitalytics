from pathlib import Path

import pytest

from gaitalytics.features import TimeSeriesFeatures, TemporalFeatures, SpatialFeatures, \
    PhaseTimeSeriesFeatures
from gaitalytics.io import MarkersInputFileReader, C3dEventInputFileReader, \
    AnalogsInputFileReader, AnalysisInputReader
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import DataCategory, Trial
from gaitalytics.segmentation import GaitEventsSegmentation

INPUT_C3D_SMALL: Path = Path('./tests/full/data/test_small.c3d')
INPUT_C3D_BIG: Path = Path('./tests/full/data/test_big.c3d')
CONFIG_FILE = Path('./tests/full/config/pig_config.yaml')


@pytest.fixture()
def configs(request):
    return MappingConfigs(CONFIG_FILE)


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

    return GaitEventsSegmentation().segment(trial)


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

    return GaitEventsSegmentation().segment(trial)


class TestTimeSeriesFeatures:

    def test_calculate_features(self, trial_small):

        for context in trial_small.get_all_cycles().keys():
            for cycle_id, cycle in trial_small.get_all_cycles()[context].items():
                features = TimeSeriesFeatures._calculate_features(cycle)
                for marker in features.channel.values:
                    if not features.loc[dict(channel=marker)].isnull().any():

                        min_value = features.loc[
                            dict(channel=marker,
                                 feature="min")]
                        max_value = features.loc[
                            dict(channel=marker,
                                 feature="max")]
                        mean_value = features.loc[
                            dict(channel=marker,
                                 feature="mean")]
                        median_value = features.loc[
                            dict(channel=marker,
                                 feature="median")]
                        amp_value = features.loc[
                            dict(channel=marker,
                                 feature="amplitude")]

                        assert min_value <= max_value
                        assert min_value <= mean_value
                        assert max_value >= mean_value
                        assert min_value <= median_value
                        assert max_value >= median_value
                        assert amp_value == (max_value - min_value)
                    else:
                        if "Power" in marker or "Force" in marker or "Moment" in marker or "GRF" in marker:
                            assert True
                        else:
                            assert False, f"Missing data for {marker} in {context} context in cycle {cycle_id}"

    def test_calculation(self, configs, trial_small):
        features = TimeSeriesFeatures(configs).calculate(trial_small)
        for feature in features.feature.values:
            if "_min" in features:
                marker = feature.replace("_min", "")
                for context in features.context.values:
                    for cycle_id in features.cycle_id.values:
                        min_value = features.loc[
                            dict(feature=f"{marker}_min",
                                 context=context,
                                 cycle_id=cycle_id)]
                        max_value = features.loc[
                            dict(feature=f"{marker}_max",
                                 context=context,
                                 cycle_id=cycle_id)]
                        mean_value = features.loc[
                            dict(feature=f"{marker}_mean",
                                 context=context,
                                 cycle_id=cycle_id)]
                        median_value = features.loc[
                            dict(feature=f"{marker}_median",
                                 context=context,
                                 cycle_id=cycle_id)]
                        amp_value = features.loc[
                            dict(feature=f"{marker}_amplitude",
                                 context=context,
                                 cycle_id=cycle_id)]

                        assert min_value <= max_value
                        assert min_value <= mean_value
                        assert max_value >= mean_value
                        assert min_value <= median_value
                        assert max_value >= median_value
                        assert amp_value == (max_value - min_value)

    def test_reshape(self, trial_small):
        features = TimeSeriesFeatures._calculate_features(
            trial_small.get_cycle("Left", 0))
        flat_feat = TimeSeriesFeatures._flatten_features(features)
        for feature in features.feature.values:
            for marker in features.channel.values:
                rec_value = flat_feat.loc[dict(feature=f"{marker}_{feature}")]
                exp_value = features.loc[dict(channel=marker, feature=feature)]
                assert rec_value == exp_value, f"Expected {exp_value}, got {rec_value}"


class TestPhaseTimeSeriesFeatures:

    def test_calculation(self, configs, trial_small):
        features = PhaseTimeSeriesFeatures(configs).calculate(trial_small)
        for phase in ["_stand", "_swing"]:
            for feature in features.feature.values:
                if "_min" in features:
                    marker = feature.replace(f"_min{phase}", "")
                    for context in features.context.values:
                        for cycle_id in features.cycle_id.values:
                            min_value = features.loc[
                                dict(feature=f"{marker}_min{phase}",
                                     context=context,
                                     cycle_id=cycle_id)]
                            max_value = features.loc[
                                dict(feature=f"{marker}_max{phase}",
                                     context=context,
                                     cycle_id=cycle_id)]
                            mean_value = features.loc[
                                dict(feature=f"{marker}_mean{phase}",
                                     context=context,
                                     cycle_id=cycle_id)]
                            median_value = features.loc[
                                dict(feature=f"{marker}_median{phase}",
                                     context=context,
                                     cycle_id=cycle_id)]
                            amp_value = features.loc[
                                dict(feature=f"{marker}_amplitude{phase}",
                                     context=context,
                                     cycle_id=cycle_id)]

                            assert min_value <= max_value
                            assert min_value <= mean_value
                            assert max_value >= mean_value
                            assert min_value <= median_value
                            assert max_value >= median_value
                            assert amp_value == (max_value - min_value)

    def test_reshape(self, trial_small):
        features = TimeSeriesFeatures._calculate_features(
            trial_small.get_cycle("Left", 0))
        flat_feat = TimeSeriesFeatures._flatten_features(features)
        for feature in features.feature.values:
            for marker in features.channel.values:
                rec_value = flat_feat.loc[dict(feature=f"{marker}_{feature}")]
                exp_value = features.loc[dict(channel=marker, feature=feature)]
                assert rec_value == exp_value


class TestTemporalFeatures:

    def test_calculation(self, configs, trial_small):
        features = TemporalFeatures(configs).calculate(trial_small)

        rec_value = features.shape[2]
        exp_value = 8
        assert rec_value == exp_value

        # Stride time
        rec_value = features.loc["Left", 0, "cadence"]
        exp_value = 113.2076
        assert rec_value == pytest.approx(exp_value, rel=1e-3)
        rec_value = features.loc["Right", 0, "cadence"]
        exp_value = 110.0917
        assert rec_value == pytest.approx(exp_value, rel=1e-3)

        # Stride time
        rec_value = features.loc["Left", 0, "stride_time"]
        exp_value = 1.06
        assert rec_value == pytest.approx(exp_value, rel=1e-2)

        rec_value = features.loc["Right", 0, "stride_time"]
        exp_value = 1.09
        assert rec_value == pytest.approx(exp_value, rel=1e-2)

        # Step time
        rec_value = features.loc["Left", 0, "step_time"]
        exp_value = 0.5
        assert rec_value == pytest.approx(exp_value, rel=1e-2)

        rec_value = features.loc["Right", 0, "step_time"]
        exp_value = 0.56
        assert rec_value == pytest.approx(exp_value, rel=1e-2)

        # Double support
        rec_value = features.loc["Left", 0, "double_support"]
        exp_value = 21.6981 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        rec_value = features.loc["Right", 0, "double_support"]
        exp_value = 23.8532 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        # Single support
        rec_value = features.loc["Left", 0, "single_support"]
        exp_value = 41.5094 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        rec_value = features.loc["Right", 0, "single_support"]
        exp_value = 35.7798 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        # Opposite foot off
        rec_value = features.loc["Left", 0, "opposite_foot_off"]
        exp_value = 11.3208 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        rec_value = features.loc["Right", 0, "opposite_foot_off"]
        exp_value = 12.8440 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        # Opposite foot contact
        rec_value = features.loc["Left", 0, "opposite_foot_contact"]
        exp_value = 52.8302 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        rec_value = features.loc["Right", 0, "opposite_foot_contact"]
        exp_value = 48.6239 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        # foot off
        rec_value = features.loc["Left", 0, "foot_off"]
        exp_value = 63.2075 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)

        rec_value = features.loc["Right", 0, "foot_off"]
        exp_value = 59.6330 / 100
        assert rec_value == pytest.approx(exp_value, rel=1e-5)


class TestSpatialFeatures:
    def test_calculation_big(self, configs, trial_big):
        features = SpatialFeatures(configs).calculate(trial_big)
        for context, cycles in trial_big.get_all_cycles().items():
            for cycle_id, cycle in cycles.items():
                event = cycle.events.attrs["end_time"]
                markers = cycle.get_data(DataCategory.MARKERS).drop_sel(axis="z").sel(
                    time=event, method="nearest")

                ipsi_label = "RHEE" if context == "Right" else "LHEE"
                contra_label = "LHEE" if context == "Right" else "RHEE"
                ipsi_heel = markers.sel(channel=ipsi_label)
                contra_heel = markers.sel(channel=contra_label)
                distances = ipsi_heel - contra_heel

                exp_value = distances.sel(axis="y")
                exp_value = exp_value.meca.abs()
                rec_value = features.loc[context, cycle_id, "step_length"]
                assert rec_value == pytest.approx(exp_value, rel=1e-0)

                exp_value = distances.sel(axis="x")
                exp_value = exp_value.meca.abs()
                rec_value = features.loc[context, cycle_id, "step_width"]
                assert rec_value == pytest.approx(exp_value, rel=1e-0)

    def test_calculation_small(self, configs, trial_small):
        features = SpatialFeatures(configs).calculate(trial_small)

        rec_value = features.loc["Left", 0, "step_length"]
        exp_value = 532.01
        assert rec_value == pytest.approx(exp_value,rel=1e-1)

        rec_value = features.loc["Right", 0, "step_length"]
        exp_value = 565.24
        assert rec_value == pytest.approx(exp_value, rel=1e-1)
