import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
from scipy import signal

import gaitalytics.model as model
import gaitalytics.utils as utils

logger = logging.getLogger(__name__)


def calculate_stats(data: np.ndarray, context: str, label: str):
    ts_max = np.nanmax(data, axis=2)
    ts_min = np.nanmin(data, axis=2)
    ts_mean = np.nanmean(data, axis=2)
    ts_std = np.nanstd(data, axis=2)
    ts_amp = ts_max - ts_min
    results = {f"{context}_{label}_mean": ts_mean,
               f"{context}_{label}_sd": ts_std,
               f"{context}_{label}_max": ts_max,
               f"{context}_{label}_min": ts_min,
               f"{context}_{label}_amplitude": ts_amp,
               }
    return results


class AbstractAnalysis(ABC):
    def __init__(self,
                 data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles],
                 configs: utils.ConfigProvider):
        self._data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles] = data_list
        self._configs: utils.ConfigProvider = configs
        self._data_condition: model.ExtractedCycleDataCondition = self.get_data_condition()

    @abstractmethod
    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        pass

    @abstractmethod
    def _analyse(self, by_phase: bool) -> dict:
        pass

    def analyse(self, **kwargs) -> dict:
        by_phase = kwargs.get("by_phase", True)
        return self._analyse(by_phase)

    def get_point_data(self, label: model.TranslatedLabel, cycle_context: model.GaitEventContext):
        data_table = None
        for cycle_point in self.get_all_point_data(cycle_context):
            if cycle_point.translated_label.name == label.value:
                data_table = cycle_point.data_table
                break

        if data_table is None:
            raise KeyError(f"{label.name} not in extracted cycles")
        return data_table

    def get_all_point_data(self, cycle_context) -> list[model.ExtractedCyclePoint]:
        extracted_cycles = self._data_list[self._data_condition]
        context_cycle = extracted_cycles.cycle_points[cycle_context]
        return context_cycle.points

    def get_subject_data(self) -> model.SubjectMeasures:
        return self._data_list[self._data_condition].subject

    def get_cycles_meta_data(self, cycle_context: model.GaitEventContext) -> dict[str, any]:
        return self._data_list[self._data_condition].cycle_points[cycle_context].meta_data

    @staticmethod
    def split_by_phase(data: np.ndarray, meta_data: dict[str: any]) -> [np.ndarray, np.ndarray]:
        events = meta_data["Foot Off_IPSI"]
        cycle_length = meta_data["end_frame"] - meta_data["start_frame"]
        standing = data.copy()
        swinging = data.copy()
        for cycle_index in range(len(events)):
            standing[:, cycle_index, int(events[cycle_index]):] = np.nan
            swinging[:, cycle_index, : int(events[cycle_index])] = np.nan

        return standing, swinging


class AbstractTimeseriesAnalysis(AbstractAnalysis):

    def __init__(
        self,
        data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles],
        configs: utils.ConfigProvider,
        data_types: [model.PointDataType],
    ):
        super().__init__(data_list, configs)
        self._point_data_types = data_types

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:

        results = None

        for key in self._data_list[self.get_data_condition()].cycle_points:
            context_cycles = self._data_list[self.get_data_condition()].cycle_points[key]
            for point in context_cycles.points:
                if self._filter_points(point):
                    data = point.data_table
                    if not by_phase:
                        result = self._do_analysis(data, point.translated_label.name, context_cycles.context,
                                                   point.point_type)
                    else:

                        standing, swinging = self.split_by_phase(data, context_cycles.meta_data)
                        result = self._do_analysis(swinging, f"{point.translated_label.name}_swing",
                                                   context_cycles.context, point.point_type)

                        result_stand = self._do_analysis(standing, f"{point.translated_label.name}_stand",
                                                         context_cycles.context, point.point_type)
                        result.update(result_stand)

                    if results is None:
                        results = result
                    else:
                        results.update(result)
        return results

    @abstractmethod
    def _do_analysis(self, data: np.ndarray, label: str, context: model.GaitEventContext,
                     point_type: model.PointDataType) -> dict:
        pass

    def _filter_points(self, point: model.ExtractedCyclePoint) -> bool:
        """Check if it's the right point data"""
        return point.point_type in self._point_data_types


class TimeseriesAnalysis(AbstractTimeseriesAnalysis):

    def __init__(self, data_list: dict, configs: utils.ConfigProvider):
        super().__init__(data_list, configs,
                         [model.PointDataType.FORCES, model.PointDataType.ANGLES, model.PointDataType.POWERS,
                          model.PointDataType.MOMENTS])

    def _do_analysis(self, data: np.ndarray, label: str, context: model.GaitEventContext,
                     point_type: model.PointDataType) -> dict:
        logger.info(f"analyse: Timeseries {label}")
        results = calculate_stats(data, context.name, label)
        if point_type == model.PointDataType.ANGLES:
            velocity = np.diff(data, axis=2)
            velo_res = calculate_stats(velocity, context.name, "angle_velocity")
            results.update(velo_res)
        return results


class CMosAnalysis(AbstractAnalysis):

    def __init__(self, data_list: dict, configs: utils.ConfigProvider):
        super().__init__(data_list, configs)

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: CMOS")

        left_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.LEFT)
        right_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.RIGHT)
        if not by_phase:
            result = calculate_stats(left_cmos, model.GaitEventContext.LEFT.name, "cmos")
            result.update(calculate_stats(right_cmos, model.GaitEventContext.RIGHT.name, "cmos"))
        else:
            left_events = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
            right_events = self.get_cycles_meta_data(model.GaitEventContext.RIGHT)["Foot Off_IPSI"]

            l_standing, l_swinging = self.split_by_phase(left_cmos,
                                                         self.get_cycles_meta_data(model.GaitEventContext.LEFT))
            r_standing, r_swinging = self.split_by_phase(right_cmos,
                                                         self.get_cycles_meta_data(model.GaitEventContext.RIGHT))

            result = calculate_stats(l_swinging, model.GaitEventContext.LEFT.name, "cmos_swing")
            result.update(calculate_stats(l_standing, model.GaitEventContext.LEFT.name, "cmos_stand"))
            result.update(calculate_stats(r_swinging, model.GaitEventContext.RIGHT.name, "cmos_swing"))
            result.update(calculate_stats(r_standing, model.GaitEventContext.RIGHT.name, "cmos_stand"))

        return result

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA


class MosAnalysis(AbstractAnalysis):

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: MOS")

        left_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.LEFT)
        left_cmos_ap = left_cmos[model.AxesNames.y.value]
        left_cmos_ml = left_cmos[model.AxesNames.x.value]

        right_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.RIGHT)
        right_cmos_ap = right_cmos[model.AxesNames.y.value]
        right_cmos_ml = right_cmos[model.AxesNames.x.value]

        left_ap = self._extract_mos_frames(left_cmos_ap, model.GaitEventContext.LEFT, "ap")
        left_ml = self._extract_mos_frames(left_cmos_ml, model.GaitEventContext.LEFT, "ml")
        right_ap = self._extract_mos_frames(right_cmos_ap, model.GaitEventContext.RIGHT, "ap")
        right_ml = self._extract_mos_frames(right_cmos_ml, model.GaitEventContext.RIGHT, "ml")
        results = left_ap
        results.update(left_ml)
        results.update(right_ap)
        results.update(right_ml)

        return results

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _extract_mos_frames(self, cmos: np.ndarray, context: model.GaitEventContext, direction: str) -> dict:
        hs_label = f"{context.name}_mos_{direction}_hs_ipsi"
        to_label = f"{context.name}_mos_{direction}_to_ipsi"
        hs_contra_label = f"{context.name}_mos_{direction}_hs_contra"
        to_contra_label = f"{context.name}_mos_{direction}_to_contra"
        results = {hs_label: [],
                   to_label: [],
                   hs_contra_label: [],
                   to_contra_label: [],
                   }

        cycle_meta_data = self.get_cycles_meta_data(context)
        for cycle_index in range(len(cmos)):
            hs_frame = cmos[cycle_index, 0]
            to_frame = cmos[cycle_index, int(cycle_meta_data["Foot Off_IPSI"][cycle_index])]
            hs_contra_frame = cmos[cycle_index, int(cycle_meta_data["Foot Strike_CONTRA"][cycle_index])]
            to_contra_frame = cmos[cycle_index, int(cycle_meta_data["Foot Off_CONTRA"][cycle_index])]
            results[hs_label].append(hs_frame)
            results[to_label].append(to_frame)
            results[hs_contra_label].append(hs_contra_frame)
            results[to_contra_label].append(to_contra_frame)
        return results


class SpatioTemporalAnalysis(AbstractAnalysis):

    def __init__(self,
                 data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles],
                 configs: utils.ConfigProvider):
        self._sub_analysis_list: list[type[AbstractAnalysis]] = [_StepWidthAnalysis, _LimbCircumductionAnalysis,
                                                                 _StepHeightAnalysis, _CycleDurationAnalysis,
                                                                 _StepLengthAnalysis]
        super().__init__(data_list, configs)

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: Spatio Temporal")

        results = {}
        for cls in self._sub_analysis_list:
            analysis_obj = cls(self._data_list, self._configs)
            results.update(analysis_obj._analyse(by_phase))
        return results

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA


class _StepWidthAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: _Step Width")

        right_heel_x_right = \
            self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI, model.GaitEventContext.RIGHT)[
                model.AxesNames.x.value]
        left_heel_x_right = \
            self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI, model.GaitEventContext.RIGHT)[
                model.AxesNames.x.value]
        right_heel_x_left = \
            self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI, model.GaitEventContext.LEFT)[
                model.AxesNames.x.value]
        left_heel_x_left = \
            self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI, model.GaitEventContext.LEFT)[
                model.AxesNames.x.value]
        right = self._calculate_step_width_side(right_heel_x_right, left_heel_x_right,
                                                model.GaitEventContext.RIGHT)
        left = self._calculate_step_width_side(left_heel_x_left, right_heel_x_left,
                                               model.GaitEventContext.LEFT)
        right.update(left)
        return right

    def _calculate_step_width_side(self, context_position_x: np.ndarray,
                                   contra_position_x: np.ndarray,
                                   context: model.GaitEventContext) -> dict[str, np.ndarray]:
        width = np.ndarray(len(context_position_x))
        for cycle_number in range(len(context_position_x)):
            width_c = abs(context_position_x[cycle_number][0] - contra_position_x[cycle_number][0])
            width[cycle_number] = width_c / self.get_subject_data().body_height
        return {f"{context.name}_step_width": width}


class _LimbCircumductionAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: _Circumduction")

        right_malleoli_right = self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI,
                                                   model.GaitEventContext.RIGHT)
        right_meta_data = self.get_cycles_meta_data(model.GaitEventContext.RIGHT)
        left_malleoli_left = self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI,
                                                 model.GaitEventContext.LEFT)
        left_meta_data = self.get_cycles_meta_data(model.GaitEventContext.LEFT)

        right_malleoli_x_right = self.split_by_phase(right_malleoli_right, right_meta_data)[1][model.AxesNames.x.value]
        left_malleoli_x_left = self.split_by_phase(left_malleoli_left, left_meta_data)[1][model.AxesNames.x.value]
        results = self._calculate_limb_circumduction_side(right_malleoli_x_right, model.GaitEventContext.RIGHT)
        results.update(self._calculate_limb_circumduction_side(left_malleoli_x_left, model.GaitEventContext.LEFT))
        return results

    def _calculate_limb_circumduction_side(self, data_malleoli_x: np.ndarray,
                                           context: model.GaitEventContext) -> dict[str, np.ndarray]:
        column_label = f"{context.name}_limb_circumduction"
        limb_circum = np.ndarray(len(data_malleoli_x))
        body_height = self.get_subject_data().body_height
        if np.nanmean(data_malleoli_x) < 0:
            data_malleoli_x = data_malleoli_x * -1

        for cycle_number in range(len(data_malleoli_x)):
            data = data_malleoli_x[cycle_number]
            data = data[~np.isnan(data)]
            max_value = np.nanmax(data)
            start_value = data[0]
            limb_circum[cycle_number] = max_value - start_value
        limb_circum = limb_circum / body_height

        return {column_label: limb_circum}


class _StepLengthAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: _Step Length")

        left_heel_y_left = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.LEFT)[
            model.AxesNames.y.value]
        right_heel_y_left = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.RIGHT)[
            model.AxesNames.y.value]

        left_heel_y_right = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.LEFT)[
            model.AxesNames.y.value]
        right_heel_y_right = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.RIGHT)[
            model.AxesNames.y.value]

        left = self._calculate_step_length(left_heel_y_left, right_heel_y_left, model.GaitEventContext.LEFT)
        right = self._calculate_step_length(left_heel_y_right, right_heel_y_right, model.GaitEventContext.RIGHT)
        left.update(right)
        return left

    @staticmethod
    def _calculate_step_length(heel_y_ipsi: np.ndarray, heel_y_contra: np.ndarray,
                               context: model.GaitEventContext) -> dict[str, np.ndarray]:
        step_length = np.ndarray(len(heel_y_ipsi))
        for cycle_number in range(len(heel_y_contra)):
            step_length[cycle_number] = abs(heel_y_ipsi[cycle_number, 0] - heel_y_contra[cycle_number, 0])
        return {f"{context.name}_step_length": step_length}


class _StepHeightAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: _Step Height")

        right_heel_z_right = \
            self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.RIGHT)[
                model.AxesNames.z.value]
        left_heel_z_left = \
            self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.LEFT)[
                model.AxesNames.z.value]

        right = self._calculate_step_height(right_heel_z_right,
                                            model.GaitEventContext.RIGHT)
        left = self._calculate_step_height(left_heel_z_left,
                                           model.GaitEventContext.LEFT)
        right.update(left)
        return right

    def _calculate_step_height(self, context_position_x: np.ndarray,
                               context: model.GaitEventContext) -> dict[str, np.ndarray]:
        body_height = self.get_subject_data().body_height
        height = np.ndarray(len(context_position_x))
        for cycle_number in range(len(context_position_x)):
            height[cycle_number] = np.nanmax(context_position_x[cycle_number]) - np.nanmin(
                context_position_x[cycle_number])
        height = height / body_height

        return {f"{context.name}_step_height": height}


class _CycleDurationAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info(f"analyse: _Duration")

        left_meta = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
        right_meta = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
        left = self._calculate_cycle_duration(left_meta, model.GaitEventContext.LEFT)
        right = self._calculate_cycle_duration(right_meta, model.GaitEventContext.RIGHT)
        right.update(left)
        return right

    def _calculate_cycle_duration(self, meta_data: dict[str, any], context: model.GaitEventContext) -> dict[
        str, np.ndarray]:
        frequency = self.get_subject_data().mocap_frequency
        cycle_length = meta_data["end_frame"] - meta_data["start_frame"]
        cycle_duration = cycle_length / frequency
        step_duration = meta_data["Foot Off_IPSI"] - meta_data["start_frame"] / frequency
        swing_length = meta_data["end_frame"] - meta_data["Foot Off_IPSI"] / frequency
        double_support_duration = meta_data["Foot Off_CONTRA"] - meta_data["start_frame"] / frequency
        single_support_duration = meta_data["Foot Off_CONTRA"] - meta_data["Foot Strike_CONTRA"] / frequency

        perc_stand_duration = step_duration / cycle_duration
        perc_swing_duration = swing_length / cycle_duration
        perc_double_support_duration = double_support_duration / cycle_duration
        perc_single_support_duration = single_support_duration / cycle_duration
        return {f"{context.name}_cycle_duration": cycle_duration,
                f"{context.name}_step_duration": step_duration,
                f"{context.name}_swing_duration": perc_swing_duration,
                f"{context.name}_stand_duration": perc_stand_duration,
                f"{context.name}_double_support_duration": perc_double_support_duration,
                f"{context.name}_single_support_duration": perc_single_support_duration,
                }

    # drag_duration_gc = np.zeros(len(data))  # %GC
    # drag_duration_swing = np.zeros(len(data))  # %swing
    # stride_speed = np.zeros(len(data))  # m/s
    # stride_length_com = np.zeros(len(data))  # %BH
    # stride_speed_com = np.zeros(len(data))  # m/s
    # length_foot_trajectory = np.zeros(len(data))  # %BH
    # length_com_trajectory = np.zeros(len(data))  # %BH
    # lateral_movement_during_swing = np.zeros(len(data))  # BH%
    # max_hip_vertical_amplitude = np.zeros(len(data))  # BH%


class MinimalToeClearance(AbstractAnalysis):
    logger.info(f"analyse: Minia")

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        right_toe_z = self.get_point_data(model.TranslatedLabel.RIGHT_META_2, model.GaitEventContext.RIGHT)
        left_toe_z = self.get_point_data(model.TranslatedLabel.LEFT_META_2, model.GaitEventContext.LEFT)
        right_toe_z = self.split_by_phase(right_toe_z, self.get_cycles_meta_data(model.GaitEventContext.RIGHT))[1][
            model.AxesNames.z.value]
        left_toe_z = self.split_by_phase(left_toe_z, self.get_cycles_meta_data(model.GaitEventContext.LEFT))[1][
            model.AxesNames.z.value]
        right = self._calculate_minimal_clearance(right_toe_z, model.GaitEventContext.RIGHT)
        left = self._calculate_minimal_clearance(left_toe_z, model.GaitEventContext.LEFT)
        right.update(left)
        return right

    @staticmethod
    def _calculate_minimal_clearance(toe: np.ndarray, side: model.GaitEventContext) -> dict[str, np.ndarray]:
        min_tc_label = f"{side.name}_minimal_toe_clearance"
        min_tc_perc_label = f"{side.name}_minimal_toe_clearance_swing_perc"
        tc_at_heel_strike_label = f"{side.name}_toe_clearance_heel_strike"

        results = {min_tc_label: np.ndarray(len(toe)),
                   min_tc_perc_label: np.ndarray(len(toe)),
                   tc_at_heel_strike_label: np.ndarray(len(toe))}
        for cycle_number in range(len(toe)):
            toe_cycle = toe[cycle_number]
            toe_cycle = toe_cycle[~np.isnan(toe_cycle)]
            peaks = signal.find_peaks(toe_cycle, distance=len(toe_cycle))
            mid_late_swing = toe_cycle[peaks[0][0]:]
            toe_clear_min = np.min(mid_late_swing)
            toe_clear_min_pos = np.argmin(mid_late_swing) + peaks[0][0]
            tc_percent = toe_clear_min_pos / len(toe_cycle)
            tc_clear_hs = toe_cycle[-1]
            results[min_tc_label][cycle_number] = toe_clear_min
            results[min_tc_perc_label][cycle_number] = tc_percent
            results[tc_at_heel_strike_label][cycle_number] = tc_clear_hs
        return results
