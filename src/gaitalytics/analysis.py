import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
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
    results = {
        f"{context}_{label}_mean": ts_mean,
        f"{context}_{label}_sd": ts_std,
        f"{context}_{label}_max": ts_max,
        f"{context}_{label}_min": ts_min,
        f"{context}_{label}_amplitude": ts_amp,
    }
    return results


class AbstractAnalysis(ABC):
    def __init__(self, data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles], configs: utils.ConfigProvider):
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

    def get_cycles_meta_data(self, cycle_context: model.GaitEventContext) -> dict[str, any]:
        return self._data_list[self._data_condition].cycle_points[cycle_context].meta_data

    @staticmethod
    def split_by_phase(data: np.ndarray, meta_data: dict[str:any]) -> [np.ndarray, np.ndarray]:
        events = meta_data["Foot Off_IPSI"]
        standing = data.copy()
        swinging = data.copy()
        for cycle_index in range(len(events)):
            standing[:, cycle_index, int(events[cycle_index]) :] = np.nan
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
                        result = self._do_analysis(data, point.translated_label.name, context_cycles.context, point.point_type)
                    else:

                        standing, swinging = self.split_by_phase(data, context_cycles.meta_data)
                        result = self._do_analysis(
                            swinging, f"{point.translated_label.name}_swing", context_cycles.context, point.point_type
                        )

                        result_stand = self._do_analysis(
                            standing, f"{point.translated_label.name}_stand", context_cycles.context, point.point_type
                        )
                        result.update(result_stand)

                    if results is None:
                        results = result
                    else:
                        results.update(result)
        return results

    @abstractmethod
    def _do_analysis(self, data: np.ndarray, label: str, context: model.GaitEventContext, point_type: model.PointDataType) -> dict:
        pass

    def _filter_points(self, point: model.ExtractedCyclePoint) -> bool:
        """Check if it's the right point data"""
        return point.point_type in self._point_data_types


class TimeseriesAnalysis(AbstractTimeseriesAnalysis):

    def __init__(self, data_list: dict, configs: utils.ConfigProvider):
        super().__init__(
            data_list,
            configs,
            [model.PointDataType.FORCES, model.PointDataType.ANGLES, model.PointDataType.POWERS, model.PointDataType.MOMENTS],
        )

    def _do_analysis(self, data: np.ndarray, label: str, context: model.GaitEventContext, point_type: model.PointDataType) -> dict:
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
        logger.info("analyse: CMOS")

        left_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.LEFT)
        right_cmos = self.get_point_data(model.TranslatedLabel.CMOS, model.GaitEventContext.RIGHT)
        if not by_phase:
            result = calculate_stats(left_cmos, model.GaitEventContext.LEFT.name, "cmos")
            result.update(calculate_stats(right_cmos, model.GaitEventContext.RIGHT.name, "cmos"))
        else:
            left_events = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
            right_events = self.get_cycles_meta_data(model.GaitEventContext.RIGHT)

            l_standing, l_swinging = self.split_by_phase(left_cmos, left_events)
            r_standing, r_swinging = self.split_by_phase(right_cmos, right_events)

            result = calculate_stats(l_swinging, model.GaitEventContext.LEFT.name, "cmos_swing")
            result.update(calculate_stats(l_standing, model.GaitEventContext.LEFT.name, "cmos_stand"))
            result.update(calculate_stats(r_swinging, model.GaitEventContext.RIGHT.name, "cmos_swing"))
            result.update(calculate_stats(r_standing, model.GaitEventContext.RIGHT.name, "cmos_stand"))

        return result

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA


class MosAnalysis(AbstractAnalysis):

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: MOS")

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
        results = {
            hs_label: [],
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

    def __init__(self, data_list: dict[model.ExtractedCycleDataCondition, model.ExtractedCycles], configs: utils.ConfigProvider):
        self._sub_analysis_list: list[type[AbstractAnalysis]] = [
            _StepWidthAnalysis,
            _LimbCircumductionAnalysis,
            _StepHeightAnalysis,
            _CycleDurationAnalysis,
            _StepLengthAnalysis,
        ]
        super().__init__(data_list, configs)

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: Spatio Temporal")

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
        logger.info("analyse: _Step Width")

        right_heel_x_right = self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI, model.GaitEventContext.RIGHT)[
            model.AxesNames.x.value
        ]
        left_heel_x_right = self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI, model.GaitEventContext.RIGHT)[
            model.AxesNames.x.value
        ]
        right_heel_x_left = self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI, model.GaitEventContext.LEFT)[
            model.AxesNames.x.value
        ]
        left_heel_x_left = self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI, model.GaitEventContext.LEFT)[
            model.AxesNames.x.value
        ]
        right = self._calculate_step_width_side(right_heel_x_right, left_heel_x_right, model.GaitEventContext.RIGHT)
        left = self._calculate_step_width_side(left_heel_x_left, right_heel_x_left, model.GaitEventContext.LEFT)
        right.update(left)
        return right

    def _calculate_step_width_side(
        self, context_position_x: np.ndarray, contra_position_x: np.ndarray, context: model.GaitEventContext
    ) -> dict[str, np.ndarray]:
        width = np.ndarray(len(context_position_x))
        for cycle_number in range(len(context_position_x)):
            width_c = abs(context_position_x[cycle_number][0] - contra_position_x[cycle_number][0])
            width[cycle_number] = width_c
        return {f"{context.name}_step_width": width}


class _LimbCircumductionAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: _Circumduction")

        right_malleoli_right = self.get_point_data(model.TranslatedLabel.RIGHT_MED_MALLEOLI, model.GaitEventContext.RIGHT)
        right_meta_data = self.get_cycles_meta_data(model.GaitEventContext.RIGHT)
        left_malleoli_left = self.get_point_data(model.TranslatedLabel.LEFT_MED_MALLEOLI, model.GaitEventContext.LEFT)
        left_meta_data = self.get_cycles_meta_data(model.GaitEventContext.LEFT)

        right_malleoli_x_right = self.split_by_phase(right_malleoli_right, right_meta_data)[1][model.AxesNames.x.value]
        left_malleoli_x_left = self.split_by_phase(left_malleoli_left, left_meta_data)[1][model.AxesNames.x.value]
        results = self._calculate_limb_circumduction_side(right_malleoli_x_right, model.GaitEventContext.RIGHT)
        results.update(self._calculate_limb_circumduction_side(left_malleoli_x_left, model.GaitEventContext.LEFT))
        return results

    def _calculate_limb_circumduction_side(self, data_malleoli_x: np.ndarray, context: model.GaitEventContext) -> dict[str, np.ndarray]:
        column_label = f"{context.name}_limb_circumduction"
        limb_circum = np.ndarray(len(data_malleoli_x))
        if np.nanmean(data_malleoli_x) < 0:
            data_malleoli_x = data_malleoli_x * -1

        for cycle_number in range(len(data_malleoli_x)):
            data = data_malleoli_x[cycle_number]
            data = data[~np.isnan(data)]
            max_value = np.nanmax(data)
            start_value = data[0]
            limb_circum[cycle_number] = max_value - start_value
        limb_circum = limb_circum

        return {column_label: limb_circum}


class _StepLengthAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: _Step Length")

        left_heel_y_left = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.LEFT)[model.AxesNames.y.value]
        right_heel_y_left = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.LEFT)[model.AxesNames.y.value]

        left_heel_y_right = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.RIGHT)[model.AxesNames.y.value]
        right_heel_y_right = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.RIGHT)[model.AxesNames.y.value]

        left = self._calculate_step_length(left_heel_y_left, right_heel_y_left, model.GaitEventContext.LEFT)
        right = self._calculate_step_length(left_heel_y_right, right_heel_y_right, model.GaitEventContext.RIGHT)
        left.update(right)
        return left

    @staticmethod
    def _calculate_step_length(
        heel_y_ipsi: np.ndarray, heel_y_contra: np.ndarray, context: model.GaitEventContext
    ) -> dict[str, np.ndarray]:
        step_length = np.ndarray(len(heel_y_ipsi))
        for cycle_number in range(len(heel_y_contra)):
            step_length[cycle_number] = abs(heel_y_ipsi[cycle_number, 0] - heel_y_contra[cycle_number, 0])
        return {f"{context.name}_step_length": step_length}


class _StepHeightAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: _Step Height")

        right_heel_z_right = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, model.GaitEventContext.RIGHT)[model.AxesNames.z.value]
        left_heel_z_left = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, model.GaitEventContext.LEFT)[model.AxesNames.z.value]

        right = self._calculate_step_height(right_heel_z_right, model.GaitEventContext.RIGHT)
        left = self._calculate_step_height(left_heel_z_left, model.GaitEventContext.LEFT)
        right.update(left)
        return right

    def _calculate_step_height(self, context_position_x: np.ndarray, context: model.GaitEventContext) -> dict[str, np.ndarray]:
        height = np.ndarray(len(context_position_x))
        for cycle_number in range(len(context_position_x)):
            height[cycle_number] = np.nanmax(context_position_x[cycle_number]) - np.nanmin(context_position_x[cycle_number])

        return {f"{context.name}_step_height": height}


class _CycleDurationAnalysis(AbstractAnalysis):

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool) -> dict:
        logger.info("analyse: _Duration")

        left_meta = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
        right_meta = self.get_cycles_meta_data(model.GaitEventContext.LEFT)
        left = self._calculate_cycle_duration(left_meta, model.GaitEventContext.LEFT)
        right = self._calculate_cycle_duration(right_meta, model.GaitEventContext.RIGHT)
        right.update(left)
        return right

    def _calculate_cycle_duration(self, meta_data: dict[str, any], context: model.GaitEventContext) -> dict[str, np.ndarray]:
        frequency = 100
        cycle_length = meta_data["end_frame"] - meta_data["start_frame"]
        cycle_duration = cycle_length / frequency
        # step_duration = meta_data["Foot Off_IPSI"] - meta_data["start_frame"] / frequency
        # TODO: Find litetrautre for step duration
        stand_duration = meta_data["Foot Off_IPSI"] / frequency
        swing_duration = (cycle_length - meta_data["Foot Off_IPSI"]) / frequency
        double_support_duration = (
            meta_data["Foot Off_CONTRA"] + (meta_data["Foot Off_IPSI"] - meta_data["Foot Strike_CONTRA"])
        ) / frequency
        single_support_duration = (
            (meta_data["Foot Strike_CONTRA"] - meta_data["Foot Off_CONTRA"]) + (cycle_length - meta_data["Foot Off_IPSI"])
        ) / frequency

        perc_stand_duration = stand_duration / cycle_duration
        perc_swing_duration = swing_duration / cycle_duration
        perc_double_support_duration = double_support_duration / cycle_duration
        perc_single_support_duration = single_support_duration / cycle_duration
        return {
            f"{context.name}_cycle_duration_s": cycle_duration,
            #   f"{context.name}_step_duration": step_duration,
            f"{context.name}_swing_duration_prec": perc_swing_duration,
            f"{context.name}_stand_duration_prec": perc_stand_duration,
            f"{context.name}_double_support_duration_prec": perc_double_support_duration,
            f"{context.name}_single_support_duration_prec": perc_single_support_duration,
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
    logger.info("analyse: Minimal Toe Clearance")

    def get_data_condition(self) -> model.ExtractedCycleDataCondition:
        return model.ExtractedCycleDataCondition.RAW_DATA

    def _analyse(self, by_phase: bool, **kwargs) -> dict:

        ### Setup the data
        RIGHT = model.GaitEventContext.RIGHT
        LEFT = model.GaitEventContext.LEFT
        plot_cycle = kwargs.get("plot_cycle", False)
        cycle_num = kwargs.get("cycle_num", -1)

        RT_meta_2 = self.get_point_data(model.TranslatedLabel.RIGHT_META_2, RIGHT)
        RT_meta_5 = self.get_point_data(model.TranslatedLabel.RIGHT_META_5, RIGHT)
        RT_hee = self.get_point_data(model.TranslatedLabel.RIGHT_HEEL, RIGHT)
        LT_meta_2 = self.get_point_data(model.TranslatedLabel.LEFT_META_2, LEFT)
        LT_meta_5 = self.get_point_data(model.TranslatedLabel.LEFT_META_5, LEFT)
        LT_hee = self.get_point_data(model.TranslatedLabel.LEFT_HEEL, LEFT)

        ## Get the swing phase
        RT_SW_meta_2 = self.split_by_phase(RT_meta_2, self.get_cycles_meta_data(model.GaitEventContext.RIGHT))[1]
        RT_SW_meta_5 = self.split_by_phase(RT_meta_5, self.get_cycles_meta_data(model.GaitEventContext.RIGHT))[1]
        RT_SW_hee = self.split_by_phase(RT_hee, self.get_cycles_meta_data(model.GaitEventContext.RIGHT))[1]
        LT_SW_meta_2 = self.split_by_phase(LT_meta_2, self.get_cycles_meta_data(model.GaitEventContext.LEFT))[1]
        LT_SW_meta_5 = self.split_by_phase(LT_meta_5, self.get_cycles_meta_data(model.GaitEventContext.LEFT))[1]
        LT_SW_hee = self.split_by_phase(LT_hee, self.get_cycles_meta_data(model.GaitEventContext.LEFT))[1]

        ## Get the Y component of the toe
        RTY_meta_2 = RT_meta_2[model.AxesNames.y.value]
        RTY_meta_5 = RT_meta_5[model.AxesNames.y.value]
        RTY_hee = RT_hee[model.AxesNames.y.value]
        LTY_meta_2 = LT_meta_2[model.AxesNames.y.value]
        LTY_meta_5 = LT_meta_5[model.AxesNames.y.value]
        LTY_hee = LT_hee[model.AxesNames.y.value]
        ## Get the Y component of the swing phase
        RTY_SW_meta_2 = RT_SW_meta_2[model.AxesNames.y.value]
        RTY_SW_meta_5 = RT_SW_meta_5[model.AxesNames.y.value]
        RTY_SW_hee = RT_SW_hee[model.AxesNames.y.value]
        LTY_SW_meta_2 = LT_SW_meta_2[model.AxesNames.y.value]
        LTY_SW_meta_5 = LT_SW_meta_5[model.AxesNames.y.value]
        LTY_SW_hee = LT_SW_hee[model.AxesNames.y.value]
        ## Get the Z component of the toe
        RTZ_meta_2 = RT_meta_2[model.AxesNames.z.value]
        RTZ_hee = RT_hee[model.AxesNames.z.value]
        LTZ_meta_2 = LT_meta_2[model.AxesNames.z.value]
        LTZ_hee = LT_hee[model.AxesNames.z.value]

        ### Process the data

        ## Get the segment Y mean speed
        RY_mean_speed = self._get_mean_segment_speed(RTY_meta_2, RTY_meta_5, RTY_hee)
        LY_mean_speed = self._get_mean_segment_speed(LTY_meta_2, LTY_meta_5, LTY_hee)

        ## Get the segment Y mean speed during the swing phase
        RTY_SW_mean_vel = self._get_mean_segment_speed(RTY_SW_meta_2, RTY_SW_meta_5, RTY_SW_hee)
        LTY_SW_mean_vel = self._get_mean_segment_speed(LTY_SW_meta_2, LTY_SW_meta_5, LTY_SW_hee)

        ## Get the upper quartile of the swing phase
        R_quartile = self._calculate_sw_quartile(RTY_SW_mean_vel, **kwargs)
        L_quartile = self._calculate_sw_quartile(LTY_SW_mean_vel, **kwargs)

        # Find where the indexes where the speed is within the quartile
        R_interval = self._calculate_speed_range(RY_mean_speed, R_quartile)
        L_interval = self._calculate_speed_range(LY_mean_speed, L_quartile)

        if plot_cycle and cycle_num != -1:
            self._visualizeGaitCycle(
                cycle_num,
                RT_SW_meta_2,
                LT_SW_meta_2,
                RTZ_meta_2,
                RTZ_hee,
                LTZ_meta_2,
                LTZ_hee,
                RY_mean_speed,
                LY_mean_speed,
                RTY_SW_mean_vel,
                LTY_SW_mean_vel,
                R_quartile,
                L_quartile,
                R_interval,
                L_interval,
            )

        mtc_right = self._calculate_MTC(RTZ_meta_2, RTZ_hee, R_interval, RIGHT, **kwargs)
        mtc_left = self._calculate_MTC(LTZ_meta_2, LTZ_hee, L_interval, LEFT, **kwargs)
        mtc_right.update(mtc_left)
        return mtc_right

    @staticmethod
    def _visualizeGaitCycle(
        cycle_num,
        RT_SW_meta_2,
        LT_SW_meta_2,
        RTZ_meta_2,
        RTZ_hee,
        LTZ_meta_2,
        LTZ_hee,
        RY_mean_speed,
        LY_mean_speed,
        RTY_SW_mean_vel,
        LTY_SW_mean_vel,
        R_quartile,
        L_quartile,
        R_interval,
        L_interval,
    ):
        RTZ_SW_meta_2 = RT_SW_meta_2[model.AxesNames.z.value]
        LTZ_SW_meta_2 = LT_SW_meta_2[model.AxesNames.z.value]

        swing_indexes_right = []
        for row in RTZ_SW_meta_2:
            sw_index = np.where(~np.isnan(row))[0]
            swing_indexes_right.append(sw_index)

        swing_indexes_left = []
        for row in LTZ_SW_meta_2:
            sw_index = np.where(~np.isnan(row))[0]
            swing_indexes_left.append(sw_index)

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(RY_mean_speed[cycle_num], label="R_mean_speed")
        plt.plot(RTY_SW_mean_vel[cycle_num], label="RTY_SW_mean_vel")
        plt.plot(
            np.arange(len(RY_mean_speed[cycle_num])),
            np.ones(len(RY_mean_speed[cycle_num])) * R_quartile[cycle_num],
            "r",
            label="R_quartile",
        )
        plt.axvline(x=R_interval[cycle_num][0], color="r", linestyle="--")
        plt.axvline(x=R_interval[cycle_num][1], color="r", linestyle="--")
        plt.legend()
        plt.title("Speed of the Y component of the RIGHT toe")

        plt.subplot(2, 2, 2)
        plt.plot(LY_mean_speed[cycle_num], label="L_mean_speed")
        plt.plot(LTY_SW_mean_vel[cycle_num], label="LTY_SW_mean_vel")
        plt.plot(
            np.arange(len(LY_mean_speed[cycle_num])),
            np.ones(len(LY_mean_speed[cycle_num])) * L_quartile[cycle_num],
            "r",
            label="L_quartile",
        )
        plt.axvline(x=L_interval[cycle_num][0], color="r", linestyle="--")
        plt.axvline(x=L_interval[cycle_num][1], color="r", linestyle="--")
        plt.legend()
        plt.title("Speed of the Y component of the LEFT toe")

        plt.subplot(2, 2, 3)
        plt.plot(RTZ_meta_2[cycle_num], label="RTZ_meta_2")
        plt.plot(RTZ_hee[cycle_num], label="RTZ_hee")
        plt.axvline(x=R_interval[cycle_num][0], color="r", linestyle="--")
        plt.axvline(x=R_interval[cycle_num][1], color="r", linestyle="--")
        plt.axvline(x=swing_indexes_right[cycle_num][0], color="g", linestyle="--", label="start of swing")
        x_bound = plt.gca().get_xbound()
        y_bound = plt.gca().get_ybound()
        plt.fill_between(
            np.arange(swing_indexes_right[cycle_num][0], int(x_bound[1])),
            y_bound[0],
            y_bound[1],
            color="green",
            alpha=0.2,
            label="swing phase",
        )
        plt.legend()
        plt.title("Right Heel and Meta_2 Z Position")

        plt.subplot(2, 2, 4)
        plt.plot(LTZ_meta_2[cycle_num], label="LTZ_meta_2")
        plt.plot(LTZ_hee[cycle_num], label="LTZ_hee")
        plt.axvline(x=L_interval[cycle_num][0], color="r", linestyle="--")
        plt.axvline(x=L_interval[cycle_num][1], color="r", linestyle="--")
        plt.axvline(x=swing_indexes_left[cycle_num][0], color="g", linestyle="--", label="start of swing")
        x_bound = plt.gca().get_xbound()
        y_bound = plt.gca().get_ybound()
        plt.fill_between(
            np.arange(swing_indexes_left[cycle_num][0], int(x_bound[1])),
            y_bound[0],
            y_bound[1],
            color="green",
            alpha=0.2,
            label="swing phase",
        )
        plt.legend()
        plt.title("Left Heel and Meta_2 Z Position")
        plt.show()

    @staticmethod
    def _calculate_MTC(toe, hee, uniform_speed_region, side: model.GaitEventContext, **kwargs):
        min_tc_label = f"{side.name}_minimal_toe_clearance"
        min_tc_perc_label = f"{side.name}_minimal_toe_clearance_swing_perc"
        tc_at_heel_strike_label = f"{side.name}_toe_clearance_heel_strike"

        results = {
            min_tc_label: np.ndarray(len(toe)),
            min_tc_perc_label: np.ndarray(len(toe)),
            tc_at_heel_strike_label: np.ndarray(len(toe)),
        }

        for cycle_number in range(len(toe)):
            toe_cycle = toe[cycle_number]
            heel_cycle = hee[cycle_number]
            usr = uniform_speed_region[cycle_number]

            Z_meta_2 = toe_cycle[usr[0] : usr[1]]
            Z_hee = heel_cycle[usr[0] : usr[1]]

            fi, li = MinimalToeClearance._is_heel_above_toe(Z_meta_2, Z_hee)

            if fi == None and li == None:
                # Use the HEE marker instead
                toe_clear_min, tc_percent, tc_clear_hs = MinimalToeClearance._find_peaks(toe_cycle, usr, Z_hee, "Z_hee", 0)
                # TODO: Put in the log that the heel was used
                # print(f"Using the heel as the foot clearance for cycle: {cycle_number}")
                results[min_tc_label][cycle_number] = toe_clear_min
                results[min_tc_perc_label][cycle_number] = tc_percent
                results[tc_at_heel_strike_label][cycle_number] = tc_clear_hs
                continue

            if fi == li:
                # If the first and last index are the same, means that i found only one point and that is the
                # min toe clearance
                toe_clear_min = Z_meta_2[fi]
                toe_clear_index = fi + usr[0]
                tc_percent = toe_clear_index / len(toe_cycle)
                tc_clear_hs = toe_cycle[-1]

                results[min_tc_label][cycle_number] = toe_clear_min
                results[min_tc_perc_label][cycle_number] = tc_percent
                results[tc_at_heel_strike_label][cycle_number] = tc_clear_hs
                continue

            Z_meta_2 = Z_meta_2[fi:li]
            toe_clear_min, tc_percent, tc_clear_hs = MinimalToeClearance._find_peaks(toe_cycle, usr, Z_meta_2, "Z_meta_2", fi)

            results[min_tc_label][cycle_number] = toe_clear_min
            results[min_tc_perc_label][cycle_number] = tc_percent
            results[tc_at_heel_strike_label][cycle_number] = tc_clear_hs

        return results

    @staticmethod
    def _find_peaks(toe_cycle, usr, Z_marker, marker_name, fi, **kwargs):
        plot_cycle = kwargs.get("plot_cycle", False)
        plot_cycle_num = kwargs.get("cycle_num", -1)

        peaks = signal.find_peaks(-Z_marker)

        if len(peaks[0]) == 0:
            # If there are no peaks, means that the region is flat
            # so I just take the middle value and the middle index
            toe_clear_min = np.min(Z_marker)
            toe_clear_index = len(Z_marker) / 2 + usr[0] + (fi - 0)
            # The (fi - 0) is to account for the fact that it could happen that
            # the first index is 0 and to get the clearance index i have to account for that
        else:
            # Get the minimum of the found peaks
            toe_clear_min = np.min(Z_marker[peaks[0]])
            peaks_min = np.argmin(Z_marker[peaks[0]])
            toe_clear_index = peaks[0][peaks_min] + usr[0] + (fi - 0)

        tc_percent = toe_clear_index / len(toe_cycle)
        tc_clear_hs = toe_cycle[-1]

        if plot_cycle and plot_cycle_num != -1:
            # Plot the peaks in the region
            plt.figure()
            plt.plot(Z_marker, label=marker_name)
            for peak in peaks[0]:
                plt.plot(peak, Z_marker[peak], "ro", label="Peaks")
            plt.legend()
            plt.title(f"Peaks {plot_cycle_num}")
            plt.show()

        return toe_clear_min, tc_percent, tc_clear_hs

    @staticmethod
    def _is_heel_above_toe(Z_meta_2, Z_hee) -> tuple:
        """
        Determines if the heel is above the toe based on the given Z coordinates.

        Args:
            Z_meta_2 (list): List of Z coordinates for the meta 2 point.
            Z_hee (list): List of Z coordinates for the heel point.

        Returns:
            tuple: A tuple containing the indices of the first and last occurrences where the heel is above the toe.
                   If no occurrences are found, returns (None, None). If only one occurrence is found, returns (index, index).
        """
        result = []
        for i, (z_meta_2, z_hee) in enumerate(zip(Z_meta_2, Z_hee, strict=False)):
            if z_meta_2 < z_hee:
                result.append(i)

        if not result:
            return (None, None)

        if len(result) < 2:
            return (result[0], result[0])

        return (result[0], result[-1])

    @staticmethod
    def _calculate_sw_quartile(marker_p: np.ndarray, **kwargs) -> list[float]:
        """
        Calculate the swing phase upper quartile value for each row in the marker velocity array.

        Args:
            marker_p (np.ndarray): The input array containing marker data.

        Keyword Args:
            plot_quartile (bool, optional): Determines whether to plot the quartile values. Default is False.

        Returns:
            list[float]: A list containing the quartile value for each row in the marker velocity array.
        """

        plot_quartile = kwargs.get("plot_quartile", False)

        result = []
        for og_row in marker_p:
            mask = ~np.isnan(og_row)
            og_row = og_row[mask]
            og_max_value = np.max(og_row)
            og_idx = np.argmax(og_row)

            # Flip the velocity and maintain the shape
            if plot_quartile == 1:
                plt.figure()
                plt.plot(og_row)
                plt.plot(og_idx, og_max_value, "ro")
                plt.title("Average Velocity")
                plt.show()

            row = og_row - og_max_value
            row = abs(row)
            quartile = np.percentile(row, 75)
            max_value = np.max(row)
            idx = np.argmax(row)

            if plot_quartile == 1:
                plt.figure()
                plt.plot(row)
                plt.plot(idx, max_value, "ro")
                plt.plot(np.arange(len(row)), quartile * np.ones(len(row)), "r--")
                plt.plot(np.arange(len(row)), max_value * np.ones(len(row)), "r--")
                plt.title("Average Velocity ABS")
                plt.show()

            translated_quartile = quartile - np.abs(og_max_value)
            translated_quartile = -translated_quartile

            if plot_quartile == 1:
                plt.figure()
                plt.plot(og_row)
                plt.plot(np.arange(len(row)), translated_quartile * np.ones(len(og_row)), "r--")
                plt.title("Average Velocity ABS")
                plt.show()

            result.append(translated_quartile)

        return result

    @staticmethod
    def _get_mean_segment_speed(meta_2: np.ndarray, meta_5: np.ndarray, hee: np.ndarray) -> np.ndarray:
        """
        Calculate the mean speed and mean position of the given markers.

        Parameters:
        - meta_2 (np.ndarray): Array representing the meta_2 markers.
        - meta_5 (np.ndarray): Array representing the meta_5 markers.
        - hee (np.ndarray): Array representing the hee markers.

        Returns:
        - mean_speed (np.ndarray): Array representing the mean speed of the markers.
        """
        meta_2p, meta_5p, hee_p = MinimalToeClearance._calculate_velocity(meta_2, meta_5, hee)

        mean_speed = (meta_2p + meta_5p + hee_p) / 3
        return mean_speed

    @staticmethod
    def _calculate_velocity(meta_2: np.ndarray, meta_5: np.ndarray, hee: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the velocity of the given parameters.

        Args:
            meta_2 (np.ndarray): Array containing the values of meta_2.
            meta_5 (np.ndarray): Array containing the values of meta_5.
            hee (np.ndarray): Array containing the values of hee.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the calculated velocities for meta_2, meta_5, and hee.

        """
        # note:
        # diff(y) = yp
        # diff(yp) = ypp
        meta_2p = np.diff(meta_2)
        meta_5p = np.diff(meta_5)
        hee_p = np.diff(hee)
        meta_2p = np.insert(meta_2p, 0, np.nan, axis=1)
        meta_5p = np.insert(meta_5p, 0, np.nan, axis=1)
        hee_p = np.insert(hee_p, 0, np.nan, axis=1)

        return meta_2p, meta_5p, hee_p

    @staticmethod
    def _calculate_speed_range(foot_mean_speed: np.ndarray, quartile: list[float]) -> list[tuple[int, int]]:
        """
        Calculate the range of where Y component of the velocity is within the quartile.

        Args:
            foot_mean_speed (np.ndarray): Array of foot mean speeds.
            quartile (np.ndarray): Array of quartile values.

        Returns:
            list[tuple[int, int]]: List of tuples representing the ranges calculated.
        """
        result = []
        for i, row in enumerate(foot_mean_speed):
            indexes = np.where(row <= quartile[i])[0]
            result.append((indexes[0], indexes[-1]))
        return result
