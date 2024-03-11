from __future__ import annotations

from enum import Enum
from enum import auto

import numpy as np


class PointDataType(Enum):
    MARKERS = 0
    ANGLES = 1
    FORCES = 2
    MOMENTS = 3
    POWERS = 4
    SCALARS = 5
    REACTIONS = 6
    MODELLED_MARKERS = 7


class AxesNames(Enum):
    x = 0
    y = 1
    z = 2


class GaitEventContext(Enum):
    """
    Representation of gait event contexts. At the moment mainly left and right
    """

    LEFT = "Left"
    RIGHT = "Right"

    @classmethod
    def get_contrary_context(cls, context: str):
        if context == cls.LEFT.value:
            return cls.RIGHT
        return cls.LEFT


class GaitEventLabel(Enum):
    FOOT_STRIKE = "Foot Strike"
    FOOT_OFF = "Foot Off"

    @classmethod
    def get_contrary_event(cls, event_label: str):
        if event_label == cls.FOOT_STRIKE.value:
            return cls.FOOT_OFF
        return cls.FOOT_STRIKE

    @classmethod
    def get_type_id(cls, event_label: str):
        if event_label == cls.FOOT_STRIKE.value:
            return 1
        return 2


class SubjectMeasures:

    def __init__(
        self,
        body_mass: float,
        body_height: float,
        left_leg_length: float,
        right_leg_length: float,
        subject: str,
        start_frame: int,
        mocap_frequency: float,
    ):
        self.body_mass = body_mass
        self.body_height = body_height
        self.left_leg_length = left_leg_length
        self.right_leg_length = right_leg_length
        self.subject = subject
        self.start_frame = start_frame
        self.mocap_frequency = mocap_frequency


class Point:
    """
    Class representing a point with various attributes.

    Attributes:
        description (str): A description of the point.
        frame_number (int): The frame number associated with the point.
        label (str): The label of the point.
        residual (float): A residual value associated with the point.
        residuals (list[float]): A list of residual values.
        timestamp (float): The timestamp of the point.
        type (PointDataType): The type of the point, as defined by PointDataType.
        values (list[float]): The values of the point.
    """

    def __init__(self) -> None:
        """Initialize a new Point instance with default None values for all attributes."""
        self._frame_number = None
        self._label = None
        self._residuals = None
        self._type = None
        self._values = None

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Label must be a string")
        self._label = value

    @property
    def residuals(self) -> list[float]:
        return self._residuals

    @residuals.setter
    def residuals(self, value: list[float]) -> None:
        self._residuals = value

    @property
    def type(self) -> PointDataType:
        return self._type

    @type.setter
    def type(self, value: PointDataType) -> None:
        # Replace PointDataType with the actual type check if needed
        self._type = value

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, value: np.ndarray) -> None:
        self._values = value


class GaitEvent:
    """
    Abstract class representing a gait event.

    Attributes:
        time (list[float], tuple[float]): The time of the gait event.
        context (str): The context of the gait event.
        label (str): The label of the gait event.
        description (str): A description of the gait event.
        subject (str): The subject related to the gait event.
        icon_id (float): An identifier for an icon associated with the gait event.
        generic_flag (int): An identifier for a generic flag associated with the event.
    """

    def __init__(self, actual_start: int, frame_frequency: int):
        """Initialize a new GaitEvent instance with None values for all attributes."""
        self._time = 0
        self._frame = 0
        self._context = ""
        self._label = ""
        self._description = ""
        self._subject = ""
        self._icon_id = 0
        self._generic_flag = 0
        self._freq = frame_frequency
        self._file_start = actual_start

    @property
    def time(self) -> float:
        """Get or set the time of the gait event in seconds."""
        return self._time

    @time.setter
    def time(self, value: float):
        if isinstance(value, float):
            self._time = value
            self._frame = int(round(value * self._freq))
        else:
            raise TypeError("Time must be a float")

    @property
    def frame(self) -> int:
        """Get or set the frame of the gait event"""
        return self._frame

    @frame.setter
    def frame(self, value: int):
        if isinstance(value, int):
            self._frame = value
            self._time = value / self._freq
        else:
            raise TypeError("Frame must be an int")

    @property
    def context(self) -> str:
        """Get or set the context of the gait event. Must be a string"""
        return self._context

    @context.setter
    def context(self, value: str):
        if isinstance(value, str):
            self._context = value
        else:
            raise TypeError("Context must be a string")

    @property
    def label(self) -> str:
        """Get or set the label of the gait event. Must be a string"""
        return self._label

    @label.setter
    def label(self, value: str):
        if isinstance(value, str):
            self._label = value
        else:
            raise TypeError("Label must be a string")

    @property
    def description(self) -> str:
        """Get or set the description of the gait event. Must be a string"""
        return self._description

    @description.setter
    def description(self, value: str):
        if isinstance(value, str):
            self._description = value
        else:
            raise TypeError("Description must be a string")

    @property
    def subject(self) -> str:
        """Get or set the subject of the gait event. Must be a string"""
        return self._subject

    @subject.setter
    def subject(self, value: str):
        if isinstance(value, str):
            self._subject = value
        else:
            raise TypeError("Subject must be a string")

    @property
    def icon_id(self) -> float | int:
        """Get or set the icon ID of the gait event. Must be a float"""
        return self._icon_id

    @icon_id.setter
    def icon_id(self, value: float | int):
        if isinstance(value, float) or isinstance(value, int):
            self._icon_id = value
        else:
            raise TypeError("Icon ID must be a float or an integer")

    @property
    def generic_flag(self) -> float | int:
        return self._generic_flag

    @generic_flag.setter
    def generic_flag(self, value: float | int):
        if isinstance(value, int) or isinstance(value, float):
            self._generic_flag = value
        else:
            raise TypeError("GenericFlag must be a float or an integer")


class GaitCycle:

    def __init__(self, number: int, context: GaitEventContext, start_frame: int, end_frame: int, unused_events: list[GaitEvent]):
        self.number: int = number
        self.context: GaitEventContext = context
        self.start_frame: int = start_frame
        self.end_frame: int = end_frame
        self.length: int = end_frame - start_frame
        self.unused_events: dict | None = None
        self._unused_events_to_dict(unused_events)

    def _unused_events_to_dict(self, unused_events: list[GaitEvent]):
        if len(unused_events) <= 3:
            self.unused_events = {}
            for unused_event in unused_events:
                key_postfix = "IPSI" if unused_event.context == self.context.value else "CONTRA"
                self.unused_events[f"{unused_event.label}_{key_postfix}"] = unused_event.frame - self.start_frame

        else:
            raise ValueError(f"too much events in cycle nr. {self.number}")


class GaitCycleList:

    def __init__(self):
        self.cycles: list[GaitCycle] = []
        self.index_left_cycles: list[int] = []
        self.index_right_cycles: list[int] = []

    def add_cycle(self, cycle: GaitCycle):
        self.cycles.append(cycle)
        index = len(self.cycles) - 1
        if cycle.context == GaitEventContext.LEFT:
            self.index_left_cycles.append(index)
        else:
            self.index_right_cycles.append(index)

    def get_longest_cycle_length(self, side: GaitEventContext) -> int:
        length = 0
        cycle_index = self.index_left_cycles if side == GaitEventContext.LEFT else self.index_right_cycles
        for index in cycle_index:
            cycle = self.cycles[index]
            if cycle.context == side:
                length = length if length > cycle.length else cycle.length
        return length

    def get_number_of_cycles(self, side: GaitEventContext) -> int:
        num = len(self.index_left_cycles) if side == GaitEventContext.LEFT else len(self.index_right_cycles)
        return num


class ExtractedCyclePoint:
    def __init__(self, translated_label: Enum, point_type: PointDataType):
        super().__init__()
        self.translated_label = translated_label
        self.point_type: PointDataType = point_type
        self.data_table: np.ndarray | None = None


class ExtractedContextCycles:

    def __init__(self, context: GaitEventContext):
        self.context: GaitEventContext = context
        self.points: list[ExtractedCyclePoint] = []
        self.meta_data: dict[str, np.ndarray] | None = None

    def add_cycle_points(self, point_cycles: ExtractedCyclePoint):
        self.points.append(point_cycles)


class ExtractedCycleDataCondition(Enum):
    RAW_DATA = "raw"
    NORM_DATA = "norm"


class ExtractedCycles:

    def __init__(
        self,
        data_condition: ExtractedCycleDataCondition,
        subject: SubjectMeasures,
        cycle_context_points: dict[GaitEventContext, ExtractedContextCycles],
    ):
        self.data_condition: ExtractedCycleDataCondition = data_condition
        self.subject: SubjectMeasures = subject
        self.cycle_points: dict[GaitEventContext, ExtractedContextCycles] = cycle_context_points


class AutoEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class TranslatedLabel(AutoEnum):
    LEFT_BACK_HIP = auto()
    RIGHT_BACK_HIP = auto()
    LEFT_FRONT_HIP = auto()
    RIGHT_FRONT_HIP = auto()

    LEFT_LAT_UPPER_LEG = auto()
    RIGHT_LAT_UPPER_LEG = auto()

    LEFT_LAT_KNEE = auto()
    RIGHT_LAT_KNEE = auto()
    LEFT_MED_KNEE = auto()
    RIGHT_MED_KNEE = auto()

    LEFT_LAT_LOWER_LEG = auto()
    RIGHT_LAT_LOWER_LEG = auto()

    LEFT_LAT_MALLEOLI = auto()
    RIGHT_LAT_MALLEOLI = auto()
    LEFT_MED_MALLEOLI = auto()
    RIGHT_MED_MALLEOLI = auto()

    RIGHT_HEEL = auto()
    LEFT_HEEL = auto()
    RIGHT_META_2 = auto()
    LEFT_META_2 = auto()
    RIGHT_META_5 = auto()
    LEFT_META_5 = auto()

    NECK = auto()
    BACK = auto()
    JUGULARIS = auto()
    XYPHOIDEUS = auto()

    # ADDITIONAL MARKERS WHICH WILL BE MODELLED IN MODELLING.PY
    COM = auto()
    XCOM = auto()
    CMOS = auto()

    LEFT_THORAX_ANGLES = auto()
    RIGHT_THORAX_ANGLES = auto()
    LEFT_SPINE_ANGLES = auto()
    RIGHT_SPINE_ANGLES = auto()
    LEFT_PELVIS_ANGLES = auto()
    RIGHT_PELVIS_ANGLES = auto()
    LEFT_FOOT_PROGRESSION_ANGLES = auto()
    RIGHT_FOOT_PROGRESSION_ANGLES = auto()
    LEFT_HIP_ANGLES = auto()
    RIGHT_HIP_ANGLES = auto()
    LEFT_KNEE_ANGLES = auto()
    RIGHT_KNEE_ANGLES = auto()
    LEFT_ANKLE_ANGLES = auto()
    RIGHT_ANKLE_ANGLES = auto()

    # FORCES
    LEFT_GRF = auto()
    RIGHT_GRF = auto()
    LEFT_NGRF = auto()
    RIGHT_NGRF = auto()
    LEFT_WAIST_FORCE = auto()
    RIGHT_WAIST_FORCE = auto()
    LEFT_HIP_FORCE = auto()
    RIGHT_HIP_FORCE = auto()
    LEFT_KNEE_FORCE = auto()
    RIGHT_KNEE_FORCE = auto()
    LEFT_ANKLE_FORCE = auto()
    RIGHT_ANKLE_FORCE = auto()

    # MOMENTS
    LEFT_GRM = auto()
    RIGHT_GRM = auto()
    LEFT_WAIST_MOMENT = auto()
    RIGHT_WAIST_MOMENT = auto()
    LEFT_HIP_MOMENT = auto()
    RIGHT_HIP_MOMENT = auto()
    LEFT_KNEE_MOMENT = auto()
    RIGHT_KNEE_MOMENT = auto()
    LEFT_ANKLE_MOMENT = auto()
    RIGHT_ANKLE_MOMENT = auto()

    # POWERS
    LEFT_WAIST_POWER = auto()
    RIGHT_WAIST_POWER = auto()
    LEFT_HIP_POWER = auto()
    RIGHT_HIP_POWER = auto()
    LEFT_KNEE_POWER = auto()
    RIGHT_KNEE_POWER = auto()
    LEFT_ANKLE_POWER = auto()
    RIGHT_ANKLE_POWER = auto()
