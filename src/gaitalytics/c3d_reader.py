import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from statistics import mean
from typing import Optional
from typing import Union

import btk

import gaitalytics.model
import gaitalytics.utils

ANALOG_VOLTAGE_PREFIX_LABEL = "Voltage."

logger = logging.getLogger(__name__)


def is_progression_axes_flip(left_heel, left_toe):
    return 0 < mean(left_toe[gaitalytics.model.AxesNames.y.value] - left_heel[gaitalytics.model.AxesNames.y.value])


class FileHandler(ABC):
    def __init__(self, file_path: Path):
        self._file_path = file_path
        self.read_file()

    def sort_events(self):
        """
        sort events in acquisition
        """
        events = self.get_events()

        value_frame: dict[int, gaitalytics.model.GaitEvent] = {}

        for index in range(len(events)):
            if events[index].frame not in value_frame:
                value_frame[events[index].frame] = events[index]

        sorted_keys: dict[int, gaitalytics.model.GaitEvent] = dict(sorted(value_frame.items()))

        self.clear_events()
        self.set_events(list(sorted_keys.values()))

    @abstractmethod
    def read_file(self):
        pass

    def write_file(self, out_file_path: Optional[Union[str, Path]] = None):
        if out_file_path is None:
            out_file_path = self._file_path
        self._write_file(out_file_path)

    @abstractmethod
    def _write_file(self, out_file_path: str):
        pass

    @abstractmethod
    def get_events_size(self) -> int:
        pass

    @abstractmethod
    def get_events(self) -> list[gaitalytics.model.GaitEvent]:
        pass

    @abstractmethod
    def set_events(self, events: list[gaitalytics.model.GaitEvent]):
        pass

    @abstractmethod
    def get_event(self, index: int) -> gaitalytics.model.GaitEvent:
        pass

    @abstractmethod
    def add_event(self, event: gaitalytics.model.GaitEvent):
        pass

    @abstractmethod
    def clear_events(self):
        pass

    @abstractmethod
    def get_point_frequency(self) -> int:
        pass

    @abstractmethod
    def get_actual_start_frame(self) -> int:
        pass

    @abstractmethod
    def get_subject_measures(self) -> gaitalytics.model.SubjectMeasures:
        pass

    @abstractmethod
    def get_points_size(self) -> int:
        pass

    @abstractmethod
    def get_point(self, marker_index: Union[int, str]) -> gaitalytics.model.Point:
        pass

    @abstractmethod
    def add_point(self, new_point: gaitalytics.model.Point):
        pass


class BtkFileHandler(FileHandler):

    def __init__(self, file_path: Path):
        self._aqc = None
        super().__init__(file_path)

    @property
    def aqc(self):
        return self._aqc

    @aqc.setter
    def aqc(self, aqc):
        self._aqc = aqc

    def get_subject_measures(self) -> gaitalytics.model.SubjectMeasures:
        body_mass = self._aqc.GetMetaData().GetChild("PROCESSING").GetChild("Bodymass").GetInfo().ToDouble()[0]
        body_height = self._aqc.GetMetaData().GetChild("PROCESSING").GetChild("Height").GetInfo().ToDouble()[0]
        left_leg_length = self._aqc.GetMetaData().GetChild("PROCESSING").GetChild("LLegLength").GetInfo().ToDouble()[0]
        right_leg_length = self._aqc.GetMetaData().GetChild("PROCESSING").GetChild("RLegLength").GetInfo().ToDouble()[0]
        name = self._aqc.GetMetaData().GetChild("SUBJECTS").GetChild("NAMES").GetInfo().ToString()[0].strip()
        start_frame = self._aqc.GetMetaData().GetChild("TRIAL").GetChild("ACTUAL_START_FIELD").GetInfo().ToInt()[0]
        subject = gaitalytics.model.SubjectMeasures(body_mass, body_height, left_leg_length, right_leg_length, name, start_frame)
        return subject

    def _write_file(self, out_file_path: Path):
        """
        write a c3d with Btk

        Args:
            out_file_path (str): filename with its path
        """
        writer = btk.btkAcquisitionFileWriter()
        writer.SetInput(self._aqc)
        writer.SetFilename(out_file_path.absolute().__str__())
        writer.Update()

    def read_file(self):
        """
        read a c3d with btk
        """
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(self._file_path.absolute().__str__())
        reader.Update()
        self._aqc = reader.GetOutput()

        # sort events
        self.sort_events()

    def get_events_size(self) -> int:
        return self._aqc.GetEventNumber()

    def get_events(self) -> list[gaitalytics.model.GaitEvent]:
        events: list[gaitalytics.model.GaitEvent] = []
        for event in btk.Iterate(self._aqc.GetEvents()):
            events.append(self.map_btk_event(event))
        return events

    def set_events(self, events: list[gaitalytics.model.GaitEvent]):
        new_events = btk.btkEventCollection()
        for gait_event in events:
            new_events.InsertItem(self.map_event(gait_event))

        self._aqc.SetEvents(new_events)

    def get_event(self, index: int) -> gaitalytics.model.GaitEvent:
        btk_event = self._aqc.GetEvent(index)
        return self.map_btk_event(btk_event)

    def clear_events(self):
        self._aqc.ClearEvents()

    def add_event(self, event: gaitalytics.model.GaitEvent):
        self._aqc.AppendEvent(self.map_event(event))

    def get_point_frequency(self) -> int:
        return self._aqc.GetPointFrequency()

    def get_actual_start_frame(self) -> int:
        return self._aqc.GetMetaData().GetChild("TRIAL").GetChild("ACTUAL_START_FIELD").GetInfo().ToInt()[0] - 1

    def get_point(self, marker_index: Union[int, str]) -> gaitalytics.model.Point:
        return self.map_btk_point(self._aqc.GetPoint(marker_index))

    def get_points_size(self) -> int:
        return self._aqc.GetPointNumber()

    def add_point(self, new_point: gaitalytics.model.Point):
        point = btk.btkPoint(new_point.type.value)
        point.SetValues(new_point.values)
        point.SetLabel(new_point.label)
        self._aqc.AppendPoint(point)

    def map_btk_event(self, btk_event: btk.btkEvent) -> gaitalytics.model.GaitEvent:
        gait_event = gaitalytics.model.GaitEvent(self.get_actual_start_frame(), self.get_point_frequency())
        gait_event.time = btk_event.GetTime()
        gait_event.context = btk_event.GetContext()
        gait_event.subject = btk_event.GetSubject()
        gait_event.icon_id = btk_event.GetId()
        gait_event.description = btk_event.GetDescription()
        gait_event.generic_flag = btk_event.GetDetectionFlags()
        gait_event.label = btk_event.GetLabel()
        return gait_event

    @staticmethod
    def map_event(gait_event: gaitalytics.model.GaitEvent) -> btk.btkEvent:
        btk_event = btk.btkEvent()
        btk_event.SetTime(gait_event.time)
        btk_event.SetContext(gait_event.context)
        btk_event.SetSubject(gait_event.subject)
        btk_event.SetId(gait_event.icon_id)
        btk_event.SetDescription(gait_event.description)
        btk_event.SetDetectionFlags(gait_event.generic_flag)
        btk_event.SetLabel(gait_event.label)
        return btk_event

    @staticmethod
    def map_btk_point(btk_point: btk.btkPoint):
        point = gaitalytics.model.Point()
        point.values = btk_point.GetValues()
        point.label = btk_point.GetLabel()
        point.residuals = btk_point.GetResiduals()
        point.type = gaitalytics.model.PointDataType(btk_point.GetType())
        return point
