import importlib
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from statistics import mean

import numpy as np

import gaitalytics.model as model

ANALOG_VOLTAGE_PREFIX_LABEL = "Voltage."

logger = logging.getLogger(__name__)


def is_progression_axes_flip(left_heel, left_toe):
    return 0 < mean(left_toe[model.AxesNames.y.value] - left_heel[model.AxesNames.y.value])


class FileHandler(ABC):

    def __init__(self, file_path: Path):
        self._file_path = str(file_path)
        self.read_file()

    def sort_events(self):
        """
        sort events in acquisition
        """
        events = self.get_events()

        value_frame: dict[int, model.GaitEvent] = {}

        for index in range(len(events)):
            if events[index].frame not in value_frame:
                value_frame[events[index].frame] = events[index]

        sorted_keys: dict[int, model.GaitEvent] = dict(sorted(value_frame.items()))

        self.clear_events()
        self.set_events(list(sorted_keys.values()))

    @abstractmethod
    def read_file(self):
        pass

    def write_file(self, out_file_path: str | Path | None = None):
        if out_file_path is None:
            out_file_path = self._file_path
        self._write_file(str(out_file_path))

    @abstractmethod
    def _write_file(self, out_file_path: str):
        pass

    @abstractmethod
    def get_events_size(self) -> int:
        pass

    @abstractmethod
    def get_events(self) -> list[model.GaitEvent]:
        pass

    @abstractmethod
    def set_events(self, events: list[model.GaitEvent]):
        pass

    @abstractmethod
    def get_event(self, index: int) -> model.GaitEvent:
        pass

    @abstractmethod
    def add_event(self, event: model.GaitEvent):
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
    def get_points_size(self) -> int:
        pass

    @abstractmethod
    def get_point(self, marker_index: int | str) -> model.Point:
        pass

    @abstractmethod
    def add_point(self, new_point: model.Point):
        pass


class BtkFileHandler(FileHandler):
    def __init__(self, file_path: Path):
        self._aqc = None
        self.btk = importlib.import_module("btk")
        super().__init__(file_path)

    @property
    def aqc(self):
        return self._aqc

    @aqc.setter
    def aqc(self, aqc):
        self._aqc = aqc

    def _write_file(self, out_file_path: str):
        """
        write a c3d with Btk

        Args:
            out_file_path (str): filename with its path
        """
        writer = self.btk.btkAcquisitionFileWriter()
        writer.SetInput(self._aqc)
        writer.SetFilename(out_file_path)
        writer.Update()

    def read_file(self):
        """
        read a c3d with btk
        """
        reader = self.btk.btkAcquisitionFileReader()
        reader.SetFilename(self._file_path)
        reader.Update()
        self._aqc = reader.GetOutput()

        # sort events
        self.sort_events()

    def get_events_size(self) -> int:
        return self._aqc.GetEventNumber()

    def get_events(self) -> list[model.GaitEvent]:
        events: list[model.GaitEvent] = []
        for event in self.btk.Iterate(self._aqc.GetEvents()):
            events.append(self.map_btk_event(event))
        return events

    def set_events(self, events: list[model.GaitEvent]):
        new_events = self.btk.btkEventCollection()
        for gait_event in events:
            new_events.InsertItem(self.map_event(gait_event))

        self._aqc.SetEvents(new_events)

    def get_event(self, index: int) -> model.GaitEvent:
        btk_event = self._aqc.GetEvent(index)
        return self.map_btk_event(btk_event)

    def clear_events(self):
        self._aqc.ClearEvents()

    def add_event(self, event: model.GaitEvent):
        self._aqc.AppendEvent(self.map_event(event))

    def get_point_frequency(self) -> int:
        return self._aqc.GetPointFrequency()

    def get_actual_start_frame(self) -> int:
        return self._aqc.GetMetaData().GetChild("TRIAL").GetChild("ACTUAL_START_FIELD").GetInfo().ToInt()[0] - 1

    def get_point(self, marker_index: int | str) -> model.Point:
        return self.map_btk_point(self._aqc.GetPoint(marker_index))

    def get_points_size(self) -> int:
        return self._aqc.GetPointNumber()

    def add_point(self, new_point: model.Point):
        point = self.btk.btkPoint(new_point.type.value)
        point.SetValues(new_point.values)
        point.SetLabel(new_point.label)
        self._aqc.AppendPoint(point)

    def map_btk_event(self, btk_event) -> model.GaitEvent:
        gait_event = model.GaitEvent(self.get_actual_start_frame(), self.get_point_frequency())
        gait_event.time = btk_event.GetTime()
        gait_event.context = btk_event.GetContext()
        gait_event.subject = btk_event.GetSubject()
        gait_event.icon_id = btk_event.GetId()
        gait_event.description = btk_event.GetDescription()
        gait_event.generic_flag = btk_event.GetDetectionFlags()
        gait_event.label = btk_event.GetLabel()
        return gait_event

    def map_event(self, gait_event: model.GaitEvent):
        btk_event = self.btk.btkEvent()
        btk_event.SetTime(gait_event.time)
        btk_event.SetContext(gait_event.context)
        btk_event.SetSubject(gait_event.subject)
        btk_event.SetId(gait_event.icon_id)
        btk_event.SetDescription(gait_event.description)
        btk_event.SetDetectionFlags(gait_event.generic_flag)
        btk_event.SetLabel(gait_event.label)
        return btk_event

    @staticmethod
    def map_btk_point(btk_point):
        point = model.Point()
        point.values = btk_point.GetValues()
        point.label = btk_point.GetLabel()
        point.residuals = btk_point.GetResiduals()
        point.type = model.PointDataType(btk_point.GetType())
        return point


class EzC3dFileHandler(FileHandler):
    _MAX_EVENTS_PER_SECTION = 255

    def __init__(self, file_path: Path):
        self._c3d = None
        self._ez = importlib.import_module("ezc3d")
        super().__init__(file_path)

    def read_file(self):
        self._c3d = self._ez.c3d(self._file_path)

    def _write_file(self, out_file_path: str):
        self._c3d.write(out_file_path)

    def get_events_size(self) -> int:
        return self._c3d["parameters"]["EVENT"]["USED"]["value"].tolist()[0]

    def get_events(self) -> list[model.GaitEvent]:
        events = []
        for index in range(self.get_events_size()):
            events.append(self.get_event(index=index))
        return events

    def set_events(self, events: list[model.GaitEvent]):
        self.clear_events()
        for event in events:
            self.add_event(event)

    def get_event(self, index: int) -> model.GaitEvent:
        para_names = self._get_parameter_name(index)
        index = index % self._MAX_EVENTS_PER_SECTION
        time = self._c3d["parameters"]["EVENT"][para_names["TIMES"]]["value"][:, index]
        context = self._c3d["parameters"]["EVENT"][para_names["CONTEXTS"]]["value"][index]
        label = self._c3d["parameters"]["EVENT"][para_names["LABELS"]]["value"][index]
        description = self._c3d["parameters"]["EVENT"][para_names["DESCRIPTIONS"]]["value"][index]
        subject = self._c3d["parameters"]["EVENT"][para_names["SUBJECTS"]]["value"][index]
        icon_id = self._c3d["parameters"]["EVENT"][para_names["ICON_IDS"]]["value"].tolist()[index]
        generic_flag = self._c3d["parameters"]["EVENT"][para_names["GENERIC_FLAGS"]]["value"].tolist()[index]
        return self.map_ez_event(time, context, label, description, subject, icon_id, generic_flag)

    def add_event(self, event: model.GaitEvent):
        self._add_event(event)

    def clear_events(self):
        if "EVENT" in self._c3d["parameters"]:
            event_param = self._c3d["parameters"]["EVENT"]
            to_delete = []
            for key in event_param.keys():
                if key != "USED":
                    to_delete.append(key)

            for key in to_delete:
                if key != "__METADATA__":
                    del self._c3d["parameters"]["EVENT"][key]

    def get_point_frequency(self) -> int:
        return self._c3d["parameters"]["TRIAL"]["CAMERA_RATE"]["value"][0]

    def get_actual_start_frame(self) -> int:
        return self._c3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0]

    def get_points_size(self) -> int:
        return self._c3d["parameters"]["POINT"]["USED"]["value"].tolist()[0]

    def get_point(self, marker_index: int | str) -> model.Point:

        if isinstance(marker_index, str):
            index = self._c3d["parameters"]["POINT"]["LABELS"]["value"].index(marker_index)
        else:
            index = marker_index
        return self.map_ez_point(index)

    def add_point(self, new_point: model.Point):
        pass

    def map_ez_event(
        self, time, context: str, label: str, description: str, subject: str, icon_id: int, generic_flag: int
    ) -> model.GaitEvent:
        event = model.GaitEvent(self.get_actual_start_frame(), self.get_point_frequency())
        event.time = time[0] * 60 + time[1]
        event.context = context
        event.label = label
        event.description = description
        event.subject = subject
        event.icon_id = icon_id
        event.generic_flag = generic_flag
        return event

    def map_ez_point(self, index: int) -> model.Point:
        label = self._c3d["parameters"]["POINT"]["LABELS"]["value"][index]
        values = self._c3d["data"]["points"][:, index].T
        residuals = self._c3d["data"]["meta_points"]["residuals"][:, index]
        type = model.PointDataType.MARKERS
        for group_name in self._c3d["parameters"]["POINT"]["TYPE_GROUPS"]["value"]:
            if group_name in self._c3d["parameters"]["POINT"]:
                group = self._c3d["parameters"]["POINT"][group_name]
                if group is not None:
                    if label in group["value"]:
                        type = model.PointDataType[group_name]
        point = model.Point()
        point.label = label
        point.values = values
        point.type = type
        point.residuals = residuals
        return point

    def _add_event(self, event: model.GaitEvent):
        """
        This function adds an event, warning two events can have the same name (it won't override it)

        """
        used = 0
        times = [[], []]
        contexts = []
        labels = []
        descriptions = []
        subjects = []
        icon_ids = []
        generic_flags = []

        if "EVENT" in self._c3d["parameters"]:
            event_param = self._c3d["parameters"]["EVENT"]
            used = event_param["USED"]["value"].tolist()[0]

            param_names = self._get_parameter_name(used)
            try:
                if used % self._MAX_EVENTS_PER_SECTION > 0:
                    times = event_param[param_names["TIMES"]]["value"].tolist()
                    contexts = event_param[param_names["CONTEXTS"]]["value"]
                    labels = event_param[param_names["LABELS"]]["value"]
                    descriptions = event_param[param_names["DESCRIPTIONS"]]["value"]
                    subjects = event_param[param_names["SUBJECTS"]]["value"]
                    icon_ids = event_param[param_names["ICON_IDS"]]["value"].tolist()
                    generic_flags = event_param[param_names["GENERIC_FLAGS"]]["value"].tolist()
            except KeyError as e:
                print(e)

            used += 1
            times[0] += [0]
            times[1] += [event.time]
            times = np.array(times)
            contexts += [event.context]
            labels += [event.label]
            descriptions += [event.description]
            subjects += [event.subject]
            icon_ids += [event.icon_id]
            generic_flags += [event.generic_flag]

            self._set_events(contexts, descriptions, generic_flags, icon_ids, labels, subjects, times, used, param_names)

    def _get_parameter_name(self, index: int) -> dict[str, str]:
        suffix = int(index / self._MAX_EVENTS_PER_SECTION)
        suffix += 1
        suffix = "" if suffix <= 1 else str(suffix)

        TIMES = "TIMES"
        CONTEXTS = "CONTEXTS"
        LABELS = "LABELS"
        DESCRIPTIONS = "DESCRIPTIONS"
        SUBJECTS = "SUBJECTS"
        ICON_IDS = "ICON_IDS"
        GENERIC_FLAGS = "GENERIC_FLAGS"

        return {
            TIMES: TIMES + suffix,
            CONTEXTS: CONTEXTS + suffix,
            LABELS: LABELS + suffix,
            DESCRIPTIONS: DESCRIPTIONS + suffix,
            SUBJECTS: SUBJECTS + suffix,
            ICON_IDS: ICON_IDS + suffix,
            GENERIC_FLAGS: GENERIC_FLAGS + suffix,
        }

    def _set_events(self, contexts, descriptions, generic_flags, icon_ids, labels, subjects, times, used, param_names):
        self._c3d.add_parameter("EVENT", "USED", used)
        self._c3d.add_parameter("EVENT", param_names["TIMES"], times)
        self._c3d.add_parameter("EVENT", param_names["CONTEXTS"], contexts)
        self._c3d.add_parameter("EVENT", param_names["LABELS"], labels)
        self._c3d.add_parameter("EVENT", param_names["DESCRIPTIONS"], descriptions)
        self._c3d.add_parameter("EVENT", param_names["SUBJECTS"], subjects)
        self._c3d.add_parameter("EVENT", param_names["ICON_IDS"], icon_ids)
        self._c3d.add_parameter("EVENT", param_names["GENERIC_FLAGS"], generic_flags)

    def print_structure(self):
        self._print_structure(self._c3d)

    @staticmethod
    def _print_structure(node, prefix: str = ""):
        for key in node.keys():
            print(f"{prefix}{key}")
            try:
                EzC3dFileHandler._print_structure(node.get(key), prefix=prefix + "  ")
            except AttributeError:
                pass
            finally:
                pass
