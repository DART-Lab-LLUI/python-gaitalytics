from enum import Enum
from pathlib import Path

from yaml import safe_load


class MappedMarkers(Enum):
    # Foot
    L_HEEL = "l_heel"
    R_HEEL = "r_heel"
    L_TOE = "l_toe"
    R_TOE = "r_toe"
    L_ANKLE = "l_lat_malleoli"
    R_ANKLE = "r_lat_malleoli"
    # Additional toe markers
    L_TOE_2 = "l_toe_2"
    R_TOE_2 = "r_toe_2"

    # Hip
    L_ANT_HIP = "l_ant_hip"
    R_POST_HIP = "r_post_hip"
    L_POST_HIP = "l_post_hip"
    R_ANT_HIP = "r_ant_hip"
    SACRUM = "sacrum"

    # Extrapolated center of mass marker for margin of stability
    XCOM = "xcom"


class MappingConfigs:
    """A class for reading the mapping configuration file.

    This class provides methods to read the mapping configuration file and get
    the markers and analogs for analysis.
    """

    _SEC_ANALYSIS: str = "analysis"
    _SEC_MARKERS_ANALYSIS: str = "markers"
    _SEC_ANALOGS_ANALYSIS: str = "analogs"

    _SEC_MAPPING: str = "mapping"
    _SEC_MARKERS_MAPPING: str = "markers"
    _SEC_ANALOGS_MAPPING: str = "analogs"

    def __init__(self, config_path: Path):
        """Initializes a new instance of the MappingConfigs class.

        Reads the yaml file into memory.

        Args:
            config_path: The path to the configuration file.
        """
        self._configs: dict[str, dict] = {}
        with open(config_path) as stream:
            self._configs = safe_load(stream)

    def get_markers_analysis(self) -> list[str]:
        """Gets the markers for analysis.

        Returns:
            A list of marker names to be used for analysis
            if present in the config file, otherwise an empty list.

        Raises:
            ValueError: If the analysis section is missing in the config file.
        """
        self._check_analysis_section()
        if self._SEC_MARKERS_ANALYSIS not in self._configs[self._SEC_ANALYSIS]:
            return []
        return self._configs[self._SEC_ANALYSIS][self._SEC_MARKERS_ANALYSIS]

    def get_analogs_analysis(self) -> list[str]:
        """Gets the analogs for analysis.

        Returns:
            A list of analog names to be used for analysis
            if present in the config file, otherwise an empty list.

        Raises:
            ValueError: If the analysis section is missing in the config file.
        """
        self._check_analysis_section()
        if self._SEC_ANALOGS_ANALYSIS not in self._configs[self._SEC_ANALYSIS]:
            return []
        return self._configs[self._SEC_ANALYSIS][self._SEC_ANALOGS_ANALYSIS]

    def _check_analysis_section(self):
        """Checks if the analysis section is present in the config file.

        Raises:
            ValueError: If the analysis section is missing in the config file.
        """
        if self._configs is None or self._SEC_ANALYSIS not in self._configs:
            raise ValueError("Analysis section is missing in the config file.")

    def get_marker_mapping(self, marker: MappedMarkers) -> str:
        """Gets the mapping of markers.

        Args:
            marker: The marker to get the mapping for.

        Returns:
            The mapped marker name if present in the config file.

        Raises:
            ValueError: If sections in the mapping are missing in the config file.
        """
        self._check_marker_mapping()

        return self._configs[self._SEC_MAPPING][self._SEC_MARKERS_MAPPING][marker.value]

    def _check_marker_mapping(self):
        """Checks if the marker mapping section is present in the config file.

        Raises:
            ValueError: If the mapping section is missing in the config file.
        """
        if self._configs is None or self._SEC_MAPPING not in self._configs:
            raise ValueError("Mapping section is missing in the config file.")
        elif self._SEC_MARKERS_MAPPING not in self._configs[self._SEC_MAPPING]:
            raise ValueError("Marker mapping section is missing in the config file.")
