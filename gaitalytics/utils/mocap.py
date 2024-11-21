import xarray as xr

import gaitalytics.mapping as mapping
import gaitalytics.model as model


def get_marker_data(
    trial: model.Trial, config: mapping.MappingConfigs, marker: mapping.MappedMarkers
) -> xr.DataArray:
    """Get the marker data for a trial.

    Args:
        trial: The trial for which to get the marker data.
        config: The mapping configurations.
        marker: The marker to get the data for.

    Returns:
        An xarray DataArray containing the marker data.
    """
    marker_label = config.get_marker_mapping(marker)
    markers = trial.get_data(model.DataCategory.MARKERS)
    marker_data = markers.sel(channel=marker_label)
    return marker_data


def get_sacrum_marker(
    trial: model.Trial, config: mapping.MappingConfigs
) -> xr.DataArray:
    """Get the sacrum marker data for a trial.

    Try to get the sacrum marker data from the trial.
    If the sacrum marker not found calculate from posterior hip markers

    Args:
        trial: The trial for which to get the marker data.
        config: The mapping configurations.

    Returns:
        An xarray DataArray containing the sacrum marker data.
    """
    try:
        return get_marker_data(trial, config, mapping.MappedMarkers.SACRUM)
    except KeyError:
        l_marker = get_marker_data(trial, config, mapping.MappedMarkers.L_POST_HIP)
        r_marker = get_marker_data(trial, config, mapping.MappedMarkers.R_POST_HIP)
        return (l_marker + r_marker) / 2


def get_progression_vector(
    trial: model.Trial, config: mapping.MappingConfigs
) -> xr.DataArray:
    """Calculate the progression vector for a trial.

    The progression vector is the vector from the sacrum to the anterior hip marker.

    Args:
        trial: The trial for which to calculate the progression vector.
        config: The mapping configurations.

    Returns:
        An xarray DataArray containing the calculated progression vector.
    """
    sacrum_marker = get_sacrum_marker(trial, config)
    r_ant_hip = get_marker_data(trial, config, mapping.MappedMarkers.R_ANT_HIP)
    l_ant_hip = get_marker_data(trial, config, mapping.MappedMarkers.L_ANT_HIP)
    ant_marker = (r_ant_hip + l_ant_hip) / 2

    return (sacrum_marker - ant_marker).mean(dim="time")
