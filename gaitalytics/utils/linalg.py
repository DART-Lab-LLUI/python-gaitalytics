import xarray as xr
import numpy as np


def calculate_distance(
    point_a: xr.DataArray,
    point_b: xr.DataArray,
) -> xr.DataArray:
    """Calculate the distance between two points.

    Args:
        point_a: The first point.
        point_b: The second point.

    Returns:
        An xarray DataArray containing the calculated distance.
    """
    return (point_b - point_a).meca.norm(dim="axis")


def project_point_on_vector(point: xr.DataArray, vector: xr.DataArray) -> xr.DataArray:
    """Project a point onto a vector.

    Args:
        point: The point to project.
        vector: The vector to project onto.

    Returns:
        An xarray DataArray containing the projected point.
    """
    return vector * point.dot(vector, dim="axis")


def get_normal_vector(vector1: xr.DataArray, vector2: xr.DataArray):
    """Create a vector with norm = 1 normal to two other vectors.

    Args:
        vector1: The first vector to be normal to.
        vector2: The second vector to be normal to.

    Returns:
        An xarray DataArray containing the normal vector.
    """
    normal_vector = xr.DataArray(
        np.cross(vector1.values, vector2.values),
        dims=vector1.dims,
        coords=vector1.coords,
    )
    return normalize_vector(normal_vector)


def normalize_vector(vector: xr.DataArray) -> xr.DataArray:
    """Normalize a vector.

    Args:
        vector: The vector to normalize.

    Returns:
        An xarray DataArray containing the normalized vector.
    """
    return vector / vector.meca.norm(dim="axis")


def calculate_speed_norm(position: xr.DataArray, dt: float = 0.01) -> np.ndarray:
    """
    Compute the speed from a 3xN position data array obtained with constant sampling rate

    Args:
        position: A 3xN xarray.DataArray where each row corresponds to an axis (x, y, z),
                          and each column represents a time point (positions in space).
        dt: Time interval between samples.

    Returns:
        An np array with the speed values.
    """
    velocity_squared_sum = sum(
        (np.diff(position.sel(axis=axis).values) / dt) ** 2
        for axis in position.coords["axis"]
    )
    speed_values = np.sqrt(velocity_squared_sum)
    speed_values = np.append(speed_values, speed_values[-1])

    return speed_values
