import xarray as xr


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


def normalize_vector(vector: xr.DataArray) -> xr.DataArray:
    """Normalize a vector.

    Args:
        vector: The vector to normalize.

    Returns:
        An xarray DataArray containing the normalized vector.
    """
    return vector / vector.meca.norm(dim="axis")
