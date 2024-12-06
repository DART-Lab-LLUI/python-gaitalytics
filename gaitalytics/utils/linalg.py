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

def signed_projection_norm(vector: xr.DataArray, onto: xr.DataArray) -> xr.DataArray:
    """Compute the signed norm of the projection of a vector onto another vector.

    Args:
        vector: The vector to be projected.
        onto: The vector to project onto.

    Returns:
        An xarray DataArray containing the signed norm of the projected vector.
    """
    projection = onto * vector.dot(onto, dim="axis") / onto.dot(onto, dim="axis")
    projection_norm = projection.meca.norm(dim="axis")
    sign = xr.where(vector.dot(onto, dim="axis") > 0, 1, -1)
    sign = xr.where(vector.dot(onto, dim="axis") == 0, 0, sign)
    return projection_norm * sign

def calculate_signed_distance_on_vector(point_a: xr.DataArray, point_b: xr.DataArray, direction_vector: xr.DataArray) -> xr.DataArray: 
    """Return the signed distance = A - B between two points projected on vector. The sign is determined based on whether point A is in front of point B according to the direction vector

    Args:
        point_a (xr.DataArray): Point A
        point_b (xr.DataArray): Point B
        direction_vector (xr.DataArray): The direction vector on which we project points

    Returns:
        xr.DataArray: The signed distance A-B
    """
    direction_vector = direction_vector / direction_vector.meca.norm(dim="axis")
    
    vector_b_to_a = point_a - point_b
    
    signed_distance = vector_b_to_a.dot(direction_vector, dim="axis")
    
    return signed_distance


def get_normal_vector(vector1: xr.DataArray, vector2: xr.DataArray) -> xr.DataArray:
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

    return xr.DataArray(speed_values, dims=["time"], coords={"time": position.coords["time"]})

def get_point_in_front(point_a: xr.DataArray, point_b: xr.DataArray, direction_vector: xr.DataArray) -> xr.DataArray:
    """Determine which point is in front of the other according to the direction vector.

    Args:
        point_a: The first point.
        point_b: The second point.
        direction_vector: The direction vector.

    Returns:
        The point that is in front according to the direction vector.
    """
    direction_vector = direction_vector / direction_vector.meca.norm(dim="axis")
    vector_b_to_a = point_a - point_b
    signed_distance = vector_b_to_a.dot(direction_vector, dim="axis")
    
    return point_a if signed_distance > 0 else point_b

def get_point_behind(point_a: xr.DataArray, point_b: xr.DataArray, direction_vector: xr.DataArray) -> xr.DataArray:
    """Determine which point is behind the other according to the direction vector.

    Args:
        point_a: The first point.
        point_b: The second point.
        direction_vector: The direction vector.

    Returns:
        The point that is behind according to the direction vector.
    """
    direction_vector = direction_vector / direction_vector.meca.norm(dim="axis")
    vector_b_to_a = point_a - point_b
    signed_distance = vector_b_to_a.dot(direction_vector, dim="axis")
    
    return point_b if signed_distance > 0 else point_a




    return speed_values
