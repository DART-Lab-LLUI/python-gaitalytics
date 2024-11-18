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

def calculate_angle(
    vector_a: xr.DataArray,
    vector_b: xr.DataArray,
) -> xr.DataArray:
    """Calculate the angle in degrees between two vectors.

    Args:
        vector_a: The first vector.
        vector_b: The second vector.

    Returns:
        An xarray DataArray containing the angle in degrees.
    """
    dot_product = (vector_a * vector_b).sum(dim="axis")
    norm_a = np.sqrt((vector_a**2).sum(dim="axis"))
    norm_b = np.sqrt((vector_b**2).sum(dim="axis"))
    cosine_angle = dot_product / (norm_a * norm_b)
    cosine_angle_clipped = cosine_angle.clip(-1, 1)
    angle_radians = np.arccos(cosine_angle_clipped)
    return np.degrees(angle_radians)


def project_point_on_vector(point: xr.DataArray, vector: xr.DataArray) -> xr.DataArray:
    """Project a point onto a vector.

    Args:
        point: The point to project.
        vector: The vector to project onto.

    Returns:
        An xarray DataArray containing the projected point.
    """
    return vector * point.dot(vector, dim="axis")

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
        coords=vector1.coords
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


def calculate_speed_norm(position: xr.DataArray,
                    dt: float=0.01) -> xr.DataArray:
    """
    Compute the speed from a 3xN position data array obtained with constant sampling rate
    
    Args:
        position_data: A 3xN xarray.DataArray where each row corresponds to an axis (x, y, z),
                          and each column represents a time point (positions in space).
        dt: Time interval between samples.
        
    Returns:
        A 1D xarray.DataArray of velocity at each time point.
    """
    velocity_squared_sum = sum((np.diff(position.sel(axis=axis).values)/dt) ** 2  for axis in position.coords["axis"])
    speed_values = np.sqrt(velocity_squared_sum)
    speed_values = np.append(speed_values, speed_values[-1])

    return xr.DataArray(speed_values, dims=["time"], coords={"time": position.coords["time"]})

