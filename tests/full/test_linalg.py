import pytest
import xarray as xr
import numpy as np

from gaitalytics.utils.linalg import (
    calculate_distance,
    project_point_on_vector,
    signed_projection_norm,
    get_normal_vector,
    normalize_vector,
    calculate_speed_norm, 
    get_point_in_front, 
    get_point_behind
)

@pytest.fixture
def sample_data():
    point_a = xr.DataArray([1, 2, 3], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    point_b = xr.DataArray([4, 5, 6], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    vector_a = xr.DataArray([1, 0, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    vector_b = xr.DataArray([0, 1, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    return point_a, point_b, vector_a, vector_b

def test_calculate_distance(sample_data):
    point_a, point_b, _, _ = sample_data
    distance = calculate_distance(point_a, point_b)
    expected_distance = np.sqrt(27)
    assert distance == pytest.approx(expected_distance)

def test_project_point_on_vector(sample_data):
    point_a, _, vector_a, _ = sample_data
    projected_point = project_point_on_vector(point_a, vector_a)
    expected_projection = xr.DataArray([1, 0, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    xr.testing.assert_allclose(projected_point, expected_projection)

def test_signed_projection_norm(sample_data):
    _, _, vector_a, vector_b = sample_data
    signed_norm = signed_projection_norm(vector_a, vector_b)
    expected_signed_norm = 0.0
    assert signed_norm == pytest.approx(expected_signed_norm)

def test_get_normal_vector(sample_data):
    _, _, vector_a, vector_b = sample_data
    normal_vector = get_normal_vector(vector_a, vector_b)
    expected_normal_vector = xr.DataArray([0, 0, 1], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    xr.testing.assert_allclose(normal_vector, expected_normal_vector)

def test_normalize_vector(sample_data):
    _, _, vector_a, _ = sample_data
    normalized_vector = normalize_vector(vector_a)
    expected_normalized_vector = xr.DataArray([1, 0, 0], dims=["axis"], coords={"axis": ["x", "y", "z"]})
    xr.testing.assert_allclose(normalized_vector, expected_normalized_vector)

def test_calculate_speed_norm():
    position = xr.DataArray(
        np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        dims=["axis", "time"],
        coords={"axis": ["x", "y", "z"], "time": [0, 1, 2]}
    )
    speed = calculate_speed_norm(position, dt=1.0)
    expected_speed = xr.DataArray([np.sqrt(3), np.sqrt(3), np.sqrt(3)], dims=["time"], coords={"time": [0, 1, 2]})
    xr.testing.assert_allclose(speed, expected_speed)
    
def test_get_point_in_front(sample_data):
    point_a, point_b, vector_a, _ = sample_data
    point_in_front = get_point_in_front(point_a, point_b, vector_a)
    expected_point_in_front = point_b
    xr.testing.assert_allclose(point_in_front, expected_point_in_front)
    
def test_get_point_behind(sample_data):
    point_a, point_b, vector_a, _ = sample_data
    point_behind = get_point_behind(point_a, point_b, vector_a)
    expected_point_behind = point_a
    xr.testing.assert_allclose(point_behind, expected_point_behind)
