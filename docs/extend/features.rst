.. meta::
   :description: Gaitalytics Feature.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, references

Extend features
===============

.. warning::
    Gaitalytics is build to be independent on marker-models and modelling mechanisms through :ref:`configurations <Configuration>`.
    In this Tutorial we will not cover the creation of new :ref:`mapping configurations <Config Mapping>` since it will alter core functionality of the library.
    Your new features therefore may not work within different setups. To make your features available to the community and integrated to gaitalytics,
    please consider :ref:`contributing <Development>` to the project and our staff will take care of the integration.

To implement new feature algorithms you need to define a new class inheriting :class:`gaitalytics.features.PointDependentFeature`.
This class internally loops though each cycle and context ('left' and 'right') and calculates the feature for each point in the cycle and calls the :meth:`gaitalytics.features.CycleFeaturesCalculation._calculate` method.
With implementing :meth:`gaitalytics.features.CycleFeaturesCalculation._calculate` in your class you can add new functionalities.

.. code-block:: python

    from gaitalytics.features import PointDependentFeature

    class NewFeature(PointDependentFeature):

        def _calculate(self, cycle):
            # Your feature calculation here
            return feature_value

Through parameter `cycle` you can access the data of the current cycle and calculate your feature, including the data of the current cycle and the context ('left' or 'right').
As return the framework expects an xarray DataArray object.

Helping functions
-----------------

To help you with the implementation of new features, gaitalytics provides a set of helper functions handle framework specific restrictions.

Markers & Analogs
^^^^^^^^^^^^^^^^^
If your planning to use markers or analogs which are mapped by Gaitalytics (:class:`gaitalytics.mapping.MappedMarkers`) you can use the following helper functions:
:meth:`gaitalytics.features.PointDependentFeature._get_marker_data`. This function returns the data of the mapped marker for the current cycle.

.. hint::
    Getting the data for the sacrum marker is handled as a special case. Since Gaitalytics allows either a single sacrum marker or a sacrum marker or two posterior pelvis markers, the method :meth:`gaitalytics.features.PointDependentFeature._get_sacrum_marker` will handle the logic to extract jut a sacrum marker.
..

If you want to use markers or analogs which are not mapped by Gaitalytics, you can find the data in the `cycle` xarray object.
Be aware that approach is not generalized and may not work with different marker models. Therefore, it is recommended to use the mapped markers whenever possible.
Future efforts will be made to generalize this approach.

Event timings
^^^^^^^^^^^^^
Ease the work with event timings in a cycle the :meth:`gaitalytics.features.PointDependentFeature.get_event_times` function can be used to extract the event timings for the current cycle.


Vectors
^^^^^^^
It is often necessary to obtain progression vectors or sagittal plane vectors. To help you with this, gaitalytics provides the following helper functions:

    - :meth:`gaitalytics.features.PointDependentFeature._get_progression_vector`
    - :meth:`gaitalytics.features.PointDependentFeature._get_sagittal_vector`

Return values
^^^^^^^^^^^^^
The expected return value of the feature calculation is an xarray DataArray object in a specific format.
To help you with the creation of this object, gaitalytics provides the following helper functions:

    - :meth:`gaitalytics.features.CycleFeaturesCalculation._create_result_from_dict` to create a DataArray object from a dictionary.
    - :meth:`gaitalytics.features.CycleFeaturesCalculation._flatten_features` to flatten an xarray DataArray object.

Including your feature
----------------------

To include your feature in the calculation of the gait metrics, you need to add it to the parameters of your :func:`gaitalytics.api.calculate_features` call.

.. code-block:: python

    from gaitalytics import api
    from gaitalytics.features import NewFeature

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example.c3d", config)

    # Calculate the only the new feature
    features = api.calculate_features(trial, config, [NewFeature])

    # Calculate all features including the new feature
    features = api.calculate_features(trial, config, [gaitalytics.features.TimeSeriesFeatures,
                                                      gaitalytics.features.PhaseTimeSeriesFeatures,
                                                      gaitalytics.features.TemporalFeatures,
                                                      gaitalytics.features.SpatialFeatures,
                                                      NewFeature])

