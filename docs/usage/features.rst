.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Feature Extraction
==================

| Several features can be extracted from the segmented data. These features can be used to analyze the gait pattern of the subject. The features are calculated for each segment of the trial. The features are calculated using the :func:`gaitalytics.api.calculate_features` function.

.. code-block:: python

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example_with_events.c3d", config)
    segmented_trial = api.segment_trial(trial)
    features = api.calculate_features(segmented_trial, config)

..

The function returns a DataArray object with the coordinates *feature, cycles, context*

Features
--------

Following features can be extracted from the segmented data:

.. csv-table::
   :file: ../_static/tables/features.csv
   :widths: 15, 70, 15
   :header-rows: 1

..

.. rubric:: Foot note

| \* Depending on the unit in the c3d file.
| ✝ Depending on the unit of the configured entity in the c3d file.
| ☨ Feature only calculated if a center of mass marker is present in the c3d file and configured in the config file.

.. rubric:: References
.. footbibliography::












