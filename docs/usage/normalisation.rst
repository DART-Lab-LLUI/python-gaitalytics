.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Time Normalisation
==================

| Gaitalytics provides functionalities to time normalise the segmented data (:func:`gaitalytics.api.time_normalise_trial`).

.. code-block:: python

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example_with_events.c3d", config)
    segmented_trial = api.segment_trial(trial)
    normalised_trial = api.time_normalise_trial(segmented_trial)

..

The function returns a :class:`gaitalytics.model.TrialCycles` object. It contains a dictionary with the cycles number as key and a :class:`gaitalytics.model.Trial` object as value, which are normalised to 100 frames.

.. warning::

    Using the parameter *n_frames* the length to which the time series should be interpolated can be set. The default value is **100** frames.

..









