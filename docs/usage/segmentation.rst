.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Segmentation
============

| Gaitalytics provides a way to segment the data into gait cycles. :func:`gaitalytics.api.segment_trial` segments data into time series form heel strike to heel strike.
| Alternatively, the parameter *method="TO"* can be used to segment the data from toe off to toe off.

.. code-block:: python

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example_with_events.c3d", config)
    segmented_trial = api.segment_trial(trial)
..

The function returns a :class:`gaitalytics.model.TrialCycles` object. It contains a dictionary with the cycles number as key and a :class:`gaitalytics.model.Trial` object as value.

.. hint::

    The loaded :class:`gaitalytics.model.Trial` object must contain gait events for segmentation.

..









