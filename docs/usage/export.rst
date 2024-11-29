.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Data export
===========

| Gaitalytics provides two ways export and store the data into a file system.

Model export
------------
| If the data should be exported to later usage with gaitalytics the function :func:`gaitalytics.model.BaseTrial.to_hdf5` can be used on the current :class:`gaitalytics.model.Trial` or :class:`gaitalytics.model.TrialCycles`.

.. code-block:: python

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example_with_events.c3d", config)
    trial.to_hdf5("./export_trial.h5")

    segmented_trial = api.segment_trial(trial)
    segmented_trial.to_hdf5("./export_segmented_trial.h5")
..

| Depending on the object class the function will store the data in a different way.
| :class:`gaitalytics.model.Trial` will store the data in a single file, where as :class:`gaitalytics.model.TrialCycles` will store the data in multiple files (one file per gait cycle).

| The data can be loaded again with the static function :func:`gaitalytics.model.trial_from_hdf5`.

.. code-block:: python

    from gaitalytics import api, model

    trial = model.trial_from_hdf5("./export_trial.h5")
    segmented_trial = model.trial_from_hdf5("./export_segmented_trial.h5")
..


Data export
-----------

| If data should be used outside of gaitalytics the function :func:`gaitalytics.api.export_trial` can be used.

.. code-block:: python

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example_with_events.c3d", config)
    api.export_trial(trial, "./export_trial.nc")

    segmented_trial = api.segment_trial(trial)
    api.export_trial(segmented_trial, "./export_segmented_trial.nc")

..

| The function will store the data in a netCDF file and can easily be loaded with the `load_dataarray <https://docs.xarray.dev/en/stable/generated/xarray.load_dataarray.html#xarray.load_dataarray>`_ function.

.. code-block:: python

    import xarray as xr

    trial = xr.load_dataarray("./export_trial.nc")
    segmented_trial = xr.load_dataarray("./export_segmented_trial.nc")

..

