Gaitalytics
===========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
    * - package
      - | |pixi-badge|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-gaitalytics/badge/?style=flat
    :target: https://python-gaitalytics.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/DART-Lab-LLUI/python-gaitalytics/actions/workflows/on_push_test.yaml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/actions/

.. |commits-since| image:: https://img.shields.io/github/commits-since/DART-Lab-LLUI/python-gaitalytics/latest.svg
    :alt: Commits since latest release
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/compare/

.. |pixi-badge| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :alt: Pixi Badge
    :target: https://pixi.sh
.. end-badges

.. image:: ./_static/images/Gaitalytics_noBackground.png
    :alt: Gaitalytics Logo
    :align: center
    :width: 200px

This Python package provides a comprehensive set of tools and advanced algorithms for analyzing 3D motion capture data.
It is specifically designed to process gait data stored in c3d format. Prior to utilizing the features of gaitalytics,
it is necessary to perform data labeling, modeling, and filtering procedures.

The library's versatility allows it to be adaptable to various marker sets and modeling algorithms,
offering high configurability.


Functionalities
---------------

Input
^^^^^
Currently only c3d files are supported.
The library provides a function to load a c3d file into a trial object for usage in the library.

.. note::
    future efforts will be made to support other file formats such as trc, mot, sto and mox files.

Event Detection
^^^^^^^^^^^^^^^

+------------+--------------------------+----------------------------------------------------------------------------+
| Method     | Description              | options                                                                    |
+============+==========================+============================================================================+
| Marker     | based on Zenis 2006      | - height: The height of peaks for events.                                  |
|            |                          | - threshold: The threshold for detecting events.                           |
|            |                          | - distance: The min distance in frames between events.                     |
|            |                          | - rel_height: The relative height of peak for events.                      |
+------------+--------------------------+----------------------------------------------------------------------------+


Event Detection Check
^^^^^^^^^^^^^^^^^^^^^

+------------+--------------------------------------------------+-------------------------+
| Method     | Description                                      | options                 |
+============+==================================================+=========================+
| sequence   | Checks gait event sequences                      |                         |
|            |  - Heel Strike - Toe off - Heel Strike - Toe off |                         |
|            |  - Left - Right - Left - Right                   |                         |
+------------+--------------------------------------------------+-------------------------+

Event Writing
^^^^^^^^^^^^^

Currently only c3d files are supported.
The main usage for this feature is the correction of detected events.

Segmentation
^^^^^^^^^^^^

Currently only the segmentation based on gait-events is supported.

+------------+--------------------------------------------------+-------------------------+
| Method     | Description                                      | options                 |
+============+==================================================+=========================+
| HS         | Segment based on heel strike                     |                         |
+------------+--------------------------------------------------+-------------------------+
| TO         | Segment based on toe off                         |                         |
+------------+--------------------------------------------------+-------------------------+

Time Normalization
^^^^^^^^^^^^^^^^^^
Currently only linear time normalization is supported.

.. note::
    future efforts will be made to support other time normalization
    methods such as dynamic time warping.

+------------+--------------------------------------------------+-------------------------+
| Method     | Description                                      | options                 |
+============+==================================================+=========================+
| linear     | Linear time-normalisation                        |                         |
+------------+--------------------------------------------------+-------------------------+


Feature calculation
^^^^^^^^^^^^^^^^^^^

+-------------------------+-------------------------------------------------------+---------------------------------------+
| Method                  | Description                                           | options                               |
+=========================+=======================================================+=======================================+
| TimeSeriesFeatures      | - min                                                 |                                       |
|                         | - max                                                 |                                       |
|                         | - mean                                                |                                       |
|                         | - sd                                                  |                                       |
|                         | - median                                              |                                       |
|                         | - amplitude                                           |                                       |
+-------------------------+-------------------------------------------------------+---------------------------------------+
| PhaseTimeSeriesFeatures | - stand_min                                           |                                       |
|                         | - stand_max                                           |                                       |
|                         | - stand_mean                                          |                                       |
|                         | - stand_sd                                            |                                       |
|                         | - stand_median                                        |                                       |
|                         | - stand_amplitude                                     |                                       |
|                         | - swing_max                                           |                                       |
|                         | - swing_mean                                          |                                       |
|                         | - swing_sd                                            |                                       |
|                         | - swing_median                                        |                                       |
|                         | - swing_amplitude                                     |                                       |
+-------------------------+-------------------------------------------------------+---------------------------------------+
| SpatialFeatures         | - step_length [1]                                     |                                       |
|                         | - stride_length [1]                                   |                                       |
+-------------------------+-------------------------------------------------------+---------------------------------------+
| TemporalFeatures        | - cycle_duration                                      |                                       |
|                         | - swing_duration_perc                                 |                                       |
|                         | - stance_duration_perc                                |                                       |
|                         | - step_width [1]                                      |                                       |
|                         | - cadence [1]                                         |                                       |
|                         | - single_support_duration_percent [2]                 |                                       |
|                         | - double_support_duration_percent [2]                 |                                       |
+-------------------------+-------------------------------------------------------+---------------------------------------+

References
""""""""""

[1] J. H. Hollman, E. M. McDade, and R. C. Petersen, “Normative Spatiotemporal
Gait Parameters in Older Adults,” Gait Posture, vol. 34, no. 1, pp. 111–118,
May 2011, doi: 10.1016/j.gaitpost.2011.03.024.

[2] A. Gouelle and F. Mégrot (2017), “Interpreting spatiotemporal
parameters, symmetry, and variability in clinical gait analysis”,
Handbook of Human Motion pp. 1-20, Publisher: Springer International
Publishing.

Quickstart
----------

Installation
^^^^^^^^^^^^

Fast install with anaconda:

.. code:: shell

    conda install gaitalytics -c DartLab-LLUI
..

Configuration
^^^^^^^^^^^^^

Gaitalytics can be used with any marker set, which at least includes
three or for hip markers (front left/right, back left/right or sacrum) and four foot
markers (left heel/toe, right heel/toe).

Additionally markers can be defined on which standard time-series features such as min max mean etc.
will be calculated.

All functionalities in the libraries only take points into account which
are configured in as specific yaml file.



Minimal requirements would look like this:

.. code:: yaml

    # Markers to analyse
    analysis:
      markers: # Markers to analyse
        # Left side
        - "LHipAngles"
        - "LKneeAngles"
        - "LAnkleAngles"
        - "LPelvisAngles"
        - "LThoraxAngles"
    mapping:
      markers:
        # Foot
        l_heel: "LHEE"
        r_heel: "RHEE"
        l_toe: "LTOE"
        r_toe: "RTOE"

        # Hip
        l_ant_hip: "LASI"
        r_ant_hip: "RASI"
        l_post_hip: "LPSI"
        r_post_hip: "RPSI"
        sacrum: "SACR"
..



Simple Pipeline
^^^^^^^^^^^^^^^^

.. code:: python

    from gaitalytics import api

    # Load configuration (yaml file from above)
    config = api.load_config("./pig_config.yaml")

    # Load trial from c3d file
    trial = api.load_c3d_trial("./test_small.c3d", config)

    # Detect events
    events = api.detect_events(trial, config)
    try:
        # detect events (marker based)
        events = api.check_events(events, config)

        # check events
        api.check_events(events, config)

        # write events to c3d in the same file
        api.write_events_to_c3d(trial, events)

        # segment trial to gait cycles. (Events are already existing in the c3d file)
        trial_segmented = api.segment_trial(trial)

        # calculate features
        features = api.calculate_features(trial_segmented, config)

        # save features
        faetures.to_netcdf("features.nc")

        # export segmented trial to netcdf
        api.export_trial(trial_segmented, config, "output.nc")

    api.export_trial(trial_segmented, config, "output.c3d")
    except ValueError as e:
        print(e)

Documentation
-------------

https://python-gaitalytics.readthedocs.org




