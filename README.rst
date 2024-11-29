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
      - | |pypi|
        | |conda|
    * - development
      - | |MIT|
        | |last-commit|
        | |commits-since|
        | |pixi-badge|

.. |docs| image:: https://img.shields.io/readthedocs/python-gaitalytics?logo=readthedocs
    :target: https://python-gaitalytics.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://img.shields.io/github/actions/workflow/status/DART-Lab-LLUI/python-gaitalytics/on_push_test.yaml?logo=pytest
    :alt: GitHub Actions Build Status
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/actions/

.. |last-commit| image:: https://img.shields.io/github/last-commit/DART-Lab-LLUI/python-gaitalytics
   :alt: GitHub last commit
   :target: https://github.com/DART-Lab-LLUI/python-gaitalytics

.. |commits-since| image:: https://img.shields.io/github/commits-since/DART-Lab-LLUI/python-gaitalytics/latest.svg
    :alt: Commits since latest release
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/compare/

.. |pixi-badge| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :alt: Pixi Badge
    :target: https://pixi.sh

.. |pypi| image:: https://img.shields.io/pypi/dm/gaitalytics?logo=pypi
   :alt: PyPI - Downloads
   :target: https://pypi.org/project/gaitalytics/

.. |conda| image:: https://img.shields.io/conda/dn/DartLab-LLUI/gaitalytics?logo=anaconda
   :alt: Conda Downloads
   :target: https://anaconda.org/dartlab-llui/gaitalytics

.. |MIT| image:: https://img.shields.io/github/license/DART-Lab-LLUI/python-gaitalytics?logo=opensourceinitiative
   :alt: GitHub License






.. end-badges

This Python package provides a comprehensive set of tools and advanced algorithms for analyzing 3D motion capture data.
It is specifically designed to process gait data stored in c3d format. Prior to utilizing the features of gaitalytics,
it is necessary to perform data labeling, modeling, and filtering procedures.

The library's versatility allows it to be adaptable to various marker sets and modeling algorithms,
offering high configurability.

Quickstart
----------

Installation
^^^^^^^^^^^^

Fast install with anaconda:

.. code:: shell

    conda install gaitalytics -c DartLab-LLUI
..

Or with pip:

.. code:: shell

    pip install gaitalytics
..

..

.. warning::
        | Manual installation of the `pyomeca <https://pyomeca.github.io/>`_ package is required. when using the pip installation method.

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

.. code-block:: yaml

    analysis:
      markers:
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
.. code-block:: python

    from gaitalytics import api
    # Load configuration (yaml file from above)
    config = api.load_config("./pig_config.yaml")

    # Load trial from c3d file
    trial = api.load_c3d_trial("./test_small.c3d", config)

    # Detect events
    events = api.detect_events(trial, config)
    try:
        # check events
        api.check_events(events)

        # write events to c3d in the same file
        api.write_events_to_c3d("./test_small.c3d", events, './test.c3d')

        # add events to trial
        trial.events = events

        # segment trial to gait cycles. (Events are already existing in the c3d file)
        trial_segmented = api.segment_trial(trial)

        # calculate features
        features = api.calculate_features(trial_segmented, config)

        # normalise time
        trial_normalized = api.time_normalise_trial(trial_segmented)

        # save features
        features.to_netcdf("features.nc")

        # export segmented trial to netcdf
        api.export_trial(trial_segmented, "output_segments")
        api.export_trial(trial_normalized, "output_norm")

    except ValueError as e:
        print(e)
..

Documentation
-------------
https://python-gaitalytics.readthedocs.org

