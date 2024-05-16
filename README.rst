===========
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
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-gaitalytics/badge/?style=flat
    :target: https://python-gaitalytics.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/DART-Lab-LLUI/python-gaitalytics/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/actions

.. |version| image:: https://img.shields.io/pypi/v/gaitalytics.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/gaitalytics

.. |wheel| image:: https://img.shields.io/pypi/wheel/gaitalytics.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/gaitalytics

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/gaitalytics.svg
    :alt: Supported versions
    :target: https://pypi.org/project/gaitalytics

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/gaitalytics.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/gaitalytics

.. |commits-since| image:: https://img.shields.io/github/commits-since/DART-Lab-LLUI/python-gaitalytics/v0.1.2.svg
    :alt: Commits since latest release
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/compare/v0.1.2...main

.. end-badges

.. image:: https://github.com/DART-Lab-LLUI/python-gaitalytics/blob/738061ef8662d1fccac9d57a0ff8a1cdc4466e34/resources/logo/Gaitalytics_noBackground.png
    :alt: Gaitalytics Logo
    :align: center
    :width: 200px

This Python package provides a comprehensive set of tools and advanced algorithms for analyzing 3D motion capture data.
It is specifically designed to process gait data stored in c3d format. Prior to utilizing the features of gaitalytics,
it is necessary to perform data labeling, modeling, and filtering procedures.

The library's versatility allows it to be adaptable to various marker sets and modeling algorithms,
offering high configurability.

  Current pre-release is only tested with data acquired with Motek Caren, HBM2 Lower Body Trunk and PIG.

Free software: MIT license

Functionalities
===============

Event Detection
---------------

+------------+--------------+----------------------------------------------------------------------------+---------+
| Method     | Description  | options                                                                    | checked |
+============+==============+============================================================================+=========+
| Marker     | Zenis 2006   | - min_distance = 100: minimum of frames between same event on same context | X       |
|            |              | - foot_strike_offset = 0: Amount of frames to offset foot strike           |         |
|            |              | - foot_off_offset = 0: Amount of frames to offset foot off                 |         |
+------------+--------------+----------------------------------------------------------------------------+---------+
| Forceplate | Split-Belt   |                                                                            |         |
+------------+--------------+----------------------------------------------------------------------------+---------+

Event Detection Check
---------------------

+------------+--------------------------------------------------+-------------------------+---------+
| Method     | Description                                      | options                 | checked |
+============+==================================================+=========================+=========+
| context    | Checks gait event sequences                      |                         |         |
|            | HS->TO-HS-TO                                     |                         |         |
+------------+--------------------------------------------------+-------------------------+---------+
| spacing    | Checks Frames between same event on same context |                         |         |
+------------+--------------------------------------------------+-------------------------+---------+

Modelling
---------

+------------+--------------------------------------------+-------------------------------+---------+
| Method     | Description                                | options                       | checked |
+============+============================================+===============================+=========+
| com        | creates Center of Mass Marker              |                               |         |
+------------+--------------------------------------------+-------------------------------+---------+
| xcom       | creates extrapolated Center of Mass Marker | - belt_speed = 1 :            |         |
|            |                                            |   speed of treadmill          |         |
|            |                                            | - dominant_leg_length = 0.1 : |         |
|            |                                            |   length of dominant leg (mm) |         |
+------------+--------------------------------------------+-------------------------------+---------+
| cmos       | create Continuous Margin of Stability      |                               |         |
+------------+--------------------------------------------+-------------------------------+---------+

Analysis
--------

+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| Method          | Description                                                | options                               | checked |
+=================+============================================================+=======================================+=========+
| angels          | - min                                                      | - by_phase = True                     |         |
|                 | - max                                                      |   If metrics should be calculated by  |         |
|                 | - mean                                                     |   standing and swinging phase         |         |
|                 | - sd                                                       |                                       |         |
|                 | - amplitude                                                |                                       |         |
|                 | - min velocity                                             |                                       |         |
|                 | - max velocity                                             |                                       |         |
|                 | - sd velocity                                              |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| forces          | - min                                                      | - by_phase = True                     |         |
|                 | - max                                                      |   If metrics should be calculated by  |         |
|                 | - mean                                                     |   standing and swinging phase         |         |
|                 | - sd                                                       |                                       |         |
|                 | - amplitude                                                |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| moments         | - min                                                      | - by_phase = True                     |         |
|                 | - max                                                      |   If metrics should be calculated by  |         |
|                 | - mean                                                     |   standing and swinging phase         |         |
|                 | - sd                                                       |                                       |         |
|                 | - amplitude                                                |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| powers          | - min                                                      | - by_phase = True                     |         |
|                 | - max                                                      |   If metrics should be calculated by  |         |
|                 | - mean                                                     |   standing and swinging phase         |         |
|                 | - sd                                                       |                                       |         |
|                 | - amplitude                                                |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| cmos            | - min                                                      | - by_phase = True                     |         |
|                 | - max                                                      |   If metrics should be calculated by  |         |
|                 | - mean                                                     |   standing and swinging phase         |         |
|                 | - sd                                                       |                                       |         |
|                 | - amplitude                                                |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| mos             | - Mos at FS                                                |                                       |         |
|                 | - Mos at TO                                                |                                       |         |
|                 | - Mos at contra HS                                         |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| Spatio-temporal | - step_length                                              |                                       | X       |
|                 | - stride_length                                            |                                       |         |
|                 | - cycle_duration                                           |                                       |         |
|                 | - swing_duration_perc                                      |                                       |         |
|                 | - stance_duration_perc                                     |                                       |         |
|                 | - step_height                                              |                                       |         |
|                 | - step_width                                               |                                       |         |
|                 | - limb_circumduction [1]                                   |                                       |         |
|                 | - single_support_duration_percent [2]                      |                                       |         |
|                 | - double_support_duration_percent [2]                      |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+
| Toe Clearance   | - minimal toe clearance                                    |                                       |         |
|                 | - Percentage in cycle where minimal toe clearance happened |                                       |         |
|                 | - minimal toe clearance at FS                              |                                       |         |
+-----------------+------------------------------------------------------------+---------------------------------------+---------+

References
~~~~~~~~~~

[1] Michael D. Lewek et al. (2012), “The influence of mechanically and
physiologically imposed stiff-knee gait patterns on the energy cost of
walking”, vol. 93, no.1, pp. 123-128. Publisher: Archives of Physical
Medicine and Rehabilitation.

[2] A. Gouelle and F. Mégrot (2017), “Interpreting spatiotemporal
parameters, symmetry, and variability in clinical gait analysis”,
Handbook of Human Motion pp. 1-20, Publisher: Springer International
Publishing.

Usage
=====

Installation
------------

Please be aware of the dependency of gaitalytics to
Biomechanical-ToolKit (BTK). To install follow the instructions
`here <https://biomechanical-toolkit.github.io/docs/Wrapping/Python/_build_instructions.html>`__
or use conda-forge version
`here <https://anaconda.org/conda-forge/btk>`__

Fast install with anaconda:

.. code:: shell

    pip install gaitalytics
    conda install -c conda-forge btk
..


You can also install the in-development version with:

.. code:: shell

    pip install https://github.com/DART-Lab-LLUI/python-gaitalytics/archive/main.zip
    conda install -c conda-forge btk
..

Configuration
-------------

Gaitalytics can be used with any marker set, which at least includes
four hip markers (left front/back, right front/back) and four foot
markers (left heel/toe, right heel/toe) and four ankle makers (left
medial/lateral, right medial lateral).

All functionalities in the libraries only take points into account which
are configured in as specific yaml file. Working example file can be
found
`here <https://github.com/DART-Lab-LLUI/python-gaitalytics/blob/defc453f95940db55f6875ae7568949daa1b67d4/settings/hbm_pig.yaml>`__

Minimal requirements would look like this:

.. code:: yaml

   marker_set_mapping:
     left_back_hip: LASIS
     right_back_hip: RASIS
     left_front_hip: LPSIS
     right_front_hip: RPSIS

     left_lat_malleoli: LLM
     right_lat_malleoli: RLM
     left_med_malleoli: LMM
     right_med_malleoli: RMM

     right_heel: RHEE
     left_heel: LHEE
     right_meta_2: RMT2
     left_meta_2: LMT2

     com: COM
     left_cmos: cmos_left
     right_cmos: cmos_right

   model_mapping:
..

   **Warning** Do not rename keys of the minimal setting

Pipeline
--------

Please take the resources in the `example
folder <https://github.com/DART-Lab-LLUI/python-gaitalytics/tree/defc453f95940db55f6875ae7568949daa1b67d4/examples>`__
for advice.

Documentation
=============

https://python-gaitalytics.readthedocs.org


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox


Release
-------

https://github.com/ionelmc/cookiecutter-pylibrary
