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
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-gaitalytics/badge/?style=flat
    :target: https://python-gaitalytics.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/cereneo-foundation/python-gaitalytics/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/cereneo-foundation/python-gaitalytics/actions

.. |codecov| image:: https://codecov.io/gh/cereneo-foundation/python-gaitalytics/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/cereneo-foundation/python-gaitalytics

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

.. |commits-since| image:: https://img.shields.io/github/commits-since/cereneo-foundation/python-gaitalytics/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/cereneo-foundation/python-gaitalytics/compare/v0.1.1...main



.. end-badges

.. image:: https://github.com/DART-Lab-LLUI/gaitalytics/blob/27ff8401295c3a05537409deb3982129ed78222c/resources/logos/Gaitalytics_noBackground.pngtox
This Python package provides a comprehensive set of tools and advanced algorithms for analyzing 3D motion capture data.
It is specifically designed to process gait data stored in c3d format. Prior to utilizing the features of gaitalytics,
it is necessary to perform data labeling, modeling, and filtering procedures.

The library's versatility allows it to be adaptable to various marker sets and modeling algorithms,
offering high configurability.

.. note::
    Current pre-release is only tested with data acquired with Motek Caren, HBM2 Lower Body Trunk and PIG.


Free software: MIT license

Functionalities
===============

Event Detection
---------------

+---+----+-------------------------------------------------------------+
| M | D  | options                                                     |
| e | es |                                                             |
| t | cr |                                                             |
| h | ip |                                                             |
| o | ti |                                                             |
| d | on |                                                             |
+===+====+=============================================================+
| M | Z  | min_distance = 100: minimum of frames between same event on |
| a | en | same contextfoot_strike_offset = 0: Amount of frames to     |
| r | is | offset foot strikefoot_off_offset = 0: Amount of frames to  |
| k | 20 | offset foot off                                             |
| e | 06 |                                                             |
| r |    |                                                             |
+---+----+-------------------------------------------------------------+
| F | S  | -                                                           |
| o | pl |                                                             |
| r | it |                                                             |
| c | Fo |                                                             |
| e | rc |                                                             |
| P | eP |                                                             |
| l | la |                                                             |
| a | te |                                                             |
| t |    |                                                             |
| e |    |                                                             |
+---+----+-------------------------------------------------------------+

Event Detection Check
---------------------

======= ================================================
Method  Description
======= ================================================
context Checks Event Sequence HS TO HS TO
spacing Checks Frames between same event on same context
======= ================================================

Modelling
---------

======= =========================================================
Method Description
======= =========================================================
com     creates Center of Mass Marker in c3d
cmos    create Continuous Margin of Stability AP ML Marker in c3d
======= =========================================================

Analysis
--------

+---+-----------------------------------------------+------------------+
| M | Description                                   | options          |
| e |                                               |                  |
| t |                                               |                  |
| h |                                               |                  |
| o |                                               |                  |
| d |                                               |                  |
| e |                                               |                  |
+===+===============================================+==================+
| a | min, max, mean, sd, amplitude,min velocity,   | by_phase = True  |
| n | max velocity, sd velocity                     | : If metrics     |
| g |                                               | should be        |
| l |                                               | calculated by    |
| e |                                               | standing and     |
| s |                                               | swinging phase   |
+---+-----------------------------------------------+------------------+
| f | min, max, mean, sd, amplitude                 | by_phase = True  |
| o |                                               | : If metrics     |
| r |                                               | should be        |
| c |                                               | calculated by    |
| e |                                               | standing and     |
| s |                                               | swinging phase   |
+---+-----------------------------------------------+------------------+
| m | min, max, mean, sd, amplitude                 | by_phase = True  |
| o |                                               | : If metrics     |
| m |                                               | should be        |
| e |                                               | calculated by    |
| n |                                               | standing and     |
| t |                                               | swinging phase   |
| s |                                               |                  |
+---+-----------------------------------------------+------------------+
| p | min, max, mean, sd, amplitude                 | by_phase = True  |
| o |                                               | : If metrics     |
| w |                                               | should be        |
| e |                                               | calculated by    |
| r |                                               | standing and     |
| s |                                               | swinging phase   |
+---+-----------------------------------------------+------------------+
| c | min, max, mean, sd, amplitude                 | by_phase = True  |
| m |                                               | : If metrics     |
| o |                                               | should be        |
| s |                                               | calculated by    |
|   |                                               | standing and     |
|   |                                               | swinging phase   |
+---+-----------------------------------------------+------------------+
| m | HS mos, TO mos, HS contra mos,TO contra mos   | -                |
| o | for both sides                                |                  |
| s |                                               |                  |
+---+-----------------------------------------------+------------------+
| t | minimal toe clearance, percent swing phase    |                  |
| o | when min toe clearance happened,toe clearance |                  |
| e | HS,                                           |                  |
| _ |                                               |                  |
| c |                                               |                  |
| l |                                               |                  |
| e |                                               |                  |
| a |                                               |                  |
| r |                                               |                  |
| a |                                               |                  |
| n |                                               |                  |
| c |                                               |                  |
| e |                                               |                  |
+---+-----------------------------------------------+------------------+
| s | step_length,stride length, cycle              |                  |
| p | duration,step duration percent, swing         |                  |
| a | duration percent, stance duration             |                  |
| t | percent,step height, step width, limb         |                  |
| i | circumduction, single/double support duration |                  |
| o |                                               |                  |
| t |                                               |                  |
| e |                                               |                  |
| m |                                               |                  |
| p |                                               |                  |
| o |                                               |                  |
| r |                                               |                  |
| a |                                               |                  |
| l |                                               |                  |
+---+-----------------------------------------------+------------------+

+---------+------------------------------------------------------------+
| Method  | Definition                                                 |
+=========+============================================================+
| limb    | maximum lateral excursion of the foot during the swing     |
| circum  | phase with respect to the position of the foot during the  |
| duction | preceding stance phase [1]                                 |
+---------+------------------------------------------------------------+
| double  | duration of the stance phase when the two feet are in      |
| support | contact with the ground. Both initial and terminal double  |
| d       | support duration were computed [2]                         |
| uration |                                                            |
+---------+------------------------------------------------------------+
| single  | duration during which one foot is in contact with the      |
| support | ground while the other foot is in the swing phase [2]      |
| d       |                                                            |
| uration |                                                            |
+---------+------------------------------------------------------------+

References
~~~~~~~~~~

[1] Michael D. Lewek et al. (2012), “The influence of mechanically and
physiologically imposed stiff-knee gait patterns on the energy cost of
walking”, vol. 93, no.1, pp. 123-128. Publisher: Archives of Physical
Medicine and Rehabilitation.

[2] A. Gouelle and F. Mégrot (2017), “Interpreting spatiotemporal
parameters, symmetry, and variability in clinical gait analysis”,
Handbook of Human Motion pp. 1-20, Publisher: Springer International
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

You can also install the in-development version with::
.. code:: shell
    pip install https://github.com/DART-Lab-LLUI/python-gaitalytics/archive/main.zip
    conda install -c conda-forge btk

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
for advice. ###





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
