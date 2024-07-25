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

.. |commits-since| image:: https://img.shields.io/github/commits-since/DART-Lab-LLUI/python-gaitalytics/v0.1.2.svg
    :alt: Commits since latest release
    :target: https://github.com/DART-Lab-LLUI/python-gaitalytics/compare/v0.1.2...main

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

Event Detection
^^^^^^^^^^^^^^^

+------------+--------------+----------------------------------------------------------------------------+
| Method     | Description  | options                                                                    |
+============+==============+============================================================================+
| Marker     | Zenis 2006   |                                                                            |
+------------+--------------+----------------------------------------------------------------------------+


Event Detection Check
^^^^^^^^^^^^^^^^^^^^^

+------------+--------------------------------------------------+-------------------------+
| Method     | Description                                      | options                 |
+============+==================================================+=========================+
| context    | Checks gait event sequences                      |                         |
|            | HS->TO-HS-TO                                     |                         |
+------------+--------------------------------------------------+-------------------------+
| spacing    | Checks Frames between same event on same context |                         |
+------------+--------------------------------------------------+-------------------------+


Analysis
^^^^^^^^

+-----------------+------------------------------------------------------------+---------------------------------------+
| Method          | Description                                                | options                               |
+=================+============================================================+=======================================+
| angels          | - min                                                      |                                       |
| forces          | - max                                                      |                                       |
| moments         | - mean                                                     |                                       |
| power           | - sd                                                       |                                       |
|                 | - median                                                   |                                       |
|                 | - amplitude                                                |                                       |
+-----------------+------------------------------------------------------------+---------------------------------------+
| Spatio-temporal | - step_length [1]                                          |                                       |
|                 | - stride_length [1]                                        |                                       |
|                 | - cycle_duration                                           |                                       |
|                 | - swing_duration_perc                                      |                                       |
|                 | - stance_duration_perc                                     |                                       |
|                 | - step_width [1]                                           |                                       |
|                 | - cadence [1]                                              |                                       |
|                 | - single_support_duration_percent [2]                      |                                       |
|                 | - double_support_duration_percent [2]                      |                                       |
+-----------------+------------------------------------------------------------+---------------------------------------+

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

    conda install gaitalytics
..


You can also install the in-development version with:

.. code:: shell

    todo
..

Configuration
^^^^^^^^^^^^^

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
^^^^^^^^

Please take the resources in the `example
folder <https://github.com/DART-Lab-LLUI/python-gaitalytics/tree/defc453f95940db55f6875ae7568949daa1b67d4/examples>`__
for advice.

Documentation
-------------

https://python-gaitalytics.readthedocs.org




