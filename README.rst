========
Overview
========

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

Gaitalytics
===========
.. image:: https://github.com/DART-Lab-LLUI/gaitalytics/blob/27ff8401295c3a05537409deb3982129ed78222c/resources/logos/Gaitalytics_noBackground.png
This Python package provides a comprehensive set of tools and advanced algorithms for analyzing 3D motion capture data.
It is specifically designed to process gait data stored in c3d format. Prior to utilizing the features of gaitalytics,
it is necessary to perform data labeling, modeling, and filtering procedures.

The library's versatility allows it to be adaptable to various marker sets and modeling algorithms,
offering high configurability.

.. note::
    Current pre-release is only tested with data acquired with Motek Caren, HBM2 Lower Body Trunk and PIG.


Free software: MIT license

Installation
============

::

    pip install gaitalytics

You can also install the in-development version with::

    pip install https://github.com/cereneo-foundation/python-gaitalytics/archive/main.zip

Features
========




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
