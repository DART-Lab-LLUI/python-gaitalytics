.. meta::
   :description: Gaitalytics Feature.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, references

Extend features
===============

.. warning::
    Gaitalytics is build to be independent on marker-models and modelling mechanisms through :ref:`configurations <Configuration>`.
    In this Tutorial we will not cover the creation of new :ref:`mapping configurations <Config Mapping>` since it will alter core functionality of the library.
    Your new features therefore may not work within different setups. To make your features available to the community and integrated to gaitalytics,
    please consider :ref:`contributing <Development>` to the project and our staff will take care of the integration.

Before you start creating new features, you should think about the following questions:

     - Will my feature be calculated for each gait cycle?
     - Are all the needed markers defined through the :ref:`mappings  <Config Mapping>`?

If you have answered all the questions, you can start creating your new feature.

Class inheritance
-----------------

With with your answers in mind, you can now decide which class you want to inherit from. The following classes are available in the library:

    - :class:`gaitalytics.features.FeatureCalculation`
    - :class:`gaitalytics.features.CycleFeaturesCalculation`
    - :class:`gaitalytics.features.PointDependentFeature`

The following flowchart will help you to decide which class you should inherit from:

.. mermaid::

    flowchart LR
        A{Per cycle?}---|Yes|B{Mapped markers?}
        A---|No|FeatureCalculation
        B---|Yes|PointDependentFeature
        B---|No|CycleFeaturesCalculation

..






