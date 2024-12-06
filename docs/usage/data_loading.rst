.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Data Loading
============
Gaitalytics is designed to be marker model independent. It does not require specific marker, analog or modelled marker names.
Nevertheless it needs some information about parts of the naming scheme to be able to identify necessary data points.
This information is provided in the configuration file.



.. warning::
    The motion capture data must be preprocessed before loading it into the library. Gap filling, filtering and calculation of kinetics / kinematics must be done before loading the data.
..

The function :func:`gaitalytics.api.load_c3d_trial` is used to load a c3d file and the corresponding configuration file.
Here is an example of how to load a c3d file and the corresponding configuration file:

.. code-block:: python
    :name: load_config.py

    from gaitalytics import api
    config = api.load_config("./config.yaml")


..


Configuration
-------------

.. note::
    The configuration file is mandatory for library to work.
..

The configuration is communicated to the framework through a yaml file.
The file is structured as follows:

.. code-block:: yaml
    :name: config.yaml

    # Mapping of model marker names to internal marker names
    mapping:
      markers:

        # Foot
        l_heel: "ExampleHeelMarker"
        r_heel: "ExampleHeelMarker"
        l_toe: "ExampleToeMarker"
        r_toe: "ExampleToeMarker"
        l_toe_2: "ExampleSecondaryToeMarker" # optional
        r_toe_2: "ExampleSecondaryToeMarker" # optional
        l_lat_malleoli: "ExampleMalleoliMarker" # optional
        r_lat_malleoli: "ExampleMalleoliMarker" # optional


        # Hip
        l_ant_hip: "ExampleHipMarker"
        r_ant_hip: "ExampleHipMarker"
        l_post_hip: "ExampleHipMarker" # optional
        r_post_hip: "ExampleHipMarker" # optional
        sacrum: "ExampleSacrumMarker"  # optional
        xcom: "ExampleXCOMMarker" # optional

    # Entities to analyse
    analysis:
      markers: # Markers to analyse
        # List of modelled markers to analyse
        - "ExampleAngle1"
        - "ExampleGRF"
        - "ExampleForce2"
      analogs:
        # List of analog channels to analyse
        - "ExampleForce3"


..

The configuration file is structured in two parts. The mapping section and the anlaysis section.

.. _Config Mapping:

Mapping
^^^^^^^
The mapping section is used to map model marker names to internal marker names.

.. csv-table::
   :file: ../_static/tables/mapping.csv
   :widths: 15, 70, 15
   :header-rows: 1


| \* Depending on the marker model, the posterior hip or sacrum may not be present. Either sacrum or both posterior hip markers must be provided.
| ✝ Only needed to get a precise calculation of the minimal toe clearance.
| ☨ Only needed to calculate the margin of stability.

Analysis
^^^^^^^^

The analysis section is used to specify which entities should be analysed. The entities are divided into markers and analogs, whereas markers will be found in the point sections of the c3d and analogs in the analogs section of the c3d file.
Configuring entities in the analysis section will have two consequences:

    1. Time series features will be extracted from these entities.
    2. The entities will be flattened (i.e. the entities will be transformed into a single time series) and saved in the analysis part of the Trial object.


..
 TODO : link to the Trial object feature
..

To define the entities to analyse, the user must provide the marker or analog of the entities which should be used.




Load c3d file
-------------
The function :func:`gaitalytics.api.load_c3d_trial` is used to load a c3d file and the corresponding configuration file.
Following code snipped illustrates how to load a c3d file with the corresponding configuration file:

.. code-block:: python
    :name: load_c3d.py

    from gaitalytics import api

    config = api.load_config("./config.yaml")
    trial = api.load_c3d_trial("./example.c3d", config)

..

| The load_trial function will return a Trial object which contains all the information of the c3d file. Internally Gaitalytics uses the pyomeca\ :footcite:p:`martinez_pyomeca_2020`, ezc3d\ :footcite:p:`michaud_ezc3d_2021` and xarray\ :footcite:p:`hoyer_xarray_2017` libraries to load and store the data.
| The object returned is a :class:`gaitalytics.model.Trial`. It contains three types of data.

    1. Markers -> time series form the Point section of the c3d
    2. Analog -> time series from the Analog section of the c3d
    3. Analysis -> flattened time series from the entities specified in the configuration file.

Additionally, the object contains the events which are stored in the c3d file.

.. rubric:: References
.. footbibliography::







