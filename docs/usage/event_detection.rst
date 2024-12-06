.. meta::
   :description: Gaitalytics User Guide.
   :keywords: gaitalytics, gait-analysis, mocap, c3d, gait-metrics, biomechanics, time-series, data-analysis, data, gait, guide, tutorial

Event Detection
===============

To detect gait events Gaitalytics currently only supports a marker based function based on :footcite:t:`zeni_two_2008` paper.
Further it provides helping functions to find errors in the detected gait events and store the events into an existing c3d file.

Detection
---------

Gaitalytics provides a function to detect the gait events in a c3d file :func:`gaitalytics.api.load_c3d_trial`.
Following snipped illustrates the usage of the detection function:

.. code-block::

        from gaitalytics import api

        config = api.load_config("./config.yaml")
        trial = api.load_c3d_trial("./example.c3d", config)
        events = api.detect_events(trial)

..


.. note::
    | The detection function is based on the assumption that the markers are correctly placed on the body. Through the configuration users can map the correct :ref:`marker names <Config Mapping>` to the internal marker names.
    |
    | Following internal markers are used for the detection: *l_heel*, *r_heel*, *l_toe*, *r_toe*, *sacrum* or *l_post_hip* and *r_post_hip*.

..

The function returns a pandas\ :footcite:p:`mckinney_pandas_2011` DataFrame with the following columns:

.. csv-table::
    :file: ../_static/tables/event_table.csv
    :widths: 10, 10, 40
    :header-rows: 1
..

.. hint::
    The marker-based function is uses on a peak detection to find the gait events. It allows to fine-tune the detection by changing the threshold and the minimum distance between two events through *kwargs*.
    Look here for more information: :class:`gaitalytics.events.MarkerEventDetection`

Checking
--------
The function :func:`gaitalytics.api.check_events` can be used to check the detected events for errors in the sequence.
To check the detected events for errors the following function can be used:

.. code-block::

        from gaitalytics import api

        events = api.detect_events(trial)
        try:
            api.check_events(events)
        except ValueError as e:
            print(f"Event errors {e}")
..

The function will check the sequence of the detected events and raise an error if the sequence is not correct.

.. hint::
    The ValueError will contain the error message which will help to identify the time range of the error.


Storing
-------
Using the :func:`gaitalytics.api.write_events_to_c3d` function the detected events can be stored into the c3d file.
The function can be used as follows:

.. code-block::

        from gaitalytics import api

        events = api.detect_events(trial)
        api.write_events_to_c3d("./example.c3d", events, './example_with_events.c3d')
..

The function will reload the *example.c3d* file and store the detected events into the new file *example_with_events.c3d*.
Alternatively the function can be used to store the events into the existing c3d file.

.. code-block::

        from gaitalytics import api

        events = api.detect_events(trial)
        api.write_events_to_c3d("./example.c3d", events, './example.c3d')
..

.. note::
    Gaitalytics allows to directly store the events in the existing :class:`gaitalytics.model.Trial` object. Nevertheless it is recommended to manually check and correct the events before continuing with the processing pipeline.
..


.. rubric:: References
.. footbibliography::





