.. py:currentmodule:: lsst.ts.m1m3.utils

.. _lsst.ts.m1m3.utils-version_history:

===============
Version History
===============

.. towncrier release notes start

v0.7.0 (2026-10-29)
===================

New Features
------------

- Add M1M3 thermocouple utils for metric calculations. (`DM-52561 <https://rubinobs.atlassian.net//browse/DM-52561>`_)
- Add ``ThermalGradients`` class and ``fit_plane_gradients`` method handling the M1M3 thermal gradients calculations. (`OSW-2472 <https://rubinobs.atlassian.net//browse/OSW-2472>`_)
- Add ``ThermocoupleCache`` to cache live M1M3 thermal scanner temperatures. (`OSW-2472 <https://rubinobs.atlassian.net//browse/OSW-2472>`_)
- Modify ``ThermocoupleAnalysis`` to use the plane-fit code from the ``ThermalGradients`` class. That fixes the Cartesian gradient error estimate (it was computed from the y coordinates instead of the temperature residuals). (`OSW-2472 <https://rubinobs.atlassian.net//browse/OSW-2472>`_)
- Add methods to retrieve not-tested FA status. (`SSW-3081 <https://rubinobs.atlassian.net//browse/SSW-3081>`_)


Other Changes and Additions
---------------------------

- Moved tests from ts_m1m3_cli. (`SSW-3081 <https://rubinobs.atlassian.net//browse/SSW-3081>`_)
- License check. (`SSW-3081 <https://rubinobs.atlassian.net//browse/SSW-3081>`_)


v0.6.2
------

* Add new feature to include a radius limit in the thermocouples module when calculating gradients. 

v0.6.1
------

* Added tool to find and count user-defined HP excesses

v0.6.0
------

* Move scripts to ts_m1m3_cli.

v0.5.0
------

* ThermocoupleAnalysis to calculate M1M3 thermocouple metrics.

v0.4.2
------

* Fixes for new lsst-efd-client API.
* find-changes tool to look for changes in data.
* correlate-timeseries to corralate data.

v0.4.1
------

* CylinderForces to calculate XYZ forces from cylinder values

v0.4.0
------

* BoosterValves class - filter for booster valves activation
* thermocouples module
* calculate_far_neighbors_factors - calculate far neighbor values

v0.3.2
------

* Added ts-salobj to Conda dependencies.

v0.3.1
------

* Improved bump_test_times querying and reporting.
* Add new script ``fcu_stats.py`` to report FCU statistics

v0.3.0
------

* BumpTest Runner - executes parallel bump tests
* m1m3-fe-outliers
* DurationTime class

v0.2.1
------

* fix conda and pip dependencies

v0.2.0
------

* m1m3-aav tool to fit acceleration and velocity forces.

v0.1.0
------

* Initial working version. Report bump tests.
