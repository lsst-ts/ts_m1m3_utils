# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import unittest

import numpy as np
from mock_air_nozzles import MockNozzlesAndOrificesDiameters

from lsst.ts.m1m3.utils import (
    fit_thermal_gradients,
    nonstandard_thermocouples,
    thermocouple_z_position,
)
from lsst.ts.xml.tables.m1m3 import (
    M3_R,
    AirNozzleTable,
    ThermocoupleTable,
    set_air_nozzles_types_and_orifice_diameters,
)

INTERCEPT = 8.5
X_GRADIENT = 0.4
Y_GRADIENT = -0.25
Z_GRADIENT = 1.5


def plane_temperatures() -> dict[str, float]:
    """Per-thermocouple temperatures sampling an exact plane."""
    return {
        tc.name: INTERCEPT
        + X_GRADIENT * tc.x_position
        + Y_GRADIENT * tc.y_position
        + Z_GRADIENT * thermocouple_z_position(tc.name)
        for tc in ThermocoupleTable
    }


class ThermalGradientsTestCase(unittest.TestCase):
    def test_fit_recovers_plane(self) -> None:
        gradients = fit_thermal_gradients(plane_temperatures(), remove_nonstandard_cells=False)

        self.assertAlmostEqual(gradients.intercept, INTERCEPT, places=6)
        self.assertAlmostEqual(gradients.x_gradient, X_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.y_gradient, Y_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.z_gradient, Z_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.x_gradient_err, 0, places=6)
        self.assertAlmostEqual(gradients.y_gradient_err, 0, places=6)
        self.assertAlmostEqual(gradients.z_gradient_err, 0, places=6)

        # A pure x/y plane has no net radial dependence, so the radial fit
        # error reflects the scatter of the plane sampled against radius.
        self.assertTrue(math.isfinite(gradients.radial_gradient))
        self.assertTrue(math.isfinite(gradients.radial_gradient_err))
        self.assertTrue(math.isfinite(gradients.radial_intercept))
        self.assertTrue(math.isfinite(gradients.radial_z_gradient))

    def test_fit_recovers_radial_field(self) -> None:
        radial_gradient = 0.7
        temperatures = {
            tc.name: INTERCEPT
            + radial_gradient * math.hypot(tc.x_position, tc.y_position)
            + Z_GRADIENT * thermocouple_z_position(tc.name)
            for tc in ThermocoupleTable
        }
        gradients = fit_thermal_gradients(temperatures, remove_nonstandard_cells=False)

        self.assertAlmostEqual(gradients.radial_gradient, radial_gradient, places=6)
        self.assertAlmostEqual(gradients.radial_intercept, INTERCEPT, places=6)
        self.assertAlmostEqual(gradients.radial_z_gradient, Z_GRADIENT, places=6)
        # A purely radial field is symmetric, so x and y gradients vanish
        # (up to the slight asymmetry of the thermocouple positions).
        self.assertAlmostEqual(gradients.x_gradient, 0, places=1)
        self.assertAlmostEqual(gradients.y_gradient, 0, places=1)

    def test_nan_temperatures_are_ignored(self) -> None:
        temperatures = plane_temperatures()
        for name in list(temperatures)[:10]:
            temperatures[name] = math.nan

        gradients = fit_thermal_gradients(temperatures, remove_nonstandard_cells=False)

        self.assertAlmostEqual(gradients.x_gradient, X_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.y_gradient, Y_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.z_gradient, Z_GRADIENT, places=6)

    def test_insufficient_data(self) -> None:
        temperatures = dict(list(plane_temperatures().items())[:3])

        gradients = fit_thermal_gradients(temperatures, remove_nonstandard_cells=False)

        self.assertTrue(math.isnan(gradients.x_gradient))
        self.assertTrue(math.isnan(gradients.radial_gradient))

    def test_radius_limit(self) -> None:
        temperatures = plane_temperatures()
        # Corrupt all thermocouples outside M3_R; a fit limited to M3_R
        # must not see them.
        for tc in ThermocoupleTable:
            if np.hypot(tc.x_position, tc.y_position) > M3_R:
                temperatures[tc.name] = 1000.0

        gradients = fit_thermal_gradients(temperatures, remove_nonstandard_cells=False, radius_limit=M3_R)

        self.assertAlmostEqual(gradients.x_gradient, X_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.y_gradient, Y_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.z_gradient, Z_GRADIENT, places=6)

    def test_nonstandard_thermocouples(self) -> None:
        # Setting nozzle types mutates the global AirNozzleTable;
        # restore it so other tests see the pristine table.
        saved = [(nozzle.nozzle, nozzle.orifice_diameter) for nozzle in AirNozzleTable]

        def restore() -> None:
            for nozzle, (nozzle_type, orifice_diameter) in zip(AirNozzleTable, saved):
                nozzle.nozzle = nozzle_type
                nozzle.orifice_diameter = orifice_diameter

        self.addCleanup(restore)

        set_air_nozzles_types_and_orifice_diameters(MockNozzlesAndOrificesDiameters)

        nonstandard = nonstandard_thermocouples()
        self.assertEqual(len(nonstandard), 4)

        # Removing nonstandard cells still recovers an exact plane.
        gradients = fit_thermal_gradients(plane_temperatures(), remove_nonstandard_cells=True)
        self.assertAlmostEqual(gradients.x_gradient, X_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.y_gradient, Y_GRADIENT, places=6)
        self.assertAlmostEqual(gradients.z_gradient, Z_GRADIENT, places=6)


if __name__ == "__main__":
    unittest.main()
