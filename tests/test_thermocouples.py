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

import os
import sys
import unittest

import pandas as pd
import vcr
from astropy.time import Time, TimeDelta
from lsst_efd_client import EfdClient
from mock_air_nozzles import MockNozzlesAndOrificesDiameters

from lsst.ts.m1m3.utils import ThermocoupleAnalysis
from lsst.ts.xml.tables.m1m3 import (
    ThermocoupleTable,
    set_air_nozzles_types_and_orifice_diameters,
)

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class ThermocouplesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")
        self.tc_analysis = ThermocoupleAnalysis(self.client)

    async def test_load(self) -> None:
        start = Time("2025-08-25T18:00:00")
        end = start + TimeDelta(3600, format="sec")

        with myvcr.use_cassette("thermocouples_test_load.yaml"):
            await self.tc_analysis.load(start, end)

        assert len(self.tc_analysis.all_thermocouples_dataframe.index) == 60

        for tc in ThermocoupleTable:
            assert 8 <= self.tc_analysis.all_thermocouples_dataframe[tc.name].mean() <= 10
            assert 425 <= self.tc_analysis.all_thermocouples_dataframe[tc.name].sum() <= 580
            assert 8 <= self.tc_analysis.all_thermocouples_dataframe[tc.name].min() <= 9.5
            assert 8.7 <= self.tc_analysis.all_thermocouples_dataframe[tc.name].max() <= 10

            diff = self.tc_analysis.all_thermocouples_dataframe.index.diff()
            assert diff[0] is pd.NaT

            diff = diff[1:]
            assert len(diff[diff != pd.Timedelta("00:01:00")]) == 0

    async def test_empty(self) -> None:
        start = Time("2025-08-21T18:00:30")
        end = start + TimeDelta(1800, format="sec")

        with myvcr.use_cassette("thermocouples_test_empty.yaml"):
            await self.tc_analysis.load(start, end)

        assert self.tc_analysis.all_thermocouples_dataframe is None

    async def test_gradients(self) -> None:
        start = Time("2025-08-25T18:00:00")
        end = start + TimeDelta(3600, format="sec")

        set_air_nozzles_types_and_orifice_diameters(MockNozzlesAndOrificesDiameters)

        with myvcr.use_cassette("thermocouples_test_load.yaml"):
            await self.tc_analysis.load(start, end, time_bin=300)

        assert len(self.tc_analysis.nonstandard_thermocouples) == 4

        assert len(self.tc_analysis.all_thermocouples_dataframe.index) == 12

        assert len(self.tc_analysis.vertical_cell_gradient_dataframe.index) == 12

        assert 0.2 <= self.tc_analysis.mean_vertical_cell_gradient.iloc[0] <= 0.22

        assert -0.02 <= self.tc_analysis.xyz_r_gradients.x_gradient.iloc[0] <= 0.02

    async def test_bulk_metrics(self) -> None:
        start = Time("2025-08-25T18:00:00")
        end = start + TimeDelta(3600, format="sec")

        with myvcr.use_cassette("thermocouples_test_load.yaml"):
            await self.tc_analysis.load(start, end, time_bin=300)

        assert len(self.tc_analysis.bulk_glass_temperature_metrics.index) == 12

        assert 8.9 <= self.tc_analysis.bulk_glass_temperature_metrics.mean_temp.iloc[0] <= 9


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}")
    unittest.main()
