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

import unittest

import pandas as pd
from astropy.time import Time, TimeDelta
from lsst.ts.m1m3.utils import thermocouples
from lsst.ts.xml.tables.m1m3 import ThermocoupleTable
from lsst_efd_client import EfdClient


class ThermocouplesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")

    # async def asyncTearDown(self) -> None:
    #    try:
    #        await self.client._influx_client.close()
    #    except AttributeError:
    #        await self.client.influx_client.close()

    async def test_load(self) -> None:
        start = Time("2025-08-25T18:00:00")
        end = start + TimeDelta(3600, format="sec")

        data = await thermocouples.get_scanner_data(self.client, start, end)

        assert len(data.index) == 120

        for tc in ThermocoupleTable:
            assert 8 <= data[tc.name].mean() <= 10
            assert 1010 <= data[tc.name].sum() <= 1100
            assert 8 <= data[tc.name].min() <= 9.5
            assert 9 <= data[tc.name].max() <= 10

            diff = data.index.diff()
            assert diff[0] is pd.NaT

            diff = diff[1:]
            assert len(diff[diff != pd.Timedelta("00:00:30")]) == 0

    async def test_empty(self) -> None:
        start = Time("2025-08-21T18:00:30")
        end = start + TimeDelta(1800, format="sec")

        data = await thermocouples.get_scanner_data(self.client, start, end)

        assert data.empty


if __name__ == "__main__":
    print(
        "This test assumes EFD connectivity setup. If it did not pass, this might be the cause."
    )
    unittest.main()
