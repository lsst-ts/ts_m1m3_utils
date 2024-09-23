# This file is part of ts_m1m3_utils.
#
# Developed for the Rubin Observatory Telescope and Site System.
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

from astropy.time import Time
from lsst.ts.m1m3.utils import BumpTestTimes
from lsst_efd_client import EfdClient


class BumpTestTimesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")
        self.btt = BumpTestTimes(self.client)

    async def asyncTearDown(self) -> None:
        await self.client.influx_client.close()

    async def test_times_saa(self) -> None:
        primary, secondary = await self.btt.find_times(
            101, Time("2024-09-09 13:28:04"), Time("2024-09-16 13:28:04")
        )

        self.assertEqual(len(primary), 10)
        self.assertEqual(len(secondary), 0)

    async def test_times_daa(self) -> None:
        primary, secondary = await self.btt.find_times(
            435, Time("2024-09-09 13:28:04"), Time("2024-09-16 13:28:04")
        )

        self.assertEqual(len(primary), 10)
        self.assertEqual(len(secondary), 10)


if __name__ == "__main__":
    print(
        "This test assumes EFD connectivity setup. If it did not pass, this might be the cause."
    )
    unittest.main()
