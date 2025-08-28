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

import os
import sys
import unittest

import vcr
from astropy.time import Time
from lsst.ts.m1m3.utils import BumpTestTimes
from lsst.ts.xml.tables.m1m3 import force_actuator_from_id
from lsst_efd_client import EfdClient

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class BumpTestTimesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")
        self.btt = BumpTestTimes(self.client)

    async def get_tests(
        self, actuator_id: int, start_t: Time, end_t: Time
    ) -> tuple[BumpTestTimes, BumpTestTimes]:
        fa = force_actuator_from_id(actuator_id)
        primary = tuple(
            [
                tt
                async for tt in self.btt.find_times(
                    fa,
                    True,
                    Time("2024-09-09 13:28:04"),
                    Time("2024-09-16 13:28:04"),
                )
            ]
        )
        secondary = tuple(
            [
                tt
                async for tt in self.btt.find_times(
                    fa,
                    False,
                    Time("2024-09-09 13:28:04"),
                    Time("2024-09-16 13:28:04"),
                )
            ]
        )
        return (primary, secondary)

    async def test_times_saa(self) -> None:
        with myvcr.use_cassette("bump_test_times_saa.yaml"):
            primary, secondary = await self.get_tests(
                101, Time("2024-09-09 13:28:04"), Time("2024-09-16 13:28:04")
            )

        self.assertEqual(len(primary), 10)
        self.assertEqual(len(secondary), 0)

    async def test_times_daa(self) -> None:
        with myvcr.use_cassette("bump_test_times_daa.yaml"):
            primary, secondary = await self.get_tests(
                435, Time("2024-09-09 13:28:04"), Time("2024-09-16 13:28:04")
            )

        self.assertEqual(len(primary), 10)
        self.assertEqual(len(secondary), 10)


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(
            f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}"
        )
    unittest.main()
