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
from astropy.time import Time, TimeDelta
from lsst.ts.m1m3.utils import BoosterValves
from lsst_efd_client import EfdClient

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class BoosterValvesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")

    async def get_tests(self, start: Time, end: Time) -> list[BoosterValves]:
        return [tt async for tt in self.bv.find_opened(start, end)]

    async def test_booster_valves(self) -> None:
        self.bv = BoosterValves(self.client)
        with myvcr.use_cassette("booster_valves.yaml"):
            ret = await self.get_tests(
                Time("2024-01-10 01:00:00"), Time("2024-01-10 02:00:00")
            )
            assert len(ret) == 58

            ret = await self.get_tests(
                Time("2024-01-10 00:00:00"), Time("2024-01-10 10:00:00")
            )
            assert len(ret) == 432

    async def test_booster_diff(self) -> None:
        with myvcr.use_cassette("booster_diff.yaml"):
            self.bv = BoosterValves(self.client, TimeDelta(3600, format="sec"))
            ret = await self.get_tests(
                Time("2024-01-10 01:00:00"), Time("2024-01-10 02:00:00")
            )
            assert len(ret) == 58

            ret = await self.get_tests(
                Time("2024-01-10 00:00:00"), Time("2024-01-10 10:00:00")
            )
            assert len(ret) == 432


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(
            f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}"
        )
    unittest.main()
