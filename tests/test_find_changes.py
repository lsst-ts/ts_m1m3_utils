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

import pandas as pd
import vcr
from astropy.time import Time
from lsst.ts.m1m3.utils import ChangedValue, find_changes
from lsst_efd_client import EfdClient

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class FindChangesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = EfdClient("usdf_efd")

    async def test_find_changes(self) -> None:
        with myvcr.use_cassette("find_changes.yaml"):
            data = await self.client.select_time_series(
                "lsst.sal.MTM1M3.logevent_softwareVersions",
                "*",
                Time("2025-01-01 12:00"),
                Time("2025-05-01 12:00"),
            )

        changes = list(find_changes(data))
        assert len(changes) == 47

        assert changes[0] == ChangedValue(
            "cscVersion",
            pd.Timestamp("2025-01-28 12:34:36.396268+00:00"),
            "v2.15.0-80-gf25b970-dirty",
            "v2.15.0-112-gabc8c159-dirty",
        )
        assert changes[1] == ChangedValue(
            "openSpliceVersion",
            pd.Timestamp("2025-01-28 12:34:36.396268+00:00"),
            "6.10.4",
            "0.0.0",
        )
        assert changes[2] == ChangedValue(
            "salVersion",
            pd.Timestamp("2025-01-28 12:34:36.396268+00:00"),
            "7.4.1-dirty",
            "10.0.0-rc2-2-ge243d92f-dirty",
        )
        assert changes[3] == ChangedValue(
            "xmlVersion",
            pd.Timestamp("2025-01-28 12:34:36.396268+00:00"),
            "20.0.0",
            "22.1.0-2-ga16574f2",
        )
        assert changes[4] == ChangedValue(
            "salVersion",
            pd.Timestamp("2025-02-24 16:53:11.756584+00:00"),
            "10.0.0-rc2-2-ge243d92f-dirty",
            "10.0.0-rc5-dirty",
        )

        assert changes[46] == ChangedValue(
            "cscVersion",
            pd.Timestamp("2025-04-30 14:20:05.086926+0000"),
            "v2.16.0-23-gdac71294-dirty",
            "v2.16.0-25-g2fdb9310-dirty",
        )


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}")
    unittest.main()
