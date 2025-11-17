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
from astropy.time import Time
from lsst.ts.m1m3.utils.compare_forces_two_tmasettings import compute_forces

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "all"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class CompareForcesTwoTMASettingsTestCase(unittest.IsolatedAsyncioTestCase):
    """Tests correlate_timeseries"""

    async def test_compare_forces_two_tmasettings(self) -> None:
        t1 = Time("2025-09-12T09:42:30Z")
        delta_t = pd.to_timedelta(10, unit="s")

        with myvcr.use_cassette("compare_forces.yaml"):
            [s1_forces, s2_forces] = await compute_forces("usdf_efd", t1, t1, delta_t, 0.0)
        residuals = s1_forces.fx - s2_forces.fx
        for value in residuals:
            self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}")
    unittest.main()
