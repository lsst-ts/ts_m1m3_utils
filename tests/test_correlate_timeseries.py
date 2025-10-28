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
from lsst.ts.m1m3.utils.correlate_timeseries import compute_time_delay

CASSETTE_DIR = os.path.join(os.path.dirname(__file__), "cassettes")

myvcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode=os.getenv("RECORD_MODE", "none"),
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


class CorrelateTimeseriesTestCase(unittest.IsolatedAsyncioTestCase):
    """Tests correlate_timeseries"""

    async def test_correlate_timeseries(self) -> None:
        t1 = Time("2025-09-12T09:42:30Z")
        t2 = Time("2025-09-12T09:59:50Z")
        extra_shift_test2 = TimeDelta(3.1, format="sec")
        delta_t = pd.to_timedelta(200, unit="s")
        sampling = 20
        overlay = TimeDelta(60, format="sec")

        with myvcr.use_cassette("correlate_timeseries.yaml"):
            [delay_test1, s1_test1, s2_test1] = await compute_time_delay(
                "usdf_efd", t1, t2, delta_t, sampling, overlay
            )
            [delay_test2, s1_test2, s2_test2] = await compute_time_delay(
                "usdf_efd", t1, t2 + extra_shift_test2, delta_t, sampling, overlay
            )

        self.assertAlmostEqual(delay_test1 / sampling, 55.60)
        self.assertAlmostEqual(delay_test2 / sampling, 52.40)


if __name__ == "__main__":
    if "RECORD_MODE" not in os.environ:
        print(f"To generate new cassettes with pre-downloaded data use: RECORD_MODE=all python {sys.argv[0]}")
    unittest.main()
