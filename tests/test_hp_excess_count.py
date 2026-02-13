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

import numpy as np
import pandas as pd
from astropy.time import TimeDelta

from lsst.ts.m1m3.utils import HPForces


class CountHPExcessesTestCase(unittest.TestCase):
    """Tests CountHPExcesses"""

    def test_calculate_excesses(self) -> None:
        nsamples = 6000
        test_sample = int(nsamples / 2)
        test_force = 1000  # N
        # Create an array of zeros
        data = np.zeros(nsamples)
        # Set the value at index 500
        data[test_sample] = test_force
        # Create the DataFrame
        sampling_freq = 20  # Hz
        freq = float(1 / sampling_freq)
        df = pd.DataFrame({"measuredForce0": data})
        df.index = pd.date_range(start="2026-01-01 00:00:00", periods=nsamples, freq=f"{freq}s")
        df.index.name = "time"
        # Initialize an HPForces instance
        time_gap_threshold = TimeDelta("1s")
        hpf = HPForces(df, sampling_freq, time_gap_threshold)
        # Test the excess counter defining some parameters first
        delta_t = 1
        delta_f_threshold = 200
        event_summary = hpf.calculate_excesses(0, delta_t, delta_f_threshold)
        self.assertAlmostEqual(event_summary["max_value"][0], test_force)
        self.assertEqual(len(event_summary), 1)


if __name__ == "__main__":
    unittest.main()
