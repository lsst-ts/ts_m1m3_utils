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

from lsst.ts.m1m3.utils import ThermocoupleCache
from lsst.ts.m1m3.utils.thermocouple_cache import CHANNELS_PER_MESSAGE
from lsst.ts.xml.tables.m1m3 import Scanner, ThermocoupleTable, find_thermocouple

# Chunks (16-channel ESS.temperature messages) per scanner that contain
# mapped thermocouples.
NUM_CHUNKS = 3

DEFAULT_TEMPERATURE = 12.5


def feed_scanner(
    cache: ThermocoupleCache,
    scanner: Scanner,
    timestamp: float,
    chunks: list[int] | None = None,
    nan_channels: set[tuple[int, int]] | None = None,
) -> None:
    """Feed all messages of one scanner scan into the cache.

    Parameters
    ----------
    nan_channels : set of (chunk_index, channel), optional
        Channels to report as NaN.
    """
    if chunks is None:
        chunks = list(range(NUM_CHUNKS))
    if nan_channels is None:
        nan_channels = set()
    for chunk_index in chunks:
        temperatures = [
            math.nan if (chunk_index, channel) in nan_channels else DEFAULT_TEMPERATURE
            for channel in range(CHANNELS_PER_MESSAGE)
        ]
        cache.add_temperatures(
            sal_index=int(scanner),
            sensor_name=f"m1m3-ts-{int(scanner) - 113:02d} {chunk_index + 1}/6",
            timestamp=timestamp,
            temperatures=temperatures,
        )


def feed_all(cache: ThermocoupleCache, timestamp: float) -> None:
    for scanner in Scanner:
        feed_scanner(cache, scanner, timestamp)


class ThermocoupleCacheTestCase(unittest.TestCase):
    def test_complete_set(self) -> None:
        cache = ThermocoupleCache()
        self.assertIsNone(cache.valid_set())

        # Feed everything except the last chunk of the last scanner.
        for scanner in list(Scanner)[:-1]:
            feed_scanner(cache, scanner, timestamp=1000.0)
        feed_scanner(cache, Scanner.TS_04, timestamp=1000.0, chunks=[0, 1])
        self.assertIsNone(cache.valid_set())

        feed_scanner(cache, Scanner.TS_04, timestamp=1000.0, chunks=[2])
        valid_set = cache.valid_set()
        self.assertIsNotNone(valid_set)
        self.assertEqual(len(valid_set), len(ThermocoupleTable))
        self.assertEqual(set(valid_set), {tc.name for tc in ThermocoupleTable})
        self.assertTrue(all(t == DEFAULT_TEMPERATURE for t in valid_set.values()))
        # Cold junctions (channel 0 of chunk 1) must not appear.
        self.assertFalse(any(name.startswith("coldJunction") for name in valid_set))
        self.assertEqual(cache.newest_timestamp, 1000.0)

    def test_staleness(self) -> None:
        cache = ThermocoupleCache(max_data_age=120.0)
        feed_all(cache, timestamp=1000.0)
        self.assertIsNotNone(cache.valid_set())

        # One scanner reports much later; the other three are now stale.
        feed_scanner(cache, Scanner.TS_01, timestamp=2000.0)
        self.assertIsNone(cache.valid_set())
        self.assertEqual(cache.newest_timestamp, 2000.0)

        # Once everything reports again, the set is valid again.
        feed_all(cache, timestamp=2000.0)
        self.assertIsNotNone(cache.valid_set())

    def test_max_missing(self) -> None:
        # A channel that never reports a finite value keeps its
        # thermocouple out of the set.
        thermocouple = find_thermocouple(Scanner.TS_01, CHANNELS_PER_MESSAGE + 1)
        self.assertIsNotNone(thermocouple)

        for max_missing, expect_valid in [(0, False), (1, True)]:
            with self.subTest(max_missing=max_missing):
                cache = ThermocoupleCache(max_missing=max_missing)
                feed_scanner(cache, Scanner.TS_01, timestamp=1000.0, nan_channels={(1, 1)})
                for scanner in list(Scanner)[1:]:
                    feed_scanner(cache, scanner, timestamp=1000.0)

                self.assertEqual(cache.missing_names(), {thermocouple.name})
                valid_set = cache.valid_set()
                if expect_valid:
                    self.assertIsNotNone(valid_set)
                    self.assertNotIn(thermocouple.name, valid_set)
                else:
                    self.assertIsNone(valid_set)

    def test_invalid_messages(self) -> None:
        cache = ThermocoupleCache()
        temperatures = [DEFAULT_TEMPERATURE] * CHANNELS_PER_MESSAGE

        with self.assertRaises(ValueError):
            cache.add_temperatures(118, "m1m3-ts-01 1/6", 1000.0, temperatures)

        with self.assertRaises(ValueError):
            cache.add_temperatures(114, "unexpected sensor", 1000.0, temperatures)

    def test_unmapped_chunks_ignored(self) -> None:
        cache = ThermocoupleCache()
        # Chunks 4-6 carry no mapped thermocouples.
        for scanner in Scanner:
            feed_scanner(cache, scanner, timestamp=1000.0, chunks=[3, 4, 5])
        self.assertTrue(math.isnan(cache.newest_timestamp))
        self.assertIsNone(cache.valid_set())

    def test_clear(self) -> None:
        cache = ThermocoupleCache()
        feed_all(cache, timestamp=1000.0)
        self.assertIsNotNone(cache.valid_set())
        cache.clear()
        self.assertIsNone(cache.valid_set())
        self.assertTrue(math.isnan(cache.newest_timestamp))


if __name__ == "__main__":
    unittest.main()
