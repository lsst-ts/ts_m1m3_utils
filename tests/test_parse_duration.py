# This file is part of ts_criopy.
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

import argparse
import unittest

from astropy.time import Time, TimeDelta
from lsst.ts.m1m3.utils import DurationTime, parse_duration


class ParseDurationTestCase(unittest.TestCase):
    def test_parse(self) -> None:
        assert parse_duration("1") == TimeDelta(1, format="sec")
        assert parse_duration("1 ") == TimeDelta(1, format="sec")
        assert parse_duration(" 1") == TimeDelta(1, format="sec")
        assert parse_duration(" 1 D ") == TimeDelta(86400, format="sec")
        assert parse_duration("2h3m6s") == TimeDelta(
            2 * 3600 + 3 * 60 + 6, format="sec"
        )
        assert parse_duration(" 10h 67   m 12345s") == TimeDelta(
            10 * 3600 + 67 * 60 + 12345, format="sec"
        )
        assert parse_duration(" 10h 67   m 12345") == TimeDelta(
            10 * 3600 + 67 * 60 + 12345, format="sec"
        )
        assert parse_duration("125u735n") == TimeDelta(0.125735, format="sec")

    def test_duration_time(self) -> None:
        assert (
            DurationTime(Time("2020-01-01T10:20"))("2m").isot
            == Time("2020-01-01T10:22").isot
        )
        assert (
            DurationTime(Time("2020-12-31T23:59"))("2m").isot
            == Time("2021-01-01T00:01").isot
        )

        assert (
            DurationTime(Time("1987-01-01T00:59"))("-59m1s").isot
            == Time("1986-12-31T23:59:59").isot
        )

        assert (
            DurationTime(Time("1986-04-26T01:23:47.53"))("-10430D5h55m36.53s").isot
            == Time("1957-10-04T19:28:34").isot
        )

    def test_failures(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_duration("1d")
            parse_duration("1Dd")
            parse_duration("239  Dd")
            parse_duration("2m@")
            parse_duration(" 2m  3 S")


if __name__ == "__main__":
    unittest.main()
