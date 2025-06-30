# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top - level directory of this distribution
# for details of code ownership.
#
# This program is free software : you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

__all__ = ["parse_duration", "DurationTime"]

import argparse
from typing import Self

from astropy.time import Time, TimeDelta


def parse_duration(duration: str) -> TimeDelta:
    """Accept string depicting duration.

    Numbers can be suffixed with character, denomination their lengths.

    Special values
    --------------
    now : current time
    yesterday: current time - 24 hours

    Length denominators
    -------------------
    W : weeks (7 * 86400 seconds)
    D : days (86400 seconds)
    h : hours (3600 seconds)
    m : minutes (60 seconds)
    s : seconds (1 second)

    Examples
    --------
    '1D 1m' = 86460 seconds
    '1h 1m 30s' = 3690 seconds

    Parameters
    ----------
    duration : `str`
        Duration string. Numbers with know suffixed. Non-sufficed number will
        be treated as seconds.

    Returns
    -------
    seconds : float
        Number of seconds in string.
    """
    muls = {
        "W": 604800,
        "D": 86400,
        "h": 3600,
        "m": 60,
        "s": 1,
        "u": 0.001,
        "n": 0.000001,
    }
    ret: float = 0.0
    current: float = 0.0
    duration = duration.strip()
    sign = 1
    fraction = 0
    if duration[0] == "-":
        sign = -1
        duration = duration[1:]
    elif duration[0] == "+":
        duration = duration[1:]

    for s in duration.strip():
        if "0" <= s <= "9":
            if fraction > 0:
                current += (0.1**fraction) * int(s)
                fraction += 1
            else:
                current = current * 10 + int(s)
        elif s == ".":
            fraction = 1
        elif s == " ":
            pass
        else:
            try:
                ret += current * muls[s]
                current = 0.0
            except KeyError:
                raise argparse.ArgumentTypeError(f"Unknown suffix: {s}")
    return TimeDelta(sign * (ret + current), format="sec")


class DurationTime(Time):
    """Converts either absolute time or relative string to Time. AS __call__
    method is overriden, the object can be used as function.

    Parameters
    ----------
    time : Time, optional
        Time for offsets. Defaults to the current system time.

    Attributes
    ----------
    base_time : Time
        Time passed as argument to the constructor.
    """

    base_time: Time | None = None

    def __init__(self, time: Time = Time.now()):
        super().__init__(time)
        self.base_time = None

    @property
    def absolute(self) -> bool:
        """True if the time was filled as absolute time."""
        return self.base_time is None

    @staticmethod
    def pair(
        t1: "DurationTime", t2: "DurationTime"
    ) -> tuple["DurationTime", "DurationTime"]:
        """Check for absolute times among input times, and produces pair of
        output times. Usefull to produce pair of start and end times parsed
        from command line, when one can be delta time.

        Parameters
        ----------
        t1 : DurationTime
            First parsed time.
        t2 : DurationTime
            Second parsed time.

        Returns
        -------
        start : DurationTime
            Start time deduced from t1 and t2.
        end : DurationTime
            End time deduced from t1 and t2.
        """

        tr1 = t1
        tr2 = t2

        if t1.absolute is True and t2.absolute is not True:
            tr2 = t1 + (t2 - t2.base_time)
        elif t1.absolute is not True and t2.absolute is True:
            tr1 = t2 + (t1 - t1.base_time)

        if tr2 < tr1:
            return (tr2, tr1)
        return (tr1, tr2)

    def __call__(self, duration_time: str) -> Self:
        """Returns time value either from time offset or from absolute time.

        Parameters
        ----------
        duration_time : str
            Input string. Shall be either absolute time, or duration string
            offset.

        Returns
        -------
        time : DurationTime
            Time either from the string, or self + offset
        """
        try:
            ls = duration_time.lower()
            if ls == "now":
                return Time.now()
            elif ls == "yesterday":
                return Time.now() - TimeDelta(1, format="jd")
            scale = "utc"
            if duration_time.count("A") > 0:
                duration_time = duration_time.replace("A", "T")
                scale = "tai"

            ret = DurationTime(Time(duration_time, scale=scale))
        except ValueError:
            ret = self + parse_duration(duration_time)
            ret.base_time = self

        return ret
