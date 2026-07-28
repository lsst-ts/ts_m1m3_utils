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

__all__ = [
    "ThermocoupleCache",
]

import math
import re
from collections.abc import Sequence

from lsst.ts.xml.tables.m1m3 import Scanner, ThermocoupleTable, find_thermocouple

# Number of temperature channels per ESS.temperature message.
CHANNELS_PER_MESSAGE = 16

# Regular expression matching the sensorName written by the
# GecThermalscannerDataClient: e.g. "m1m3-ts-01 2/6".
SENSOR_NAME_RE = re.compile(r"m1m3-ts-0\d (\d+)/(\d+)")


class ThermocoupleCache:
    """Cache live M1M3 thermal scanner temperatures until a valid set is
    completed.

    Ingests ESS.temperature telemetry messages published by the four M1M3
    GEC thermal scanner instances (SAL indices 114-117, see
    `lsst.ts.xml.tables.m1m3.Scanner`) and maps every channel to its
    thermocouple. A valid set is completed when all expected thermocouples
    have a finite sample not older than ``max_data_age`` seconds relative
    to the newest cached sample.

    Parameters
    ----------
    max_data_age : `float`, optional
        Maximum age (seconds) of a cached sample, relative to the newest
        cached sample, for it to count towards a valid set. Defaults to
        120 seconds - four times the nominal 30 second scanner cadence.
    max_missing : `int`, optional
        Maximum number of expected thermocouples that can be missing or
        stale while the set is still considered valid. Defaults to 0.
    expected_names : `set` [`str`], optional
        Names of the thermocouples required for a valid set. Defaults to
        all thermocouples in `lsst.ts.xml.tables.m1m3.ThermocoupleTable`.
    """

    def __init__(
        self,
        max_data_age: float = 120.0,
        max_missing: int = 0,
        expected_names: set[str] | None = None,
    ):
        self.max_data_age = max_data_age
        self.max_missing = max_missing
        if expected_names is None:
            expected_names = {thermocouple.name for thermocouple in ThermocoupleTable}
        self.expected_names = expected_names

        # Thermocouple name: (temperature (deg C), timestamp (TAI unix sec)).
        self._samples: dict[str, tuple[float, float]] = {}

    def add_temperatures(
        self,
        sal_index: int,
        sensor_name: str,
        timestamp: float,
        temperatures: Sequence[float],
    ) -> bool:
        """Ingest one ESS.temperature telemetry message.

        Non-finite temperatures and channels without an assigned
        thermocouple (e.g. the cold junction) are ignored.

        Parameters
        ----------
        sal_index : `int`
            SAL index of the ESS instance that published the message.
            Must be one of the `Scanner` values (114-117).
        sensor_name : `str`
            The sensorName field of the message, e.g. "m1m3-ts-01 2/6".
        timestamp : `float`
            The timestamp field of the message (TAI unix seconds).
        temperatures : `Sequence` [`float`]
            The temperatureItem field of the message - 16 channel
            temperatures in deg C.

        Returns
        -------
        `bool`
            True if any cached thermocouple value was updated.

        Raises
        ------
        ValueError
            If sal_index is not a thermal scanner index or sensor_name
            does not match the expected M1M3 thermal scanner format.
        """
        scanner = Scanner(sal_index)

        match = SENSOR_NAME_RE.match(sensor_name)
        if match is None:
            raise ValueError(f"Unexpected sensorName for index {sal_index}: {sensor_name!r}")
        chunk_index = int(match[1]) - 1

        updated = False
        for channel, temperature in enumerate(temperatures[:CHANNELS_PER_MESSAGE]):
            thermocouple = find_thermocouple(scanner, chunk_index * CHANNELS_PER_MESSAGE + channel)
            if thermocouple is None or not math.isfinite(temperature):
                continue
            self._samples[thermocouple.name] = (temperature, timestamp)
            updated = True
        return updated

    @property
    def newest_timestamp(self) -> float:
        """Timestamp of the newest cached sample, or NaN if empty."""
        if not self._samples:
            return math.nan
        return max(timestamp for _, timestamp in self._samples.values())

    def missing_names(self) -> set[str]:
        """Return the expected thermocouples that are missing or stale."""
        newest = self.newest_timestamp
        if math.isnan(newest):
            return set(self.expected_names)
        return {
            name
            for name in self.expected_names
            if name not in self._samples or newest - self._samples[name][1] > self.max_data_age
        }

    def valid_set(self) -> dict[str, float] | None:
        """Return the completed set of thermocouple temperatures, if valid.

        Returns
        -------
        `dict` [`str`, `float`] or None
            Temperature (deg C) per thermocouple name for all expected
            thermocouples with a fresh sample, or None if more than
            ``max_missing`` of them are missing or stale.
        """
        missing = self.missing_names()
        if len(missing) > self.max_missing:
            return None
        return {name: self._samples[name][0] for name in self.expected_names - missing}

    def clear(self) -> None:
        """Drop all cached samples."""
        self._samples.clear()
