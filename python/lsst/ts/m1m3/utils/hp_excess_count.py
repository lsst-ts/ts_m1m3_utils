# This file is part of ts_m1m3_utils.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org). See the COPYRIGHT file at the top - level directory
# of this distribution for details of code ownership.
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

__all__ = ["HPForces"]

import pandas as pd
from astropy.time import TimeDelta


class HPForces:
    """Class to handle excess measured in data on hardpoints

    Parameters
    ----------
    forces: `pd.DataFrame`
        Should contain "measuredForce[0-5]" fields,
        corresponding to forces on hardpoints (HPs)
    sampling_freq: `int`
        Number of samples per second in the dataframe
    time_gap_threshold: `Timedelta`
        Minimum time between events to consider a grouped excess.
        1s is a good value.
    """

    def __init__(self, forces: pd.DataFrame, sampling_freq: int, time_gap_threshold: TimeDelta):
        self.forces = forces
        self.sampling_freq = sampling_freq
        self.time_gap_threshold = pd.to_timedelta(time_gap_threshold.sec, unit="s")

    def calculate_excesses(self, hp_id: int, delta_t: float, delta_f_threshold: float) -> tuple[pd.DataFrame]:
        """Count excesses on hardpoints defined as events in which
        the force changes above a certain threshold in a given time

        Parameters
        ----------
        hp_id: `int`
            Hardpoint identifier [1-6]
        delta_t: `float`
            Reference period (in seconds) in which an 'excess' is defined
        delta_f_threshold: `float`
            Minimum increment of force (in N) above which (in absolute value)
            an excess is considered in delta_t time
        """

        # number of samples is calculated assuming sampling frequency is in Hz
        nsamples = int(self.sampling_freq * delta_t)
        # compute difference in force from beginning to end of period defined
        # by delta_t
        hp_index = hp_id - 1
        df_hps = self.forces[f"measuredForce{hp_index}"] - self.forces[f"measuredForce{hp_index}"].shift(
            nsamples
        )
        df_delta = abs(df_hps.dropna())
        # count in how many points od the dataframe the condition is met,
        # this will result in some grouped events
        above = df_delta[df_delta > delta_f_threshold]
        # measure time between events passing requirement above
        time_gap = above.index.to_series().diff()
        # aggregate those events that are separated by self.time_gap_threshold
        event_id = (time_gap > self.time_gap_threshold).cumsum()
        events = above.groupby(event_id)
        event_summary = pd.DataFrame(
            {
                "start": events.apply(lambda x: x.index[0]),
                "end": events.apply(lambda x: x.index[-1]),
                "duration": events.apply(lambda x: x.index[-1] - x.index[0]),
                "samples": events.size(),
                "max_value": events.apply(lambda x: x.values.max()),
            }
        )

        return event_summary
