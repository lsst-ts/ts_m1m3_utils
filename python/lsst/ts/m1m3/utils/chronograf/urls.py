# This file is part of ts_m1m3_utils
#
# Developed for the LSST Telescope and Site.
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

from urllib.parse import urlencode, urlunparse

from astropy.time import Time
from lsst.ts.xml.tables.m1m3 import ForceActuatorData

__all__ = ["DashboardURL", "M1M3FCUStats", "M1M3ForceActuatorForces"]


class DashboardURL:
    """Construct links to given dashboard. Keywords arguments passed to the
    constructor method serves as pointer to dashboard number. Server addresses
    are set based on the name of the EFD instance passed to the constructor.

    The expected usage is

    Parameters
    ----------
    efd : `str`
        Name of the EFD connection. Either summit_efd or usfd_efd.

    kwargs : `dict[str, int]`
        Dashboard number for the EFD/Chronograf instance specified in key.
    """

    def __init__(self, efd: str, **kwargs: int):
        self.efd = efd
        try:
            self.dashboard = kwargs[efd]
        except KeyError:
            raise RuntimeError(f"Unknow dashboard number for EFD {efd}.")

    def hostname(self) -> str:
        if self.efd == "summit_efd":
            return "summit-lsp.lsst.codes"
        if self.efd == "usdf_efd":
            return "usdf-rsp.slac.stanford.edu"
        raise RuntimeError(f"Unknown EFD name: {self.efd}.")

    def url(self, lower: Time, upper: Time, **kwargs: str | int | float) -> str:
        params = {
            "refresh": "Paused",
            "lower": lower.isot + "Z",
            "upper": upper.isot + "Z",
        } | dict([(f"tempVars[{key}]", value) for key, value in kwargs.items()])

        return urlunparse(
            (
                "https",
                self.hostname(),
                f"/chronograf/sources/1/dashboards/{self.dashboard}",
                "",
                urlencode(params),
                "",
            )
        )


class M1M3ForceActuatorForces(DashboardURL):
    """Generator for the Force Actuator Forces URLs.

    Parameters
    ----------
    efd : `str`
        Name of the EFD instance.
    """

    def __init__(self, efd: str):
        super().__init__(efd, summit_efd=199, usdf_efd=61)

    def fa_url(self, lower: Time, upper: Time, fa: ForceActuatorData) -> str:
        """Generate URL for given actuator, linking to Chronograf with plots
        containing forces and errors.

        Parameters
        ----------
        lower : `Time`
            Start time.
        upper : `Time`
            End time.
        fa : `ForceActuatorData`
            Force actuator to plot.

        Returns
        -------
        url : `str`
            URL to Chronograf site containing given Force Actuator Forces
            plots.
        """

        def test_index(index: int | None) -> int:
            return 0 if index is None else fa.actuator_id

        return self.url(
            lower,
            upper,
            x_index=test_index(fa.x_index),
            y_index=test_index(fa.y_index),
            s_index=test_index(fa.s_index),
            z_index=fa.actuator_id,
        )


class M1M3FCUStats(DashboardURL):
    def __init__(self, efd: str):
        super().__init__(efd, summit_efd=390, usdf_efd=125)
