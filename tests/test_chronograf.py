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

from astropy.time import Time
import unittest

from lsst.ts.m1m3.utils.chronograf import M1M3_FA

class TestFAs(unittest.IsolatedAsyncioTestCase):
    def test_url(self) -> None:
        summit = M1M3_FA("summit_efd")
        assert summit.url(Time("2025-05-05T21:13:18.23"), Time("2025-05-05T21:13:17.456"), xIndex=101) == "https://summit-lsp.lsst.codes/chronograf/sources/1/dashboards/199?refresh=Paused&lower=2025-05-05T21%3A13%3A18.230Z&upper=2025-05-05T21%3A13%3A17.456Z&tempVars%5BxIndex%5D=101"
