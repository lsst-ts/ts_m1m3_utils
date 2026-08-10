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
    "ThermalGradients",
    "fit_plane_gradients",
    "fit_thermal_gradients",
    "nonstandard_thermocouples",
    "thermocouple_z_position",
]

import dataclasses
from collections.abc import Mapping

import numpy as np

from lsst.ts.xml.enums.MTM1M3TS import AirNozzle
from lsst.ts.xml.tables.m1m3 import AirNozzleTable, ThermocoupleData, ThermocoupleTable

# Air nozzle types that make a cell nonstandard for gradient fits.
NONSTANDARD_NOZZLES = frozenset([AirNozzle.BLOCKED, AirNozzle.SUPER_SHORT, AirNozzle.COVERED])


@dataclasses.dataclass(frozen=True)
class ThermalGradients:
    """Result of plane fits to a single set of M1M3 thermocouple
    temperatures.

    Gradients are in deg C/m (z gradient in deg C per normalized
    back-to-front mirror thickness). Fields of the radial fit that do not
    exist for a 2D fit are NaN.
    """

    intercept: float
    intercept_err: float
    x_gradient: float
    x_gradient_err: float
    y_gradient: float
    y_gradient_err: float
    z_gradient: float
    z_gradient_err: float
    radial_gradient: float
    radial_gradient_err: float
    radial_intercept: float
    radial_intercept_err: float
    radial_z_gradient: float
    radial_z_gradient_err: float


def thermocouple_z_position(name: str) -> float:
    """Return the normalized z position of a thermocouple from its name.

    Back thermocouples (including B1/B2 calibration pairs) are at 0,
    middle at 0.5 and front at 1.

    Parameters
    ----------
    name : `str`
        Thermocouple name, e.g. "MTC001F".
    """
    if name[-1] == "M":
        return 0.5
    if name[-1] == "F":
        return 1
    return 0


def nonstandard_thermocouples() -> list[ThermocoupleData]:
    """Return all thermocouples in cells with nonstandard air nozzle
    configurations.

    Uses the current content of `AirNozzleTable`, so call
    `lsst.ts.xml.tables.m1m3.set_air_nozzles_types_and_orifice_diameters`
    first to get meaningful results.
    """
    ret = []
    for thermocouple in ThermocoupleTable:
        nozzle_status = [s.nozzle for s in AirNozzleTable if s.cell == thermocouple.core_location]
        if len(nozzle_status) > 0 and nozzle_status[0] in NONSTANDARD_NOZZLES:
            ret.append(thermocouple)
    return ret


def fit_plane_gradients(positions: np.ndarray, temperatures: np.ndarray) -> ThermalGradients:
    """Fit Cartesian and radial temperature planes to a single sample.

    Solves the least-squares problems t = a + gx*x + gy*y (+ gz*z)
    and t = ar + gr*r (+ grz*z), with parameter errors estimated from the
    fit residuals. Sensors with non-finite temperatures are ignored.

    Parameters
    ----------
    positions : `np.ndarray`
        Sensor positions, shape (n, 2) for a 2D fit or (n, 3) for a 3D fit.
    temperatures : `np.ndarray`
        Sensor temperatures in deg C, shape (n,).

    Returns
    -------
    `ThermalGradients`
        Fitted gradients; all fields NaN if there are not enough finite
        samples to constrain the fit.
    """
    positions = np.asarray(positions, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)

    use_3d = positions.shape[1] == 3

    valid = np.isfinite(temperatures) & np.all(np.isfinite(positions), axis=1)
    positions = positions[valid]
    temperatures = temperatures[valid]

    n = temperatures.size
    num_parameters = 4 if use_3d else 3
    if n <= num_parameters:
        return ThermalGradients(*([np.nan] * 14))

    x = positions[:, 0]
    y = positions[:, 1]
    r = np.hypot(x, y)

    if use_3d:
        z = positions[:, 2]
        A = np.column_stack([np.ones(n), x, y, z])
        Ar = np.column_stack([np.ones(n), r, z])
    else:
        A = np.column_stack([np.ones(n), x, y])
        Ar = np.column_stack([np.ones(n), r])

    def solve(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dtd = design.T @ design
        beta = np.linalg.lstsq(dtd, design.T @ temperatures, rcond=None)[0]
        residuals = temperatures - design @ beta
        dof = max(n - design.shape[1], 1)
        sigma2 = float(np.sum(residuals**2) / dof)
        cov = sigma2 * np.linalg.pinv(dtd)
        return beta, np.sqrt(np.diag(cov))

    beta, errs = solve(A)
    beta_r, errs_r = solve(Ar)

    return ThermalGradients(
        intercept=beta[0],
        intercept_err=errs[0],
        x_gradient=beta[1],
        x_gradient_err=errs[1],
        y_gradient=beta[2],
        y_gradient_err=errs[2],
        z_gradient=beta[3] if use_3d else np.nan,
        z_gradient_err=errs[3] if use_3d else np.nan,
        radial_gradient=beta_r[1],
        radial_gradient_err=errs_r[1],
        radial_intercept=beta_r[0],
        radial_intercept_err=errs_r[0],
        radial_z_gradient=beta_r[2] if use_3d else np.nan,
        radial_z_gradient_err=errs_r[2] if use_3d else np.nan,
    )


def fit_thermal_gradients(
    temperatures: Mapping[str, float],
    remove_nonstandard_cells: bool = True,
    radius_limit: float | None = None,
) -> ThermalGradients:
    """Fit thermal gradients to one set of per-thermocouple temperatures.

    Parameters
    ----------
    temperatures : `Mapping` [`str`, `float`]
        Temperature in deg C per thermocouple name (e.g. as returned by
        `ThermocoupleCache.valid_set`). Names not in `ThermocoupleTable`
        are ignored.
    remove_nonstandard_cells : `bool`, optional
        If true (the default), exclude thermocouples in cells with
        nonstandard air nozzle configurations - see
        `nonstandard_thermocouples`.
    radius_limit : `float`, optional
        Only use thermocouples within this radius (meters) from the mirror
        center. Defaults to None - use all thermocouples.

    Returns
    -------
    `ThermalGradients`
        Fitted 3D gradients.
    """
    excluded = {tc.name for tc in nonstandard_thermocouples()} if remove_nonstandard_cells else set()

    positions = []
    values = []
    for thermocouple in ThermocoupleTable:
        if thermocouple.name not in temperatures or thermocouple.name in excluded:
            continue
        if (
            radius_limit is not None
            and np.hypot(thermocouple.x_position, thermocouple.y_position) > radius_limit
        ):
            continue
        positions.append(
            (
                thermocouple.x_position,
                thermocouple.y_position,
                thermocouple_z_position(thermocouple.name),
            )
        )
        values.append(temperatures[thermocouple.name])

    if len(positions) == 0:
        return ThermalGradients(*([np.nan] * 14))

    return fit_plane_gradients(np.array(positions), np.array(values))
