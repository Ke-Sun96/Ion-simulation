#!/usr/bin/python

# Copyright (c) 2022 - 2022 Yichao Yu <yyc1992@gmail.com>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy import optimize

def _optimize_minmax(A, y):
    """
    Find the `x` that satisfies `A @ x = y` while having the smallest maximum element.
    """
    x0 = np.linalg.lstsq(A, y, rcond=None)[0]
    ny, nx = A.shape
    nt = nx - ny
    if nt <= 0:
        return x0
    # With the A @ x = y constraints,
    # the degrees of freedom left in x are the ones that satisfies A * x = 0
    # In another word, these are the x's that are orthogonal to all rows of A.
    # We can find the basis set that spans such space using QR decomposition.
    B = np.linalg.qr(A.T, mode='complete')[0][:, ny:]

    # Now our job is to find `t` that gives the smallest maximum element in
    # x0 + B @ t.

    # Formally, we have `nt + 1` variables including `nt` elements in `t`
    # and `maxv` variable that we'll use to compute the maximum voltage.
    # The target function is simply to minimize `maxv`.
    C = np.zeros(nt + 1)
    C[nt] = 1
    # We have `2 * nx` constraints that corresponds to
    # `maxv >= x0 + B @ t` and `x0 + B @ t >= -maxv`
    # or equivalently
    # `B @ t - maxv <= -x0` and `-B @ t - maxv <= x0`
    A_ub = np.zeros((nx * 2, nt + 1))
    A_ub[:nx, :nt] = B
    A_ub[nx:, :nt] = -B
    A_ub[:, nt] = -1

    b_ub = np.zeros(nx * 2)
    b_ub[:nx] = -x0
    b_ub[nx:] = x0

    res = optimize.linprog(C, A_ub=A_ub, b_ub=b_ub,
                           bounds=[(None, None) for i in range(1 + nt)])
    return B @ res.x[:nt] + x0

def optimize_minmax(A, y):
    if y.ndim == 1:
        return _optimize_minmax(A, y)
    ny, nx = A.shape
    assert y.shape[0] == ny
    nc = y.shape[1]
    res = np.empty((nx, nc))
    for i in range(nc):
        res[:, i] = _optimize_minmax(A, y[:, i])
    return res
