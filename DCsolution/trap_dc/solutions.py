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

from . import fitting, mapping, optimizers

import os
import os.path
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import collections
import h5py
import numpy as np
from scipy.optimize import fsolve

def find_flat_point(data, init=None):
    N = data.ndim
    if init is None:
        init = [(s - 1) / 2 for s in data.shape]
    init = np.array(init)
    # 3rd order fit
    fitter = fitting.PolyFitter(tuple(3 for i in range(N)))
    cache = fitting.PolyFitCache(fitter, data)
    def model(x):
        return np.array([cache.gradient(i, x) for i in range(N)])
    return fsolve(model, init)

def find_all_flat_points(all_data, init=None):
    N = all_data.ndim
    npoints = all_data.shape[0]
    all_res = np.empty((N - 1, npoints))
    if init is None:
        init = [(s - 1) / 2 for s in all_data.shape[1:]]
    for i in range(npoints):
        idx_range = (i,) + tuple(slice(None) for i in range(N - 1))
        init = find_flat_point(all_data[idx_range], init=init)
        all_res[:, i] = init
    return all_res

# EURIQA unit:
# Unit such that electric potential that creates 1MHz trapping frequency
# for Yb171 has the form X^2/2,
# and the electric potential between two ions is 1/r.

N_A = 6.02214076e23
m_Yb171 = 170.9363315e-3 / N_A # kg
q_e = 1.60217663e-19 # C
epsilon_0 = 8.8541878128e-12

A = m_Yb171 * (2 * np.pi * 1e6)**2 / q_e
B = q_e / (4 * np.pi * epsilon_0)

V_unit = np.cbrt(A * B**2) # V
l_unit = np.cbrt(B / A) # m

del A
del B

l_unit_um = l_unit * 1e6 # um
V_unit_uV = V_unit * 1e6 # uV

class CenterTracker:
    def __init__(self, trap=None, filename=None):
        if filename is None:
            if trap is None:
                raise ValueError('Must specify either "trap" or "filename"')
            filename = os.path.join(root_path, "data", f"rf_center_{trap}.h5")
        with h5py.File(filename, 'r') as fh:
            self.yz_index = np.array(fh['yz_index'])

    def get(self, xidx):
        # return (y, z)
        nx = self.yz_index.shape[1]
        lb_idx = min(max(int(np.floor(xidx)), 0), nx - 1)
        ub_idx = min(max(int(np.ceil(xidx)), 0), nx - 1)
        y_lb = self.yz_index[0, lb_idx]
        z_lb = self.yz_index[1, lb_idx]
        if lb_idx == ub_idx:
            return y_lb, z_lb
        assert ub_idx == lb_idx + 1
        y_ub = self.yz_index[0, ub_idx]
        z_ub = self.yz_index[1, ub_idx]
        c_ub = xidx - lb_idx
        c_lb = ub_idx - xidx
        return y_lb * c_lb + y_ub * c_ub, z_lb * c_lb + z_ub * c_ub

def load_short_map(fname):
    m = np.loadtxt(fname, dtype=str, delimiter=',')
    res = {}
    for i in range(m.shape[0]):
        res[m[i, 0]] = m[i, 1]
    return res

CompensateTerms1 = collections.namedtuple("CompensateTerms1",
                                          ["dx", "dy", "dz",
                                           "xy", "yz", "zx",
                                           "z2", "x2", "x3", "x4"])

# Terms we care about
# x, y, z, xy, yz, xz, (z^2 - y^2) / 2, (x^2 - (y^2 + z^2) / 2) / 2, x^3 / 3!, x^4 / 4!
# Since we care about the symmetry of the x^2 and z^2 term,
# we actually do need to scale the x, y and z correctly.
# stride should be in um, voltage should be in V
def get_compensate_terms1(res, stride):
    # axis order of fitting result and stride are both is x, y, z
    raw_x = res[1, 0, 0]
    raw_y = res[0, 1, 0]
    raw_z = res[0, 0, 1]

    raw_xy = res[1, 1, 0]
    raw_yz = res[0, 1, 1]
    raw_zx = res[1, 0, 1]

    raw_x2 = res[2, 0, 0]
    raw_y2 = res[0, 2, 0]
    raw_z2 = res[0, 0, 2]

    raw_x3 = res[3, 0, 0]
    raw_x4 = res[4, 0, 0]

    scaled_x = raw_x / stride[0]
    scaled_y = raw_y / stride[1]
    scaled_z = raw_z / stride[2]

    # We need to divide the xy/yz/zx terms by 2 relative to the x2, y2, z2 terms.
    # This makes sure that, e.g., the z^2 term is a direct rotation of the
    # xy/yz/zx terms.
    scaled_xy = raw_xy / stride[0] / stride[1]
    scaled_yz = raw_yz / stride[1] / stride[2]
    scaled_zx = raw_zx / stride[2] / stride[0]

    scaled_x2 = raw_x2 / stride[0]**2 * 2
    scaled_y2 = raw_y2 / stride[1]**2 * 2
    scaled_z2 = raw_z2 / stride[2]**2 * 2

    scaled_x3 = raw_x3 / stride[0]**3 * 6
    scaled_x4 = raw_x4 / stride[0]**4 * 24

    # The two legal quadratic terms are `x^2 - (y^2 + z^2) / 2` and `z^2 - y^2`
    # which are also orthogonal to each other.
    # The orthogonal illegal term is `x^2 + y^2 + z^2`.
    # Here we just need to find the transfermation to go from the taylor expansion
    # basis to the new basis.
    # Since the three terms are orthogonal, we can just compute the dot product
    # with these three terms and apply the correct normalization coefficient.
    xx = (2 * scaled_x2 - scaled_y2 - scaled_z2) / 3
    zz = (scaled_z2 - scaled_y2) / 2

    # Current units are V/um^n
    # Expected units
    # DX/DY/DZ: V/m
    # XY, YZ, ZX, ZZ, XX: 525 uV / (2.74 um)^2
    # X3: 525 uV / (2.74 um)^3
    # X4: 525 uV / (2.74 um)^4
    scale_1 = 1e6
    scale_2 = (l_unit_um**2 / V_unit)
    scale_3 = (l_unit_um**3 / V_unit)
    scale_4 = (l_unit_um**4 / V_unit)
    return CompensateTerms1(scaled_x * scale_1, scaled_y * scale_1, scaled_z * scale_1,
                            scaled_xy * scale_2, scaled_yz * scale_2,
                            scaled_zx * scale_2, zz * scale_2,
                            xx * scale_2, scaled_x3 * scale_3, scaled_x4 * scale_4)

def compensate_fitter1(potential, sizes=(129, 5, 5)):
    fitter = fitting.PolyFitter((4, 2, 2), sizes=sizes)
    return potential.get_cache(fitter)

def get_compensate_coeff1(cache, pos, electrode_min_num=20, electrode_min_dist=350):
    # pos is in xyz index
    x_coord = cache.potential.x_index_to_axis(pos[0]) * 1000
    ele_select = mapping.find_electrodes(cache.potential.electrode_index, x_coord,
                                         min_num=electrode_min_num,
                                         min_dist=electrode_min_dist)
    ele_select = list(ele_select)
    ele_select.sort()
    fits = [cache.get(e, pos) for e in ele_select]

    # Change stride to um in unit
    stride_um = np.array(cache.potential.stride) * 1000
    nfits = len(fits)
    coefficient = np.empty((10, nfits))
    for i in range(nfits):
        coefficient[:, i] = tuple(get_compensate_terms1(fits[i], stride_um))
    return ele_select, coefficient

def solve_compensate1(cache, pos, electrode_min_num=20, electrode_min_dist=350):
    ele_select, coefficient = get_compensate_coeff1(
        cache, pos, electrode_min_num=electrode_min_num,
        electrode_min_dist=electrode_min_dist)
    X = optimizers.optimize_minmax(coefficient, np.eye(10))
    assert X.shape[1] == 10
    return ele_select, CompensateTerms1(X[:, 0], X[:, 1], X[:, 2],
                                        X[:, 3], X[:, 4], X[:, 5],
                                        X[:, 6], X[:, 7], X[:, 8], X[:, 9])

CompensateTerms2 = collections.namedtuple("CompensateTerms2",
                                          ["dx", "dy", "dz",
                                           "xy", "yz", "zx",
                                           "z2", "x2", "x3", "x4", "x2z"])

# Terms we care about on pheonix
# x, y, z, xy, yz, xz, (z^2 - y^2) / 2, (x^2 - (y^2 + z^2) / 2) / 2,
# x^3 / 3!, x^4 / 4!, x^2z / 2
# Compared to HOA, we are able to compensate for x^2z due to the outer electrodes.
# stride should be in um, voltage should be in V
def get_compensate_terms2(res, stride):
    # axis order of fitting result and stride are both is x, y, z
    raw_x = res[1, 0, 0]
    raw_y = res[0, 1, 0]
    raw_z = res[0, 0, 1]

    raw_xy = res[1, 1, 0]
    raw_yz = res[0, 1, 1]
    raw_zx = res[1, 0, 1]

    raw_x2 = res[2, 0, 0]
    raw_y2 = res[0, 2, 0]
    raw_z2 = res[0, 0, 2]

    raw_x3 = res[3, 0, 0]
    raw_x4 = res[4, 0, 0]

    raw_x2z = res[2, 0, 1]

    scaled_x = raw_x / stride[0]
    scaled_y = raw_y / stride[1]
    scaled_z = raw_z / stride[2]

    # We need to divide the xy/yz/zx terms by 2 relative to the x2, y2, z2 terms.
    # This makes sure that, e.g., the z^2 term is a direct rotation of the
    # xy/yz/zx terms.
    scaled_xy = raw_xy / stride[0] / stride[1]
    scaled_yz = raw_yz / stride[1] / stride[2]
    scaled_zx = raw_zx / stride[2] / stride[0]

    scaled_x2 = raw_x2 / stride[0]**2 * 2
    scaled_y2 = raw_y2 / stride[1]**2 * 2
    scaled_z2 = raw_z2 / stride[2]**2 * 2

    scaled_x3 = raw_x3 / stride[0]**3 * 6
    scaled_x4 = raw_x4 / stride[0]**4 * 24

    scaled_x2z = raw_x2z / stride[0]**2 / stride[2] * 2

    # The two legal quadratic terms are `x^2 - (y^2 + z^2) / 2` and `z^2 - y^2`
    # which are also orthogonal to each other.
    # The orthogonal illegal term is `x^2 + y^2 + z^2`.
    # Here we just need to find the transfermation to go from the taylor expansion
    # basis to the new basis.
    # Since the three terms are orthogonal, we can just compute the dot product
    # with these three terms and apply the correct normalization coefficient.
    xx = (2 * scaled_x2 - scaled_y2 - scaled_z2) / 3
    zz = (scaled_z2 - scaled_y2) / 2

    # Current units are V/um^n
    # Expected units
    # DX/DY/DZ: V/m
    # XY, YZ, ZX, ZZ, XX: 525 uV / (2.74 um)^2
    # X3: 525 uV / (2.74 um)^3
    # X4: 525 uV / (2.74 um)^4
    scale_1 = 1e6
    scale_2 = (l_unit_um**2 / V_unit)
    scale_3 = (l_unit_um**3 / V_unit)
    scale_4 = (l_unit_um**4 / V_unit)
    return CompensateTerms2(scaled_x * scale_1, scaled_y * scale_1, scaled_z * scale_1,
                            scaled_xy * scale_2, scaled_yz * scale_2,
                            scaled_zx * scale_2, zz * scale_2,
                            xx * scale_2, scaled_x3 * scale_3, scaled_x4 * scale_4,
                            scaled_x2z * scale_3)

def get_compensate_coeff2(cache, pos, electrode_min_num=20, electrode_min_dist=350):
    # pos is in xyz index
    x_coord = cache.potential.x_index_to_axis(pos[0]) * 1000
    ele_select = mapping.find_electrodes(cache.potential.electrode_index, x_coord,
                                         min_num=electrode_min_num,
                                         min_dist=electrode_min_dist)
    ele_select = list(ele_select)
    ele_select.sort()
    fits = [cache.get(e, pos) for e in ele_select]

    # Change stride to um in unit
    stride_um = np.array(cache.potential.stride) * 1000
    nfits = len(fits)
    coefficient = np.empty((11, nfits))
    for i in range(nfits):
        coefficient[:, i] = tuple(get_compensate_terms2(fits[i], stride_um))
    return ele_select, coefficient

def solve_compensate2(cache, pos, electrode_min_num=20, electrode_min_dist=350):
    ele_select, coefficient = get_compensate_coeff2(
        cache, pos, electrode_min_num=electrode_min_num,
        electrode_min_dist=electrode_min_dist)
    X = optimizers.optimize_minmax(coefficient, np.eye(11))
    assert X.shape[1] == 11
    return ele_select, CompensateTerms2(X[:, 0], X[:, 1], X[:, 2],
                                        X[:, 3], X[:, 4], X[:, 5],
                                        X[:, 6], X[:, 7], X[:, 8], X[:, 9],
                                        X[:, 10])

def compensate_fitter3(potential, sizes=(77, 5, 5)):
    fitter = fitting.PolyFitter((8, 2, 2), sizes=sizes)
    return potential.get_cache(fitter)
