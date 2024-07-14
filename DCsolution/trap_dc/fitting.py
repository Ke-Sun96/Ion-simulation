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

import math
import numpy as np
from scipy.special import binom

if hasattr(math, 'prod'):
    math_prod = math.prod
else:
    def math_prod(a):
        p = 1
        for v in a:
            p = p * v
        return p

def to_tuple(v):
    if isinstance(v, tuple):
        return v
    return (v,)

# Both the cartesian indices and linear indicies are 0-based
# Note that we don't perform most of the boundcheck since I'm too lazy...
def _linear_to_cartesian(sizes, lidx):
    res = ()
    for s in reversed(sizes):
        res = (lidx % s, *res)
        lidx = lidx // s
    return res

def _cartesian_to_linear(sizes, cidx):
    res = 0
    for (s, i) in zip(sizes, cidx):
        res = res * s + i
    return res

class LinearIndices:
    def __init__(self, sizes):
        self.__sizes = sizes

    def __iter__(self):
        return iter(range(len(self)))
    def __reversed__(self):
        return iter(reversed(range(len(self))))

    def __len__(self):
        return math_prod(self.__sizes)
    def __getitem__(self, idx):
        idx = to_tuple(idx)
        if len(idx) == 1:
            return idx[0]
        return _cartesian_to_linear(self.__sizes, idx)

class CartisianIndicesIter:
    """
    Iterate through Cartisian index in the row major order
    """
    def __init__(self, sizes):
        self.__sizes = sizes
        self.__value = np.zeros(len(sizes), dtype='q') # Next value

    def __next__(self):
        if self.__value[0] >= self.__sizes[0]:
            raise StopIteration
        res = tuple(self.__value)
        for i in reversed(range(len(self.__sizes))):
            self.__value[i] += 1
            if self.__value[i] < self.__sizes[i]:
                break
            if i == 0:
                break
            self.__value[i] = 0
        return res

    def __iter__(self):
        return self

class CartesianIndices:
    def __init__(self, sizes):
        self.__sizes = sizes

    def __iter__(self):
        return CartisianIndicesIter(self.__sizes)

    def __len__(self):
        return math_prod(self.__sizes)
    def __getitem__(self, idx):
        idx = to_tuple(idx)
        if len(idx) == 1:
            return _linear_to_cartesian(self.__sizes, idx[0])
        return idx

class PolyFitter:
    # center is the origin of the polynomial in index (0-based)
    def __init__(self, orders, sizes=None, center=None):
        orders = np.array(orders, dtype='q')
        self.orders = orders
        if sizes is None:
            sizes = orders + 1
        else:
            sizes = np.array(sizes, dtype='q')
        self.sizes = sizes
        if center is None:
            center = (sizes - 1) / 2
        else:
            center = np.array(center, dtype='d')

        assert (sizes > orders).all()
        nterms = math_prod(orders + 1)
        npoints = math_prod(sizes)

        self.coefficient = np.empty((npoints, nterms))
        pos_lidxs = LinearIndices(sizes)
        pos_cidxs = CartesianIndices(sizes)
        ord_lidxs = LinearIndices(orders + 1)
        ord_cidxs = CartesianIndices(orders + 1)
        self.scales = np.empty(nterms)
        scale_max = np.maximum((sizes - 1) / 2, 1.0)
        for iorder in ord_lidxs:
            order = np.array(ord_cidxs[iorder])
            self.scales[iorder] = 1 / math_prod(scale_max**order)

        # Index for position
        for ipos in pos_lidxs:
            # Position of the point, with the origin in the middle of the grid.
            pos = np.array(pos_cidxs[ipos]) - center
            # Index for the polynomial order
            for iorder in ord_lidxs:
                order = np.array(ord_cidxs[iorder])
                self.coefficient[ipos, iorder] = (math_prod(pos**order) *
                                                  self.scales[iorder])

    def fit(self, data):
        res = np.linalg.lstsq(self.coefficient,
                              np.reshape(data, -1), rcond=None)[0] * self.scales
        return PolyFitResult(self.orders, res)

def _shifted_term(max_order, term_order, shift):
    return shift**(max_order - term_order) * binom(max_order, term_order)

class PolyFitResult:
    def __init__(self, orders, coefficient):
        self.orders = np.array(orders)
        self.coefficient = coefficient
    def __assert_order(self, v):
        assert (self.orders == v.orders).all()
    def __pos__(self):
        return self
    def __neg__(self):
        return PolyFitResult(self.orders, -self.coefficient)
    def __add__(self, v):
        self.__assert_order(v)
        return PolyFitResult(self.orders, self.coefficient + v.coefficient)
    def __sub__(self, v):
        self.__assert_order(v)
        return PolyFitResult(self.orders, self.coefficient - v.coefficient)

    def __mul__(self, s):
        return PolyFitResult(self.orders, self.coefficient * s)
    def __rmul__(self, s):
        return PolyFitResult(self.orders, s * self.coefficient)

    def __truediv__(self, s):
        return PolyFitResult(self.orders, self.coefficient / s)

    def __call__(self, *pos):
        assert len(pos) == len(self.orders)
        sizes = self.orders + 1
        lindices = LinearIndices(sizes)
        cindices = CartesianIndices(sizes)
        v = 0.0
        pos = np.array(pos)
        for iorder in lindices:
            order = cindices[iorder]
            v += self.coefficient[iorder] * math_prod(pos**order)
        return v

    def __getitem__(self, order):
        order = to_tuple(order)
        return self.coefficient[_cartesian_to_linear(self.orders + 1, order)]

    def __setitem__(self, order, v):
        order = to_tuple(order)
        self.coefficient[_cartesian_to_linear(self.orders + 1, order)] = v

    def _shifted_coefficient(self, shift, order):
        v = 0.0
        sizes = self.orders + 1
        lindices = LinearIndices(sizes)
        cindices = CartesianIndices(sizes)
        for lidx in lindices:
            term_order = np.array(cindices[lidx])
            if not (term_order >= order).all():
                continue
            v += math_prod(_shifted_term(t, o, s) for (t, o, s)
                           in zip(term_order, order, shift)) * self.coefficient[lidx]
        return v

    # shift the solution to get the polynomial representing the same function
    # but with the origin shifted to `shift`.
    # `x` with a shift of `1` becomes `x + 1`.
    def shift(self, shift):
        coefficient = np.empty(self.coefficient.shape)
        sizes = self.orders + 1
        lindices = LinearIndices(sizes)
        cindices = CartesianIndices(sizes)
        for lidx in lindices:
            order = cindices[lidx]
            coefficient[lidx] = self._shifted_coefficient(shift, order)
        return PolyFitResult(self.orders, coefficient)

    def gradient(self, dim, pos):
        order = tuple(int(i == dim) for i in range(len(self.orders)))
        return self._shifted_coefficient(pos, order)

# pos is in index unit
def _best_fit_idx(ntotal, kernel_size, pos):
    # for fit at `index`, the data covered is `index:(index + kernel_size - 1)`
    # with the center at index `index + (kernel_size - 1) / 2`.
    # Therefore, the ideal index to use for `pos` is `pos - (kernel_size - 1) / 2`
    idx = round(pos - (kernel_size - 1) / 2)
    if idx <= 0:
        return 0
    elif idx >= ntotal - kernel_size:
        return ntotal - kernel_size
    return idx

class PolyFitCache:
    def __init__(self, fitter, data):
        self.fitter = fitter
        self.data = data
        self.cache = {}

    def __get_internal(self, idx):
        # idx is the start index
        if idx in self.cache:
            return self.cache[idx]
        data = self.data[tuple(slice(i, i + s) for (i, s)
                         in zip(idx, self.fitter.sizes))]
        res = self.fitter.fit(data)
        self.cache[idx] = res
        return res

    def get(self, pos, fit_center=None):
        pos = np.array(pos)
        if fit_center is None:
            fit_center = pos
        kernel_sizes = np.array(self.fitter.sizes)
        data_sizes = self.data.shape
        idxs = tuple(_best_fit_idx(n, k, p) for (n, k, p)
                     in zip(data_sizes, kernel_sizes, fit_center))
        fit = self.__get_internal(idxs)
        return fit.shift(pos - (kernel_sizes - 1) / 2 - idxs)

    def get_single(self, pos, orders, fit_center=None):
        pos = np.array(pos)
        if fit_center is None:
            fit_center = pos
        kernel_sizes = np.array(self.fitter.sizes)
        data_sizes = self.data.shape
        idxs = tuple(_best_fit_idx(n, k, p) for (n, k, p)
                     in zip(data_sizes, kernel_sizes, fit_center))
        fit = self.__get_internal(idxs)
        return fit._shifted_coefficient(pos - (kernel_sizes - 1) / 2 - idxs, orders)

    def gradient(self, dim, pos):
        order = tuple(int(i == dim) for i in range(len(self.fitter.sizes)))
        return self.get_single(pos, order)