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
import struct

from .fitting import PolyFitCache

##
# Electrode names for Phoenix and Peregrine
_raw_electrode_names_px = ["GND", "RF"]
_raw_electrode_index_px = {"GND": 0, "RF": 1}
for i in range(10):
    name = f"L{i}"
    _raw_electrode_index_px[name] = len(_raw_electrode_names_px)
    _raw_electrode_names_px.append(name)
for i in range(2):
    name = f"O{i}"
    _raw_electrode_index_px[name] = len(_raw_electrode_names_px)
    _raw_electrode_names_px.append(name)
for i in range(66):
    name = f"Q{i}"
    _raw_electrode_index_px[name] = len(_raw_electrode_names_px)
    _raw_electrode_names_px.append(name)
for i in range(12):
    name = f"S{i}"
    _raw_electrode_index_px[name] = len(_raw_electrode_names_px)
    _raw_electrode_names_px.append(name)

##
# Electrode names for HOA
_raw_electrode_names_hoa = ["GND", "RF"]
_raw_electrode_index_hoa = {"GND": 0, "RF": 1}
for i in range(8):
    name = f"G{i + 1}"
    _raw_electrode_index_hoa[name] = len(_raw_electrode_names_hoa)
    _raw_electrode_names_hoa.append(name)
for i in range(16):
    name = f"L{i + 1}"
    _raw_electrode_index_hoa[name] = len(_raw_electrode_names_hoa)
    _raw_electrode_names_hoa.append(name)
for i in range(40):
    name = f"Q{i + 1}"
    _raw_electrode_index_hoa[name] = len(_raw_electrode_names_hoa)
    _raw_electrode_names_hoa.append(name)
for i in range(6):
    name = f"T{i + 1}"
    _raw_electrode_index_hoa[name] = len(_raw_electrode_names_hoa)
    _raw_electrode_names_hoa.append(name)
for i in range(24):
    name = f"Y{i + 1}"
    _raw_electrode_index_hoa[name] = len(_raw_electrode_names_hoa)
    _raw_electrode_names_hoa.append(name)

def _raw_electrode_names(trap):
    if trap == "phoenix" or trap == "peregrine":
        return _raw_electrode_names_px
    if trap == "hoa":
        return _raw_electrode_names_hoa
    raise ValueError(f"Unknown trap name {trap}")

def _raw_electrode_index(trap):
    if trap == "phoenix" or trap == "peregrine":
        return _raw_electrode_index_px
    if trap == "hoa":
        return _raw_electrode_index_hoa
    raise ValueError(f"Unknown trap name {trap}")

def _alias_to_names(_aliases, trap):
    """
    Translate an alias map (i.e. to account for electrodes shorting together)
    to a list of electrode names which will be stored in the potential object.

    The keys for the alias map should be the electrodes that are shorted
    to something else and the corresponding values
    should be the ones they got shorted to.
    The order should not make much of a difference except when
    multiple ones are shorted together, in which case the values of the aliases
    should be the same, or when an electrode is effectively shorted to ground,
    in which case the value should be the ground electrode.
    """
    aliases = {}
    raw_electrode_index = _raw_electrode_index(trap)
    for (k, v) in _aliases.items():
        if not isinstance(k, int):
            k = raw_electrode_index[k]
        if not isinstance(v, int):
            v = raw_electrode_index[v]
        aliases[k] = v
    raw_electrode_names = _raw_electrode_names(trap)
    nraw_electrodes = len(raw_electrode_names)
    # This is the mapping between the old electrode index and the new ones.
    id_map = [-1 for i in range(nraw_electrodes)]
    cur_id = 0
    nnew_electrodes = nraw_electrodes - len(aliases)
    electrode_names = [[] for i in range(nnew_electrodes)]
    for i in range(nraw_electrodes):
        if i in aliases:
            continue
        id_map[i] = cur_id
        electrode_names[cur_id].append(raw_electrode_names[i])
        cur_id += 1
    assert nnew_electrodes == cur_id
    for (k, v) in aliases.items():
        # The user should connect directly to the final one
        assert v not in aliases
        new_id = id_map[v]
        assert new_id != -1
        electrode_names[new_id].append(raw_electrode_names[k])
    return electrode_names

def _get_electrode_names(aliases, electrode_names, trap):
    if electrode_names is not None:
        assert aliases is None
        return electrode_names
    if aliases is None:
        return [[name] for name in _raw_electrode_names(trap)]
    return _alias_to_names(aliases, trap)

class RawPotential:
    @classmethod
    def import_v0(cls, filename):
        self = cls()
        with open(filename, mode="rb") as fh:
            fh.read(4) # discard
            self.electrodes = struct.unpack('<I', fh.read(4))[0]
            self.nx = struct.unpack('<I', fh.read(4))[0]
            self.ny = struct.unpack('<I', fh.read(4))[0]
            self.nz = struct.unpack('<I', fh.read(4))[0]
            vsets = struct.unpack('<I', fh.read(4))[0]
            # Use mm instead of m
            self.stride = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # Use mm instead of m
            self.origin = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # I have no idea what's stored in these
            fh.read(4)
            fh.read(4)
            self.electrodemapping = np.fromfile(fh, np.dtype('<I'), self.electrodes)
            data = np.fromfile(fh, np.dtype('<d'))
            if len(data) != self.electrodes * self.nx * self.ny * self.nz:
                raise ValueError("Did not find the right number of samples")
            self.data = np.reshape(data, (self.electrodes, self.nx, self.ny, self.nz))
        return self

    @classmethod
    def import_v1(cls, filename):
        self = cls()
        with open(filename, mode="rb") as fh:
            fh.read(4) # discard
            self.electrodes = struct.unpack('<I', fh.read(4))[0]
            self.nx = struct.unpack('<I', fh.read(4))[0]
            self.ny = struct.unpack('<I', fh.read(4))[0]
            self.nz = struct.unpack('<I', fh.read(4))[0]
            vsets = struct.unpack('<I', fh.read(4))[0]
            xaxis = (1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0])
            yaxis = (1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0])
            # Use mm instead of m
            self.stride = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # Use mm instead of m
            self.origin = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # I have no idea what's stored in these
            fh.read(4)
            fh.read(4)
            self.electrodemapping = np.fromfile(fh, np.dtype('<I'), self.electrodes)
            data = np.fromfile(fh, np.dtype('<d'))
            if len(data) != self.electrodes * self.nx * self.ny * self.nz:
                raise ValueError("Did not find the right number of samples")
            self.data = np.reshape(data, (self.electrodes, self.nx, self.ny, self.nz))
        return self

    @classmethod
    def import_64(cls, filename):
        self = cls()
        with open(filename, mode="rb") as fh:
            fh.read(8) # discard
            self.electrodes = struct.unpack('<Q', fh.read(8))[0]
            self.nx = struct.unpack('<Q', fh.read(8))[0]
            self.ny = struct.unpack('<Q', fh.read(8))[0]
            self.nz = struct.unpack('<Q', fh.read(8))[0]
            vsets = struct.unpack('<Q', fh.read(8))[0]
            xaxis = (1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0])
            yaxis = (1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0],
                     1000 * struct.unpack('<d', fh.read(8))[0])
            # Use mm instead of m
            self.stride = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # Use mm instead of m
            self.origin = (1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0],
                           1000 * struct.unpack('<d', fh.read(8))[0])
            # I have no idea what's stored in these
            fh.read(8)
            fh.read(8)
            electrodemapping = np.fromfile(fh, np.dtype('<Q'), self.electrodes)
            data = np.fromfile(fh, np.dtype('<d'))
            if len(data) != self.electrodes * self.nx * self.ny * self.nz:
                raise ValueError("Did not find the right number of samples")
            self.data = np.reshape(data, (self.electrodes, self.nx, self.ny, self.nz))
        return self

    def x_index_to_axis(self, i):
        return i * self.stride[0] + self.origin[0]
    def y_index_to_axis(self, i):
        return i * self.stride[1] + self.origin[1]
    def z_index_to_axis(self, i):
        return i * self.stride[2] + self.origin[2]

    def x_axis_to_index(self, a):
        return (a - self.origin[0]) / self.stride[0]
    def y_axis_to_index(self, a):
        return (a - self.origin[1]) / self.stride[1]
    def z_axis_to_index(self, a):
        return (a - self.origin[2]) / self.stride[2]

class Potential(RawPotential):
    def __init_alias(self, electrode_names, trap):
        raw_electrode_names = _raw_electrode_names(trap)
        raw_electrode_index = _raw_electrode_index(trap)
        assert self.electrodes == len(raw_electrode_names)
        new_electrodes = len(electrode_names)
        new_data = np.empty((new_electrodes, self.nx, self.ny, self.nz))
        electrode_index = {}
        for i in range(new_electrodes):
            electrodes = electrode_names[i]
            first = True
            for elec in electrodes:
                electrode_index[elec] = i
                raw_idx = raw_electrode_index[elec]
                if first:
                    new_data[i, :, :, :] = self.data[raw_idx, :, :, :]
                    first = False
                else:
                    new_data[i, :, :, :] += self.data[raw_idx, :, :, :]
            assert not first
        self.data = new_data
        self.electrodes = new_electrodes
        self.electrode_index = electrode_index
        self.electrode_names = electrode_names

    @classmethod
    def import_v0(cls, filename, trap="phoenix", aliases=None, electrode_names=None):
        self = super(Potential, cls).import_v0(filename)
        self.__init_alias(_get_electrode_names(aliases, electrode_names, trap), trap)
        return self

    @classmethod
    def import_v1(cls, filename, trap="phoenix", aliases=None, electrode_names=None):
        self = super(Potential, cls).import_v1(filename)
        self.__init_alias(_get_electrode_names(aliases, electrode_names, trap), trap)
        return self

    @classmethod
    def import_64(cls, filename, trap="phoenix", aliases=None, electrode_names=None):
        self = super(Potential, cls).import_64(filename)
        self.__init_alias(_get_electrode_names(aliases, electrode_names, trap), trap)
        return self

    def get_cache(self, fitter):
        return FitCache(fitter, self)

class FitCache:
    def __init__(self, fitter, potential):
        self.fitter = fitter
        self.potential = potential
        self.cache = {}

    def __get_internal(self, ele):
        if not isinstance(ele, int):
            ele = self.potential.electrode_index[ele]
        if ele in self.cache:
            return self.cache[ele]
        res = PolyFitCache(self.fitter, self.potential.data[ele, :, :, :])
        self.cache[ele] = res
        return res

    def get(self, ele, *args, **kwargs):
        fit_cache = self.__get_internal(ele)
        if args or kwargs:
            return fit_cache.get(*args, **kwargs)
        return fit_cache

    def get_single(self, ele, *args, **kwargs):
        fit_cache = self.__get_internal(ele)
        return fit_cache.get_single(*args, **kwargs)

    def gradient(self, ele, *args, **kwargs):
        fit_cache = self.__get_internal(ele)
        return fit_cache.gradient(*args, **kwargs)
