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

import bisect
import numpy as np

# X position of electrodes
# We need to know the axial (X) positions of the electrodes so that we can figure out
# which electrodes to use for generating potentials at a given location.

# From looking at the Sandia 3D model, each inner electrode is 70um wide in X direction
# (67um + 3um gap) and each outer quantum is 2x this (140um total).
# All the odd electrode are always located at the same X position as
# the electrode numbered one less than it.

# In unit of 70um and showing only the even electrodes, the order/positions
# of the electrodes are,

# Outer: |               46.5(O0)            |22(Q44-64)|          14.5(O0)         |
# Inner: |2(gap)|10(GND)|5(L0-8)|30(S10-0 x 5)|22(Q0-42) |8(S0-10,0-3)|6(GND)|2(gap)|

# where the number outside the parenthesis is the width in unit of 70um
# and the parenthesis marks the corresponding (even) electrode.
# The origin is located in the middle of the quantum region (11 from left and right).
# This distribution is cross-checked with the potential data
# by setting two pairs of electrode to opposite values and measuring the position
# of the zero crossing.

class ElectrodePosition:
    def __init__(self, name, left, right, up):
        self.name = name
        self.left = left
        self.right = right
        self.up = up
    def distance(self, x):
        if x < self.left:
            return self.left - x
        elif x > self.right:
            return x - self.right
        else:
            return 0.0

# For Phoenix and Peregrine only since I'm way too lazy to do this for HOA
outer_positions_px = []
inner_positions_px = []

def _populate_positions_px():
    begin_gnd = 12
    nL = 5
    nS = 6
    S_rep1 = 5
    nQ = 22
    S_rep2 = 1
    end_gnd = 8

    unit_um = 70

    assert nQ % 2 == 0
    nQ_outer = nQ // 2
    left_edge = -(begin_gnd + nL + nS * S_rep1 + nQ // 2)

    pos_inner = left_edge + begin_gnd
    pos_outer = left_edge

    # Loading
    for i in range(nL):
        inner_positions_px.append(ElectrodePosition(f"L{i * 2}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um, True))
        inner_positions_px.append(ElectrodePosition(f"L{i * 2 + 1}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um, False))
        pos_inner += 1

    # Transition 1
    for j in range(S_rep1):
        for i in range(nS - 1, -1, -1):
            inner_positions_px.append(ElectrodePosition(f"S{i * 2}",
                                                        pos_inner * unit_um,
                                                        (pos_inner + 1) * unit_um,
                                                        True))
            inner_positions_px.append(ElectrodePosition(f"S{i * 2 + 1}",
                                                        pos_inner * unit_um,
                                                        (pos_inner + 1) * unit_um,
                                                        False))
            pos_inner += 1

    # Outer 1
    outer_positions_px.append(ElectrodePosition("O0", pos_outer * unit_um,
                                                (pos_inner - 0.5) * unit_um, True))
    outer_positions_px.append(ElectrodePosition("O1", pos_outer * unit_um,
                                                (pos_inner - 0.5) * unit_um, False))
    pos_outer = pos_inner - 0.5

    # Quantum inner
    for i in range(nQ):
        inner_positions_px.append(ElectrodePosition(f"Q{i * 2}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um, True))
        inner_positions_px.append(ElectrodePosition(f"Q{i * 2 + 1}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um, False))
        pos_inner += 1

    # Quantum outer
    for i in range(nQ_outer):
        i += nQ
        outer_positions_px.append(ElectrodePosition(f"Q{i * 2}", pos_outer * unit_um,
                                                    (pos_outer + 2) * unit_um, True))
        outer_positions_px.append(ElectrodePosition(f"Q{i * 2 + 1}",
                                                    pos_outer * unit_um,
                                                    (pos_outer + 2) * unit_um, False))
        pos_outer += 2
    assert pos_inner - 0.5 == pos_outer

    # Transition 2
    for j in range(S_rep2):
        for i in range(nS):
            inner_positions_px.append(ElectrodePosition(f"S{i * 2}",
                                                        pos_inner * unit_um,
                                                        (pos_inner + 1) * unit_um,
                                                        True))
            inner_positions_px.append(ElectrodePosition(f"S{i * 2 + 1}",
                                                        pos_inner * unit_um,
                                                        (pos_inner + 1) * unit_um,
                                                        False))
            pos_inner += 1

    # S0-S3 appeared again at the end (shouldn't really matter......)
    for i in range(2):
        inner_positions_px.append(ElectrodePosition(f"S{i * 2}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um,
                                                    True))
        inner_positions_px.append(ElectrodePosition(f"S{i * 2 + 1}",
                                                    pos_inner * unit_um,
                                                    (pos_inner + 1) * unit_um,
                                                    False))
        pos_inner += 1

    # Outer 2
    outer_positions_px.append(ElectrodePosition("O0", pos_outer * unit_um,
                                                (pos_inner + end_gnd) * unit_um, True))
    outer_positions_px.append(ElectrodePosition("O1", pos_outer * unit_um,
                                                (pos_inner + end_gnd) * unit_um, False))

_populate_positions_px()

# We can avoid computing this on 3.10 using the `key=` argument
# for bisect.bisect
inner_position_right = [p.right for p in inner_positions_px]
outer_position_right = [p.right for p in outer_positions_px]

class ElectrodeSearchState:
    def __init__(self, pos):
        self.pos = pos
        self.inner_candidates = []
        self.outer_candidates = []

        self.inner_idx2 = bisect.bisect_right(inner_position_right, pos)
        self.outer_idx2 = bisect.bisect_right(outer_position_right, pos)
        self.inner_idx1 = self.inner_idx2 - 1
        self.outer_idx1 = self.outer_idx2 - 1

    def _find_next_distance(self):
        # First find the closest distance
        dist = np.inf
        if self.inner_idx2 < len(inner_positions_px):
            dist = min(dist, inner_positions_px[self.inner_idx2].distance(self.pos))
        if self.inner_idx1 >= 0:
            dist = min(dist, inner_positions_px[self.inner_idx1].distance(self.pos))
        if self.outer_idx2 < len(outer_positions_px):
            dist = min(dist, outer_positions_px[self.outer_idx2].distance(self.pos))
        if self.outer_idx1 >= 0:
            dist = min(dist, outer_positions_px[self.outer_idx1].distance(self.pos))
        if not np.isfinite(dist):
            return dist
        self.inner_candidates.clear()
        self.outer_candidates.clear()
        while self.inner_idx2 < len(inner_positions_px):
            epos = inner_positions_px[self.inner_idx2]
            if epos.distance(self.pos) > dist:
                break
            self.inner_candidates.append(epos)
            self.inner_idx2 += 1
        while self.inner_idx1 >= 0:
            epos = inner_positions_px[self.inner_idx1]
            if epos.distance(self.pos) > dist:
                break
            self.inner_candidates.append(epos)
            self.inner_idx1 -= 1
        while self.outer_idx2 < len(outer_positions_px):
            epos = outer_positions_px[self.outer_idx2]
            if epos.distance(self.pos) > dist:
                break
            self.outer_candidates.append(epos)
            self.outer_idx2 += 1
        while self.outer_idx1 >= 0:
            epos = outer_positions_px[self.outer_idx1]
            if epos.distance(self.pos) > dist:
                break
            self.outer_candidates.append(epos)
            self.outer_idx1 -= 1
        assert self.inner_candidates or self.outer_candidates
        return dist

def find_electrodes(electrode_index, pos, min_num=0, min_dist=0):
    """
    Find at least `min_num` electrodes that are the closest in axial (X) position
    to `pos` (in um). All electrodes within `min_dist` will also be included.

    `electrode_index` is a map from electrode name to a unique ID.
    ID 0 will be ignored (assumed to be ground)
    electrodes with the same ID are assumed to be shorted together
    and therefore will be treated as the same one.
    """
    res = set()

    search_state = ElectrodeSearchState(pos)
    dist_satisfied = False
    num_satisfied = False

    while True:
        num_satisfied = min_num <= len(res)
        if num_satisfied and dist_satisfied:
            return res

        dist = search_state._find_next_distance()
        if not np.isfinite(dist):
            if num_satisfied:
                return res
            raise ValueError("Unable to find enough terms")
        dist_satisfied = dist >= min_dist

        for p in search_state.inner_candidates:
            id = electrode_index[p.name]
            # Ground
            if id == 0:
                continue
            res.add(id)

        for p in search_state.outer_candidates:
            id = electrode_index[p.name]
            # Ground
            if id == 0:
                continue
            res.add(id)
