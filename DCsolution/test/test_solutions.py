#!/usr/bin/python

from trap_dc import solutions

import numpy as np
import pytest

def test_find_flat():
    x, y = np.meshgrid(np.arange(10), np.arange(12), indexing='ij')
    for x0 in np.linspace(0, 8, 10):
        for y0 in np.linspace(0, 8, 10):
            for scale in np.array([-1, -0.1, 0.1, 1]):
                data = (x - x0)**2 + (y - y0)**2 * scale
                assert solutions.find_flat_point(data) == pytest.approx([x0, y0])

    z, x, y = np.meshgrid(np.arange(1000), np.arange(10), np.arange(12), indexing='ij')
    def x0_z(z):
        z = (z - 500) / 500
        return z * 0.5
    def y0_z(z):
        z = (z - 500) / 500
        return z**3 * 0.3
    for scale in np.array([-1, -0.1, 0.1, 1]):
        data = (x - x0_z(z))**2 + (y - y0_z(z))**2 * scale
        xy0 = solutions.find_all_flat_points(data)
        for zi in range(z.shape[0]):
            assert xy0[0, zi] == pytest.approx(x0_z(zi), abs=2e-3)
            assert xy0[1, zi] == pytest.approx(y0_z(zi), abs=2e-3)

def test_units():
    assert solutions.V_unit_uV == solutions.V_unit * 1e6
    assert solutions.l_unit_um == solutions.l_unit * 1e6

    assert solutions.V_unit_uV == pytest.approx(525.3, abs=0.2)
    assert solutions.l_unit_um == pytest.approx(2.741, abs=0.002)

def test_center_tracker():
    center_phoenix = solutions.CenterTracker(trap="phoenix")
    assert center_phoenix.get(1000.5) == pytest.approx(center_phoenix.get(1000),
                                                       abs=0.005)
    assert center_phoenix.get(-1) == center_phoenix.get(-1.5)
    assert center_phoenix.get(10000) == center_phoenix.get(20000)

    center_peregrine = solutions.CenterTracker(trap="peregrine")
    assert center_peregrine.get(1000.5) == pytest.approx(center_peregrine.get(1000),
                                                         abs=0.005)
    assert center_peregrine.get(-1) == center_peregrine.get(-1.5)
    assert center_peregrine.get(10000) == center_peregrine.get(20000)

    center_hoa = solutions.CenterTracker(trap="hoa")
    assert center_hoa.get(1000.5) == pytest.approx(center_hoa.get(1000), abs=0.005)
    assert center_hoa.get(-1) == center_hoa.get(-1.5)
    assert center_hoa.get(10000) == center_hoa.get(20000)
