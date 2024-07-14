#!/usr/bin/python

from trap_dc import fitting

import numpy as np
import pytest

def test_linearindex():
    lidx = fitting.LinearIndices((2,))
    assert len(lidx) == 2
    assert lidx[0] == 0
    assert lidx[1] == 1
    assert list(lidx) == [0, 1]
    assert list(reversed(lidx)) == [1, 0]

    lidx = fitting.LinearIndices((2, 4))
    assert len(lidx) == 8
    for i in range(8):
        assert lidx[i] == i
    for i in range(2):
        for j in range(4):
            assert lidx[i, j] == i * 4 + j
    assert list(lidx) == list(range(8))
    assert list(reversed(lidx)) == list(reversed(range(8)))

    lidx = fitting.LinearIndices((2, 3, 4))
    assert len(lidx) == 24
    for i in range(24):
        assert lidx[i] == i
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert lidx[i, j, k] == i * 3 * 4 + j * 4 + k
    assert list(lidx) == list(range(24))
    assert list(reversed(lidx)) == list(reversed(range(24)))

def test_cartesianindex():
    cidx = fitting.CartesianIndices((2,))
    assert len(cidx) == 2
    assert cidx[0] == (0,)
    assert cidx[1] == (1,)
    assert list(cidx) == [(0,), (1,)]
    assert list(reversed(cidx)) == [(1,), (0,)]

    cidx = fitting.CartesianIndices((2, 4))
    assert len(cidx) == 8
    expected = []
    for i in range(2):
        for j in range(4):
            expected.append((i, j))
            assert cidx[i * 4 + j] == (i, j)
            assert cidx[i, j] == (i, j)
    assert list(cidx) == expected
    assert list(reversed(cidx)) == list(reversed(expected))

    cidx = fitting.CartesianIndices((2, 3, 4))
    assert len(cidx) == 24
    expected = []
    for i in range(2):
        for j in range(3):
            for k in range(4):
                expected.append((i, j, k))
                assert cidx[i * 3 * 4 + j * 4 + k] == (i, j, k)
                assert cidx[i, j, k] == (i, j, k)
    assert list(cidx) == expected
    assert list(reversed(cidx)) == list(reversed(expected))

def test_fitresult_math():
    res1 = fitting.PolyFitResult((1,), np.zeros(2))
    assert (res1.coefficient == np.zeros(2)).all()
    res2 = fitting.PolyFitResult((1,), np.array([1, 0]))
    assert (res2.coefficient == [1, 0]).all()
    res3 = fitting.PolyFitResult((1,), np.array([0, 2]))
    assert (res3.coefficient == [0, 2]).all()

    assert +res1 is res1
    assert ((-res1).coefficient == -(res1.coefficient)).all()
    assert ((-res2).coefficient == -(res2.coefficient)).all()
    assert ((-res3).coefficient == -(res3.coefficient)).all()

    assert ((res1 + res2).coefficient == (res2.coefficient)).all()
    assert ((res2 + res3).coefficient == [1, 2]).all()

    assert ((res1 - res2).coefficient == -(res2.coefficient)).all()
    assert ((res3 - res1).coefficient == (res3.coefficient)).all()
    assert ((res2 - res3).coefficient == [1, -2]).all()

    assert ((res2 * 2).coefficient == [2, 0]).all()
    assert ((5 * res3).coefficient == [0, 10]).all()

    assert ((res2 / 2).coefficient == [0.5, 0]).all()

def test_fitresult_eval():
    # 1 + 2x + x^3
    res1 = fitting.PolyFitResult((3,), np.array([1.0, 2, 0, 1]))
    for x in np.arange(-2, 2.1, 0.25):
        assert res1(x) == 1 + 2 * x + x**3
    assert res1[0] == 1
    assert res1[1] == 2
    assert res1[2] == 0
    assert res1[3] == 1
    res1[0] = 1.5
    res1[1] = 0
    res1[2] = -0.5
    res1[3] = 0.25
    for x in np.arange(-2, 2.1, 0.25):
        assert res1(x) == 1.5 - 0.5 * x**2 + 0.25 * x**3

    # x^2 + xy - 3x^2y^2 - y
    res2 = fitting.PolyFitResult((2, 2), np.array([0.0, -1, 0, # y^n
                                                   0, 1, 0, # x * y^n
                                                   1, 0, -3])) # x^2 * y^n
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2(x, y) == x**2 + x * y - 3 * x**2 * y**2 - y
    assert res2[0, 0] == 0
    assert res2[0, 1] == -1
    assert res2[0, 2] == 0
    assert res2[1, 0] == 0
    assert res2[1, 1] == 1
    assert res2[1, 2] == 0
    assert res2[2, 0] == 1
    assert res2[2, 1] == 0
    assert res2[2, 2] == -3

    res2[0, 0] = 1
    res2[0, 1] = 0
    res2[0, 2] = 2
    res2[1, 0] = -1
    res2[1, 1] = 0
    res2[1, 2] = 3
    res2[2, 0] = 0
    res2[2, 1] = 0.5
    res2[2, 2] = 0
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2(x, y) == 1 + 2 * y**2 - x + 3 * x * y**2 + 0.5 * x**2 * y

def test_fitresult_shift():
    res1 = fitting.PolyFitResult((3,), np.array([1.0, -1, 0, 1]))
    for x in np.arange(-2, 2.1, 0.25):
        assert res1(x) == 1 - x + x**3
    res1_2 = res1.shift((1.5,))
    for x in np.arange(-2, 2.1, 0.25):
        assert res1_2(x - 1.5) == 1 - x + x**3

    res2 = fitting.PolyFitResult((2, 2), np.array([0.0, -1, 0, # y^n
                                                   0, 1, 0, # x * y^n
                                                   1, 0, -3])) # x^2 * y^n
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2(x, y) == x**2 + x * y - 3 * x**2 * y**2 - y
    res2_2 = res2.shift((1.5, -0.5))
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2_2(x - 1.5, y + 0.5) == x**2 + x * y - 3 * x**2 * y**2 - y
    res2_3 = res2.shift((1.5, 0))
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2_3(x - 1.5, y) == x**2 + x * y - 3 * x**2 * y**2 - y
    res2_4 = res2_3.shift((1.5, -1.25))
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2_4(x - 3, y + 1.25) == x**2 + x * y - 3 * x**2 * y**2 - y

def test_fitter():
    fitter1 = fitting.PolyFitter((2,))
    x1 = np.array([-1, 0, 1])
    y1 = 1.25 + x1 / 2 + x1**2 * 3
    res1 = fitter1.fit(y1)
    assert res1.coefficient == pytest.approx([1.25, 0.5, 3])

    fitter1 = fitting.PolyFitter((4,), sizes=(11,))
    x1 = np.arange(-5, 6)
    y1 = 1 - x1 * 2 + x1**2 / 3 + 0.4 * x1**4
    res1 = fitter1.fit(y1)
    assert res1.coefficient == pytest.approx([1, -2, 1 / 3, 0, 0.4])

    fitter1 = fitting.PolyFitter((4,), sizes=(11,), center=(2.5,))
    x1 = np.arange(11) - 2.5
    y1 = 1 - x1 * 2 + x1**2 / 3 + 0.4 * x1**4
    res1 = fitter1.fit(y1)
    assert res1.coefficient == pytest.approx([1, -2, 1 / 3, 0, 0.4])

    fitter2 = fitting.PolyFitter((2, 2, 4))
    x2, y2, z2 = np.meshgrid(np.arange(3) - 1, np.arange(3) - 1, np.arange(5) - 2, indexing='ij')
    v2 = -y2**2 - 3 * z2 + y2**2 * z2**3 - 3 * x2**2 * z2**3 + y2**2 * z2**4 - x2**2 * y2**2 * z2**4
    res2 = fitter2.fit(v2)
    for i in range(3):
        for j in range(3):
            for k in range(5):
                c = res2[i, j, k]
                if i == 0 and j == 2 and k == 0:
                    assert c == pytest.approx(-1)
                elif i == 0 and j == 0 and k == 1:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 3:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 0 and k == 3:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 4:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 2 and k == 4:
                    assert c == pytest.approx(-1)
                else:
                    assert c == pytest.approx(0)

    fitter2 = fitting.PolyFitter((2, 2, 4), sizes=(10, 21, 51))
    x2, y2, z2 = np.meshgrid(np.arange(10) - 4.5, np.arange(21) - 10,
                             np.arange(51) - 25, indexing='ij')
    v2 = -y2**2 - 3 * z2 + y2**2 * z2**3 - 3 * x2**2 * z2**3 + y2**2 * z2**4 - x2**2 * y2**2 * z2**4
    res2 = fitter2.fit(v2)
    for i in range(3):
        for j in range(3):
            for k in range(5):
                c = res2[i, j, k]
                if i == 0 and j == 2 and k == 0:
                    assert c == pytest.approx(-1)
                elif i == 0 and j == 0 and k == 1:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 3:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 0 and k == 3:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 4:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 2 and k == 4:
                    assert c == pytest.approx(-1)
                else:
                    assert c == pytest.approx(0, abs=1e-5)

    fitter2 = fitting.PolyFitter((2, 2, 4), sizes=(10, 21, 51), center=(3.4, 12.6, 30))
    x2, y2, z2 = np.meshgrid(np.arange(10) - 3.4, np.arange(21) - 12.6,
                             np.arange(51) - 30, indexing='ij')
    v2 = -y2**2 - 3 * z2 + y2**2 * z2**3 - 3 * x2**2 * z2**3 + y2**2 * z2**4 - x2**2 * y2**2 * z2**4
    res2 = fitter2.fit(v2)
    for i in range(3):
        for j in range(3):
            for k in range(5):
                c = res2[i, j, k]
                if i == 0 and j == 2 and k == 0:
                    assert c == pytest.approx(-1)
                elif i == 0 and j == 0 and k == 1:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 3:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 0 and k == 3:
                    assert c == pytest.approx(-3)
                elif i == 0 and j == 2 and k == 4:
                    assert c == pytest.approx(1)
                elif i == 2 and j == 2 and k == 4:
                    assert c == pytest.approx(-1)
                else:
                    assert c == pytest.approx(0, abs=1e-5)

def test_fit_cache():
    def check_fit(fit_cache, pos, **kwargs):
        fit = fit_cache.get(pos, **kwargs)
        for order in fitting.CartesianIndices(fit_cache.fitter.orders + 1):
            fit_single = fit_cache.get_single(pos, order, **kwargs)
            fit_full = fit[order]
            assert fit_single == pytest.approx(fit_full, abs=1e-5)
        return fit

    fitter1 = fitting.PolyFitter((4,), sizes=(11,))
    x1 = np.arange(101)
    def f1(x):
        return 1 - x - x * 2 + x**2 / 3 + 0.4 * x**4
    v1 = f1(x1)
    fit_cache1 = fitting.PolyFitCache(fitter1, v1)
    def check_fit1(pos, **kwargs):
        fit = check_fit(fit_cache1, (pos,), **kwargs)
        assert fit[4] == pytest.approx(0.4, abs=1e-6)
        for x in np.linspace(-10, 10):
            assert fit(x) == pytest.approx(f1(x + pos), abs=0.1)
    for pos in np.linspace(-20, 120, 20):
        check_fit1(pos)
        for fit_center in np.linspace(-20, 120, 20):
            check_fit1(pos, fit_center=(fit_center,))

    fitter2 = fitting.PolyFitter((4, 5), sizes=(11, 20))
    x2, y2 = np.meshgrid(np.arange(30), np.arange(100), indexing='ij')
    def f2(x, y):
        x = x - 15
        y = y - 50
        return (1 + x + y / 2 + x * 2 + x * y + x**2 / 3 + (x / 5)**2 * (y / 10)**3
                    + (x / 5)**4 * (y / 10)**5)
    v2 = f2(x2, y2)
    fit_cache2 = fitting.PolyFitCache(fitter2, v2)
    def check_fit2(xpos, ypos, **kwargs):
        fit = check_fit(fit_cache2, (xpos, ypos), **kwargs)
        assert fit[4, 5] == pytest.approx(1 / 5**4 / 10**5, abs=1e-8)
        for x in np.linspace(-2, 2, 10):
            for y in np.linspace(-2, 2, 10):
                assert fit(x, y) == pytest.approx(f2(x + xpos, y + ypos),
                                                     abs=1.5, rel=2e-3)
    for xpos in np.linspace(-5, 35, 7):
        for ypos in np.linspace(-10, 110, 7):
            check_fit2(xpos, ypos)
            for xfit_center in np.linspace(-5, 35, 7):
                for yfit_center in np.linspace(-10, 110, 7):
                    check_fit2(xpos, ypos, fit_center=(xfit_center, yfit_center))

def test_gradient():
    # 1 + 2x + x^3
    res1 = fitting.PolyFitResult((3,), np.array([1.0, 2, 0, 1]))
    for x in np.arange(-2, 2.1, 0.25):
        assert res1.gradient(0, (x,)) == 2 + 3 * x**2

    # x^2 + xy - 3x^2y^2 - y
    res2 = fitting.PolyFitResult((2, 2), np.array([0.0, -1, 0, # y^n
                                                   0, 1, 0, # x * y^n
                                                   1, 0, -3])) # x^2 * y^n
    for x in np.arange(-2, 2.1, 0.25):
        for y in np.arange(-2, 2.1, 0.25):
            assert res2.gradient(0, (x, y)) == 2 * x + y - 6 * x * y**2
            assert res2.gradient(1, (x, y)) == x - 6 * x**2 * y - 1

    fitter1 = fitting.PolyFitter((4,), sizes=(11,))
    x1 = np.arange(101)
    def f1(x):
        return 1 - x - x * 2 + x**2 / 3 + 0.4 * x**4
    def f1_dx(x):
        return -1 - 2 + 2 * x / 3 + 0.4 * 4 * x**3
    v1 = f1(x1)
    fit_cache1 = fitting.PolyFitCache(fitter1, v1)
    for pos in np.linspace(-20, 120):
        assert fit_cache1.gradient(0, (pos,)) == pytest.approx(f1_dx(pos))

    fitter2 = fitting.PolyFitter((4, 5), sizes=(11, 20))
    x2, y2 = np.meshgrid(np.arange(30), np.arange(100), indexing='ij')
    def f2(x, y):
        x = x - 15
        y = y - 50
        return (1 + x + y / 2 + x * 2 + x * y + x**2 / 3 + (x / 5)**2 * (y / 10)**3
                    + (x / 5)**4 * (y / 10)**5)
    def f2_dx(x, y):
        x = x - 15
        y = y - 50
        return (1 + 2 + y + 2 * x / 3 + 2 / 5 * (x / 5) * (y / 10)**3
                    + 4 / 5 * (x / 5)**3 * (y / 10)**5)
    def f2_dy(x, y):
        x = x - 15
        y = y - 50
        return (1 / 2 + x + 3 / 10 * (x / 5)**2 * (y / 10)**2
                    + 5 / 10 * (x / 5)**4 * (y / 10)**4)
    v2 = f2(x2, y2)
    fit_cache2 = fitting.PolyFitCache(fitter2, v2)
    for xpos in np.linspace(-5, 35):
        for ypos in np.linspace(-10, 110):
            assert (fit_cache2.gradient(0, (xpos, ypos)) ==
                        pytest.approx(f2_dx(xpos, ypos)))
            assert (fit_cache2.gradient(1, (xpos, ypos)) ==
                        pytest.approx(f2_dy(xpos, ypos)))
