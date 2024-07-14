"""A FM Molmer Sorensen gate sequence solver

It is a solver of FM MS gate based on P.H. Leung's robust MS gate paper.
It is updated to fit system which has DDS as RF source.

Originally written by Shilin Huang, then Bichen Zhang.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumtrapz
import time
import pylab
from matplotlib import collections
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.optimize
import pickle


def modes(posi, beta):
    n = len(posi)
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = beta**2 - np.sum(np.abs(posi[j]-posi[:j])
                                           ** (-3)) - np.sum(np.abs(posi[j]-posi[j+1:])**(-3))
            else:
                A[i][j] = 1. / np.abs(posi[i]-posi[j]) ** 3
    w, v = LA.eig(A)
    idx = w.argsort()
    v = v[:, idx]
    return v


def force(u):
    n = len(u)
    y = np.zeros(n)
    for i in range(n):
        y[i] = u[i] - np.sum((u[:i]-u[i])**(-2)) + np.sum((u[i+1:]-u[i])**(-2))
    return y


def coupling_strength_table(n, equi=np.array([-1.50775, -0.476161, 0.480104, 1.5118])):
    # equi = scipy.optimize.root(force, np.arange(-np.floor(n/2),np.floor(n/2)+1), method='hybr').x
    # The real ion spaceing
    eta = 0.1
    return eta * modes(equi, 11.37037037)


def trajectory(detuning, gate_time, mode='all'):
    n = len(detuning)
    delta_t = gate_time / n
    theta = np.zeros(n+1)
    theta[1:] = np.cumsum(detuning) * delta_t
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    s0d1 = sin_theta[:-1] / detuning  # sin theta[i] / detuning[i]
    s1d1 = sin_theta[1:] / detuning  # sin theta[i+1] / detuning[i]
    c0d1 = cos_theta[:-1] / detuning  # cos theta[i] / detuning[i]
    c1d1 = cos_theta[1:] / detuning  # cos theta[i+1] / detuning[i]
    s0d2 = s0d1 / detuning       # sin theta[i] / detuning[i]^2
    s1d2 = s1d1 / detuning       # sin theta[i+1] / detuning[i]^2
    c0d2 = c0d1 / detuning       # cos theta[i] / detuning[i]^2
    c1d2 = c1d1 / detuning       # cos theta[i] / detuning[i]^2
    s0d3 = s0d2 / detuning       # sin theta[i] / detuning[i]^3
    s1d3 = s1d2 / detuning       # sin theta[i+1] / detuning[i]^3
    c0d3 = c0d2 / detuning       # cos theta[i] / detuning[i]^3
    c1d3 = c1d2 / detuning       # cos theta[i] / detuning[i]^3

    if mode in ['all', 'center_of_mass']:

        # IIc: Integrate[Integrate[Cos[theta[t'],{t',0,t}], {t,0,gate_time}]
        IIc = np.sum((s1d1 * (n-1-np.arange(n)) - s0d1 *
                      (n-np.arange(n))) * delta_t + c0d2 - c1d2) / gate_time
        # IIs: Integrate[Integrate[Sin[theta[t'],{t',0,t}], {t,0,gate_time}]
        IIs = np.sum((c0d1 * (n-np.arange(n)) - c1d1 *
                      (n-1-np.arange(n))) * delta_t + s0d2 - s1d2) / gate_time

        grad_IIc = np.zeros(n)
        grad_IIc[1:] = delta_t * ((c1d1 * (n-1-np.arange(n)) -
                                   c0d1*(n-np.arange(n))) * delta_t + s1d2 - s0d2)[::-1][:-1]
        grad_IIc = np.cumsum(grad_IIc)[::-1]
        grad_IIc += c1d1 * (n-1-np.arange(n)) * (delta_t**2)
        grad_IIc -= s1d2 * (n-2-np.arange(n)) * delta_t
        grad_IIc += s0d2 * (n-np.arange(n)) * delta_t
        grad_IIc += (c1d3-c0d3)*2

        # gradient of IIs
        grad_IIs = np.zeros(n)
        grad_IIs[1:] = delta_t * ((s1d1*(n-1-np.arange(n))-s0d1 *
                                   (n-np.arange(n))) * delta_t - c1d2 + c0d2)[::-1][:-1]
        grad_IIs = np.cumsum(grad_IIs)[::-1]
        grad_IIs += s1d1 * (n-1-np.arange(n)) * (delta_t**2)
        grad_IIs += c1d2 * (n-2-np.arange(n)) * delta_t
        grad_IIs -= c0d2 * (n-np.arange(n)) * delta_t
        grad_IIs += (s1d3 - s0d3)*2

        # center of mass
        center_of_mass = IIc**2 + IIs**2
        grad_center_of_mass = 2*(IIc*grad_IIc + IIs*grad_IIs) / gate_time

    if mode == 'center_of_mass':
        return center_of_mass, grad_center_of_mass

    if mode in ['all', 'area']:
        # area enclosed by the trajectory
        sum_diff_sd1 = np.cumsum(s1d1 - s0d1)
        sum_diff_cd1 = np.cumsum(c0d1 - c1d1)
        sum_diff_sd1_shift = np.zeros(n)
        sum_diff_sd1_shift[1:] = sum_diff_sd1[:-1]
        sum_diff_cd1_shift = np.zeros(n)
        sum_diff_cd1_shift[1:] = sum_diff_cd1[:-1]

        area = np.sum(detuning**(-1)) * delta_t
        area -= np.dot(c1d1, sum_diff_sd1)
        area -= np.dot(s1d1, sum_diff_cd1)
        area += np.dot(c0d1, sum_diff_sd1_shift)
        area += np.dot(s0d1, sum_diff_cd1_shift)
        area *= 0.5

        # gradient of the area
        grad_area = -detuning**(-2) * delta_t
        grad_area += c1d2 * sum_diff_sd1 + s1d2 * sum_diff_cd1
        grad_area -= c0d2 * sum_diff_sd1_shift + s0d2 * sum_diff_cd1_shift
        grad_area += np.cumsum((s1d1 * sum_diff_sd1)[::-1])[::-1] * delta_t
        grad_area -= np.cumsum((c1d1 * sum_diff_cd1)[::-1])[::-1] * delta_t
        grad_area[:-1] -= np.cumsum((s0d1 * sum_diff_sd1_shift)
                                    [::-1])[::-1][1:] * delta_t
        grad_area[:-1] += np.cumsum((c0d1 * sum_diff_cd1_shift)
                                    [::-1])[::-1][1:] * delta_t

        grad_diff_sd1 = c1d1 * delta_t - s1d2 + s0d2
        grad_diff_cd1 = s1d1 * delta_t + c1d2 - c0d2

        temp = -c1d1
        temp[:-1] += c0d1[1:]
        temp = np.cumsum(temp[::-1])[::-1]
        grad_area += temp * grad_diff_sd1
        grad_area[:-1] += np.cumsum((temp * (c1d1 - c0d1)
                                     * delta_t)[::-1])[::-1][1:]

        temp = -s1d1
        temp[:-1] += s0d1[1:]
        temp = np.cumsum(temp[::-1])[::-1]
        grad_area += temp * grad_diff_cd1
        grad_area[:-1] += np.cumsum((temp * (s1d1 - s0d1)
                                     * delta_t)[::-1])[::-1][1:]
        grad_area *= 0.5

    if mode == 'area':
        return area, grad_area

    return center_of_mass, grad_center_of_mass, area, grad_area


def cost_func(Rabi_freq, pulse, mode_freq, strength, gate_time, with_grad=False):
    m = len(mode_freq)
    pulse = np.append(pulse, np.flip(pulse))
    n = len(pulse)
    total_cm = 0
    total_grad_cm = np.zeros(n)
    total_area = 0
    total_grad_area = np.zeros(n)
    weight = [1]*m
    for i in range(m):
        cm, grad_cm, area, grad_area = trajectory(
            pulse - mode_freq[i], gate_time)
        cm2, grad_cm2 = trajectory(
            pulse - mode_freq[i]-0.5e3*2*np.pi, gate_time, 'center_of_mass')
        cm3, grad_cm3 = trajectory(
            pulse - mode_freq[i]+0.5e3*2*np.pi, gate_time, 'center_of_mass')
        total_cm += (cm+cm2+cm3)*weight[i]
        total_grad_cm += (grad_cm+grad_cm2+grad_cm3)*weight[i]
        total_area += area * strength[i]
        total_grad_area += grad_area * strength[i]

    alpha = 1e14
    beta = 1
    gamma = 20
    cost = alpha * total_cm + beta * \
        np.exp(gamma*(np.pi/4-Rabi_freq**2/2 * total_area))
    grad = alpha * total_grad_cm - beta * \
        np.exp(gamma*(np.pi/4-0.5*total_area*Rabi_freq**2)) * \
        gamma*0.5*Rabi_freq**2*total_grad_area
    grad_1, grad_2 = np.split(grad, 2)
    grad = grad_1 + np.flip(grad_2)

    if with_grad:
        return cost, grad
    return cost


def plot_solution(solution, mode_freq, gate_time):
    plt.rcParams['figure.figsize'] = [3, 3]
    for omega in mode_freq:
        plt.hlines(y=omega/(2*np.pi), xmin=0,
                   xmax=gate_time, color='r', linestyle='-')

    steps = 5000
    x = np.linspace(0, gate_time, steps*len(solution)*2)
    y = np.kron(np.append(solution, np.flip(solution)), np.ones(steps))
    plt.plot(x, y/(2*np.pi))
    plt.show()


def area_total(pulse, mode_freq, strength, gate_time):
    m = len(mode_freq)
    pulse = np.append(pulse, np.flip(pulse))
    n = len(pulse)
    total = 0
    for i in range(m):
        _, _, area, _ = trajectory(pulse - mode_freq[i], gate_time)
        total += area * strength[i]
    return total


def error_rate(pulse_freq, mode_freq, gate_time):
    pulse_freq = np.append(pulse_freq, np.flip(pulse_freq))
    n = len(pulse_freq)
    delta_t = gate_time / n
    err = 0
    for omega in mode_freq:
        detuning = pulse_freq - omega
        theta = np.zeros(n+1)
        theta[1:] = np.cumsum(detuning) * delta_t
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        err += np.sum((sin_theta[1:] - sin_theta[:-1] +
                       cos_theta[:-1] - cos_theta[1:]) / detuning) ** 2
    return err


def optimize(num_slices, mode_freq, gate_time, Rabi_freq, threshold, index1, index2, equi_posi):
    n = len(mode_freq)
    mode_freq.sort()
    strength = coupling_strength_table(n, equi_posi)
    solutions = []
    # for i in range(n-1):
    #    for j in range(i+1,n):
    i = index1
    j = index2
    def func_grad(x): return cost_func(Rabi_freq, x, mode_freq,
                                       strength[i]*strength[j], gate_time, True)
    def func(x): return cost_func(Rabi_freq, x, mode_freq,
                                  strength[i]*strength[j], gate_time)
    best = np.zeros(num_slices) + mode_freq[0] - 1e3
    best_value = func(best)

    for k in range(len(mode_freq)-1):
        init_guess = mode_freq[k] - 1e3
        solution = minimize(func_grad, np.zeros(num_slices) + init_guess,
                            method="BFGS", jac=True,  options={'disp': False, 'maxiter': 1000})
        value = func(solution.x)
        if best_value > value:
            best = solution.x
            best_value = value
            if best_value < threshold:
                break
        init_guess = mode_freq[k] + 1e3
        solution = minimize(func_grad, np.zeros(num_slices) + init_guess,
                            method="BFGS", jac=True,  options={'disp': False, 'maxiter': 1000})
        value = func(solution.x)
        if best_value > value:
            best = solution.x
            best_value = value
            if best_value < threshold:
                break
    # print (func(best))
    solutions.append(best)
    area = area_total(best, mode_freq, strength[i]*strength[j], gate_time)
    Rabi = np.sqrt((np.pi/4 / np.abs(area))) / np.pi/2
    plot_solution(best, mode_freq, gate_time)
    err = error_rate(best, mode_freq, gate_time) * \
        (strength[j][-1] * Rabi * 2*np.pi)**2
#    print(func(best))
#    print('(%d,%d): Carrier Rabi = %.3f Hz' % (i, j, Rabi))
#     print('(%d,%d): Center ion #5 mode Pi time = %.3f us' %
#           (i, j, 500000/((Rabi*np.abs(strength[2, 0])))))
#    print('Error rate = %.5f%%' % (err / 1e-2))
#    print(best/(2*np.pi))
    return solutions, Rabi*0.2*np.pi
