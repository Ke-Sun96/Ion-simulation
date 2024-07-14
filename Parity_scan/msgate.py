"""A modulated Molmer Sorensen gate simulator

It is a model of FM/AM/PM MS gates by solving time evolution of
spins-phonons system.

For developer, redefining Modulated._hamiltonian() funtion gives you
the freedom to tweak params adaptively at each step during the simulation.

Written by Bichen Zhang.
Modified by Ke Sun
"""

import qutip as qt
import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import Utilities.quantum_gates as qg

# Modulated MS gate
 
class ModulatedMSGate(object):
    def __init__(self, n_max=20):
        self.gate_time = 0.50
        self.pulse_seq = -np.ones(2)*2*2*np.pi  # frequency sequence
        self.mode_freq = np.zeros(1)*2*2*np.pi
        self.rabi_freq = 1*2*np.pi
        self.pwrBalance = np.ones([2,2])
        self.ion0 = 0
        self.ion1 = 1
        self.n_max = n_max
        self.posi = np.array([-1., 1.])  # ion position for coulping strength
        self.eta = self._get_coupling()*2/np.sqrt(2)  # lamb-dicke
        self.freq_sol = np.ones(1)*-1  # solution from solver
        self.phase_sol = np.ones(2)*self.gate_time*2*np.pi
        # highly depend on frequency sequence if not using PM gates
        self.phase_list = np.arange(2)*self.gate_time*2*np.pi
        self.amp_list = np.ones(2)

    # set attributions for FM gates
    def set_attr_fm(self,
                    ion0,
                    ion1,
                    gate_time,
                    freq_sol,
                    rabi_freq,
                    mode_freq,
                    posi,
                    pwrBalance = np.ones([2,2])):
        self.ion0 = ion0
        self.ion1 = ion1
        self.gate_time = gate_time
        self.freq_sol = freq_sol
        self.pulse_seq = np.append(freq_sol, np.flip(freq_sol))
        print('Pulse seq: ', self.pulse_seq)
        self.amp_list = np.ones(self.pulse_seq.shape[0])
        self.mode_freq = mode_freq
        self.mode_freq.sort()
        self.rabi_freq = rabi_freq
        self.pwrBalance = pwrBalance
        self.posi = posi
        self.eta = self._get_coupling()
        self.phase_list = self._get_phase_fm(
            self.pulse_seq, self.gate_time/self.pulse_seq.shape[0])


    # set attributions for FMPM gates
    def set_attr_fmpm(self,
                      ion0,
                      ion1,
                      gate_time,
                      freq_sol,
                      phase_sol,
                      rabi_freq,
                      mode_freq,
                      posi,
                      pwrBalance = np.ones([2,2])):
        self.ion0 = ion0
        self.ion1 = ion1
        self.gate_time = gate_time
        self.phase_sol = phase_sol
        self.pulse_seq = freq_sol
        self.mode_freq = mode_freq
        self.rabi_freq = rabi_freq
        self.pwrBalance = pwrBalance
        self.amp_list = np.ones(self.pulse_seq.shape[0])
        self.posi = posi
        self.eta = self._get_coupling()
        self.phase_list = self._get_phase_fm(self.pulse_seq, self.gate_time/self.pulse_seq.shape[0]) +\
            self._get_phase_pm(
                self.phase_sol, self.gate_time/self.pulse_seq.shape[0])

    # calculate phase
    def _get_phase_fm(self, freq_list, delta_t):
        return np.append(0.0, (freq_list.cumsum()[:-1]-freq_list[1:]*np.arange(1, freq_list.shape[0]))*delta_t)

    def _get_phase_pm(self, phase_sol, delta_t):
        # + np.arange(phase_input.shape[0])*delta_t*(self.pulse_seq[0]-mode)
        return phase_sol.cumsum()

    # calculate coupling strength
    def _get_coupling(self, beta=3/0.27):
        posi = self.posi
        n = posi.shape[0]
        A = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i][j] = beta**2 - np.sum(np.abs(posi[j]-posi[:j])
                                               ** (-3)) - np.sum(np.abs(posi[j]-posi[j+1:])**(-3))
                else:
                    A[i][j] = 1. / np.abs(posi[i]-posi[j]) ** 3
        w, v = linalg.eig(A)
        idx = w.argsort()
        v = v[:, idx]
        return v

    def _hamiltonian(self, mode_id, step_id, isReversed=False):

        eta = 1
        ion0 = self.ion0
        ion1 = self.ion1
        pulse = self.pulse_seq
        mode_freq = self.mode_freq
        phase_list = self.phase_list
        b = self.eta
        mode = mode_id
        Nmax = self.n_max

        delta = pulse[step_id] - mode_freq[mode_id]

        delta_r = np.array([-delta, -delta])
        delta_b = np.array([delta, delta])

#         print(delta_r)

        omega_r = np.array([self.rabi_freq, self.rabi_freq]
                           )*self.pwrBalance[0]*self.amp_list[step_id]
        omega_b = np.array([self.rabi_freq, self.rabi_freq]
                           )*self.pwrBalance[1]*self.amp_list[step_id]

#         delta_phi = np.pi if step_id %2 == 0 else 0
        delta_phi = 0

        phi_r = phase_list[step_id] + delta_phi
        phi_b = -phi_r

        H0 = b[ion0, mode]*eta/2*omega_r[0] * \
            np.exp(1j*phi_r)*qt.tensor(qt.sigmap(),
                                       qt.qeye(2), qt.destroy(Nmax))

        def H0_coeff(t, args):
            return np.exp(-1j*delta_r[0]*t)

        H1 = b[ion0, mode]*eta/2*omega_r[0] * \
            np.exp(-1j*phi_r)*qt.tensor(qt.sigmam(),
                                        qt.qeye(2), qt.create(Nmax))

        def H1_coeff(t, args):
            return np.exp(1j*delta_r[0]*t)

        H2 = b[ion0, mode]*eta/2*omega_b[0] * \
            np.exp(1j*phi_b)*qt.tensor(qt.sigmap(),
                                       qt.qeye(2), qt.create(Nmax))

        def H2_coeff(t, args):
            return np.exp(-1j*delta_b[0]*t)

        H3 = b[ion0, mode]*eta/2*omega_b[0] * \
            np.exp(-1j*phi_b)*qt.tensor(qt.sigmam(),
                                        qt.qeye(2), qt.destroy(Nmax))

        def H3_coeff(t, args):
            return np.exp(1j*delta_b[0]*t)
        
        if isReversed:
            #delta_phi = 0#np.random.random()
            phi_r = phase_list[step_id] + delta_phi +np.pi
            phi_b = -phi_r

        H4 = b[ion1, mode]*eta/2*omega_r[1] * \
            np.exp(1j*phi_r)*qt.tensor(qt.qeye(2),
                                       qt.sigmap(), qt.destroy(Nmax))

        def H4_coeff(t, args):
            return np.exp(-1j*delta_r[1]*t)

        H5 = b[ion1, mode]*eta/2*omega_r[1] * \
            np.exp(-1j*phi_r)*qt.tensor(qt.qeye(2),
                                        qt.sigmam(), qt.create(Nmax))

        def H5_coeff(t, args):
            return np.exp(1j*delta_r[1]*t)

        H6 = b[ion1, mode]*eta/2*omega_b[1] * \
            np.exp(1j*phi_b)*qt.tensor(qt.qeye(2),
                                       qt.sigmap(), qt.create(Nmax))

        def H6_coeff(t, args):
            return np.exp(-1j*delta_b[1]*t)

        H7 = b[ion1, mode]*eta/2*omega_b[1] * \
            np.exp(-1j*phi_b)*qt.tensor(qt.qeye(2),
                                        qt.sigmam(), qt.destroy(Nmax))

        def H7_coeff(t, args):
            return np.exp(1j*delta_b[1]*t)

        return [[H0, H0_coeff],
                [H1, H1_coeff],
                [H2, H2_coeff],
                [H3, H3_coeff],
                [H4, H4_coeff],
                [H5, H5_coeff],
                [H6, H6_coeff],
                [H7, H7_coeff]]

    def _get_fidelity(self, s0, nth, Gamma, coherence_t, Nsample, verbose, laser_dp, spin_init = None, isReversed=False, isPlot = False):
        gate_time = self.gate_time
        pulse = self.pulse_seq
        Nmax = self.n_max
        tau = coherence_t

        if not Gamma:
            Gamma = np.zeros(self.mode_freq.shape[0])
        if not spin_init:
            s_init = qt.tensor(qt.ket2dm(
                qt.tensor(qt.fock(2, 0), qt.fock(2, 0))), qt.thermal_dm(Nmax, nth[0]))
        else:
            if spin_init.dims != [[2, 2], [1, 1]]:
                s_init = qt.tensor(spin_init, qt.thermal_dm(Nmax, nth[0]))
            else:
                s_init = qt.tensor(qt.ket2dm(spin_init), qt.thermal_dm(Nmax, nth[0]))
#         s_init = qt.tensor(ms_half,qt.thermal_dm(Nmax,nth[0]))
        t = np.linspace(0, gate_time, Nsample)

        t_step = gate_time/pulse.shape[0]
        t_list = []
        Plot_output = []
        Plot_time = []
        for i in range(pulse.shape[0]):
            t_list.append(np.linspace(i*t_step, (i+1)*t_step, Nsample))
        for i in range(self.mode_freq.shape[0]):
            for j in range(pulse.shape[0]):
                H = self._hamiltonian(mode_id=i, step_id=j,isReversed=isReversed)
                c_ops = [qt.tensor(qt.tensor(qt.qeye(2), qt.qeye(2)), np.sqrt(Gamma[i])*qt.destroy(Nmax)),
                         qt.tensor(qt.tensor(qt.qeye(2), qt.qeye(2)), np.sqrt(Gamma[i])*qt.create(Nmax))]
                if tau != 0:
                    c_ops.append(qt.tensor(qt.qeye(2), qt.qeye(2), np.sqrt(
                        2/tau)*qt.create(Nmax)*qt.destroy(Nmax)))
                if laser_dp != 0 and i==0:
                    c_ops.append(qt.tensor(np.sqrt(1/laser_dp)*qt.sigmaz(), qt.qeye(2), qt.qeye(Nmax)))
                    c_ops.append(qt.tensor(qt.qeye(2), np.sqrt(1/laser_dp)*qt.sigmaz(),qt.qeye(Nmax)))
                output = qt.mesolve(H, s_init, t_list[j], c_ops, [])
                Plot_output.append(output)
                s_init = output.states[-1]
                if verbose:
                    print('.', end='')
            if verbose:
                print('\n', end='')
            if i < self.mode_freq.shape[0] - 1:
                s_init = qt.tensor(output.states[-1].ptrace([0, 1]), qt.thermal_dm(Nmax, nth[i+1]))
            spin = output.states[-1].ptrace([0, 1])
        if not spin_init:
            final = qg.xx(np.pi/4)*qt.tensor(qt.fock(2,0),qt.fock(2,0))*(qg.xx(np.pi/4)*qt.tensor(qt.fock(2,0),qt.fock(2,0))).dag()
        else:
            final = qg.xx(np.pi/4)*qt.tensor(spin_init)*(qg.xx(np.pi/4).dag())
        if qt.fidelity(spin, final)**2 < 0.5:
            if not spin_init:
                final = qg.xx(-np.pi/4)*qt.tensor(qt.fock(2,0),qt.fock(2,0))*(qg.xx(-np.pi/4)*qt.tensor(qt.fock(2,0),qt.fock(2,0))).dag()
            else:
                final = qg.xx(-np.pi/4)*qt.tensor(spin_init)*(qg.xx(-np.pi/4).dag())
                
        if isPlot:
            sz = []
            for out in Plot_output:
                print(out)
            
        return qt.fidelity(spin, final), spin


    def get_fidelity_fm(self, nth=[], Gamma=[], coherence_t=0, laser_dp = 0, Nsample=50, verbose=True, spin_init = None,isReversed=False):
        if not nth:
            nth = np.zeros(self.mode_freq.shape[0])
        s0 = qt.tensor(qt.ket2dm(qt.tensor(qt.fock(2, 0), qt.fock(
            2, 0))), qt.thermal_dm(self.n_max, nth[0]))
        print('coherence_t = '+str(coherence_t)+', Gamma = '+str(Gamma)+', laser_dp = '+str(laser_dp))
        return self._get_fidelity(s0, nth, Gamma, coherence_t, Nsample, verbose, laser_dp, spin_init = spin_init,isReversed=isReversed, isPlot=False)

    def _ps_x(self, state):
        x = 1/np.sqrt(2)*(qt.destroy(self.n_max)+qt.create(self.n_max))
        state_ptr = state.ptrace([2])
        x_expect = np.real((state_ptr*x).tr())
        return x_expect

    def _ps_p(self, state):
        p = 1j/np.sqrt(2)*(qt.create(self.n_max)-qt.destroy(self.n_max))
        state_ptr = state.ptrace([2])
        p_expect = np.real((state_ptr*p).tr())
        return p_expect

    def get_trajectory(self, nth=0, Gamma=0, coherence_t=0, Nsample=100):
        v_phase_plot_x = np.vectorize(self._ps_x)
        v_phase_plot_p = np.vectorize(self._ps_p)
        nth = nth
        Gamma = Gamma
        gate_time = self.gate_time
        pulse = self.pulse_seq
        Nmax = self.n_max
        tau = coherence_t

        eigen_state = (qt.tensor(qt.sigmap(), qt.qeye(2)) - qt.tensor(qt.sigmam(), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.sigmap()) - qt.tensor(qt.qeye(2), qt.sigmam())).eigenstates()

        s_init = qt.tensor(eigen_state[1][3], qt.fock(Nmax, 0))
        t = np.linspace(0, gate_time, Nsample)

        t_step = gate_time/pulse.shape[0]
        t_list = []
        for i in range(pulse.shape[0]):
            t_list.append(np.linspace(i*t_step, (i+1)*t_step, Nsample))

        states_list = []
        figures = []
        for i in range(self.mode_freq.shape[0]):
            figures.append(plt.figure())
            states = []
            for j in range(pulse.shape[0]):
                H = self._hamiltonian(mode_id=i, step_id=j)
                c_ops = [qt.tensor(qt.tensor(qt.qeye(2), qt.qeye(2)), np.sqrt(Gamma*(nth+1))*qt.destroy(Nmax)),
                         qt.tensor(qt.tensor(qt.qeye(2), qt.qeye(2)), np.sqrt(Gamma*nth)*qt.create(Nmax))]
                if tau != 0:
                    c_ops.append(qt.tensor(qt.qeye(2), qt.qeye(2), np.sqrt(
                        2/tau)*qt.create(Nmax)*qt.destroy(Nmax)))
                output = qt.mesolve(H, s_init, t_list[j], c_ops, [])
                states += output.states
                s_init = output.states[-1]
            states_list.append(states)
            s_init = qt.tensor(
                output.states[-1].ptrace([0, 1]), qt.thermal_dm(Nmax, nth))

        ps_x = []
        ps_p = []
        for i in range(self.mode_freq.shape[0]):
            ax = figures[i].add_subplot(111)
            temp_x = []
            temp_p = []
            for j in range(len(states_list[i])):
                # print(states_list[i][j],self._ps_x(states_list[i][j]))
                temp_x.append(self._ps_x(states_list[i][j]))
                temp_p.append(self._ps_p(states_list[i][j]))
            ps_x.append(temp_x)
            ps_p.append(temp_p)
            ax.axis('square')
            ax.set(xlim=(-.2, .2), ylim=(-.2, .2))
            ax.grid(True)
            ax.set(xlabel='x')
            ax.set(ylabel='p')
            ax.plot(ps_x[-1], ps_p[-1], 'o', ls='-',
                    ms=4, mec='r', markevery=[0, -1])
#             ax.set(tight_layout = True)
            print('mode', i, 'completed')
            figures[i].tight_layout()
            #figures[i].savefig('mode_%d.png' % i)
#         return ps_p, ps_x, output.states[-1].ptrace([0,1])

    def scan_rabi(self, rabi_freqs, modulation='fm'):
        res = []
        rabi_origin = self.rabi_freq
        for freq in rabi_freqs:
            self.rabi_freq = freq
            if modulation == 'pm':
                fidelity = self.get_fidelity_pm(verbose=False)[0]
            if modulation == 'fm':
                fidelity = self.get_fidelity_fm(verbose=False)[0]
            res.append(fidelity)
#         plt.plot(rabi_freqs, res)
        table = np.concatenate(
            (rabi_freqs.reshape(-1, 1), np.array(res).reshape(-1, 1)), axis=1)
        table = pd.DataFrame(table)
        self.rabi_freq = rabi_origin
        return table
