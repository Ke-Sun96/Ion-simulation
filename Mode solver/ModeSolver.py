##################################
# Author: Laird Egan
# Email: laird.egan@gmail.com
##################################
# Revised by: Ke Sun
# Email: ke.sun621@outlook.com


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy.linalg as lin
import math

# def modes(posi, beta):
#     n = len(posi)
#     A = np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 A[i,j] = beta**2 - np.sum(np.abs(posi[j]-posi[:j])**(-3)) - np.sum(np.abs(posi[j]-posi[j+1:])**(-3))
#             else:
#                 A[i,j] = 1. / np.abs(posi[i]-posi[j]) ** 3
#     w, v = np.linalg.eig(A)
#     idx = w.argsort() 
#     v = v[:,idx]
#     return w, v

def modes(posi, beta):
    b = 0.01
    n = len(posi)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i,j] = beta**2 - np.sum(np.abs(posi[j]-posi[:j])**(-3)) - np.sum(np.abs(posi[j]-posi[j+1:])**(-3))+beta**2*6*b*posi[j]
            else:
                A[i,j] = 1. / np.abs(posi[i]-posi[j]) ** 3
    w, v = np.linalg.eig(A)
    idx = w.argsort() 
    v = v[:,idx]
    return w, v

def get_normal_coeffs(posi, beta=11.37):
    _, eigvecs = modes(posi, 1)
    return eigvecs.T


class TrapModel:
    # This class calculates a trap model that is a required argument to calculate two qubit gates (in gate_solver.py)
    # The potential can be specified to be quadratic, quartic, or numeric (TO-DO)
    #
    # TO-DO: It would also be nice to be able to specify a constant ion spacing and have the
    #  calculator figure out the potential
    #
    # Note: For experimental reason, inputs and output frequencies of this class are Hz, NOT angular. (Compare with gate calculations)
    #

    CONST_E = 1.60217662e-19  # electron charge in Coulomb
    CONST_MP = 1.6605390e-27  # atomic mass unit in kg
    CONST_M = CONST_MP * 171.0  # Yb ion mass in kg
    CONST_EPS = 8.854187818e-12  # vacuum permittivity in F / m
    CONST_COU = CONST_E**2 / (4 * math.pi * CONST_EPS)  # Coulombs Constant e^2/(4*pi*eps)

    def __init__(self, n_ions, com_radial, com_axial, potShape, q3 = 1, q4 = 1, source='model', real_units=False):
        self.n_ions = n_ions
        self.pos = np.zeros((n_ions, 1))  # Ion Positions
        self.f_ax_com = com_axial  #COM axial frequency in Hz or in normalized units of COM Radial
        self.f_rad_com = com_radial #COM radial frequency in Hz or in normalized units of COM Radial (i.e. = 1)
        self.f_modes = np.zeros((n_ions, 1))  # Transverse Normal Mode Frequencies
        self.mode_p = np.zeros((n_ions, n_ions))  # Mode Participation
        self.V = []
        self.q3 = q3
        self.q4 = q4
        self.define_potential(v_type=potShape)  # Trap potential model

        self.real_units = real_units
        if source == 'model':
            pass
        elif source == 'data':
            pass


    def define_potential(self, v_type='quartic', fname=''):
        self.V = TrapPotential(v_type, self.q3, self.q4, fname)


    def get_measured_modes(self):
        # For further extensibility. We might want to directly measure the secular and then compute
        # the mode participation to feed into the gate solver. This function could either be used to read a database
        # or file with the value or directly run an experiment through ARTIQ
        pass


    def calc_ion_pos(self):
        x0 = np.linspace(-(self.n_ions-1)/2, (self.n_ions-1)/2, self.n_ions)
        self.V.getVtot(x0)
        options = {'maxiter': 1000, 'disp': False}
        result = op.minimize(self.V.getVtot, x0, jac=self.V.getVjac, hess=self.V.getVhess,
                             method='trust-ncg', options=options)
        x = result["x"]

        if self.real_units:
            ls = (self.CONST_COU / (self.CONST_M * (2 * math.pi * self.f_ax_com) ** 2)) ** (1 / 3)
        else:
            ls = 1

        self.pos = x * ls
        self.ls = ls
        return self.pos

    def calc_modes(self):
        xii = np.reshape(np.tile(self.pos, self.n_ions), (self.n_ions, self.n_ions))
        xjj = xii.T
        xij = np.abs(xii - xjj)  # xi - xj for each ion in matrix form
        np.seterr(divide='ignore')  # Ignore divide by zero on diagonal
        R_inv = 1 / xij  # 1/Rij distance matrix
        np.fill_diagonal(R_inv, 0)  # Set Diagonal Terms to 0

        if self.real_units:
#             Knm = self.CONST_COU*np.power(R_inv, 3) + np.identity(self.n_ions)*(
#                 self.CONST_M * (2*math.pi*self.f_rad_com)**2 - self.CONST_COU*np.sum(np.power(R_inv, 3), axis=0))
            Knm = self.CONST_COU*np.power(R_inv, 3) + np.identity(self.n_ions)*(
                self.CONST_M * (2*math.pi*self.f_rad_com)**2*(1+6*self.q3*self.pos/self.ls) - self.CONST_COU*np.sum(np.power(R_inv, 3), axis=0))
        else:
#             Knm = np.power(R_inv, 3) + np.identity(self.n_ions)*(
#                 (self.f_rad_com/self.f_ax_com)**2 - np.sum(np.power(R_inv, 3), axis=0))
            Knm = np.power(R_inv, 3) + np.identity(self.n_ions)*(
                (self.f_rad_com/self.f_ax_com)**2*(1+6*self.q3*self.pos) - np.sum(np.power(R_inv, 3), axis=0))

        eigenval, eigenvec = lin.eig(Knm)
        ind = np.argsort(eigenval, kind='mergesort')

        if np.any(eigenval < 0):
            print("Knm is not positive definite. Some of the modes are not stable solutions. " +
                  "Try decreasing the axial confinement")
            return
        elif self.real_units:
            self.f_modes = np.sqrt(eigenval[ind]/self.CONST_M)/2/math.pi
            self.mode_p = eigenvec[:, ind]
        else:
            self.f_modes = np.sqrt(eigenval[ind])
            self.f_modes /= np.max(self.f_modes)

        self.mode_p = eigenvec[:, ind]
        self.mode_p[np.abs(self.mode_p) < 1e-10] = 0
        return self.f_modes


    def plot_modes(self):
#         fig, (ax0, ax1) = plt.subplots(2, 1)
        fig, ax0 = plt.subplots(1, 1)
        if self.real_units:
            plot_wz = self.f_modes/1e6
            ax0.set_xlabel("Normal Mode Freq (MHz)")
            ax0.set_title("Normal Mode Spectrum")
        else:
            plot_wz = self.f_modes
            ax0.set_xlabel("Normal Mode Frequency (in units of COM)")
            ax0.set_title("Normal Mode Spectrum")
        
        ax0.bar(plot_wz, np.ones(self.n_ions), width=0.001)
        ax0.get_yaxis().set_visible(False)
        ax0.grid(True)
        
#         cax = ax1.matshow(self.mode_p, aspect=0.3, cmap="bwr")
#         ax1.set_xlabel("Normal Mode #")
#         ax1.set_ylabel("Ion #")
#         fig.colorbar(cax, ax=ax1)

#         fig.tight_layout()
#         plt.show()
        
        ion_ls = np.linspace(1,self.n_ions, self.n_ions)
        normal_coeffs = [coeffs for coeffs in self.mode_p.T]
        plt.figure(figsize = (20,4))
        for i in range(len(normal_coeffs)):
            plt.subplot(1,self.n_ions,i+1)
            plt.bar(ion_ls, abs(normal_coeffs[i]))
            plt.xlabel('Ion number')
            plt.title('Mode '+str(i+1)+' (Abs)')
            plt.ylim(0,0.8)
        plt.figure(figsize = (20,4))
        for i in range(len(normal_coeffs)):
            plt.subplot(1,self.n_ions,i+1)
#             plt.bar(ion_ls, normal_coeffs[i])
            plt.plot(ion_ls, normal_coeffs[i],'.-')
            plt.xlabel('Ion number')
            plt.title('Mode '+str(i+1))
            plt.ylim(-0.8,0.8)
        return normal_coeffs
    def plot_pos(self):

        fig, (ax0, ax1) = plt.subplots(2, 1)
        if self.real_units:
            plot_pos = self.pos*1e6
            ax0.set_xlabel("Ion Positions (um)")
            ax0.set_title("{0} Ions,  Mean Spacing ="
                          " {1} um".format(self.n_ions, np.round(np.mean(np.diff(plot_pos)) * 100) / 100))
        else:
            plot_pos = self.pos
            ax0.set_xlabel("Ion Positions (arb.)")
            ax0.set_title("{0} Ions.".format(self.n_ions))

        dx = np.diff(plot_pos)
        pctdx = 100*(dx-np.mean(dx))/np.mean(dx)

        ax0.plot(plot_pos, np.zeros([self.n_ions, 1]), 'bo')
        ax0.get_yaxis().set_visible(False)
        ax0.set_xlim(-(max(plot_pos)+np.mean(dx)), (max(plot_pos)+np.mean(dx)))
        ax0.grid(True)

        ax1.bar(np.linspace(-self.n_ions/2+1, self.n_ions/2-1, self.n_ions-1), pctdx)
        ax1.set_ylabel("% Deviation from Equal Spacing")
        ax1.get_xaxis().set_visible(False)

        fig.tight_layout()
        plt.show()


class TrapPotential:

    def __init__(self, v_type='quartic', q3 = 1, q4=1, fname=''):
        self.v_type = v_type
        self.Vtot = np.array([])
        self.Vjac = np.array([])
        self.Vhess = np.array([])

        if self.v_type == 'quadratic':
            self.q3 = 0
            self.q4 = 0
        elif self.v_type == 'cubic':
            self.q3 = q3
            self.q4 = 0
        elif self.v_type == 'quartic':
            self.q3 = 0
            self.q4 = q4
        elif self.v_type == 'solution':
            self.filename = fname


    def getVtot(self, x):
        N = len(x)
        xii = np.reshape(np.tile(x, N), (N, N))
        xjj = xii.T
        xij = xii - xjj  # xi - xj for each ion in matrix form
        sign_xij = np.sign(xij)
        np.seterr(divide='ignore')  # Ignore divide by zero on diagonal
        R_inv = 1 / np.abs(xij)  # 1/Rij distance matrix
        np.fill_diagonal(R_inv, 0)  # Set Diagonal Terms to 0
        V1 = np.sum(R_inv) / 2  # Couloumb Interaction Potential
        
        if self.v_type == 'quadratic':
            V2 = np.sum(np.power(x,2))/2 # Static Trap potential
            Vtot = V1+V2
            Vjac = x - np.sum(sign_xij * np.power(R_inv, 2), axis=0) 
            Vhess = -2*np.power(R_inv,3)
            np.fill_diagonal(Vhess, 1 + np.sum(-1*Vhess,axis=1))
            self.Vtot = Vtot
            self.Vjac = Vjac
            self.Vhess = Vhess
        
        if self.v_type == 'cubic':
            V2 = np.sum(np.power(x,2))/2 + self.q3*np.sum(np.power(x,3)) # Static Trap potential
            Vtot = V1+V2
            Vjac = x + 3 * self.q3 * np.power(x, 2) - np.sum(sign_xij * np.power(R_inv, 2), axis=0)
            Vhess = -2*np.power(R_inv,3)
            np.fill_diagonal(Vhess, 1 + 3*2*self.q3*np.power(x,1) + np.sum(-1*Vhess,axis=1))
            self.Vtot=Vtot
            self.Vjac = Vjac
            self.Vhess = Vhess
            
        if self.v_type == 'quartic':
            V2 = -np.sum(np.power(x,2))/2 + self.q4*np.sum(np.power(x,4))/4 # Static Trap potential
            Vtot = V1+V2
            Vjac = -x + self.q4 * np.power(x, 3) - np.sum(sign_xij * np.power(R_inv, 2), axis=0)
            Vhess = -2*np.power(R_inv,3)
            np.fill_diagonal(Vhess, -1 + 3*self.q4*np.power(x,2) + np.sum(-1*Vhess,axis=1))
            self.Vtot=Vtot
            self.Vjac = Vjac
            self.Vhess = Vhess

        elif self.v_type == 'solution':
            # For further extensibility we should allow a voltage solution calculated by Sandia or otherwise
            # Load solution file here and evaluate at x locations. Numerically calculate Jacobian/Hessian.
            # Will need to find a way to deal with units because position is arbitrary here
            self.Vtot= 0
            self.Vjac = 0
            self.Vhess = 0

        return Vtot


    def getVjac(self, x):
        Vjac = self.Vjac
        return Vjac


    def getVhess(self, x):
        Vhess = self.Vhess
        return Vhess
