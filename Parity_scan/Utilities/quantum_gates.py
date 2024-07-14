from qutip import *
from math import *

def xx(t):
    matrix = [[cos(t), 0, 0, -1j*sin(t)],
              [0, cos(t), -1j*sin(t), 0],
              [0, -1j*sin(t), cos(t), 0],
              [-1j*sin(t), 0, 0, cos(t)]]
    ans = Qobj(matrix)
    ans.dims = [[2,2],[2,2]]
    return ans

def yy(t):
    matrix = [[cos(t), 0, 0, 1j*sin(t)],
              [0, cos(t), -1j*sin(t), 0],
              [0, -1j*sin(t), cos(t), 0],
              [1j*sin(t), 0, 0, cos(t)]]
    ans = Qobj(matrix)
    ans.dims = [[2,2],[2,2]]
    return ans

def MS(t, axis='XX'):
    if axis == 'XX':
        return xx(t/2)
    elif axis == 'YY':
        return yy(t/2)
    else:
        raise ValueError("Molmer-Sorensen gate rotation axis error")
    
