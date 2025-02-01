#%%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from vectorfield import VectorFieldSE3
from vectorfield.lie import SE3, se3
from scipy.linalg import expm

def hd(s, r=1.0, b=1.0, d=0.2):
    """Curve parametrization used in paper. This is based on the hyperbolic 
    paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.
    
    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4) # initialize the homogeneous transformation matrix
    position = [r * np.cos(theta), r * np.sin(theta), b]
    hds[:3, 3] = np.array(position)
    orientation = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])
    hds[:3, :3] = orientation
    return hds

def hd_dot(s, r=1.0, b=1.0, d=0.2):
    """Curve derivative parametrization used in paper. This is based on the 
    hyperbolic paraboloid.

    Parameters
    ----------
    s : float
        Parameter of the curve. It must be in the interval [0, 1].
    r : float, optional
        Radius of the curve in XY plane. The default is 1.
    b : float, optional
        Height of the curve. The default is 1.
    d : float, optional
        Curvature of the curve. The default is 0.2.
    
    Returns
    -------
    hds : np.array
        Homogeneous transformation matrix of the curve evaluated at parameter s.
        This is a 'list' of elements of the SE(3) group.
    """
    theta = 2 * np.pi * s
    hds = np.identity(4) # initialize the homogeneous transformation matrix
    position = [-r * np.sin(theta), r * np.cos(theta), 0]
    hds[:3, 3] = np.array(position)
    orientation = np.array([[1, 0, 0], [0, -np.sin(theta), np.cos(theta)], [0, -np.cos(theta), -np.sin(theta)]])
    hds[:3, :3] = orientation
    return hds

def precomputed_hd(n_points, fun, *args, **kwargs):
    """Function that precomputes the curve for each parameter s.

    Parameters
    ----------
    n_points : int
        Number of points in the curve.
    *args : list
        Arguments of the curve function.
    **kwargs : dict
        Keyword arguments of the curve function.
    
    Returns
    -------
    precomputed : np.array
        Array with the precomputed curve. The shape is (n_points, 4, 4).
    """
    s = np.linspace(0, 1, num=n_points)
    precomputed = []
    for si in s:
        precomputed.append(fun(si, *args, **kwargs))
    precomputed = np.array(precomputed)
    return precomputed

def progress_bar(i, imax):
    """Prints a progress bar in the terminal.

    Parameters
    ----------
    i : int
        Current iteration.
    imax : int
        Maximum number of iterations.
    """
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()
    
#%%
H0 = SE3()
T = 8.0
dt = 1e-2
epsilon, ds = 1e-3, 1e-3
n_points = 5000

curve = precomputed_hd(n_points, hd)
curve_dot = precomputed_hd(n_points, hd_dot)

gain_N1, gain_N2, gain_T1, gain_T2, gain_T3 = 1.0, 1.0, 1.0, 1.0, 1.0
vf = VectorFieldSE3(curve, curve_dot, epsilon, ds)

H = SE3(H0)
imax = int(T/dt)
H_hist = []
distance_hist = []
nearest_point_hist = []
vector_field_hist = []
for i in range(imax):
    progress_bar(i, imax)
    psi = vf.eval(H, True, gain_N1, gain_N2, gain_T1, gain_T2, gain_T3).reshape(-1, 1)
    # Update the system
    H = SE3((H.algebra().S(psi) * dt).exp() @ H.matrix())
    # Save the variables
    H_hist.append(H)

# %%
