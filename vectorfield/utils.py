import sys
import numpy as np

def dot_J(robot, qdot, q=None):
    r"""Compute the end effetctor jacobian matrix time derivative.
    robot.jac_jac_geo returns a list of the jacobians of the i-th jacobian column
    by the configurations q. The jacobian of the jacobian (jac_jac_geo) is a 
    nx6xn tensor. Since \dot{J} = \frac{\partial J}{\partial q}\dot{q}, we can 
    compute the jacobian time derivative as jac_jac_geo @ qdot.
    """
    if q is None:
        q = robot.q
    jj_geo, *_ = robot.jac_jac_geo(q=q, axis='eef')
    dotJ = np.array(jj_geo) @ np.array(qdot).reshape(-1, 1)
    dotJ = dotJ[:, :, 0].T
    
    return dotJ

def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = np.array(q).ravel()
    return np.array([[0, -q[2], q[1]],
                     [q[2], 0, -q[0]],
                     [-q[1], q[0], 0]])

def vee(R):
    """Maps a skew-symmetric matrix to a vector"""
    return np.array([[R[2,1]], [R[0,2]], [R[1,0]]])

def to_htm(p=None, R=None):
    """Converts a position and orientation to a homogeneous transformation matrix.
    """
    if p is None:
        p = np.zeros((3, 1))
    if R is None:
        R = np.eye(3)
    p = p.ravel()
    htm = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], p[0]],
            [R[1, 0], R[1, 1], R[1, 2], p[1]],
            [R[2, 0], R[2, 1], R[2, 2], p[2]],
            [0, 0, 0, 1],
        ]
    )
    return htm

def progress_bar(i, imax):
    sys.stdout.write("\r")
    sys.stdout.write(
        "[%-20s] %d%%" % ("=" * round(20 * i / (imax - 1)), round(100 * i / (imax - 1)))
    )
    sys.stdout.flush()