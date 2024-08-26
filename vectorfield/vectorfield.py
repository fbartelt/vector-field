import warnings
import numpy as np
from scipy.linalg import logm, expm
from .utils import skew, vee

_INVHALFPI = 0.63660

class VectorField:
    """Vector Field class. Uses extension to the vector field presented in:
    A. M. C. Rezende, V. M. Goncalves and L. C. A. Pimenta, "Constructive Time-
    Varying Vector Fields for Robot Navigation," in IEEE Transactions on
    Robotics, vol. 38, no. 2, pp. 852-867, April 2022,
    doi: 10.1109/TRO.2021.3093674.

    Parameters
    ----------
    parametric_equation : function
        A function that represents the parametric equation of the curve. It must
        return a tuple whose first component is a NxM array of coordinates of
        the curve, and the second component is a Nx3x3 tensor corresponding to
        the N orientations frames. For the first component, each one of the n rows
        should contain a m-dimensional float vector that is the n-th
        m-dimensional sampled point of the curve.
    time_dependent : bool, optional
        Whether the curve (consequently the vector field) is time dependent or
        not. The default is False.
    beta: float, optional
        A parameter that weights the contribution of rotation error in the
        distance function. It also ensures dimensionality consistency. The
        default is 1.
    dt : float, optional
        The time step, in seconds, used to compute the time derivative of the
        vector field. The default is 1e-3.
    ds : float, optional
        The step size betweem samples of the curve. It is used to compute the
        tangent vector components. The default is 1.
    sense : int, optional
        The sense of circulation of the curve. If 1, the curve is circulated in
        anti-clockwise sense. If -1, the curve is circulated in clockwise sense.
        The default is 1.
    g_function : function, optional
        A function that computes the function G of the vector field. The default
        is None, which uses the default function used in the paper. The instance
        will be passe to the user-defined function as an argument 'instance'.
    h_function : function, optional
        A function that computes the function H of the vector field. The default
        is None, which uses the default function used in the paper. The instance
        will be passe to the user-defined function as an argument 'instance'.
    *args:
        Additional positional arguments to be passed to the user-defined
        functions or used to configure the default functions.
    **kwargs:
        Additional keyword arguments to be passed to the user-defined functions
        or used to configure the default functions.

    Methods
    -------
    __call__(position, time=0)
        Returns the vector field value at the given position and time. It is the
        same as calling the psi method.
    __repr__()
        Returns the string representation of the vector field.
    psi(position, time=0, store_points=True)
        Returns the vector field value at the given position and time. If
        store_points is True, the nearest points of the curve are stored in the
        nearest_points attribute.
    acceleration(position, velocity, time=0)
        Returns the acceleration of the vector field at the given position,
        velocity and time.
    """

    def __init__(
        self,
        parametric_equation,
        time_dependent=False,
        beta=1,
        dt=1e-3,
        ds=1,
        sense=1,
        g_function=None,
        h_function=None,
        *args,
        **kwargs,
    ):
        self.parametric_equation = parametric_equation
        self.time_dependent = time_dependent
        self.beta = beta
        self.dt = dt
        self.ds = ds
        self.sense = sense
        self.G = self._wrap_function(g_function or self._default_G, *args, **kwargs)
        self.H = self._wrap_function(h_function or self._default_H, *args, **kwargs)
        self.nearest_points = []

        if time_dependent:
            raise NotImplementedError(
                "Time dependent vector fields are not implemented"
            )

    def __call__(self, position, orientation, time=0, store_points=True):
        return self.psi(position, orientation, time, store_points)

    def __repr__(self):
        return f"Time-{('In'*(not self.time_dependent)+'variant').capitalize()} Vector Field.\n Alpha: {self.alpha},\n Constant Velocity: {self.const_vel},\n dt: {self.dt},\n Parametric Equation: {self.parametric_equation.__name__}"

    def _wrap_function(self, func, *args, **kwargs):
        def wrapped_function(min_dist):
            return func(min_dist, instance=self, *args, **kwargs)

        return wrapped_function

    @staticmethod
    def _default_G(min_dist, k_0=5, k_1=1, *args, **kwargs):
        return k_1 * _INVHALFPI * np.arctan(k_0 * np.sqrt(min_dist))

    @staticmethod
    def _default_H(min_dist, instance, k_2=14, *args, **kwargs):
        fun_g = instance._default_G(min_dist, *args, **kwargs)
        return k_2 * np.sqrt(max(1 - fun_g**2, 0))

    @staticmethod
    def distance_fun(p, R, pd, Rd, beta=1):
        r"""Computes the distance between the current point and the desired point
        and the current orientation and the desired orientation. The distance is
        described by :math: `D=||p-p_d||^2 + \frac{1}{2}||I - Rd^TR||^2_F`
        """
        p = np.array(p).reshape(-1, 1)
        pd = np.array(pd).reshape(-1, 1)
        lin_dist = np.linalg.norm(p - pd) ** 2
        rot_dist = 0.5 * np.linalg.norm(Rd.T @ R, "fro") ** 2
        return lin_dist + beta * rot_dist

    def _divide_conquer(self, curve_segment, p, R):
        curve_points = np.array(curve_segment[0])
        curve_frames = curve_segment[1]
        npoints = curve_segment[0].shape[0]

        if npoints == 1:
            return (
                self.distance_fun(
                    p, R, curve_points[0, :], curve_frames[0, :, :], self.beta
                ),
                0,
            )

        mid_index = npoints // 2
        left_segment = (
            curve_segment[0][:mid_index, :],
            curve_segment[1][:mid_index, :, :],
        )
        right_segment = (
            curve_segment[0][mid_index:, :],
            curve_segment[1][mid_index:, :, :],
        )

        left_dist, left_index = self._divide_conquer(left_segment, p, R)
        right_dist, right_index = self._divide_conquer(right_segment, p, R)

        if left_dist < right_dist:
            return left_dist, left_index
        else:
            return right_dist, right_index + mid_index

    def _add_nearest_point(self, point):
        self.nearest_points.append(point)

    def psi(self, position, orientation, time=0, store_points=True):
        """Computes the vector field value at the given position and time. It is
        the same as calling the __call__ method. If store_points is True, the
        nearest points of the curve are stored in the nearest_points attribute.

        Parameters
        ----------
        position : list or np.array
            The position where the vector field will be computed.
        orientation : np.array
            The orientation at the given position, represented by a 3x3 rotation
            matrix.
        time : float, optional
            The time at which the vector field will be computed. The default is 0.
        store_points : bool, optional
            Whether to store the nearest points of the curve. The default is True.
        """
        psi_s = self._psi_s(position, orientation, time, store_points=store_points)
        return psi_s

    def _psi_s(self, position, orientation, time=0, store_points=True):
        p = np.array(position).reshape(-1, 1)
        curve = self.parametric_equation(time=time)
        return self._vector_field_vel(
            p,
            orientation,
            curve,
            store_points=store_points,
        )

    def _vector_field_vel(self, p, R, curve, store_points=True):
        vec_n, vec_t, min_dist = self._compute_ntd(
            curve, p, R, store_points=store_points
        )
        fun_g = self.G(min_dist)
        fun_h = self.H(min_dist)
        return -fun_g * vec_n + self.sense * fun_h * vec_t

    def _compute_ntd(self, curve, p, R, store_points=True):
        min_dist = float("inf")
        ind_min = -1

        pr = np.array(p).ravel()  # row vector to simplify computations
        min_dist, ind_min = self._divide_conquer(curve, pr, R)

        vec_n_p = pr - curve[0][ind_min, :]
        Rd = curve[1][ind_min, :, :]
        sigma = self.beta * (
            skew(Rd[:, 0]) @ R[:, 0].reshape(-1, 1)
            + skew(Rd[:, 1]) @ R[:, 1].reshape(-1, 1)
            + skew(Rd[:, 2]) @ R[:, 2].reshape(-1, 1)
        )
        vec_n = np.vstack((vec_n_p.reshape(-1, 1), sigma))

        # Handles closed curves (next index is the first point in the curve)
        if ind_min == np.shape(curve[0])[0] - 1:
            vec_t_p = curve[0][1, :] - curve[0][ind_min, :]
            vec_t_rot = vee(logm(curve[1][1, :, :] @ Rd.T) / self.ds)
        else:
            vec_t_p = curve[0][ind_min + 1, :] - curve[0][ind_min, :]
            vec_t_rot = vee(logm(curve[1][ind_min + 1, :, :] @ Rd.T) / self.ds)

        vec_t = np.vstack((vec_t_p.reshape(-1, 1), vec_t_rot))

        if store_points:
            self._add_nearest_point((curve[0][ind_min, :], curve[1][ind_min, :, :]))

        if vec_n.shape != (6, 1) or vec_t.shape != (6, 1):
            raise ValueError(f"Error in vec_n or vec_t: {vec_n.shape}, {vec_t.shape}")

        return vec_n, vec_t, min_dist

    def acceleration(self, position, orientation, v, w, time=0):
        r"""Returns the acceleration of the vector field at the given position,
        orientation, linear velocity, angular velocity and time.

        It computes the nummerical approximation for the vector field time derivative
        \dot{\psi} \approx (\psi[k + 1] - \psi[k])/dt

        Parameters
        ----------
        position : list or np.array
            The position where the acceleration will be computed.
        orientation : np.array
            The orientation at the given position, represented by a 3x3 rotation matrix.
        v : list or np.array
            The linear velocity at the given position.
        w : list or np.array
            The angular velocity at the given position.
        time : float, optional
            The time at which the acceleration will be computed. The default is 0.
        """
        position = np.array(position).reshape(-1, 1)
        l_velocity = np.array(v).reshape(-1, 1)
        a_velocity = np.array(w).reshape(-1, 1)

        # Compute the numerical approximation of the time derivative
        current_vf = self.psi(position, orientation, time, store_points=False)
        next_vf = self.psi(position + l_velocity * self.dt, expm(skew(a_velocity) * self.dt) @ orientation, time + self.dt, store_points=False)
        dot_psi = (next_vf - current_vf) / self.dt

        return dot_psi