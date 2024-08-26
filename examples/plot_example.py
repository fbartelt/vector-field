# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.spatial.transform import Rotation
from vectorfield.vectorfield import VectorField
from scipy.spatial.transform import Rotation
from vectorfield.utils import skew, progress_bar, to_htm
from scipy.linalg import expm

# If Uaibot is installed, also create a uaibot simulation.
import importlib.util

uaibot_spec = importlib.util.find_spec("uaibot")
uaibot_installed = uaibot_spec is not None

if uaibot_installed:
    from uaibot.simulation import Simulation
    from uaibot.simobjects.ball import Ball
    from uaibot.simobjects.frame import Frame
    from uaibot.simobjects.pointcloud import PointCloud
    from uaibot.simobjects.pointlight import PointLight
    from uaibot.utils import Utils

# Parametric equation definition
def parametric_eq_factory(c1=1 / 8, c2=0.4, maxtheta=500):
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    precomputed = ()
    rotz = np.identity(3)

    curve = np.empty((3, len(theta)))
    for i, _ in enumerate(theta):
        curve[:, i] = rotz @ np.array(
            [
                c1 * (sin_theta[i] + 2 * np.sin(2 * theta[i])),
                c1 * (cos_theta[i] - 2 * np.cos(2 * theta[i])),
                c2 + c1 * (-np.sin(3 * theta[i])),
            ]
        )
    orientations = np.empty((len(theta), 3, 3))
    for i, ang in enumerate(theta):
        orientations[i, :, :] = (
            Rotation.from_euler("z", ang).as_matrix()
            @ Rotation.from_euler("x", 2 * ang).as_matrix()
        )

    precomputed = (curve.T, orientations)

    def parametric_eq(time=0):
        return precomputed

    return parametric_eq

maxtheta = 500
T = 20
dt = 1e-2
eq = parametric_eq_factory(c1=1 / 8, c2=0.4, maxtheta=maxtheta)
vf = VectorField(eq, time_dependent=False, k_0=5, k_1=1, k_2=70, beta=1, dt=1e-2)

curve = eq(0)
curve_points = curve[0]
curve_ori = curve[1]
p = np.array([0.3, 0.3, 0.1]).reshape(-1, 1)
R = (
    Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    @ Rotation.from_euler("x", np.deg2rad(12)).as_matrix()
)

if uaibot_installed:
    obj = Ball(htm=Utils.trn(p), radius=0.05, color="#8a2be2")
    frame_ball = Frame(to_htm(p, R), "axis", 0.2)
    light1 = PointLight(
        name="light1", color="white", intensity=2.5, htm=Utils.trn([-1, -1, 1.5])
    )
    light2 = PointLight(
        name="light2", color="white", intensity=2.5, htm=Utils.trn([-1, 1, 1.5])
    )
    light3 = PointLight(
        name="light3", color="white", intensity=2.5, htm=Utils.trn([1, -1, 1.5])
    )
    light4 = PointLight(
        name="light4", color="white", intensity=2.5, htm=Utils.trn([1, 1, 1.5])
    )
    curve_draw = PointCloud(name="curve", points=curve_points.T, size=8, color="orange")
    curve_frames = []
    for i, c in enumerate(zip(curve_points, curve_ori)):
        pos, ori = c
        if i % 50 == 0:
            curve_frames.append(Frame(to_htm(pos, ori), f"curveframe{i}", 0.1))

imax = int(T / dt)
p_hist = []
R_hist = []
v_hist, w_hist = [np.zeros((3, 1))], [np.zeros((3, 1))]

for i in range(imax):
    progress_bar(i, imax)
    xi = vf.psi(p, R)
    vd = xi[:3, :]
    wd = xi[3:, :]
    if np.iscomplex(xi).any():
        print("Complex number found")
    p = p + vd * dt
    R = expm(dt * skew(wd)) @ R

    if uaibot_installed:
        obj.add_ani_frame(i * dt, to_htm(p, None))
        frame_ball.add_ani_frame(i * dt, to_htm(p, R))
        curve_draw.add_ani_frame(i * dt, 0, 500)
        # _, ind_min = vf._divide_conquer(curve, p, R)
        # for frame in curve_frames[ind_min + 1 :]:
        #     frame.add_ani_frame(i * dt, htm=Utils.trn([0, 0, 0]))

    p_hist.append(p)
    R_hist.append(R)
    v_hist.append(vd)
    w_hist.append(wd)

if uaibot_installed:
    sim = Simulation.create_sim_grid(
        [obj, frame_ball, curve_draw, light1, light2, light3, light4]
    )
    sim.add([curve_frames])
    sim.set_parameters(
        width=1200, height=600, ambient_light_intensity=4, show_world_frame=False
    )
    sim.run()
# %%
""" Plot vector field and object path"""
from vectorfield.plotting import vector_field_plot

orientations = R_hist
fig = vector_field_plot(
    p_hist,
    v_hist,
    R_hist,
    curve,
    num_arrows=10,
    init_ball=0,
    final_ball=int((T) / dt) - 1,
    num_balls=10,
    show_curve=True,
    sizemode="absolute",
    sizeref=3e-2,
    anchor="tail",
)

cam = np.array([-0.3, 1.4, 1.4])
cam = 2.5 * cam / np.linalg.norm(cam)
camera = dict(eye=dict(x=cam[0], y=cam[1], z=cam[2]))
yticks = [-0.4, 0.4]  # [-0.4, -0.2, 0, 0.1, 4]
zticks = [0.0, 0.6]  # [0.2, 0.4, 0.6]
xticks = [-0.4, 0.4]
args = dict(
    margin=dict(t=0, b=0, r=0, l=0, pad=0),
    scene_camera=camera,
    showlegend=False,
    scene_aspectmode="cube",
    scene_yaxis=dict(
        range=[-0.4, 0.4],
        ticks="outside",
        tickvals=yticks,
        ticktext=yticks,
        gridcolor="rgba(148, 150, 153, 1)",
        showticklabels=False,
        title="",
    ),
    scene_zaxis=dict(
        range=[0, 0.6],
        ticks="outside",
        tickvals=zticks,
        ticktext=zticks,
        gridcolor="rgba(148, 150, 153, 1)",
        showticklabels=False,
        title="",
    ),
    scene_xaxis=dict(
        range=[-0.4, 0.4],
        tickvals=xticks,
        gridcolor="rgba(148, 150, 153, 1)",
        showticklabels=False,
        title="",
    ),
    width=1080,
    height=1080,
)
fig.update_layout(**args)
fig.show()
# %%
"""Create an animation of the distance metric evolution"""
from vectorfield.plotting import animate_distance

fig = animate_distance(p_hist, R_hist, vf.nearest_points, dt, T)
fig.show()

# %%
