#%%
import uaibot as ub
from uaibot.utils import Utils
from uaibot.simulation import Simulation
from uaibot.simobjects.frame import Frame
from uaibot.simobjects.pointcloud import PointCloud
from uaibot.simobjects.ball import Ball
import numpy as np
import csv

def read_csv_and_restore(filename, isvfdata=True):
    iteration_results = []

    # Define the dimensions of the matrix and vectors
    matrix_shape = (4, 4)  # Example for a 4x4 matrix
    vector_size = 6  # Example for two 6x1 vectors

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')

        for row in csvreader:
            # Convert the row (list of strings) to a list of floats
            data = list(map(float, row))

            # Extract the matrix elements and reshape them into a 4x4 matrix
            matrix_size = matrix_shape[0] * matrix_shape[1]
            matrix_data = np.array(data[:matrix_size]).reshape(matrix_shape)

            if isvfdata:
                # Extract the first vector (6x1)
                tangent_data = np.array(data[matrix_size:matrix_size + vector_size])

                # Extract the second vector (6x1)
                normal_data = np.array(data[matrix_size + vector_size:matrix_size + 2 * vector_size])

                # Store the results as a tuple of matrix and two vectors
                iteration_results.append((matrix_data, tangent_data, normal_data, data[-1]))
            else:
                iteration_results.append(matrix_data)
    return iteration_results

# Example usage:
option = ''
results = read_csv_and_restore(f'/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/vf_data{option}.csv')
curve = read_csv_and_restore(f'/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/curve_data{option}.csv', isvfdata=False)
states = read_csv_and_restore(f'/home/fbartelt/Documents/Projetos/vector-field/vfcpp/logs/iteration_data{option}.csv', isvfdata=False)

#%%
dt = 0.01

H0 = Utils.trn([-2, -1, 0]) @ Utils.rotz(np.pi / 4)

ball = Ball(htm=H0, name='ball', radius=0.05, color='magenta')
frame_ball = Frame(htm=H0, name="axis", size=0.2)

curve_points = []
curve_frames = []
for i, htm in enumerate(curve):
    curve_points.append(htm[:3, 3])
    if i % 50 == 0: 
        curve_frames.append(Frame(htm, f"curveframe{i}", 0.1))

curve_points = np.array(curve_points).T
curve_draw = PointCloud(name="curve", points=curve_points, size=8, color="orange")
sim = Simulation.create_sim_grid([ball, frame_ball, curve_draw, curve_frames])

for i, state in enumerate(states):
    H = state
    ball.add_ani_frame(i * dt, H)
    frame_ball.add_ani_frame(i * dt, H)
    curve_draw.add_ani_frame(i * dt, 0, curve_points.shape[1])

sim.set_parameters(
        width=1200, height=800, ambient_light_intensity=4, show_world_frame=False
    )
sim.run()
# %%
import plotly.express as px
closest_points, tangents, normals, distances = zip(*results)

norms = []
# Computes the norm of each element in normals and tanges list:
for normal, tangent in zip(normals, tangents):
    xin = np.array(normal).reshape(-1, 1)
    xit = np.array(tangent).reshape(-1, 1)
    norms.append(float(180/np.pi * np.arccos((xin.T @ xit) / (1e-6 + np.linalg.norm(xin) * np.linalg.norm(xit)))))

px.line(norms)
# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

closest_points, tangents, normals, distances = zip(*results)

ori_errs = []
pos_errs = []
for closest_point, state in zip(closest_points, states):
    p_near = closest_point[:3, 3]
    ori_near = closest_point[:3, :3]
    p_curr = state[:3, 3]
    ori_curr = state[:3, :3]
    pos_errs.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ ori_curr.T)
    acos = np.arccos((trace_ - 1) / 2)
    # checks if acos is nan
    if np.isnan(acos):
        acos = 0
    ori_errs.append(acos * 180 / np.pi)
    # ori_errs.append(np.linalg.norm(np.eye(3) - ori_near @ ori_curr.T, 'fro'))

# makes a figure with two plots, one above another. First the position error, then the orientation error
dt = 0.01
time_vec = np.arange(0, len(pos_errs) * dt, dt)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=time_vec, y=distances, showlegend=False, line=dict(width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=time_vec, y=pos_errs, showlegend=False, line=dict(width=3)), row=2, col=1)
fig.add_trace(go.Scatter(x=time_vec, y=ori_errs, showlegend=False, line=dict(width=3)), row=3, col=1)
fig.update_xaxes(title_text="Time (s)", gridcolor='gray', zerolinecolor='gray', row=3, col=1)
fig.update_xaxes(title_text="", gridcolor='gray', zerolinecolor='gray', row=1, col=1)
fig.update_xaxes(title_text="", gridcolor='gray', zerolinecolor='gray', row=2, col=1)
fig.update_yaxes(title_text="Distance D", gridcolor='gray', zerolinecolor='gray', row=1, col=1, title_standoff=30)
fig.update_yaxes(title_text="Pos. error (cm)", gridcolor='gray', zerolinecolor='gray', row=2, col=1, title_standoff=30)
fig.update_yaxes(title_text="Ori. error (deg)", gridcolor='gray', zerolinecolor='gray', row=3, col=1, title_standoff=30)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                  width=718.110, height=605.9155)
fig.show()
fig.write_image("/home/fbartelt/Documents/Artigos/figures/distance_pos_ori_D.svg")
#%%
"""Plot distance D"""
import plotly.graph_objects as go
closest_points, tangents, normals, distances = zip(*results)

dt = 0.01
time_vec = np.arange(0, len(distances) * dt, dt)
fig = go.Figure(go.Scatter(x=time_vec ,y=distances, showlegend=False, line=dict(width=3)))
fig.update_xaxes(title_text="Time (s)", gridcolor='gray', zerolinecolor='gray')
fig.update_yaxes(title_text="Distance D", gridcolor='gray', zerolinecolor='gray', )#zerolinewidth=1, zeroline=True)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# Add more spacing between y-axis title and ticks
fig.update_yaxes(title_standoff=30)

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                  width=718.110, height=403.937)

fig.show()
fig.write_image("/home/fbartelt/Documents/Artigos/figures/distanceD.svg")
# %%
import plotly.graph_objects as go
closest_points, tangents, normals, distances = zip(*results)

curve_positions = []
obj_frames = []
obj_positions = []
vfs = []
for i, H in enumerate(states[:-1]):
    R = H[:3, :3]
    p = H[:3, 3]
    obj_frames.append(R)
    obj_positions.append(p)
    tangent_ = tangents[i][:3]
    normal_ = normals[i][:3]
    vf_ =  tangent_ + normal_
    vf_ = vf_ / (np.linalg.norm(vf_))
    vfs.append(vf_)

obj_positions = np.array(obj_positions).reshape(-1, 3)
curve_positions = np.array(curve_positions).reshape(-1, 3)
vfs = np.array(vfs).reshape(-1, 3)

fig = go.Figure(go.Cone(x=obj_positions[:, 0], y=obj_positions[:, 1], 
                        z=obj_positions[:, 2], u=vfs[:, 0], v=vfs[:, 1], 
                        w=vfs[:, 2], sizemode="absolute", sizeref=5, 
                        anchor="tail"))
fig.show()
# %%
""" PLOT VECTOR FIELD
 ---- HOW TO GET CURRENT CAMERA CONFIGURATION
    f = go.FigureWidget(fig)
    f
    print(f.get_state()['_layout']['scene']['camera']['eye'])
    print(f.get_state()['_layout']['scene']['camera']['center'])
"""
import sys
import plotly.express as px
sys.path.append('/home/fbartelt/Documents/Projetos/vector-field/')

from vectorfield.plotting import vector_field_plot
closest_points, tangents, normals, distances = zip(*results)

curve_positions = []
obj_frames = []
obj_positions = []
vfs = []
for i, H in enumerate(states[:-1]):
    R = H[:3, :3]
    p = H[:3, 3]
    obj_frames.append(R)
    obj_positions.append(p)
    tangent_ = tangents[i][:3]
    normal_ = normals[i][:3]
    vf_ = tangent_ + normal_ 
    vf_ = vf_ / (np.linalg.norm(vf_))
    vfs.append(vf_)

for i, htm in enumerate(curve):
    curve_positions.append(htm[:3, 3])

obj_positions = np.array(obj_positions).reshape(-1, 3)
curve_positions = np.array(curve_positions).reshape(-1, 3)
vfs = np.array(vfs).reshape(-1, 3)
obj_frames = np.array(obj_frames).reshape(-1, 3, 3)

final_ball = 1450 # 499 for 1st, 970 for 2nd, 1450 for 3rd
init_ball = 970  # 0 for 1st, 499 for 2nd, 970 for 3rd
xticks = [-2, 1.1]
yticks = [-1.2, 1.1]
zticks = [0, 1.3] 
frame_scale = [abs(xticks[1] - xticks[0]), abs(yticks[1] - yticks[0]), abs(zticks[1] - zticks[0])]
frame_scale = .3 * 1/(np.max(frame_scale) / frame_scale)
fig = vector_field_plot(obj_positions, vfs, obj_frames, curve_positions, 
                        num_arrows=0, init_ball=init_ball, final_ball=final_ball,
                        curr_path_style="solid", prev_path_style="dash",
                        sizemode="absolute", sizeref=6e-2, anchor="tail",
                        ball_size=12, curve_width=5, path_width=10, frame_scale=frame_scale,
                        frame_width=3)
# cam = np.array([-2, -1.2, 1.3]) # first plot
# center = dict(x=0, y=0.06, z=-0.175) # first plot
# zoom = 1.95 # first plot
# cam = np.array([0.2, 2, 0.8]) # second plot
# center = dict(x=-0.15, y=-0.1, z=-0.2) # second plot
# zoom = 1.75 # second plot
cam = np.array([-3, 0.1, 2.5]) # third plot
center = dict(x=0, y=-0.1, z=-0.05) # third plot
zoom = 1.8 # third plot
cam = zoom * cam / np.linalg.norm(cam)
eye = dict(x=cam[0], y=cam[1], z=cam[2])
# xticks = [-2, 2]
# yticks = [-2, 2]
# zticks = [0, 1.3]

## New results:
# eye = {'x': -1.1895395264042192, 'y': -0.7763234687127192,'z': 1.4869053271198966} # FIRST PLOT  # [-1.1895395264042192, -0.7763234687127192, 1.4869053271198966]
# center = {'x': 0.03903242065155699, 'y': 0.07130795959536744, 'z': -0.13962461747321292} # FIRST PLOT
# eye = {'x': -0.005417756765127929, 'y': 1.5295635461969768, 'z': 0.852109769210902} # SECOND PLOT
# center = {'x': 0.08777933955447563, 'y': -0.13216780957261906, 'z': -0.16458099064064602} # SECOND PLOT
eye = {'x': 0.9282984011542231, 'y': -0.08422883056475791, 'z': 1.609796084764507} # THIRD PLOT
center = {'x': -0.07743803067418094, 'y': 0.08348353761188693, 'z': -0.08028295936982242} # THIRD PLOT
camera = dict(eye=eye, center=center, up=dict(x=0, y=0, z=1))
args = dict(
    margin=dict(t=0, b=0, r=0, l=0, pad=0),
    scene_camera=camera,
    showlegend=False,
    scene_aspectmode="cube",
    scene_yaxis=dict(
        range=yticks,
        ticks="outside",
        tickvals=yticks,
        ticktext=yticks,
        showticklabels=False,
        title="",
        gridcolor="black",
        backgroundcolor="rgba(0, 0, 0, 0)",
    ),
    scene_zaxis=dict(
        range=zticks,
        ticks="outside",
        tickvals=zticks,
        ticktext=zticks,
        showticklabels=False,
        title="",
        gridcolor="black",
        backgroundcolor="rgba(0, 0, 0, 0)",
    ),
    scene_xaxis=dict(
        range=xticks,
        tickvals=xticks,
        showticklabels=False,
        title="",
        gridcolor="black",
        backgroundcolor="rgba(0, 0, 0, 0)",
    ),
    width=800,
    height=800,
)
fig.update_layout(**args)

# fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
fig.show()
fig.write_image("/home/fbartelt/Documents/Artigos/figures/vf_automatica_3.pdf")
# %%
""" PLOT CURVE WITH FRAMES -- CBA presentation"""
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

colorscale = pc.qualitative.Plotly

curve_pos, curve_ori = [], []
for htm in curve:
    curve_pos.append(htm[:3, 3])
    curve_ori.append(htm[:3, :3])

curve_pos = np.array(curve_pos).T
curve_ori = np.array(curve_ori).reshape(-1, 3, 3)
fig = go.Figure(go.Scatter3d(x=curve_pos[0, :], y=curve_pos[1, :], z=curve_pos[2, :], mode='markers', marker=dict(size=4, color=colorscale[9]), showlegend=False,))

scale_frame = 0.1
for i, ori in enumerate(curve_ori):
    if i % 70 == 0:
        px, py, pz = curve_pos[:, i]
        ux, uy, uz = scale_frame * (ori[:, 0]) #/ (np.linalg.norm(ori[:, 0]) + 1e-6)
        vx, vy, vz = scale_frame * (ori[:, 1]) #/ (np.linalg.norm(ori[:, 1]) + 1e-6)
        wx, wy, wz = scale_frame * (ori[:, 2]) #/ (np.linalg.norm(ori[:, 2]) + 1e-6)
        fig.add_trace(
            go.Scatter3d(
                x=[px, px + ux],
                y=[py, py + uy],
                z=[pz, pz + uz],
                mode="lines",
                line=dict(color="red", width=3),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[px, px + vx],
                y=[py, py + vy],
                z=[pz, pz + vz],
                mode="lines",
                line=dict(color="lime", width=3),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[px, px + wx],
                y=[py, py + wy],
                z=[pz, pz + wz],
                mode="lines",
                line=dict(color="blue", width=3),
                showlegend=False,
            )
        )

fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), width=1200, height=800)
fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, nticks=2), yaxis=dict(showticklabels=False, nticks=2), zaxis=dict(showticklabels=False, nticks=2)))
# Changes camera position to a better view (less blank space around the plot)
fig.update_layout(scene_camera=dict(eye=dict(x=1.3, y=1.3, z=1.3), center=dict(x=0, y=0, z=-0.3), up=dict(x=0, y=0, z=1)), )
# Add zoom to the plot
# fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.5, y=0.5, z=0.5)))
# fig.show()
# save figure as pdf file
fig.write_image("/home/fbartelt/Documents/Artigos/figures/cba/curve_with_frames.pdf")

# %%
