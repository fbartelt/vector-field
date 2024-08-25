import plotly.colors as pc
import plotly.graph_objects as go
import numpy as np

def vector_field_plot(coordinates, field_values, orientations, curve, num_arrows=10, init_ball=0, final_ball=50,
                      num_balls=10, add_lineplot=False, camera=None, **kwargs):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function.

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the velocity of the field in each direction.
        The column entries are the respective values.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
    """
    coordinates = np.array(coordinates).reshape(-1, 3)
    skip_arrows = int(len(coordinates) / num_arrows)
    coord_field = coordinates[::skip_arrows].T
    field_values = np.array(field_values).reshape(-1, 3)[::skip_arrows].T
    skip_balls = int(len(coordinates[init_ball : final_ball]) / num_balls)
    coord_balls = coordinates[init_ball : final_ball + skip_balls : skip_balls]
    ori_balls = orientations[init_ball : final_ball + skip_balls : skip_balls]
    coordinates = coordinates.T
    # npoints = coordinates.shape[1]
    _, cscale = zip(*pc.make_colorscale(pc.qualitative.Plotly))
    if isinstance(curve, tuple):
        curve = curve[0]

    # curve
    fig = go.Figure(go.Scatter3d(x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], 
                                 mode="lines", line=dict(width=2, color=cscale[1])))
    # Ball path
    if init_ball > 0:
        fig.add_trace((go.Scatter3d(x=coordinates[0, 0:init_ball], y=coordinates[1, 0:init_ball], 
                                    z=coordinates[2, 0:init_ball], mode="lines", line=dict(width=5, dash='dash', color=cscale[5]))))
    # Workaround for first plot
    # fig.add_trace(go.Scatter3d(x=coordinates[0, init_ball:final_ball-100], y=coordinates[1, init_ball:final_ball-100], 
    #                            z=coordinates[2, init_ball:final_ball-100], mode="lines", line=dict(width=5, color=cscale[0])))
    fig.add_trace(go.Scatter3d(x=coordinates[0, init_ball:final_ball], y=coordinates[1, init_ball:final_ball], 
                               z=coordinates[2, init_ball:final_ball], mode="lines", line=dict(width=5, dash='solid', color=cscale[0])))
    
    
    # Vector field
    fig.add_trace(
        go.Cone(
            x=coord_field[0, :],
            y=coord_field[1, :],
            z=coord_field[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            # colorscale=[[i / max(index), c[1]] for i, c in zip(index, plasma_cscale)],
            colorscale=[[0, cscale[5]], [1, cscale[5]]],  # Set the colorscale
            showscale=False,
            **kwargs,
        )
    )

    # Orientation frames
    scale_frame = 0.05
    if orientations is not None:
        for i, ori in enumerate(ori_balls):
            px, py, pz = coord_balls[i, :]
            ux, uy, uz = scale_frame*(ori[:, 0])
            vx, vy, vz = scale_frame*(ori[:, 1])
            wx, wy, wz = scale_frame*(ori[:, 2])
            fig.add_trace(go.Scatter3d(x=[px, px+ux], y=[py, py+uy], z=[pz, pz+uz], mode='lines', line=dict(color='red')))
            fig.add_trace(go.Scatter3d(x=[px, px+vx], y=[py, py+vy], z=[pz, pz+vz], mode='lines', line=dict(color='lime')))
            fig.add_trace(go.Scatter3d(x=[px, px+wx], y=[py, py+wy], z=[pz, pz+wz], mode='lines', line=dict(color='blue'))
            )

    # Object
    for i, coord in enumerate(coord_balls):
        if i == 0:
            color = cscale[3]
        elif i == len(coord_balls) - 1:
            color = cscale[4]
        else:
            color = 'rgba(172, 99, 250, 0.6)'
        fig.add_trace(go.Scatter3d(x=[coord[0]], y=[coord[1]], z=[coord[2]], mode="markers", marker=dict(size=15, color=color)))
    # fig.add_trace(go.Scatter3d(x=[coordinates[0, 0]], y=[coordinates[1, 0]], z=[coordinates[2, 0]], mode="markers", marker=dict(size=10, color='magenta')))
    # fig.add_trace(go.Scatter3d(x=[coordinates[0, i2]], y=[coordinates[1, i2]], z=[coordinates[2, i2]], mode="markers", marker=dict(size=10, color='orange')))
    # fig.add_trace(go.Scatter3d(x=[coordinates[0, i3]], y=[coordinates[1, i3]], z=[coordinates[2, i3]], mode="markers", marker=dict(size=15, color='magenta')))

    #  sizemode=sizemode, sizeref=2.5, anchor='tail'))
    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )
    # camera = dict(eye=dict(x=-0.3, y=2.2, z=0.5))
    # # camera = dict(eye=dict(x=-0.4, y=1.4, z=1.6))
    # camera = dict(eye=dict(x=1.7, y=0.01, z=1.6)) # second plot
    yticks = [-0.4, 0.4]#[-0.4, -0.2, 0, 0.1, 4]
    zticks = [0., 0.6]#[0.2, 0.4, 0.6]
    xticks = [-0.4, 0.4]
    fig.update_layout(margin=dict(t=0, b=0, r=0, l=0, pad=0), scene_camera=camera, 
                      showlegend=False, scene_aspectmode='cube', 
                      scene_yaxis=dict(range=[-0.4, 0.4],   ticks='outside',
                                       tickvals=yticks, ticktext=yticks,
                                       gridcolor='rgba(148, 150, 153, 1)',
                                       showticklabels=False, title=''),
                      scene_zaxis=dict(range=[0, 0.6],   ticks='outside',
                                       tickvals=zticks, ticktext=zticks,
                                       gridcolor='rgba(148, 150, 153, 1)',
                                       showticklabels=False, title=''),
                      scene_xaxis=dict(range=[-0.4, 0.4], tickvals=xticks, 
                                       gridcolor='rgba(148, 150, 153, 1)',
                                       showticklabels=False, title=''),
                      width=1080, height=1080) # Last value makes the background transparent

    return fig