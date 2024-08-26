import plotly.colors as pc
import plotly.graph_objects as go
import numpy as np

def vector_field_plot(coordinates, field_values, orientations, curve, num_arrows=10, init_ball=0, final_ball=None,
                      num_balls=10, add_lineplot=False, colorscale=None, show_curve=True, **kwargs):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function. Also plots the target curve, and the path of the
    object. The object is represented as a sphere. The orientations are represented
    as frames with the x, y and z axis of the frame. 

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the velocity of the field in each direction.
        The column entries are the respective values.
    orientations : list or np.array
        Mx3x3 array of orientations of the object. Each row corresponds to the
        orientation of the object at that point. The 'column' entries are the
        respective 3x3 rotation matrices.
    curve : np.array
        Nx3 array of the curve points. Each row corresponds to x,y,z respectively.
    num_arrows : int, optional
        Number of vector field arrows (cones) to plot. The default is 10.
    init_ball : int, optional
        Initial ball index to plot. The default is 0.
    final_ball : int, optional
        Final ball index to plot. The default is None, which plots until the end.
    num_balls : int, optional
        Number of balls to plot. The default is 10.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
        This is used to connect the vector field arrows.
    colorscale : list, optional
        List of colors to use in the plot. The default is None, which uses the
        Plotly default colors. The list must have at least 6 colors, which are
        used for the curve, previous path, current path, initial ball, final ball
        and the object, respectively.
    show_curve : bool, optional
        Whether to show the target curve. The default is True.
    **kwargs
        Additional keyword arguments to pass to the go.Cone function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    if final_ball is None:
        final_ball = len(coordinates) - 1

    coordinates = np.array(coordinates).reshape(-1, 3)
    arrows_idx = np.round(np.linspace(0, len(coordinates) - 1, num_arrows)).astype(int)
    coord_field = coordinates[arrows_idx].T
    field_values = np.array(field_values).reshape(-1, 3)[arrows_idx].T
    ball_idx = np.round(np.linspace(init_ball, final_ball, num_balls)).astype(int)
    coord_balls = coordinates[ball_idx]
    ori_balls = np.array(orientations)[ball_idx]
    coordinates = coordinates.T
    
    if colorscale is None:
        colorscale = pc.qualitative.Plotly
    
    if isinstance(curve, tuple):
        curve = curve[0]
    
    fig = go.Figure()
    
    # Curve
    if show_curve:
        fig.add_trace(go.Scatter3d(x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], 
                                 mode="lines", line=dict(width=2, color=colorscale[1])))    
    # Previous path
    if init_ball > 0:
        fig.add_trace((go.Scatter3d(x=coordinates[0, 0:init_ball], y=coordinates[1, 0:init_ball], 
                                    z=coordinates[2, 0:init_ball], mode="lines", line=dict(width=5, dash='dash', color=colorscale[5]))))

    # Current path
    fig.add_trace(go.Scatter3d(x=coordinates[0, init_ball:final_ball], y=coordinates[1, init_ball:final_ball], 
                               z=coordinates[2, init_ball:final_ball], mode="lines", line=dict(width=5, dash='solid', color=colorscale[0])))
    
    
    # Vector field arrows
    fig.add_trace(
        go.Cone(
            x=coord_field[0, :],
            y=coord_field[1, :],
            z=coord_field[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            colorscale=[[0, colorscale[5]], [1, colorscale[5]]],  # Set the colorscale
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
            color = colorscale[3]
        elif i == len(coord_balls) - 1:
            color = colorscale[4]
        else:
            color = 'rgba(172, 99, 250, 0.6)'
        fig.add_trace(go.Scatter3d(x=[coord[0]], y=[coord[1]], z=[coord[2]], mode="markers", marker=dict(size=15, color=color)))

    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    return fig