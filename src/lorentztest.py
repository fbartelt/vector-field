#%%
import numpy as np
from scipy.linalg import expm, logm
import plotly.express as px
import plotly.graph_objects as go

def S(xi):
    # Create an nxn matrix filled with zeros using Eigen
    S_ = np.zeros((4, 4))

    S_[0, 1] = xi[0]
    S_[0, 2] = xi[1]
    S_[1, 2] = xi[2]
    S_ = S_ - S_.T

    S_[0, 3] = xi[3]
    S_[1, 3] = xi[4]
    S_[2, 3] = xi[5]
    S_[3, 0] = xi[3]
    S_[3, 1] = xi[4]
    S_[3, 2] = xi[5]

    return S_

s = np.linspace(0, 1, 100)
Xs = []

for s_ in s:
    xi = np.array([np.cos(s_), np.sin(s_), np.cos(s_), np.sin(s_), np.cos(s_), np.sin(s_)])
    S_ = S(xi)
    Xs.append(expm(S_))


Ipq = np.eye(4)
Ipq[-1, -1] = -1
s = 1/8
beta_x = 0.9999 * np.cos(2*np.pi * s)
gamma_x = 1 / np.sqrt(1 - beta_x**2)

def boost(beta, gamma, axis='x'):
    if axis == 'x':
        boost = np.array([
            [gamma, 0, 0, -gamma * beta],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-gamma * beta, 0, 0, gamma]
        ])
    elif axis == 'y':
        boost = np.array([
                    [1, 0, 0, 0],
                    [0, gamma, 0, -gamma * beta],
                    [0, 0, 1, 0],
                    [0, -gamma * beta, 0, gamma]
                ])
    return boost
    
beta_y = 0.9999 * np.sin(2*np.pi * s)
gamma_y = 1 / np.sqrt(1 - beta_y**2)

x = np.array([0,0,0, 1]).reshape(-1, 1)
# x_ = boost_x @ x

hist_t = []
hist_x = []
hist_y = []
curve = []
v = 0.9999
for ss in np.linspace(0, 1, 1000):
    x = np.array([0,0,0, 1]).reshape(-1, 1)
    beta = (v * np.cos(2*np.pi * ss))
    gamma = (1 / np.sqrt(1 - beta**2))
    boost_x = boost(beta, gamma, axis='x')
    beta_y = (v * np.sin(2*np.pi * ss))
    gamma_y = (1 / np.sqrt(1 - beta_y**2))
    boost_y = boost(beta_y, gamma_y, axis='y')
    boost_ = boost_x
    x_ = boost_ @ x

    curve.append(boost_)
    hist_t.append(x_[-1].astype(np.float64))
    hist_x.append(x_[0])
    hist_y.append(x_[1])

    if x_[-1] < 0:
        print(boost_[-1, -1])
        if np.abs(np.linalg.det(boost_) - 1) > 1e-10:
            print('Det not 1: ', np.linalg.det(boost_))
        if not np.allclose(boost_.T @ Ipq @ boost_, Ipq):
            print('Not Lorentz: ', boost_.T @ Ipq @ boost_)
        if boost_[-1, -1] < 0:
            print('Not positive: ', boost_[-1, -1])

print(x[0]**2 + x[1]**2 + x[2]**2 - x[3]**2)
print(x_[0]**2 + x_[1]**2 + x_[2]**2 - x_[3]**2)

px.line(hist_t).show()
px.line(x=np.array(hist_x).ravel(), y=np.array(hist_y).ravel()).show()
go.Figure(go.Scatter3d(x=np.array(hist_x).ravel(), y=np.array(hist_y).ravel(), z=np.array(hist_t).ravel()))
# %%
