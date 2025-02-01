#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import pylie as pl
import numpy as np

H = pl.SE3()

print(H)
print(H * H)
print(H @ H)
# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from vectorfield import VectorFieldSE3

# %%
