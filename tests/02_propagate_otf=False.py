import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from testtools import differ
from pyslice import Loader,Probe,Propagate,grid_from_trajectory,Potential
from pyslice.backend import make_backend, to_numpy

import numpy as np

backend = make_backend()

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=5,eV=100e3,backend=backend)

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, backend=backend, kind="kirkland")

# TEST PROPAGATION
result = Propagate(probe,potential,backend,onthefly=False)
ary = to_numpy(result)
ary=ary[0,:,:]

differ(ary[::10,::10],"outputs/propagate-test.npy","EXIT WAVE")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#ax.imshow(np.absolute(ary), cmap="inferno")
#plt.show()
ax.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(ary)))**.1, cmap="inferno")
plt.savefig("outputs/figs/02_propagate_otf=False.png")
