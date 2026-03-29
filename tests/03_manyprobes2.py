import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import Probe,Propagate,create_batched_probes

import pyslice.backend as backend

from pyslice.multislice.multislice2 import Probe as Probe2
from pyslice.multislice.multislice2 import Propagate as Propagate2
from pyslice.multislice.multislice2 import create_batched_probes as create_batched_probes2

from pyslice.multislice.potentials2 import gridFromTrajectory,Potential
from testtools import differ, differ_phase, differ_abs
import numpy as np
import matplotlib.pyplot as plt
#from ..pyslice.tacaw.ms_calculator_npy import gridFromTrajectory
#from pyslice.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from pyslice.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from pyslice.tacaw.potential import Potential

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=30,eV=100e3)
probe2=Probe2(xs,ys,mrad=30,eV=100e3)

arr = backend.to_cpu(probe.array)
arr2 = backend.to_cpu(probe2.array)

assert(np.all(np.isclose(arr, arr2)))
probe = probe2

x, y = np.meshgrid(np.linspace(a,3*a,16),np.linspace(b,3*b,16))
xy = np.reshape([x,y], (2,len(x.flat))).T
#print(xy)
probes_many=create_batched_probes(probe,xy)
probes_many2=create_batched_probes2(probe,xy)
print('print', probes_many.array.shape)
print('print', probes_many2.array.shape)

arr = backend.to_cpu(probes_many.array)
arr2 = backend.to_cpu(probes_many2.array)

assert(np.all(np.isclose(arr, arr2)))

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types = trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# SANITY CHECK THAT OUR CROPPED TRAJECTORY IS CORRECT
#ary=np.asarray(potential.array)
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#dx=potential.xs[1]-potential.xs[0] ; dy=potential.ys[1]-potential.ys[0]
#ax.plot(x/dx,y/dy)
#plt.show()
#pr=probes_many.array[5,:,:]
#po=np.sum(potential.array,axis=2)
#fig, ax = plt.subplots()
#ax.imshow(np.absolute(pr)*np.absolute(po), cmap="inferno")
#plt.show()

# TEST PROPAGATION
# Handle device conversion properly for PyTorch tensors
result = Propagate(probes_many, potential)
result2 = Propagate(probes_many2, potential)

ary = backend.to_cpu(result)     # Convert PyTorch tensor to numpy
ary2 = backend.to_cpu(result2)     # Convert PyTorch tensor to numpy

differ(ary[::2,::2,::2],"outputs/manyprobes-test.npy","EXIT WAVE")
differ(ary2[::2,::2,::2],"outputs/manyprobes-test.npy","EXIT WAVE")
differ_phase(ary2[::2,::2,::2],"outputs/manyprobes-test.npy","EXIT WAVE")
differ_abs(ary2[::2,::2,::2],"outputs/manyprobes-test.npy","EXIT WAVE")


print(ary.shape)
ary3 = np.load('outputs/manyprobes-test.npy')
print(ary3.shape)

# ASSEMBLE HAADF IMAGE
# Convert PyTorch tensors to numpy arrays for k-space calculations
kxs = backend.to_numpy(potential.kxs)
kys = backend.to_numpy(potential.kys)
q=np.sqrt(kxs[:,None]**2+kys[None,:]**2)
fig, ax = plt.subplots()
fft = np.fft.fft2(ary,axes=(1,2))[:,::2,::2];
print(fft.shape)
fft[:,q[::2,::2]<2] = 0 # mask in reciprocal space (keep only high scattering angles)
ax.imshow(np.absolute(np.fft.fftshift(fft[0]))**.1, cmap="inferno")
plt.savefig("outputs/figs/03_manyprobes2_fft.png")
print(fft.shape)

fig, ax = plt.subplots()
HAADF=np.sum(np.absolute(fft),axis=(1,2)).reshape((len(x),len(y)))[::2,::2]
ax.imshow(HAADF, cmap="inferno")
plt.savefig("outputs/figs/03_manyprobes2_haadf_fft.png")


fft = np.fft.fft2(ary3,axes=(1,2));
fft[:,q[::2,::2]<2] = 0 # mask in reciprocal space (keep only high scattering angles)
print(fft.shape)
ax.imshow(np.absolute(np.fft.fftshift(fft[0]))**.1, cmap="inferno")
plt.savefig("outputs/figs/03_manyprobes2_fft3.png")
#plt.show()
#fig, ax = plt.subplots()
#HAADF=np.sum(np.absolute(fft),axis=(1,2)).reshape((len(x)//2,len(y)//2))
#ax.imshow(HAADF, cmap="inferno")
#plt.savefig("outputs/figs/03_manyprobes2_haadf_fft3.png")
