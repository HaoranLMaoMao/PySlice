import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,HAADFData,differ

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,5*a],[0,5*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(2,seed=5)
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],14,16)
#calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_levels=[])
probe_xs = np.linspace(a,3*a,28)
probe_ys = np.linspace(b,3*b,27)
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,prism=True,loop_probes=100,use_memmap=True)#,cache_levels=[])
# RUN MULTISLICE
exitwaves = calculator.run()
#exitwaves.plot_realspace(whichProbe=500)

#fig, ax = plt.subplots()
#ary = calculator.base_probe.array[0,25,:,:]
#ax.imshow(ary.T, cmap="inferno")
#plt.show()

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
xs=haadf.xs ; ys=haadf.ys

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
haadf.plot("outputs/figs/22_prism.png")
