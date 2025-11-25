import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,probe_grid,MultisliceCalculator,HAADFData,differ

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# cache_level options include: ["exitwaves","slices","potentials"]

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])

# ONE TIMESTEPS, ONE PROBE:
print("1. one timestep, one probe, normal caching")
traj1=trajectory.get_random_timesteps(11,seed=1)
calculator=MultisliceCalculator()
calculator.setup(traj1,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::5,::5,:],"outputs/caching/01-test.npy","01") # p,t,x,y,l indices

# ONE TIMESTEPS, ONE PROBE:
print("2. one timestep, one probe, cache potentials only")
traj2=trajectory.get_random_timesteps(11,seed=2)
calculator=MultisliceCalculator()
calculator.setup(traj2,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,cache_levels=["potentials"])
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::5,::5,:],"outputs/caching/02-test.npy","02") # p,t,x,y,l indices

# ONE TIMESTEP, MANY PROBES:
print("3. one timestep, many probes, normal caching")
traj3=trajectory.get_random_timesteps(1,seed=3)
calculator=MultisliceCalculator()
probe_xs = np.linspace(a,3*a,14)
probe_ys = np.linspace(b,3*b,16)
calculator.setup(traj3,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
exitwaves = calculator.run()
differ(exitwaves.array[::5,:,::5,::5,:],"outputs/caching/03-test.npy","03") # p,t,x,y,l indices

# MANY TIMESTEPS, ONE PROBE:
print("4. many timesteps, one probe, normal caching")
traj4=trajectory.get_random_timesteps(10,seed=4)
calculator=MultisliceCalculator()
calculator.setup(traj4,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::5,::5,:],"outputs/caching/04-test.npy","04") # p,t,x,y,l indices

# MANY TIMESTEPS, MANY PROBES:
print("5. many timesteps, many probes, normal caching")
traj5=trajectory.get_random_timesteps(5,seed=5)
calculator=MultisliceCalculator()
probe_xs = np.linspace(a,3*a,9)
probe_ys = np.linspace(b,3*b,10)
calculator.setup(traj5,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::10,::10,:],"outputs/caching/05-test.npy","05") # p,t,x,y,l indices

# CACHING TURNED OFF:
print("6. many timesteps, many probes, no caching")
traj6=trajectory.get_random_timesteps(5,seed=6)
calculator=MultisliceCalculator()
probe_xs = np.linspace(a,3*a,9)
probe_ys = np.linspace(b,3*b,10)
calculator.setup(traj6,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,cache_levels=[])
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::10,::10,:],"outputs/caching/06-test.npy","06") # p,t,x,y,l indices

# OR WITH THE POTENTIAL SAVED OFF ONLY
print("7. many timesteps, one probe, caching potentials only")
traj7=trajectory.get_random_timesteps(10,seed=7)
calculator=MultisliceCalculator()
calculator.setup(traj7,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,cache_levels=["potentials"])
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::5,::5,:],"outputs/caching/07-test.npy","07") # p,t,x,y,l indices

# LAYERWISE CACHING
print("8. many timesteps, many probes, layerwise caching")
traj8=trajectory.get_random_timesteps(5,seed=8)
calculator=MultisliceCalculator()
probe_xs = np.linspace(a,3*a,6)
probe_ys = np.linspace(b,3*b,7)
calculator.setup(traj8,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,cache_levels=["slices","exitwaves"])
exitwaves = calculator.run()
differ(exitwaves.array[:,::3,::20,::20,::5],"outputs/caching/08-test.npy","08") # p,t,x,y,l indices

# LAYERWISE CACHING, WITH ONE PROBE
print("9. many timesteps, one probe, layerwise caching")
traj9=trajectory.get_random_timesteps(5,seed=9)
calculator=MultisliceCalculator()
calculator.setup(traj9,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,cache_levels=["slices","exitwaves"])
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::5,::5,::5],"outputs/caching/09-test.npy","09") # p,t,x,y,l indices

# LAYERWISE CACHING OR WITH ONE TIMESTEP
print("10. one timestep, many probes, layerwise caching")
traj10=trajectory.get_random_timesteps(1,seed=10)
calculator=MultisliceCalculator()
probe_xs = np.linspace(a,3*a,9)
probe_ys = np.linspace(b,3*b,10)
calculator.setup(traj10,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,cache_levels=["slices","exitwaves"])
exitwaves = calculator.run()
differ(exitwaves.array[:,:,::10,::10,::5],"outputs/caching/10-test.npy","10") # p,t,x,y,l indices

