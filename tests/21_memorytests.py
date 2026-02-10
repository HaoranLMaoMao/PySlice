import sys,os
try:
	import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,HAADFData,differ,gridFromTrajectory,Potential

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

#run = "reference"
#run = "memmap"
run = "probeloop"

shutil.rmtree("psi_data")

if run == "reference": # same as 04_haadf.py, no modifications
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	#positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	#atom_types=trajectory.atom_types
	#xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	#potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_reference.png")


if run == "memmap": # same as 04_haadf.py, but using memmap
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,use_memmap=True)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_memmap.png")

if run == "probeloop":
# same as 04_haadf.py, but with chunked looped probes
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,loop_probes=True)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_probeloop1.png")
