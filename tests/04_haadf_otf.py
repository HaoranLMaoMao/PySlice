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


trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()                   # LOAD TRAJECTORY
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])                        # TRIM TO 10x10 UC
trajectory=trajectory.get_random_timesteps(3,seed=5)                            # SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
calculator=MultisliceCalculator()                                               # CREATE CALCULATOR OBJECT
probe_xs = np.linspace(10*a-a,10*a-3*a,14)                                      # SET UP GRID OF HAADF SCAN POINTS
probe_ys = np.linspace(10*b-b,10*b-3*b,16)
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,ADF=True,loop_probes=10)
exitwaves,haadf = calculator.run()                                              # RUN MULTISLICE, RETURNS HAADF OBJECT SINCE WE USED ADF=True FLAG
#haadf=HAADFData(exitwaves)                                                     # NO NEED FOR HAADF CALCULATOR SINCE WE DID IT ON THE FLY
#ary=haadf.calculateADF()
ary = haadf.array
haadf.plot("outputs/figs/04_haadf_otf.png")
ary=np.asarray(ary)
differ(ary[::4,::4],"outputs/haadf-test.npy","HAADF")
