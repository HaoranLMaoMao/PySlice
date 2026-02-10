import sys,os,time
try:
    import pyslice
except ModuleNotFoundError:
    print("import failed, falling back to relative paths")
    sys.path.insert(0, '../src')
from pyslice.backend import zeros,fft,fftshift,asarray
from pyslice import Probe,Propagate,Potential,to_cpu
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
    xp = torch
except:
    xp = np

xs=np.linspace(0,11,111)
ys=np.linspace(0,10,101)

array = zeros((len(xs),len(ys),1))+1+.2*xp.sin(10*asarray(xs)[:,None,None]+13*asarray(ys)[None,:,None])
array*=1000
#plt.imshow(array[:,:,0])
#plt.show()

O = Potential(xs, ys, [0], array=array)
P = probe=Probe(xs,ys,mrad=30,eV=100e3)
E = Propagate(P,O)

#E.plot()
#plt.imshow(np.absolute(E[0,:,:]))
#plt.show()
plt.imshow(np.absolute(to_cpu(fftshift(xp.fft.fft2(E[0,:,:])))))
plt.show()
