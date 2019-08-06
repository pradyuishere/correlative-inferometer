import numpy as np
import matplotlib.pyplot as plt
from cmath import e,pi,sin,cos

N=1
M=8
p=100
fc=1e6
fs=1e7

#Generating Source Signal : Nxp

s = np.zeros((N,p),dtype=complex)
for t in np.arange(start=1,stop=p+1):
    t_val = t/fs
    s[:,t-1] = np.exp(1j*2*pi*fc*t_val)
print("Source Signal s : ",s.shape)

np.save('source_signal_data.npy',s)