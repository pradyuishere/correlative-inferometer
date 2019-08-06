import numpy as np
import matplotlib.pyplot as plt
from cmath import e,pi,sin,cos


M=8
fc=1e6
fs=1e7

c = 3e8
d = 150

phaseAngles = np.arange(180)*np.pi/180

#Steering Vector as a function of theta
def a(theta, p, q):
    a1 = np.exp(-1j*2*pi*fc*d*(np.cos(theta)/c) * (p - q) )
    return a1

def nc2(n):
    return int(n*(n-1)/2)

phaseDiffVectors = np.zeros([180, nc2(M)], dtype=complex)
phaseDiffVectorsIndex = 0

for theta in phaseAngles:
    phaseDiffVector = np.zeros(nc2(M), dtype=complex)
    phaseDiffVectorIndex = 0
    for iter in range(M):
        for iter2 in range(M-iter-1):
            phaseDiffVector[phaseDiffVectorIndex] = a(theta, iter, iter + iter2+1)
            phaseDiffVectorIndex += 1
    phaseDiffVectors[phaseDiffVectorsIndex, :] = phaseDiffVector
    phaseDiffVectorsIndex += 1

print("Shape of the phaseDiffVectors : ", phaseDiffVectors.shape)
np.save('phase_difference_vectors.npy', phaseDiffVectors)