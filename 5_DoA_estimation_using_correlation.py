
import numpy as np
import matplotlib.pyplot as plt
from cmath import e,pi,sin,cos

N=1
M=8
p=100
fc=1e6
fs=1e7
c = 3e8
d = 150

#function to correlate two given phase difference vectors
def correlatePhaseDiffVectors(vecX, vecY):
    return np.abs(np.matmul(vecX.conjugate(), vecY)/(np.linalg.norm(vecX) * np.linalg.norm(vecY)))

#function to find the phase difference between two vectors
def phiXY(vecX, vecY):
    return (np.matmul(vecX.conjugate(), vecY)/(np.linalg.norm(vecX) * np.linalg.norm(vecY)))

#function to find the nC2 for a given n
def nc2(n):
    return int(n*(n-1)/2)

#reference phase difference vectors
phiR = np.load('phase_difference_vectors.npy')
#print(phiR)

#recieved signal data
X = np.load('recieved_signal_data.npy')
print("Recieved Signal X: ",X.shape)

#calculate phase difference vectors
calculatedPhaseDiffVector = np.zeros([nc2(M)], dtype=complex)
calculatedPhaseDiffVectorIndex = 0
for iter in range(M):
    for iter2 in range(M-iter-1):
        calculatedPhaseDiffVector[calculatedPhaseDiffVectorIndex] = phiXY(X[iter + iter2 + 1, :], X[iter, :])
        calculatedPhaseDiffVectorIndex += 1

print("Shape of phase difference vectors : ", calculatedPhaseDiffVector.shape)
#calculate correlation with the reference data
correlationValues = np.zeros(phiR.shape[0])
print("Shape of phase reference vectors : ", phiR[0, :].shape)

print(phiR.shape[0])
for iter in range(phiR.shape[0]):
    correlationValues[iter] = correlatePhaseDiffVectors(calculatedPhaseDiffVector, phiR[iter, :])

#plotting original DOAs for comparison with peaks
fig, ax = plt.subplots(figsize=(10,4))
doa = np.array([85])
print("Original Directions of Arrival (degrees): \n",doa)
for k in range(len(doa)):
	plt.axvline(x=doa[k],color='red',linestyle='--')

#Plotting P_vals vs theta to find peaks
theta_vals = np.arange(0,180,1)

plt.plot(np.abs(theta_vals), correlationValues)
plt.xticks(np.arange(0, 180, 10))
plt.xlabel('$\\theta$')
plt.ylabel('$P(\\theta)$')
plt.title('Dotted Lines = Actual DOA    Peaks = Estimated DOA')

plt.legend()
plt.grid()
#else
plt.show()
