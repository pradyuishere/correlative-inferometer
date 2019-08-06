
import numpy as np
import matplotlib.pyplot as plt
from cmath import e,pi,sin,cos

N=5
M=10
p=100
fc=1e6
fs=1e7
c = 3e8
d = 150

#function to find the phase difference between two vectors
def phiXY(vecX, vecY):
    return np.angle(vecX.transpose()@vecY/vecX.shape[0])

#function to find the nC2 for a given n
def nc2(n):
    return int(n*(n-1)/2)

#reference phase difference vectors
phiR = np.load('phase_difference_vectors.npy')

#recieved signal data
X = np.load('recieved_signal_data.npy')
print("Recieved Signal X: ",X.shape)

#calculate phase difference vectors
calculatedPhaseDifferenceVectors = np.zeros([180, nc2(M)], dtype=complex)
calculatedPhaseDiffVectorsIndex = 0

for theta in phaseAngles:
    calculatedPhaseDiffVector = np.zeros(nc2(M), dtype=complex)
    calculatedPhaseDiffVectorIndex = 0
    for iter in range(M):
        for iter2 in range(M-iter-1):
            calculatedPhaseDiffVector[calculatedPhaseDiffVectorIndex] = phiXY(theta, iter, iter + iter2+1)
            calculatedPhaseDiffVectorIndex += 1
    calculatedPhaseDiffVectors[calculatedPhaseDiffVectorsIndex] = calculatedPhaseDiffVector
    calculatedPhaseDiffVectorsIndex += 1


#finding eigen values and eigen vectors
eigvals, eigvecs = np.linalg.eig(S)
#eigen values are real as S is Hermitian matrix
eigvals = eigvals.real

#sorting eig vals and eig vecs in decreasing order of eig vals
idx = eigvals.argsort()[::-1]   
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]

#Plotting Eigen Values
fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(np.arange(N),eigvals[:N],label="N EigVals from Source")
ax.scatter(np.arange(N,M),eigvals[N:],label="M-N EigVals from Noise")
plt.title('Visualize Source and Noise Eigenvalues')
plt.legend()

#separating source and noise eigvectors
Us, Un = eigvecs[:,:N], eigvecs[:,N:]
print("Source Eigen Values : Us: ",Us.shape)
print("Noise Eigen Values : Un: ",Un.shape)

#plotting original DOAs for comparison with peaks
fig, ax = plt.subplots(figsize=(10,4))
doa = np.array([20,50,85,110,145])
print("Original Directions of Arrival (degrees): \n",doa)
for k in range(len(doa)):
	plt.axvline(x=doa[k],color='red',linestyle='--')

def P(theta):
    return(1/(a(theta).conj().T@Un@Un.conj().T@a(theta)))[0,0]

#searching for all possible theta
theta_vals = np.arange(0,181,1)
P_vals = np.array([P(val*pi/180.0) for val in theta_vals]).real

#Plotting P_vals vs theta to find peaks
plt.plot(np.abs(theta_vals),P_vals)
plt.xticks(np.arange(0, 181, 10))
plt.xlabel('$\\theta$')
plt.ylabel('$P(\\theta)$')
plt.title('Dotted Lines = Actual DOA    Peaks = Estimated DOA')

plt.legend()
plt.grid()
#else
plt.show()
