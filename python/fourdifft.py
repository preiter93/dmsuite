import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})
n=20
L=2*np.pi
dx=L/n
x = np.arange(0, L, dx,dtype="complex_")

#Create a Function
#f=np.cos(x) * np.exp(-np.power(x,2)/25)    
f=np.sin(x)
#Analytically obtain the Derivative
#df= -(np.sin(x) * np.exp(-np.power(x,2)/25 + (2/25)*x*f))
df=np.cos(x)

##Approximate derivative by FFT
fhat = np.fft.fft(f)
kappa = (2*np.pi/L)*np.arange(-n/2,n/2)

#Re-order fft frequencies
kappa = np.fft.fftshift(kappa) 
#Obtain real part of the function for plotting
dfhat = kappa*fhat*(1j)
#Inverse Fourier Transform
dfFFT = np.real(np.fft.ifft(dfhat))

##Plot results
plt.plot(x, df.real, color='k', LineWidth=2, label='True Derivative')
# plt.plot(x, dfFD.real, '--', color='b', LineWidth=1.5, label='Finite Difference')
plt.plot(x, dfFFT.real, '--', color='c', LineWidth=1.5, label='Spectral Derivative')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()