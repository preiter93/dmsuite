import numpy as np 
import matplotlib.pyplot as plt
import time
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

n = 20
k = np.arange(1,n+1)
x = np.cos((k-1)*np.pi/(n-1))
#Input function
f=np.sin(np.pi*x)
#Analytically obtain the Derivative
df=np.pi*np.cos(np.pi*x)
df2=-np.pi**2*np.sin(np.pi*x)
# -- Differentiation via chebfft based on 
# -- chebdifft.m matlab version
def chebdifft(f,M):
	N=len(f)

	a = np.flipud(f[1:N-1])
	a = np.concatenate((f,a))
	a0 = np.fft.fft(a)

	ones = np.ones(N-2)
	a = np.concatenate(([0.5],ones,[0.5] ))
	a0 = a0[0:N]*a/(N-1)  #a0 contains Chebyshev coefficients of f
	#print(a0)

	# Recursion formula for computing coefficients of ell'th derivative 
	a = np.zeros((N,M+1),dtype="complex_")
	a[:,0] = a0
	for ell in np.arange(1,M+1):
		a[N-ell-1,ell]=2*(N-ell)*a[N-ell,ell-1];
		for k in np.arange(N-ell-2,0,-1):

			a[k,ell]=a[k+2,ell]+2*(k+1)*a[k+1,ell-1]
		a[0,ell]=a[1,ell-1]+a[2,ell]/2

	# Transform back to nphysical space
	b1 = [2*a[0,M]]
	b2 = a[1:N-1,M]
	b3 = [2*a[N-1,M]]
	b4 = np.flipud(b2)
	back = np.concatenate((b1,b2,b3,b4))
	Dmf = 0.5*np.fft.fft(back)
	# Real data in, real derivative out
	Dmf = Dmf[0:N]
	return np.real(Dmf)

t0 = time.time()
dfFFT = chebdifft(f,1)
dfFFT2 = chebdifft(f,2)
t1 = time.time()

print("Total time: {:}".format(t1-t0))
##Plot results
plt.plot(x, df.real, color='k', LineWidth=2, label='True Derivative')
# plt.plot(x, dfFD.real, '--', color='b', LineWidth=1.5, label='Finite Difference')
plt.plot(x, dfFFT.real, '--', color='c', LineWidth=1.5, label='Spectral Derivative')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

##Plot results
plt.plot(x, df2.real, color='k', LineWidth=2, label='True Derivative')
# plt.plot(x, dfFD.real, '--', color='b', LineWidth=1.5, label='Finite Difference')
plt.plot(x, dfFFT2.real, '--', color='c', LineWidth=1.5, label='2nd Spectral Derivative')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.ylim(-np.pi**2,+np.pi**2)
plt.show()



