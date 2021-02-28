import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def fourdif(nfou, mder):
    """
    Fourier spectral differentiation.
    Spectral differentiation matrix on a grid with nfou equispaced points in [0,2pi)
    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)
    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix
    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.
    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.
    """
    # grid points
    xxt = 2*np.pi*np.arange(nfou)/nfou
    # grid spacing
    dhh = 2*np.pi/nfou

    nn1 = np.int(np.floor((nfou-1)/2.))
    nn2 = np.int(np.ceil((nfou-1)/2.))
    if mder == 0:
        # compute first column of zeroth derivative matrix, which is identity
        col1 = np.zeros(nfou)
        col1[0] = 1
        row1 = np.copy(col1)

    elif mder == 1:
        # compute first column of 1st derivative matrix
        col1 = 0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack((0, col1))
        else:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = np.hstack((0, col1*np.hstack((topc, np.flipud(topc[0:nn1])))))
        # first row
        row1 = -col1

    elif mder == 2:
        # compute first column of 1st derivative matrix
        col1 = -0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)**2.
            col1 = col1*np.hstack((topc, np.flipud(topc[0:nn1])))
            col1 = np.hstack((-np.pi**2/3/dhh**2-1/6, col1))
        else:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack(([-np.pi**2/3/dhh**2+1/12], col1))

        # first row
        row1 = col1

    else:
        # employ FFT to compute 1st column of matrix for mder > 2
        nfo1 = np.int(np.floor((nfou-1)/2.))
        nfo2 = -nfou/2*(mder+1)%2*np.ones((nfou+1)%2)
        mwave = 1j*np.concatenate((np.arange(nfo1+1), nfo2, np.arange(-nfo1, 0)))
        col1 = np.real(np.fft.ifft(mwave**mder*np.fft.fft(np.hstack(([1], np.zeros(nfou-1))))))
        if mder%2 == 0:
            row1 = col1
        else:
            col1 = np.hstack(([0], col1[1:nfou+1]))
            row1 = -col1

    ddm = toeplitz(col1, row1)

    return xxt, ddm

ncheb = 3; pi = np.pi
x, D1 = fourdif(ncheb, 1)      # first derivatives
x, D2 = fourdif(ncheb, 2)      # second derivatives
x, D3 = fourdif(ncheb, 3)      # second derivatives
y = np.sin(x)              # function at Chebyshev nodes
yd = np.cos(x)    # theoretical first derivative
# print(x)
# print("")
# print(D1)
# plt.plot(x,D1@y,"k")
# plt.plot(x,yd,"r--")
# plt.show()

print(D3)