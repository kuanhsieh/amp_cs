# AMP for signal recovery in compressed sensing.
# CS model: y = Ax + noise
# Goal: recover x given y and A
# Copyright (c) 2021 Kuan Hsieh

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# Need to fix the partial fourier matrix thing.
# Shouldn't sample first row. And also can do DCT matrix,
# shouldn't sample row 1 and row N/2.

##### Compressed Sensing #####

def initialise_CS(N, M, K, sigma=0, x_choice=0, A_choice=0):
    ''' Initialises compressed sensing (CS) problem, y = Ax + noise.

    Inputs:
      N: Signal dimension
      M: Number of measurements
      K: Sparsity, number of non-zero entries in signal vector
      sigma: noise standard deviation
      A_choice: type of sensing matrix
          0: With iid normal entries N(0,1/M)
          1: With iid entries in {+1/sqrt(M), -1/sqrt(M)} with
             uniform probability
          2: Random partial Fourier matrix
      x_choice: type of signal
          0: Elements in {+1, 0, -1}, with K +-1's with
             uniform probability of +1 and -1's
          1: K non-zero entries drawn from the standard normal distribution

    Outputs:
        y: measurment vector
        A: sensing matrix
        x: signal vector
    '''

    # Generate signal x
    if x_choice == 0:
        x = 1*(np.random.rand(K)<0.5)
        x[np.logical_not(x)] = -1
        x = np.concatenate((x, np.zeros((N-K))))
    elif x_choice == 1:
        x = np.random.randn(K)
        x = np.concatenate((x, np.zeros((N-K))))
    else:
        print('x_choice must be either 0 or 1')
    np.random.shuffle(x)

    # Generate sensing matrix A
    if A_choice == 0:
        A = np.random.randn(M,N)/np.sqrt(M)
    elif A_choice == 1:
        A = np.random.rand(M,N)
        A = (A<0.5)/np.sqrt(M)
        A[np.logical_not(A)] = -1/np.sqrt(M)
    elif A_choice == 2:
        A = fft(np.eye(N)) # N-by-N DFT matrix
        rows = random.choice(N,size=M,replace=False) # Without replacement, include first row
        A = A[rows,:]/np.sqrt(M) # Normalize so that L2 norm of each column = 1
    else:
        # With odd number of rows cannot normaize columns to have unit norm
        print('M must be an even number for random partial Fourier sensing matrix ')
    # print(LA.norm(A,axis=0)) # Checks the norm of the columns are 1
    
    # Calculate measurement vector y
    y = np.dot(A,x)
    if sigma != 0:
        y += sigma * np.random.randn(M)

    return y, A, x

##### AMP #####

### AMP with soft-thresholding denoiser

def plus_op(x):
    '''Plus operator.

    plus_op(x) = x if x>0, = 0 otherwise'''
    return x*(x>0)

def soft_thresh(x, L):
    '''Soft-thresholding function.

    x is the signal, L is the threshold lambda
    x = sign(x)(abs(x)-lambda)_+
    ()_+ is the element-wise plus operator which equals
    the +ve part of x if x>0, otherwise = 0.'''
    return np.multiply(np.sign(x), plus_op(np.absolute(x)-L))

def opt_tuning_param(eps, limit=3):
    '''Find optimal tuning parameter for given sparsity ratio (epsilon) 
       for the AMP algorithm.

    Equation on p8 of "Graphical Models Concepts in Compressed Sensing"
    Inputs
        eps: epsilon = K/N = delta*rho
        limit: Limits the search for optimal tuning parameter alpha
    '''
    res   = minimize_scalar(M_pm,bracket=(0,limit),args=(eps))
    alpha = res.x
    return alpha

def M_pm(alpha, eps):
    return eps*(1+alpha**2) + (1-eps)*(2*(1+alpha**2)*norm.cdf(-alpha)-2*alpha*norm.pdf(alpha))

def amp(y, A, x, z, alpha):
    '''Approximate message passing (AMP) iteration 
       with soft-thresholding denoiser.
    
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        alpha: threshold tuning parameter
        
    Outputs
        x: signal estimate
        z: residual
        
    Note 
        Need to initialise AMP iteration with 
        x = np.zeros(N)
        z = y
    '''
    
    M = len(y)
    
    # Estimate vector
    theta = alpha*np.sqrt(LA.norm(z)**2/M) # alpha*tau
    x     = soft_thresh(x + np.dot(A.T,z), theta)
    
    # Calculate residual with the Onsager term
    b = LA.norm(x,0)/M
    z = y - np.dot(A,x) + b*z
    
    # L = theta*(1 - b) # The last L is the actual lambda of the LASSO we're minimizing

    return (x, z)

### AMP with Bayes-optimal denoiser for different signals

def amp_3pt(y, A, x, z, eps, c=1):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE)
    denoiser for signals with iid entries drawn from the 3-point distribution:
    probability (1-eps) equal to 0 and probability eps/2 equal to each +c and 
    -c.

    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
        c: the values of the non-zero entries of the signal equal to +c and -c
        
    Outputs
        x: signal estimate
        z: residual
        
    Note 
        Need to initialise AMP iteration with 
        x = np.zeros(N)
        z = y
    '''
        
    M,N = np.shape(A)
    tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation
    
    # Estimate vector
    s = x + np.dot(A.T, z) # Effective (noisy) observation of signal x
    u = s*c / tau**2       # Temporary variable
    top = c * eps*np.sinh(u,dtype=np.float128) 
    bot = eps*np.cosh(u,dtype=np.float128) + (1-eps)*np.exp(c**2/(2*tau**2),dtype=np.float128)
    x   = (top / bot).astype(np.float)
    
    # Calculate residual with the Onsager term
    b = (N/M) * np.mean(x * (c/np.tanh(u) - x)) / tau**2
    z = y - np.dot(A,x) + b*z

    return (x, z)

def amp_bg(y, A, x, z, eps, v=1):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE)
    denoiser for signals with iid entries drawn from the Bernoulli-Gaussian 
    distribution: probability (1-eps) equal to 0 and probability eps drawn
    from a Gaussian distribution with standard deviation v.

    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
        v: the standard deviation of the non-zero entries of the signal which
           are drawn from a Gaussian distribution
        
    Outputs
        x: signal estimate
        z: residual
        
    Note 
        Need to initialise AMP iteration with 
        x = np.zeros(N)
        z = y
    '''
        
    M,N = np.shape(A)
    tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation
    
    # Estimate vector
    s = x + np.dot(A.T, z)   # Effective (noisy) observation of signal x
    u = v**2 / tau**2        # Temporary variable
    term1 = 1 + tau**2/v**2
    term2 = 1 + (1-eps)/eps * np.sqrt(1+u) * np.exp(-(s**2/(2*tau**2))*u/(1+u))
    denom = term1 * term2
    x = s / denom
    
    # Calculate residual with the Onsager term
    eta_der = (1/denom) + (x/tau**2) * (s/(1+tau**2/v**2) - x)
    b = (N/M) * np.mean(eta_der)
    z = y - np.dot(A,x) + b*z

    return (x, z)