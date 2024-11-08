"""
Finite Difference Heat Solver
Author: @SpencerDFrancis

Everything that is needed to solve the 1D Heat Equation using the finite
difference method in 1D and 2D.
"""

import numpy as np
import matplotlib.pyplot as plt


def fd1dNeumman(t_span, x_span,c,u0,duL,u_initial = 0):
    
    N = np.size(x_span)
    if u_initial == 0:
        u_initial = np.zeros(N)
    
    # Inialize array of 
    u_span = np.zeros( (np.size(t_span),N) )
    
    # Set boundary conditions
    u_span[0] = u_initial
    u_span[:,0] = u0
    for i in range(1,np.size(t_span)):
        g = t_span[i]-t_span[i-1]
        u_span[i,N-1] = u_span[i-1,N-1] + g*duL
    
    # We itterate through all elements of u
    for i in range(np.size(t_span)-1):
        for j in range(1,N-1):
            g = t_span[i]-t_span[i-1]
            h1 = x_span[j]-x_span[j-1]
            h2 = x_span[j+1]-x_span[j]
            
            u_span[i+1,j] = g*c*(u_span[i,j+1] -2*u_span[i,j] + u_span[i,j-1])/(h1*h2) + u_span[i,j]
            
    return u_span


def fd1dDirichlet(t_span,x_span,u_initial,c,u0,uL):
    N = np.size(x_span)
    if u_initial == 0:
        u_initial = np.zeros(N)
    
    # Inialize array of value
    u_span = np.zeros( (np.size(t_span),N) )
    
    # Set boundary conditions
    u_span[0] = u_initial
    u_span[:,0] = u0
    u_span[:,N] = uL
    
    # We itterate through all elements of u
    for i in range(np.size(t_span)-1):
        for j in range(1,N-1):
            g = t_span[i]-t_span[i-1]
            h1 = x_span[j]-x_span[j-1]
            h2 = x_span[j+1]-x_span[j]
            
            u_span[i+1,j] = g*c*(u_span[i,j+1] -2*u_span[i,j] + u_span[i,j-1])/(h1*h2) + u_span[i,j]
            
    return u_span

def test():
    t_span = np.linspace(0,1,10000)
    x_span = np.linspace(0,1,10)
    
    u_span = fd1dNeumman(t_span, x_span, 0.05, 100, 0)
    
    
    plt.plot(x_span, u_span[0,:])
    plt.plot(x_span, u_span[5,:])
    plt.plot(x_span, u_span[25,:])
    plt.plot(x_span, u_span[60,:])
    plt.plot(x_span, u_span[99,:])
    plt.show()
    return

def main():
    return


if __name__ == "__main__":
    test()