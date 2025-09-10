import numpy as np

def ph_wrap(phase0):
    # Phase wrapping
    return np.mod(phase0+np.pi,2*np.pi)-np.pi

def project_nd(f, p):
    # Project point p onto plane f' x = 0, all in R^M
    # x: coordinate vector
    # p1 <= p - (f' p)/||f|| f/||f||, ||f||^2=f' f
    t0=-np.dot(f,p)/np.dot(f,f)
    return p+ f*t0
        
def calculate_projection_points(distance_vector):
    # Algorithm 1, [Hui Chen et al. 2021]
    # in: distance vector (normalized by half wavelength)
    phi0=np.pi*np.sin(-np.pi/2)*distance_vector
    phimax=np.pi*np.sin(np.pi/2)*distance_vector
    h0=phi0-ph_wrap(phi0)
    psi0=ph_wrap(phi0)

    # storage
    hk=list()
    pk=list()
    cj=1
    EPS=1e-6
    #replace np.all(phimax > phi0) with a threshold
    while np.all(phimax - phi0 > EPS):
            hk.append(h0.copy())
            pk.append(project_nd(distance_vector,psi0))
            idx=np.argmin((np.pi-psi0)/distance_vector)
            psi0=distance_vector*(np.pi-psi0[idx])/(distance_vector[idx]) + psi0
            psi0[idx]=psi0[idx]-2*np.pi
            h0[idx]=h0[idx]+2*np.pi
            phi0=psi0+h0
            ci=cj+1

    return hk,pk
