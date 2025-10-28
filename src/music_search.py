import numpy as np

def steering_vector(az,el,xyz,wavelength):
    rfi_xyz=[np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)]
    
    # spatial array factor for full array
    v_a=xyz @ rfi_xyz
    af=np.exp(1j*2*np.pi/wavelength*v_a)
    # for full pol, replicate
    ubeam=np.ones(2)
    af_p=np.kron(ubeam,af)
    return af_p
 

def run_music(R_xx, array_coords, wavelength, az_grid, el_grid):
    music_spectrum=np.zeros((az_grid.size,el_grid.size))
    eig, evec = np.linalg.eig(R_xx)
    # sort and remove the dominant eigenvalue
    idx= eig.argsort()[::-1]
    # noise subspace
    evec=evec[:,idx]
    E=evec[:,1:]
    EE=np.matmul(E,np.conjugate(E.T))
    for ci in range(az_grid.size):
        for cj in range(el_grid.size):
           af_p=steering_vector(az_grid[ci],el_grid[cj],array_coords,wavelength)
           Ea=np.matmul(EE,af_p)
           aRa=np.matmul(np.conjugate(af_p.T),Ea)
           music_spectrum[ci,cj]=1/np.abs(aRa)
    
    return music_spectrum
