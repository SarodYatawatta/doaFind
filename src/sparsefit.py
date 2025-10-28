import numpy as np
try:
    import cvxpy as cp
except ImportError:
    print('CVXPY not installed, some routines will not work')

def steering_vector_dual_pol(az, el, array_coords, wavelength):
    """
    Calculates the V and H steering vectors for a dual-polarized array.

    Args:
        N : total receivers
        az (float): Azimuth angle in rad.
        el (float): Elevation angle in rad.
        array_coords (np.ndarray): A (N x 3) numpy array of element coords.
        wavelength (float): Wavelength of the signal.

    Returns:
        tuple: A tuple containing (a_v, a_h), the vertical (2N x 1) and
               horizontal (2N x 1) steering vectors.
    """
    N = array_coords.shape[0]
    
    # Standard spatial steering vector (N x 1)
    k = 2 * np.pi / wavelength
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    v_a = np.exp(-1j * k * (array_coords @ u)).reshape(-1, 1)

    # Polarization basis vectors (2 x 1)
    v_p_V = np.array([[1], [0]]) # Vertical
    v_p_H = np.array([[0], [1]]) # Horizontal

    # Combine using Kronecker product to get the full 2N x 1 steering vectors
    a_v = np.kron(v_p_V, v_a)
    a_h = np.kron(v_p_H, v_a)
    
    return a_v, a_h

def sparsity_doa(R_xx, array_coords, wavelength, az_grid, el_grid, lambda_reg):
    """
    Performs 2D DOA estimation from the covariance matrix of a dual-polarized array.

    y = A s + n, 
    N: stations, Ng: grid points in 1 dimension, total: Ng^2
    A : steering vectors for all directions to search N x Ng^2
    s: sparse source vector Ng^2 possible locations, 
    y: voltage, N for single polarization

    R=E{ y y^H } : NxN 
    R=A (s s^H) A^H, r=vec(R)=vec(A (s s^H) A^H) = A^* \\kron A vec(s s^H)
    r = A^* \\kron A vec( s s^H) = A^ s^
    r: N^2,
    A^ : N^2 x Ng^4
    s^ : Ng^4 sparse vector
    """
    num_elements = array_coords.shape[0]
    num_total_channels = 2 * num_elements
    
    # 1. Vectorize the sample covariance matrix (now 4N^2 x 1)
    r_vec = R_xx.T.flatten().reshape(-1, 1)

    # 2. Build the new dictionary matrix (B_grid) for dual polarization
    grid_atoms = []
    for el in el_grid:
        for az in az_grid:
            # Get both V and H steering vectors for this grid point
            a_v, a_h = steering_vector_dual_pol(az, el, array_coords, wavelength)
            
            # Create the two corresponding atoms and add them to the dictionary
            # elements of A^* \kron A
            atom_v = np.kron(a_v.conj(), a_v)
            atom_h = np.kron(a_h.conj(), a_h)
            grid_atoms.append(atom_v)
            grid_atoms.append(atom_h)
            
    B_grid = np.hstack(grid_atoms)
    num_grid_points = B_grid.shape[1]
    print(f"problem size {r_vec.shape} {B_grid.shape}")

    # 3. Set up the convex optimization problem (LASSO)
    # min || r_vec - B_grid s^ ||_2^2 + \lambda ||s^||_1, s^ is sparse
    # 
    # The variable 'p' contains interleaved V and H powers. It must be non-negative.
    p_grid = cp.Variable((num_grid_points, 1), nonneg=True)
    data_fidelity = cp.sum_squares(r_vec - B_grid @ p_grid)
    sparsity_term = cp.norm(p_grid, 1)
    objective = cp.Minimize(data_fidelity + lambda_reg * sparsity_term)

    # 4. Solve the problem
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Solver failed with status: {problem.status}")
        return np.zeros((len(el_grid), len(az_grid)))

    # 5. De-interleave and reshape the recovered power vector
    p_recovered = p_grid.value
    p_v_flat = p_recovered[0::2] # Power in Vertical channel
    p_h_flat = p_recovered[1::2] # Power in Horizontal channel

    power_spectrum_v = np.abs(p_v_flat).reshape(len(el_grid), len(az_grid))
    power_spectrum_h = np.abs(p_h_flat).reshape(len(el_grid), len(az_grid))
    
    # Return total power
    return power_spectrum_v + power_spectrum_h
