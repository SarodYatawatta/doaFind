import numpy as np

def ransac_line_fit_3d(points, threshold=10, max_trials=1000):
    """
    Find points that best fit a line using RANSAC
    
    Args:
        points: Nx3 array of 3D points
        threshold: Maximum distance from line to be considered inlier
        max_trials: Maximum RANSAC iterations
    
    Returns:
        inliers: Boolean mask of points on the line
        line_params: (point_on_line, direction_vector)
    """
    best_inliers = None
    best_count = 0
    
    for _ in range(max_trials):
        # Randomly select 2 points to define a line
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        
        # Line direction vector
        direction = (p2 - p1) / np.linalg.norm(p2 - p1)
        
        # Calculate distances from all points to the line
        distances = np.array([
            np.linalg.norm(np.cross(p - p1, direction))
            for p in points
        ])
        
        # Find inliers
        inliers = distances < threshold
        count = np.sum(inliers)
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_line = (p1, direction)
    
    return best_inliers, best_line


def is_present(array_list,new_array):
    """
    array_list: list of arrays
    new_array: array to compare

    check if new_array is already included in the array_list,
    if so return true, else false
    """
    for arr in array_list:
        if (arr==new_array).all():
           return True
    return False


def linearity_metrics(pts):
    P = np.asarray(pts, dtype=float)
    c = P.mean(axis=0)               # centroid
    Q = P - c
    # PCA / SVD
    _, s, vh = np.linalg.svd(Q, full_matrices=False)
    # singular values squared are eigenvalues (up to 1/(N-1))
    lam1, lam2, lam3 = s**2
    non_lin_ratio = (lam2 + lam3) / lam1

    u = vh[0]                        # direction of best line
    # Perpendicular distances
    dists = np.linalg.norm(np.cross(Q, u), axis=1)
    rms = np.sqrt((dists**2).mean())
    length = (Q @ u).ptp()           # span along the line
    rel_rms = rms / length if length else 0.0

    return dict(non_lin_ratio=non_lin_ratio,
                rms=rms,
                length=length,
                rel_rms=rel_rms)

n_stat=6
#xyz=np.loadtxt('layout_c032.txt',delimiter=' ',dtype=float)
#from scipy.io import savemat
#pos=xyz
#mydict={'pos':pos}
#savemat('pp.mat',mydict)
#np.save('skalow.npy',xyz)
xyz=np.load('skalow.npy')
lines=list()
for ci in range(80):
  best_liners,best_line=ransac_line_fit_3d(xyz,threshold=0.25)
  line=np.where(best_liners)
  line=line[0]
  N=line.size
  if N>n_stat:
      line1=line[:n_stat]
      line2=line[-n_stat:]
      if not is_present(lines,line1):
          lines.append(line1)
      if not is_present(lines,line2):
          lines.append(line2)

n_lines=len(lines)
linearity=np.zeros(n_lines)
lengths=np.zeros(n_lines)
ci=0
for line in lines:
  pts=xyz[line]
  result=linearity_metrics(pts)
  linearity[ci]=result['non_lin_ratio']
  lengths[ci]=result['length']
  ci = ci+1

max_length=80
max_nonlin=5e-3
print(linearity)
print(lengths)  
print(f'found {n_lines} lines')
fv=np.zeros((n_lines,n_stat),dtype=np.int32)
ci=0
for line in lines:
    fv[ci]=line
    ci=ci+1
fv=fv[linearity<max_nonlin]

do_plot=False
if do_plot:
   # exclude long arrays
   #fv=fv[lengths<max_length]
   # exclude nonlinear arrays
   n_selected=fv.shape[0]
   print(fv)
   import matplotlib.pyplot as plt

   fig=plt.figure()
   ax=fig.add_subplot(projection='3d')
   ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],marker='x')
   idx=0
   for ci in range(n_selected):
     if linearity[ci]<max_nonlin:
        ax.scatter(xyz[fv[ci],0],xyz[fv[ci],1],xyz[fv[ci],2],marker="$"+str(ci)+"$",s=100,c=(ci+1)*np.ones(n_stat)/11.0)
   plt.show()

n_arr=fv.shape[0]
### print as input to python
print('fv=np.array([')
for row in range(n_arr):
    print('           [',end='')
    for col in range(n_stat):
        print(str(fv[row,col])+', ',end='')
    print('],')
print('           ], dtype=np.int32)')
