#! /usr/bin/env python

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.matlib
import torch
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import argparse

from replaybuffer import ReplayBuffer,ReplayBuffer3D
import telescope_config
import phase_utils
import sparsefit

fig=plt.figure(1)
fig.tight_layout()
font={'family': 'serif',
      'color': 'black',
      'weight': 'normal',
      'size': 12,
      }

# Env for simulating an RFI signal for DOA estimation 
# using the full array, suitable for sparse DOA comparison

VERBOSE=False

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

class RFISparse(gym.Env):
    metadata={'render.modes':['human']}
    # nearfield RFI range (m)
    R_LOW=10e3
    R_HIGH=10000e3

    def __init__(self,T=1000,buffer_size=20000,telescope='A12',nfraction=0.3,simulate_range=False):
        super(RFISparse,self).__init__()

        self.n_time=T
        # sample ~1sec
        self.time_coord=np.arange(0,1,1/self.n_time)

        self.SNR=100

        self.telescope=telescope
        assert(self.telescope=='A12' or self.telescope=='SKA')

        # random parameters are used for: 
        # RFI direction, power(fixed), polarization, frequency
        self.rfi_phi=0 # [0,2pi]
        self.rfi_theta=0 # [0,pi/2]
        self.rfi_E0=10
        self.rfi_pol_gamma=0 # [0,pi/2]
        self.rfi_pol_eta=0 # [-pi,pi]
        self.rfi_freq=100e6 # [10,180] MHz
        self.wavelength=3e8/self.rfi_freq
        self.rfi_xyz=None
        # Beam polarization
        self.rfi_beam_E=np.eye(2)

        # if simulate_range, sample in range as well 
        # in addition to az, el
        self.simulate_range=simulate_range

        # nearfield RFI parameters
        if self.simulate_range:
           self.nf_fraction=1.0
           self.nearfield=True
        else:
           self.nf_fraction=nfraction
           self.nearfield=False

        # Array settings
        if self.telescope=='SKA':
           self.N=6 # select stations for each subarray
           self.L=53 # how many sub arrays
        else:
           self.N=6 # select 6 stations for each subarray
           self.L=48 # how many sub arrays
 
        self.B=self.N*(self.N-1)//2

        if self.telescope=='SKA':
           self.position_file='skalow.npy'
        else:
           self.position_file='a12pos.npy'
        self.antpos=None
        self.fv=None
        self.Rxx=None

        # evaluation settings
        self.n_grid=128
        self.patch_size=16
        # if range is simulated (simulate_range=True)
        if self.simulate_range:
           self.n_range=4
           self.log_r=np.log(np.arange(self.R_LOW,self.R_HIGH,(self.R_HIGH-self.R_LOW)/self.n_range))
        else:
           self.n_range=1

        # variables to store and pass
        self.arr_dir=None
        self.arr_distance=None
        self.arr_sintheta=None

        # replaybuffer
        self.buffer_size=buffer_size
        if self.simulate_range:
           self.buffer=ReplayBuffer3D(self.buffer_size,self.L,self.N,self.n_grid,self.n_range)
        else:
           self.buffer=ReplayBuffer(self.buffer_size,self.L,self.N,self.n_grid)

    def __reset_rfi(self):
        self.rfi_phi=np.random.random_sample()*2*np.pi
        self.rfi_theta=np.random.random_sample()*np.pi/2
        self.rfi_pol_gamma=np.random.random_sample()*np.pi/2
        self.rfi_pol_eta=np.random.random_sample()*2*np.pi-np.pi
        self.rfi_freq=np.random.random_sample()*170e6+10e6
        self.wavelength=3e8/self.rfi_freq
        self.rfi_beam_E=np.random.randn(2,2)+1j*np.random.randn(2,2)

        if np.random.rand() < self.nf_fraction:
            self.nearfield=True
            # range is calculated from array centroid
            self.rfi_range=np.random.random_sample()*(self.R_HIGH-self.R_LOW)+self.R_LOW
        else:
            self.nearfield=False
            self.rfi_range=np.inf

        # direction vector
        rfi_x=np.cos(self.rfi_theta)*np.cos(self.rfi_phi)
        rfi_y=np.cos(self.rfi_theta)*np.sin(self.rfi_phi)
        rfi_z=np.sin(self.rfi_theta)
        self.rfi_xyz=np.array([rfi_x, rfi_y, rfi_z])

    def __load_array(self):
        # Load full array antenna positions, also select the sub-arrays
        self.antpos,self.fv=telescope_config.load_array(position_file=self.position_file,L=self.L,N=self.N,telescope=self.telescope)

    def __get_bl_pos(self,idx):
        # get centroid of each baseline (m)
        xyz=np.zeros((self.N,3))
        ants=np.zeros(self.N,dtype=int)
        for ci in range(self.N):
          xyz[ci]=self.antpos[self.fv[idx,ci]]
          ants[ci]=self.fv[idx,ci]

        # find the mapping in the true array to Rxx
        fvu=np.unique(self.fv.reshape(-1,1))
        N=fvu.size

        # iterate over baselines (cross-corr) for this array
        bl=0
        bl_pos=np.zeros((self.B,3))
        for p in range(self.N-1):
            # find position of p in fvu
            pidx=np.where(fvu==ants[p])[0][0]
            for q in range(p+1,self.N):
                qidx=np.where(fvu==ants[q])[0][0]
                cen_xyz=(xyz[q]+xyz[p])/2
                bl_pos[bl]=cen_xyz
                bl += 1

        return bl_pos

    def __process_array(self,idx):
        # - simulate RFI signal for given array
        # - perform correlation - average
        # - find dominant eigenvector
        # - estimate array factor
        # - calculate phase projection transforms
        # - get projected phase
        # return:
        # bl_direction: baseline direction vectors Bx3
        # sintheta: sin(theta) estimage Bx1
        # bl_distance: baseline distance (wavelangths/2) Bx1

        xyz=np.zeros((self.N,3))
        ants=np.zeros(self.N,dtype=int)
        for ci in range(self.N):
          xyz[ci]=self.antpos[self.fv[idx,ci]]
          ants[ci]=self.fv[idx,ci]

        # find the mapping in the true array to Rxx
        fvu=np.unique(self.fv.reshape(-1,1))
        N=fvu.size

        # iterate over baselines (cross-corr) for this array
        bl=0
        bl_distance=np.zeros((self.B))
        bl_psi=np.zeros((self.B))
        bl_direction=np.zeros((self.B,3))
        for p in range(self.N-1):
            # find position of p in fvu
            pidx=np.where(fvu==ants[p])[0][0]
            for q in range(p+1,self.N):
                qidx=np.where(fvu==ants[q])[0][0]
                del_xyz=xyz[q]-xyz[p]
                # baseline distance
                delta=np.linalg.norm(del_xyz)
                # store baseline distance (normalized by half wavelength)
                bl_distance[bl]=delta/(0.5*self.wavelength)
                # store unit vector of baseline
                bl_direction[bl]=del_xyz/delta

                # correlation, 4x4, extract from full correlation
                R=np.zeros((4,4),dtype=complex)
                # Since : Rxx = [ XX, XY; YX, YY ] block matrices,
                # need to select values from correct quadrant
                R[0,0]=self.Rxx[pidx,pidx]
                R[1,0]=self.Rxx[pidx+N,pidx]
                R[0,1]=self.Rxx[pidx,pidx+N]
                R[1,1]=self.Rxx[pidx+N,pidx+N]
                R[2,2]=self.Rxx[qidx,qidx]
                R[3,2]=self.Rxx[qidx+N,qidx]
                R[2,3]=self.Rxx[qidx,qidx+N]
                R[3,3]=self.Rxx[qidx+N,qidx+N]
                R[0,2]=self.Rxx[pidx,qidx]
                R[1,2]=self.Rxx[pidx+N,qidx]
                R[0,3]=self.Rxx[pidx,qidx+N]
                R[1,3]=self.Rxx[pidx+N,qidx+N]
                R[2,0]=self.Rxx[qidx,pidx]
                R[2,1]=self.Rxx[qidx,pidx+N]
                R[3,0]=self.Rxx[qidx+N,pidx]
                R[3,1]=self.Rxx[qidx+N,pidx+N]

                rv,rV=np.linalg.eig(R)
                # find the max eigenvalue (assume positive and real, because R is Hermitian)
                imax=np.argmax(rv)
                Es=rV[:,imax]
                Ex=Es[0:2]
                Ey=Es[2:4]
                # Ey = Ex Psi, so Psi = pinv(Ex)*Ey 
                # = (Ex' Ex)^-1 Ex' Ey =  Ex' Ey / (Ex' Ex)
                Psi= np.dot(np.conj(Ex),Ey)/np.dot(np.conj(Ex),Ex)
                if VERBOSE:
                   print(f'{bl} array factor angle estimated {np.angle(Psi)}')
                bl_psi[bl]=np.angle(Psi)

                bl += 1
        H,P=phase_utils.calculate_projection_points(bl_distance)
        K=len(H)

        phat=bl_psi-np.dot(bl_distance,bl_psi)*bl_distance/np.dot(bl_distance,bl_distance)
        # compare phat over all P{} and find closest
        pp=np.zeros((K))
        for ci in range(K):
           pp[ci]=np.linalg.norm(P[ci]-phat)

        zi=np.argmin(pp)
        pz=P[zi]
        psi_tilde=pz+np.dot(bl_distance,bl_psi)*bl_distance/np.dot(bl_distance,bl_distance)
        phi_hat=psi_tilde+H[zi]
        sintheta=phi_hat/(np.pi*bl_distance)

        return bl_direction,sintheta,bl_distance


    def __total_error(self,directions,sintheta,theta,phi):
        # theta, phi: grid vectors Ngrid
        # directions: direction vectors B x 3 (B=total baselines of all arrays)
        # sintheta: sin(theta) estimate, B x 1
        # output: cost, theta, phi (Ntheta, Nphi, 3) 
        Ntheta=theta.size
        Nphi=phi.size
        F=np.zeros((3,Nphi,Ntheta),dtype=np.float32)
        for ci in range(Ntheta):
            for cj in range(Nphi):
                doa=[np.cos(theta[ci])*np.cos(phi[cj]), np.cos(theta[ci])*np.sin(phi[cj]), np.sin(theta[ci])]
                fval=(np.matmul(directions,doa)**2-(sintheta**2))**2
                F[0,cj,ci]=np.sum(fval)
                F[1,cj,ci]=theta[ci]
                F[2,cj,ci]=phi[cj]

        # normalize channels 
        F[0]=F[0]/np.max(np.abs(F[0]))
        F[1]=F[1]/(np.pi)
        F[2]=F[2]/(np.pi)
        return F

    def __total_error_withrange(self,directions,positions,sintheta,theta,phi,logrange):
        # theta, phi: grid vectors Ngrid
        # logrange : log(range) Nr != Ngrid
        # directions: direction vectors B x 3 (B=total baselines of all arrays)
        # positions: centroid position B x 3 of each baseline
        # sintheta: sin(theta) estimate, B x 1
        # output: cost, theta, phi (Ntheta, Nphi, 3) 
        Ntheta=theta.size
        Nphi=phi.size
        Nr=logrange.size
        r=np.exp(logrange)+self.array_cen
        F=np.zeros((4,Nphi,Ntheta,Nr),dtype=np.float32)
        B=positions.shape[0]
        for ci in range(Ntheta):
            for cj in range(Nphi):
                shat=np.array([np.cos(theta[ci])*np.cos(phi[cj]), np.cos(theta[ci])*np.sin(phi[cj]), np.sin(theta[ci])])
                for cr in range(Nr):
                    # find unit vec =R s^ - (pos), for all baselines
                    rr=np.matlib.repmat(r[cr]*shat,B,1)-positions
                    ff=np.zeros(B)
                    for b in range(B):
                        ff[b]=np.dot(rr[b],directions[b])/(np.linalg.norm(rr[b])+1e-6)
                    fval=(ff**2-sintheta**2)**2
                    F[0,cj,ci,cr]=np.sum(fval)
                    F[1,cj,ci,cr]=theta[ci]
                    F[2,cj,ci,cr]=phi[cj]
                    F[3,cj,ci,cr]=logrange[cr]

        # normalize channels 
        for cr in range(Nr):
            F[0,:,:,cr]=F[0,:,:,cr]/np.max(np.abs(F[0,:,:,cr]))
        F[1]=F[1]/(np.pi)
        F[2]=F[2]/(np.pi)
        return F

    def __simulate(self):
        # Build full correlation matrix
        # find unique stations
        fvu=np.unique(self.fv.reshape(-1,1))
        N=fvu.size
        xyz=np.zeros((N,3))
        for ci in range(N):
          xyz[ci]=self.antpos[fvu[ci]]
        # centroid of array
        xyz_cen=np.mean(xyz,axis=0)
        # distance to centroid of array
        self.array_cen=np.linalg.norm(xyz_cen)
       
        # spatial array factor for full array
        v_a=xyz @ self.rfi_xyz
        af=np.exp(1j*2*np.pi/self.wavelength*v_a)
        # source signal
        # polarization vector
        u=np.array([-np.cos(self.rfi_pol_gamma), np.sin(self.rfi_pol_gamma)*np.exp(1j*self.rfi_pol_eta)])
        ubeam=np.matmul(self.rfi_beam_E , u)
        # combined product
        A=np.kron(ubeam,af)
        # source signal (unplolarized)
        phi0=np.random.random_sample()*2*np.pi
        ef=self.rfi_E0*np.exp(1j*(2*np.pi*self.rfi_freq*self.time_coord+phi0))
        # noiseless signal received by full array
        noiseless_signal= np.outer(A , ef)

        noise=np.random.randn(2*N,self.n_time)+1j*np.random.randn(2*N,self.n_time)
        v=noiseless_signal+(self.rfi_E0/self.SNR)*noise

        R=np.matmul(v,np.conjugate(v.T))/self.n_time
        # Note: Rxx = [ XX, XY; YX, YY ] block matrices
        # each block size N x N
        self.Rxx=R

    def __simulate_nearfield(self):
        # Build full correlation matrix
        # find unique stations
        fvu=np.unique(self.fv.reshape(-1,1))
        N=fvu.size
        xyz=np.zeros((N,3))
        for ci in range(N):
          xyz[ci]=self.antpos[fvu[ci]]

        # centroid of array
        xyz_cen=np.mean(xyz,axis=0)
        # distance to centroid of array
        self.array_cen=np.linalg.norm(xyz_cen)
        # distance to source from coordinate origin
        rfi_range=self.rfi_range+self.array_cen
        assert(np.isfinite(rfi_range))
        # source xyz
        rfi_xyz = rfi_range*self.rfi_xyz
        # distance from array positions to source
        distances=np.sqrt((xyz[:,0]-rfi_xyz[0])**2+(xyz[:,1]-rfi_xyz[1])**2+(xyz[:,2]-rfi_xyz[2])**2)
        # delays to each receiver 
        C=3e8
        delays=distances/C
        
        # spatial array factor for full array (freq=C/wavelength)
        attenuation=rfi_range/distances
        af=np.exp(1j*2*np.pi*self.rfi_freq*delays)*attenuation

        # source signal
        # polarization vector
        u=np.array([-np.cos(self.rfi_pol_gamma), np.sin(self.rfi_pol_gamma)*np.exp(1j*self.rfi_pol_eta)])
        ubeam=np.matmul(self.rfi_beam_E , u)
        # combined product
        A=np.kron(ubeam,af)
        # source signal (unplolarized)
        phi0=np.random.random_sample()*2*np.pi
        ef=self.rfi_E0*np.exp(1j*(2*np.pi*self.rfi_freq*self.time_coord+phi0))
        # noiseless signal received by full array
        noiseless_signal= np.outer(A , ef)

        noise=np.random.randn(2*N,self.n_time)+1j*np.random.randn(2*N,self.n_time)
        v=noiseless_signal+(self.rfi_E0/self.SNR)*noise

        R=np.matmul(v,np.conjugate(v.T))/self.n_time
        # Note: Rxx = [ XX, XY; YX, YY ] block matrices
        # each block size N x N
        self.Rxx=R

    def process(self):
        # simulate full array to create full correlation matrix
        if self.nearfield:
           self.__simulate_nearfield()
        else:
           self.__simulate()

        # vectors to store results
        self.arr_dir=np.zeros((self.L*self.B,3),dtype=np.float32)
        self.arr_sintheta=np.zeros((self.L*self.B),dtype=np.float32)
        self.arr_distance=np.zeros((self.L),dtype=np.float32)

        if self.simulate_range:
            arr_pos=np.zeros(self.arr_dir.shape)

        for idx in range(self.L):
            if self.simulate_range:
               bl_pos=self.__get_bl_pos(idx)
               arr_pos[idx*self.B:(idx+1)*self.B]=bl_pos

            bl_direction,sintheta,bl_distance=self.__process_array(idx)
            self.arr_dir[idx*self.B:(idx+1)*self.B]=bl_direction
            self.arr_sintheta[idx*self.B:(idx+1)*self.B]=sintheta
            self.arr_distance[idx]=np.log(bl_distance[0]) # only copy first value
        theta=np.arange(0,np.pi/2,np.pi/2/self.n_grid)
        phi=np.arange(0,2*np.pi,2*np.pi/self.n_grid)
        if self.simulate_range:
            self.arr_F=self.__total_error_withrange(self.arr_dir,arr_pos,self.arr_sintheta,theta,phi,self.log_r)
        else:
            self.arr_F=self.__total_error(self.arr_dir,self.arr_sintheta,theta,phi)

        if self.simulate_range:
            self.buffer.store_observation(self.arr_F,self.arr_dir[::self.B],self.arr_distance,self.arr_sintheta[::self.B], np.array([self.rfi_theta, self.rfi_phi, np.log(min(self.rfi_range,self.R_HIGH))]), self.rfi_freq)
        else:
            self.buffer.store_observation(self.arr_F,self.arr_dir[::self.B],self.arr_distance,self.arr_sintheta[::self.B], np.array([self.rfi_theta, self.rfi_phi]), self.rfi_freq)

    def sparse_fit(self,filename='R.png'):
        fvu=np.unique(self.fv.reshape(-1,1))
        N=fvu.size
        xyz=np.zeros((N,3))
        for ci in range(N):
          xyz[ci]=self.antpos[fvu[ci]]

        az_grid=np.arange(0,361,10)
        el_grid=np.arange(0,91,10)
        lambda_reg=10

        total_power=sparsefit.sparsity_doa(
                self.Rxx, xyz, self.wavelength, az_grid, el_grid, lambda_reg
        )
        plt.clf()
        plt.imshow(total_power,aspect='auto',origin='lower',extent=[0,np.pi/2,0,2*np.pi])
        plt.xlabel('Elevation (rad)',fontdict=font)
        plt.ylabel('Azimuth (rad)',fontdict=font)
        plt.savefig(filename)


    def reset(self):
        self.__reset_rfi()
        self.__load_array()

        observation=1
        return observation

    def sample(self,batch_size):
        if batch_size > self.buffer.mem_cntr:
          return

        return self.buffer.sample_observation(batch_size,self.patch_size)

    def step(self,action):
        done=False
        info={}
        reward=0
        observation=1

        return observation, reward, done, info
        

    def render(self,filename_prefix='pp',mode='human'):
        if self.simulate_range:
           print(f'RFI range 3D {self.rfi_range/1e3} km DOA {self.rfi_theta} {self.rfi_phi} (rad) pol {self.rfi_pol_gamma} {self.rfi_pol_eta} (rad) freq {self.rfi_freq/1e6} MHz')
        else:
           print(f'RFI range {self.rfi_range/1e3} km DOA {self.rfi_theta} {self.rfi_phi} (rad) pol {self.rfi_pol_gamma} {self.rfi_pol_eta} (rad) freq {self.rfi_freq/1e6} MHz')
        for ci in range(self.n_range):
           plt.clf()
           if self.simulate_range:
              plt.imshow(self.arr_F[0,:,:,ci],aspect='auto',origin='lower',extent=[0,np.pi/2,0,2*np.pi])
           else:
              plt.imshow(self.arr_F[0,:,:],aspect='auto',origin='lower',extent=[0,np.pi/2,0,2*np.pi])
           plt.xlabel('Elevation (rad)',fontdict=font)
           plt.ylabel('Azimuth (rad)',fontdict=font)
           cbar=plt.colorbar()
           cbar.set_label('Cost',fontdict=font)
           if self.simulate_range:
              plt.suptitle(f'Range {np.exp(self.log_r[ci])/1e3} {self.rfi_range/1e3:.2f} km Freq {self.rfi_freq/1e6:.2f} MHz Pol {self.rfi_pol_gamma:.2f} {self.rfi_pol_eta:.2f} rad')
           else:
              plt.suptitle(f'Range {self.rfi_range/1e3:.2f} km Freq {self.rfi_freq/1e6:.2f} MHz Pol {self.rfi_pol_gamma:.2f} {self.rfi_pol_eta:.2f} rad')
           plt.scatter(self.rfi_theta,self.rfi_phi,color='r')
           filename=filename_prefix+'_'+str(ci)+'.png'
           plt.savefig(filename)


    def close(self):
        pass


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Generate signal for DOA estimation',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--array',default="A12",type=str,metavar='s',
                        help='which array to use: A12 or SKA')
    parser.add_argument('--seed',default=0,type=int,metavar='i',
       help='random seed to use')
    parser.add_argument('--episodes',default=10000,type=int,metavar='i',
       help='number of episodes to simulate')
    parser.add_argument('--render', action='store_true', default=False,
       help='produce graphical output (slow)')
    parser.add_argument('--sparsefit', action='store_true', default=False,
       help='solve sparsity constrained DOA (slow)')
    parser.add_argument('--nearfield_fraction',default=0.3,type=float,metavar='f',
       help='fraction (out of 1) of nearfield simulations')
    parser.add_argument('--simulate_range', action='store_true', default=False,
       help='generate data including the range (3 dimensional, 100% nearfield)')

    args=parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env=RFISparse(T=1000,buffer_size=args.episodes,telescope=args.array,nfraction=args.nearfield_fraction,simulate_range=args.simulate_range)

    for loop in range(args.episodes):
       env.reset()
       env.process()
       if args.sparsefit:
          env.sparse_fit(filename='R_'+str(loop)+'.png')

       if loop%1000==0:
          print(f'iteration {loop}')
          env.buffer.save_checkpoint()

       if args.render:
          env.render(filename_prefix='pp_'+str(loop))

    env.buffer.save_checkpoint()
