#! /usr/bin/env python

import numpy as np
import torch
from torch import nn
import argparse

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

fig=plt.figure(1)
fig.tight_layout()
font={'family': 'serif',
      'color': 'black',
      'weight': 'normal',
      'size': 12,
      }

from replaybuffer import ReplayBuffer,ReplayBuffer3D
from transformer import ManyAttention

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


def render(array_F,theta0,phi0,theta,phi,freq,filename='pp.png',telescope='A12'):
  # array_F : 2D function on theta/phi grid
  # theta0,phi0: ground truth
  # theta,phi: estimate
  plt.clf()
  plt.imshow(array_F[0],aspect='auto',origin='lower',extent=[0,np.pi/2,0,2*np.pi])
  plt.xlabel('Elevation (rad)',fontdict=font)
  plt.ylabel('Azimuth (rad)',fontdict=font)
  cbar=plt.colorbar()
  cbar.set_label('Cost',fontdict=font)
  plt.scatter(theta0,phi0,color='r',s=50)
  plt.scatter(np.mod(theta,np.pi/2),np.mod(phi,np.pi*2),color='b',marker='x',s=50)
  if telescope=='SKA':
     plt.scatter(np.mod(theta,np.pi/2),np.mod(phi+np.pi,np.pi*2),color='b',marker='x',s=50)
  # find minimum value
  idx=np.unravel_index(np.argmin(array_F[0]),array_F[0].shape)
  x=idx[1]/array_F[0].shape[1]*np.pi/2
  y=idx[0]/array_F[0].shape[0]*np.pi*2
  plt.scatter(x,y,color='g',marker='x',s=50)
  plt.savefig(filename)
  # calculate errors (arc length ~ angle in radians)
  doa0=np.array([np.cos(theta0)*np.cos(phi0), np.cos(theta0)*np.sin(phi0), np.sin(theta0)])
  doa1=np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
  error=np.arccos(np.dot(doa0,doa1))
  # for telescope==SKA, +pi ambiguity in azimuth
  if telescope=='SKA':
     doa1p=np.array([np.cos(theta)*np.cos(phi+np.pi), np.cos(theta)*np.sin(phi+np.pi), np.sin(theta)])
     errorp=np.arccos(np.dot(doa0,doa1p))
     error=min(error,errorp)

  doa2=np.array([np.cos(x)*np.cos(y), np.cos(x)*np.sin(y), np.sin(x)])
  # Print arccos(cosine_similarity) of two directions 
  # freq/MHz (transformer) (minimum)
  print(f'{freq/1e6} {error} {np.arccos(np.dot(doa0,doa2))}')

def render3D(array_F,theta0,phi0,r0,theta,phi,r,freq,filename='pp.png',telescope='A12'):
  # array_F : 2D function on theta/phi grid
  # theta0,phi0,r0: ground truth
  # theta,phi,r: estimate
  plt.clf()
  plt.imshow(array_F[0],aspect='auto',origin='lower',extent=[0,np.pi/2,0,2*np.pi])
  plt.xlabel('Elevation (rad)',fontdict=font)
  plt.ylabel('Azimuth (rad)',fontdict=font)
  cbar=plt.colorbar()
  cbar.set_label('Cost',fontdict=font)
  plt.scatter(theta0,phi0,color='r',s=50)
  plt.scatter(np.mod(theta,np.pi/2),np.mod(phi,np.pi*2),color='b',marker='x',s=50)
  if telescope=='SKA':
     plt.scatter(np.mod(theta,np.pi/2),np.mod(phi+np.pi,np.pi*2),color='b',marker='x',s=50)
  # find minimum value
  idx=np.unravel_index(np.argmin(array_F[0]),array_F[0].shape)
  x=idx[1]/array_F[0].shape[1]*np.pi/2
  y=idx[0]/array_F[0].shape[0]*np.pi*2
  plt.scatter(x,y,color='g',marker='x',s=50)
  plt.savefig(filename)
  # calculate errors (arc length ~ angle in radians)
  doa0=np.array([np.cos(theta0)*np.cos(phi0), np.cos(theta0)*np.sin(phi0), np.sin(theta0)])
  doa1=np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
  error=np.arccos(np.dot(doa0,doa1))
  # for telescope==SKA, +pi ambiguity in azimuth
  if telescope=='SKA':
     doa1p=np.array([np.cos(theta)*np.cos(phi+np.pi), np.cos(theta)*np.sin(phi+np.pi), np.sin(theta)])
     errorp=np.arccos(np.dot(doa0,doa1p))
     error=min(error,errorp)

  doa2=np.array([np.cos(x)*np.cos(y), np.cos(x)*np.sin(y), np.sin(x)])
  # Print arccos(cosine_similarity) of two directions 
  # freq/MHz (transformer) (minimum)
  print(f'{freq/1e6} {error} {np.arccos(np.dot(doa0,doa2))} {(r-r0)**2/r0**2}')



if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Evaluate trained transformer model',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--array',default="A12",type=str,metavar='s',
                        help='which array to use: A12 or SKA')
    parser.add_argument('--episodes',default=30000,type=int,metavar='g',
       help='number of episodes stored in the buffer')
    parser.add_argument('--estimate_range', action='store_true', default=False,
       help='use data including the range (3 dimensional, 100%% nearfield)')
    parser.add_argument('--iterations',default=1,type=int,metavar='g',
       help='number of evaluations')
 
    args=parser.parse_args()
 
    if args.array=='SKA':
       net=ManyAttention(depth=6, embed_dim=96, num_heads=8, n_arrays=53, n_stations=6, estimate_range=args.estimate_range).to(mydevice)
    else:
       net=ManyAttention(depth=6, embed_dim=64, num_heads=8, n_arrays=48, n_stations=6, estimate_range=args.estimate_range).to(mydevice)
    net.load_checkpoint()
    net.eval()
    if args.estimate_range:
       if args.array=='SKA':
          buffer=ReplayBuffer3D(args.episodes, n_arrays=53, n_stations=6, n_grid=128)
       else:
          buffer=ReplayBuffer3D(args.episodes, n_arrays=48, n_stations=6, n_grid=128)
    else:
       if args.array=='SKA':
          buffer=ReplayBuffer(args.episodes, n_arrays=53, n_stations=6, n_grid=128)
       else:
          buffer=ReplayBuffer(args.episodes, n_arrays=48, n_stations=6, n_grid=128)
    buffer.load_checkpoint()

    patch_size=16
    for ci in range(args.iterations):
       values,keys,targets,freqs=buffer.sample_observation(batch_size=1,patch_size=patch_size)
       values=values.to(mydevice)
       keys=keys.to(mydevice)
       output=net(keys,values)
       # undo transfroms to values # num_patches, n_range*batch(=1), channel(=3 or 4)*patch_size*patch_size
       values=values.permute(1,2,0)
       # to n_range*batch, num_patches, channel*patch_size*patch_size 
       fold=nn.Fold(output_size=(128,128),kernel_size=[patch_size,patch_size],stride=[patch_size,patch_size])
       if args.estimate_range:
           n_range=values.shape[0] # batch=1
           output=output.cpu().data.numpy()
           for cn in range(n_range):
               val=fold(values[cn])
               val=val.cpu().data.numpy()
               render3D(val[0],targets[0][0],targets[0][1],np.exp(targets[0][2]),output[0][0],output[0][1],np.exp(output[0][2]),freqs[0],filename='eval_'+str(ci)+'_'+str(cn)+'.png',telescope=args.array)
       else:
           values=fold(values)
           values=values.cpu().data.numpy()
           output=output.cpu().data.numpy()
           render(values[0],targets[0][0],targets[0][1],output[0][0],output[0][1],freqs[0],filename='eval_'+str(ci)+'.png',telescope=args.array)

