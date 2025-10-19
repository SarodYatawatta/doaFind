#! /usr/bin/env python

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

from replaybuffer import ReplayBuffer,ReplayBuffer3D
from transformer import ManyAttention

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

def loss_function(x,y,telescope='A12'):
    batch_size=x.shape[0]
    # convert DOA to direction cosines
    theta0=x[:,0]
    phi0=x[:,1]
    theta=y[:,0]
    phi=y[:,1]
    doa0=[torch.cos(theta0)*torch.cos(phi0), torch.cos(theta0)*torch.sin(phi0), torch.sin(theta0)]
    doa=[torch.cos(theta)*torch.cos(phi), torch.cos(theta)*torch.sin(phi), torch.sin(theta)]

    # 1-(cosine_similarity)
    dotp=doa0[0]*doa[0]+doa0[1]*doa[1]+doa0[2]*doa[2]
    if telescope=='A12':
      loss=1-dotp
    else:
      # azimuth can have pi ambiguity, consider both and take minimum
      loss1=1-dotp
      doa0=[torch.cos(theta0)*torch.cos(phi0+torch.pi), torch.cos(theta0)*torch.sin(phi0+torch.pi), torch.sin(theta0)]
      dotp=doa0[0]*doa[0]+doa0[1]*doa[1]+doa0[2]*doa[2]
      loss2=1-dotp
      loss=torch.min(loss1,loss2)

    return torch.sum(loss)/batch_size

def loss_function_3d(x,y,telescope='A12'):
    batch_size=x.shape[0]
    # convert DOA to direction cosines
    theta0=x[:,0]
    phi0=x[:,1]
    r0=x[:,2]
    theta=y[:,0]
    phi=y[:,1]
    r=y[:,2]
    doa0=[torch.cos(theta0)*torch.cos(phi0), torch.cos(theta0)*torch.sin(phi0), torch.sin(theta0)]
    doa=[torch.cos(theta)*torch.cos(phi), torch.cos(theta)*torch.sin(phi), torch.sin(theta)]

    # 1-(cosine_similarity)
    dotp=doa0[0]*doa[0]+doa0[1]*doa[1]+doa0[2]*doa[2]
    if telescope=='A12':
      loss=1-dotp
    else:
      # azimuth can have pi ambiguity, consider both and take minimum
      loss1=1-dotp
      doa0=[torch.cos(theta0)*torch.cos(phi0+torch.pi), torch.cos(theta0)*torch.sin(phi0+torch.pi), torch.sin(theta0)]
      dotp=doa0[0]*doa[0]+doa0[1]*doa[1]+doa0[2]*doa[2]
      loss2=1-dotp
      loss=torch.min(loss1,loss2)

    # log(range) loss
    range_loss=(r-r0)**2

    return torch.sum(loss+0.1*range_loss)/batch_size


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Train transformer model using simlated DOA data',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--array',default="A12",type=str,metavar='s',
                        help='which array to use: A12 or SKA')
    parser.add_argument('--seed',default=0,type=int,metavar='s',
       help='random seed to use')
    parser.add_argument('--episodes',default=10000,type=int,metavar='g',
       help='number of episodes stored in the buffer')
    parser.add_argument('--iterations',default=30000,type=int,metavar='g',
       help='number of learning iterations')
    parser.add_argument('--learning_rate',default=1e-6,type=float,metavar='g',
       help='learning rate')
    parser.add_argument('--estimate_range', action='store_true', default=False,
       help='use data including the range (3 dimensional, 100%% nearfield)')
    parser.add_argument('--load', action='store_true', default=False,
       help='load saved model')
 
    args=parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.array=='SKA':
       net=ManyAttention(depth=6, embed_dim=96, num_heads=8, n_arrays=53, n_stations=6, estimate_range=args.estimate_range).to(mydevice)
    else:
       net=ManyAttention(depth=6, embed_dim=64, num_heads=8, n_arrays=48, n_stations=6, estimate_range=args.estimate_range).to(mydevice)
    if args.load:
       net.load_checkpoint()
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

    optimizer = optim.Adam(net.parameters(),lr=args.learning_rate)

    for n_iter in range(args.iterations):
       values,keys,targets,_=buffer.sample_observation(batch_size=256,patch_size=16)
       values=Variable(values).to(mydevice)
       keys=Variable(keys).to(mydevice)
       targets=Variable(targets).to(mydevice)
       def closure():
          if torch.is_grad_enabled():
             optimizer.zero_grad()
          output=net(keys,values)
          if args.estimate_range:
              loss=loss_function_3d(output,targets,telescope=args.array)
          else:
              loss=loss_function(output,targets,telescope=args.array)
          if loss.requires_grad:
             loss.backward()
          print(f'{n_iter} {loss.data.item()}')
          return loss
       optimizer.step(closure)

       if n_iter%10000==0:
          net.save_checkpoint()

    net.save_checkpoint()
