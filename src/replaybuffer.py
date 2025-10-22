import numpy as np
import pickle
import torch
from torch import nn


class ReplayBuffer(object):
    def __init__(self, max_size, n_arrays=48, n_stations=6, n_grid=128):
        # n_arrays: number of ULAs
        # n_stations: stations in each ULA (=6)
        # n_grid: grid point (theta, phi) where error evaluated
        self.mem_size = max_size
        self.mem_cntr = 0

        self.n_arrays=n_arrays
        self.n_stations=n_stations
        self.n_grid=n_grid

        # value: error, theta, phi (3 channels), n_grid x n_grid
        self.value_= np.zeros((self.mem_size, 3, n_grid, n_grid), dtype=np.float32)
        # key : consits of
        # 1) directions : direction vector of first baseline, n_arrays x 3
        self.direction_ = np.zeros((self.mem_size, n_arrays, 3), dtype=np.float32)
        # 2) distance : log(distance) of first baseline, n_arrays x 1 (normalized by lambda/2)
        self.distance_ = np.zeros((self.mem_size, n_arrays), dtype=np.float32)
        # 3) sin(theta) : estimate of each array, n_arrays x 1 
        self.sintheta_ = np.zeros((self.mem_size, n_arrays), dtype=np.float32)

        # target: true DOA (theta,phi) 2 x 1
        self.target_ = np.zeros((self.mem_size, 2), dtype=np.float32)
        # frequency: not used for training, but useful for checking wideband
        # performance
        self.frequency_ = np.zeros(self.mem_size,dtype=np.float32)

        self.filename='databuffer.npy' # for saving object

    def store_observation(self, value, direction, distance, sintheta, target, freq=0):
        index = self.mem_cntr % self.mem_size
        self.value_[index] = value
        self.direction_[index] = direction
        self.distance_[index] = distance
        self.sintheta_[index] = sintheta
        self.target_[index] = target
        self.frequency_[index] = freq

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # return value, direction, log(distance), sintheta, target as separate numpy arrays
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        values = self.value_[batch]
        directions = self.direction_[batch]
        distances = self.distance_[batch]
        sinthetas = self.sintheta_[batch]
        targets = self.target_[batch]
        freqs = self.frequency_[batch]

        return values, directions, distances, sinthetas, targets, freqs


    def sample_observation(self, batch_size, patch_size):
        # return transformed versions of 
        # value, direction, log(distance), sintheta as torch.tensors
        # value: unfolded into patches: num_patches, batch, channel*patch_size*patch_size
        # keys: concat(direction: 3*n_arrays, log(distance): n_arrays, sintheta: n_arrays)
        # target: 2x1
        # freqs: 1, not used for training, but useful for visualizing
        values, directions, distances, sinthetas, targets, freqs = self.sample_buffer(batch_size)
        unfold=nn.Unfold(kernel_size=[patch_size,patch_size], stride=[patch_size,patch_size])
        patches=unfold(torch.tensor(values)) # batch, channel*patch_size*patch_size, num_patches
        value=patches.permute(2,0,1) # num_patches(=sequence),batch,channel*patch_size*patch_size
        key=torch.cat((torch.tensor(directions.reshape(batch_size,3*self.n_arrays)),torch.tensor(distances),torch.tensor(sinthetas)),dim=1) # batch x (3*n_arrays+n_arrays+n_arrays)
        target=torch.tensor(targets)

        # value: num_patches x batch x channel(=3)*patch_size*patch_size
        # key: batch x (3+1+1)*n_arrays
        # target: batch x 2
        return value,key,target,freqs


    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
        
    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.n_arrays=temp.n_arrays
          self.n_stations=temp.n_stations
          self.n_grid=temp.n_grid
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.value_=temp.value_
          self.direction_=temp.direction_
          self.distance_=temp.distance_
          self.sintheta_=temp.sintheta_
          self.target_=temp.target_
          self.frequency_=temp.frequency_

    def reset(self):
        self.mem_cntr=0


class ReplayBuffer3D(object):
    def __init__(self, max_size, n_arrays=48, n_stations=6, n_grid=128, n_range=2):
        # n_arrays: number of ULAs
        # n_stations: stations in each ULA (=6)
        # n_grid: grid point (theta, phi) where error evaluated
        # n_range: grid points in log(range)
        self.mem_size = max_size
        self.mem_cntr = 0

        self.n_arrays=n_arrays
        self.n_stations=n_stations
        self.n_grid=n_grid
        self.n_range=n_range

        # value: error, theta, phi, logrange (4 channels), n_grid x n_grid x n_range
        self.value_= np.zeros((self.mem_size, 4, n_grid, n_grid, n_range), dtype=np.float32)
        # key : consits of
        # 1) directions : direction vector of first baseline, n_arrays x 3
        self.direction_ = np.zeros((self.mem_size, n_arrays, 3), dtype=np.float32)
        # 2) distance : log(distance) of first baseline, n_arrays x 1 (normalized by lambda/2)
        self.distance_ = np.zeros((self.mem_size, n_arrays), dtype=np.float32)
        # 3) sin(theta) : estimate of each array, n_arrays x 1 
        self.sintheta_ = np.zeros((self.mem_size, n_arrays), dtype=np.float32)

        # target: true DOA (theta,phi,log(range)) 3 x 1
        self.target_ = np.zeros((self.mem_size, 3), dtype=np.float32)
        # frequency: not used for training, but useful for checking wideband
        # performance
        self.frequency_ = np.zeros(self.mem_size,dtype=np.float32)

        self.filename='databuffer.npy' # for saving object

    def store_observation(self, value, direction, distance, sintheta, target, freq=0):
        index = self.mem_cntr % self.mem_size
        self.value_[index] = value
        self.direction_[index] = direction
        self.distance_[index] = distance
        self.sintheta_[index] = sintheta
        self.target_[index] = target
        self.frequency_[index] = freq

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # return value, direction, log(distance), sintheta, target as separate numpy arrays
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        values = self.value_[batch]
        directions = self.direction_[batch]
        distances = self.distance_[batch]
        sinthetas = self.sintheta_[batch]
        targets = self.target_[batch]
        freqs = self.frequency_[batch]

        return values, directions, distances, sinthetas, targets, freqs


    def sample_observation(self, batch_size, patch_size):
        # return transformed versions of 
        # value, direction, log(distance), sintheta as torch.tensors
        # value: unfolded into patches: num_patches, n_range*batch, channel*patch_size*patch_size
        # keys: concat(direction: 3*n_arrays, log(distance): n_arrays, sintheta: n_arrays)
        # target: 3x1
        # freqs: 1, not used for training, but useful for visualizing
        values, directions, distances, sinthetas, targets, freqs = self.sample_buffer(batch_size)
        unfold=nn.Unfold(kernel_size=[patch_size,patch_size], stride=[patch_size,patch_size])
        # batch x channel x n_grid x n_grid x n_range
        values=torch.tensor(values)
        patches=[unfold(values[:,:,:,:,ci]) for ci in range(self.n_range)] # list of n_range, each batch, channel*patch_size*patch_size, num_patches
        # concat over num_patches
        patches=torch.cat(patches,dim=2)
        value=patches.permute(2,0,1) # num_patches(=sequence),n_range*batch,channel*patch_size*patch_size
        key=torch.cat((torch.tensor(directions.reshape(batch_size,3*self.n_arrays)),torch.tensor(distances),torch.tensor(sinthetas)),dim=1) # batch x (3*n_arrays+n_arrays+n_arrays)
        target=torch.tensor(targets)

        # value: num_patches x n_range*batch x channel(=4)*patch_size*patch_size
        # key: batch x (3+1+1)*n_arrays
        # target: batch x 3
        return value,key,target,freqs


    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
        
    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.n_arrays=temp.n_arrays
          self.n_stations=temp.n_stations
          self.n_grid=temp.n_grid
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.value_=temp.value_
          self.direction_=temp.direction_
          self.distance_=temp.distance_
          self.sintheta_=temp.sintheta_
          self.target_=temp.target_
          self.frequency_=temp.frequency_

    def reset(self):
        self.mem_cntr=0
