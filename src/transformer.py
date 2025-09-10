import numpy as np
import torch
from torch import nn

from replaybuffer import ReplayBuffer

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# initialize all layer weights, based on the fan in
def init_layer(layer,sc=None):
  sc = sc or 1./np.sqrt(layer.weight.data.size()[0])
  torch.nn.init.uniform_(layer.weight.data, -sc, sc)
  torch.nn.init.uniform_(layer.bias.data, -sc, sc)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        # Multi-head attention with residual
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        # Residual connection for attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.norm1(x)
        
        # Residual connection for MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Another residual connection
        x = self.norm2(x)
        
        return x

class ManyAttention(nn.Module):
    def __init__(self, depth=6, embed_dim=64, num_heads=8, n_arrays=48, n_stations=6, n_grid=128, patch_size=16):
        super().__init__()
        self.n_arrays=n_arrays
        self.embed_dim=embed_dim
        # input transforms
        self.num_patches=(n_grid//patch_size)**2
        # keys: (3+1+1)*n_arrays
        self.proj1=nn.Linear(5*n_arrays, embed_dim*self.num_patches)
        # values: channel(=3)*patch_size*patch_size
        self.proj2=nn.Linear(3*patch_size*patch_size, embed_dim)

        # attention blocks
        self.mha1=nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1=nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(depth)
        ])

        # output transforms
        self.out_linear=nn.Linear(embed_dim,2)

        # dropout
        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.1)

        self.checkpoint_file='transformer_model.npy'

        # constans for final transform
        self.A=torch.tensor([1, 0]).float().to(mydevice)
        self.B=torch.tensor([0.5*np.pi/2, np.pi]).float().to(mydevice)

        init_layer(self.proj1)
        init_layer(self.proj2)
        init_layer(self.out_linear)

    def forward(self,x,y):
        batch_size=x.shape[0]
        # x: batch x (3+1+1)*n_arrays (key=query)
        keys=self.dropout1(self.proj1(x))
        keys=keys.reshape(batch_size,self.num_patches,self.embed_dim)
        keys=keys.permute(1,0,2)
        # y: seq, batch, channel(=3)*patch_size*patch_size (value)
        values=self.dropout2(self.proj2(y))
        query=keys
        attn1,_=self.mha1(query,keys,values)
        x=self.norm1(values+attn1) # residual

        for block in self.blocks:
            x = block(x)  # Each block has internal residuals
 
        # pool over sequences (attention pooling ?)
        pooled=x.mean(dim=0)
        # map to [-1,1]
        x=torch.tanh(self.out_linear(pooled))
        # map x[0] to [0,pi/2] and x[1] to [-pi,pi]
        x = (x + self.A)*self.B
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
