import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention


class MSA(Attention):
    def __init__(self, dim=384, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0., return_attn=True):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.return_attn = return_attn
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5
        

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm =  nn.Identity()
        self.k_norm =  nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.return_attn:
            return x, attn
        else:
            return x
        
        
class BottleNeckBlock(nn.Module):
    def __init__(self,in_chan, mid_chan, out_chan, down_sample=False, flatten=False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=1)
        
        
        if down_sample:
            self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=2, padding=1)
        else:
            self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1)
        
        
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.bn3 = nn.BatchNorm2d(out_chan)
        if down_sample:
            conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=2)
            bn = nn.BatchNorm2d(out_chan)
            dwn = nn.Sequential(conv,bn)
        else:
            conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1)
            bn = nn.BatchNorm2d(out_chan)
            dwn = nn.Sequential(conv,bn)
        self.down = dwn
        self.flatten = flatten
        
    def forward(self,x):
        i = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.down is not None:
            i = self.down(i)
        
        x += i
        x = self.relu(x)
        
        if self.flatten:
            x = torch.mean(x.view(x.size(0),x.size(1),-1),dim=2)
        
        return x