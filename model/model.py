from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as nnf
import torch
import torch.nn as nn
from model.modules import MSA, BottleNeckBlock

class MyModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384,num_heads=6, attn_drop=0.,
                 depth=1, block_out_list=[32,64,128],down_list=[True,True,True], block_mid_chan=64):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_w = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim),requires_grad=False)
        
        if depth == 1:
            self.mhsa = MSA(dim=embed_dim, num_heads=num_heads, return_attn=True)
        else:
            first = [MSA(dim=embed_dim,num_heads=num_heads,return_attn=False,attn_drop=attn_drop) for i in range(depth-1)]
            second = MSA(dim=embed_dim,num_heads=num_heads,return_attn=True,attn_drop=attn_drop)
            self.mhsa = nn.Sequential(*first, second)
        self.att_classifier = nn.Linear(embed_dim, 2)
        self.img_size = img_size
        self.in_chans = in_chans
        
        self.block, self.classifier = self._make_block(block_out_list, block_mid_chan, down_list) 
        self.relu = nn.ReLU(inplace=True)
        
        self.initialize_weight()
        
        
    def initialize_weight(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def _make_block(self, block_out_list, block_mid_chan, down_list):
        assert len(block_out_list) == len(down_list) ,"block_out_list & down_list should have same length"
        
        block_in_chan_list = [self.in_chans+1]
        flt_list = [False for i in range(len(down_list))]
        flt_list[-1] = True
        for i in range(len(block_out_list)-1):
            block_in_chan_list.append(block_out_list[i])
        
        made_block = nn.Sequential(*[BottleNeckBlock(in_chan=block_in_chan_list[i], mid_chan=block_mid_chan,out_chan=block_out_list[i], down_sample=down_list[i],
                                                    flatten=flt_list[i]) 
                                     for i in range(len(block_out_list))])
        made_clf = nn.Linear(block_out_list[-1],2, bias=True)
        
        return made_block, made_clf
        
            
        
        
    def forward(self,x):
        img = x
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed[:,1:,:]
        
        cls_token = self.cls_token + self.pos_embed[:,:1,:]
        cls_tokens = cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)
        
        attn_x, attn = self.mhsa(x)
        att_out = self.att_classifier(attn_x[:,0,:])
        # print(attn.shape)
        attn = attn.sum(axis=1)[:,0,1:].reshape(-1,1,self.patch_w,self.patch_w)
        attn = nnf.interpolate(attn, size=(self.img_size,self.img_size),mode='bicubic') 
        
        x = torch.concatenate([img, attn], dim=1)
        
        x1 = x
        x = self.block(x)
        out = self.classifier(x)
        
        return out, x1, attn, att_out

class MyinferModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384,num_heads=6, attn_drop=0.,
                 depth=1, block_out_list=[32,64,128],down_list=[True,True,True], block_mid_chan=64):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_w = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim),requires_grad=False)
        
        if depth == 1:
            self.mhsa = MSA(dim=embed_dim, num_heads=num_heads, return_attn=True)
        else:
            first = [MSA(dim=embed_dim,num_heads=num_heads,return_attn=False,attn_drop=attn_drop) for i in range(depth-1)]
            second = MSA(dim=embed_dim,num_heads=num_heads,return_attn=True,attn_drop=attn_drop)
            self.mhsa = nn.Sequential(*first, second)
        # self.att_classifier = nn.Linear(embed_dim, 2)
        self.img_size = img_size
        self.in_chans = in_chans
        
        self.block, self.classifier = self._make_block(block_out_list, block_mid_chan, down_list) 
        self.relu = nn.ReLU(inplace=True)
        
        self.initialize_weight()
        
        
    def initialize_weight(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def _make_block(self, block_out_list, block_mid_chan, down_list):
        assert len(block_out_list) == len(down_list) ,"block_out_list & down_list should have same length"
        
        block_in_chan_list = [self.in_chans+1]
        flt_list = [False for i in range(len(down_list))]
        flt_list[-1] = True
        for i in range(len(block_out_list)-1):
            block_in_chan_list.append(block_out_list[i])
        
        made_block = nn.Sequential(*[BottleNeckBlock(in_chan=block_in_chan_list[i], mid_chan=block_mid_chan,out_chan=block_out_list[i], down_sample=down_list[i],
                                                    flatten=flt_list[i]) 
                                     for i in range(len(block_out_list))])
        made_clf = nn.Linear(block_out_list[-1],2, bias=True)
        
        return made_block, made_clf
        
            
        
        
    def forward(self,x):
        img = x
        
        x = self.patch_embed(x)
        
        x = x + self.pos_embed[:,1:,:]
        
        cls_token = self.cls_token + self.pos_embed[:,:1,:]
        cls_tokens = cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)
        
        _,attn = self.mhsa(x)

        attn = attn.sum(axis=1)[:,0,1:].reshape(-1,1,self.patch_w,self.patch_w)
        attn = nnf.interpolate(attn, size=(self.img_size,self.img_size),mode='bicubic') 
        
        x = torch.concatenate([img, attn], dim=1)
        
        x1 = x
        x = self.block(x)
        out = self.classifier(x)
        
        return out
    
    
class AblationModel(nn.Module):
    def __init__(self, in_chans=1,block_out_list=[32,64,128],down_list=[True,True,True], block_mid_chan=64):
        super().__init__()
        self.in_chans = in_chans
        self.block, self.classifier = self._make_block(block_out_list, block_mid_chan, down_list) 
        self.relu = nn.ReLU(inplace=True)
        
        
        self.initialize_weight()
        
        
    def initialize_weight(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    
    def _make_block(self, block_out_list, block_mid_chan, down_list):
        assert len(block_out_list) == len(down_list) ,"block_out_list & down_list should have same length"
        
        block_in_chan_list = [self.in_chans]
        flt_list = [False for i in range(len(down_list))]
        flt_list[-1] = True
        for i in range(len(block_out_list)-1):
            block_in_chan_list.append(block_out_list[i])
        
        made_block = nn.Sequential(*[BottleNeckBlock(in_chan=block_in_chan_list[i], mid_chan=block_mid_chan,out_chan=block_out_list[i], down_sample=down_list[i],
                                                    flatten=flt_list[i]) 
                                     for i in range(len(block_out_list))])
        made_clf = nn.Linear(block_out_list[-1],2, bias=True)
        
        return made_block, made_clf
        
            
        
        
    def forward(self,x):
        
        x = self.block(x)
        out = self.classifier(x)
        
        return out
    
