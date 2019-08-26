# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:04:19 2019

@author: shanbhsa
"""

import torch
import torch.nn as nn
import math

def default_conv3D(in_channels,out_channels,kernel_size,bias=True):
    return nn.Conv3d(in_channels,out_channels,kernel_size,stride=1,padding=(kernel_size//2),bias=bias)

def default_conv2D(in_channels,out_channels,kernel_size,bias=True):
    return nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=(kernel_size//2),bias=bias)    

class EResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(EResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
        
class ResBlock(nn.Module):
    def __init__(self,conv3D,con_feats,n_feats,kernel_size,act=nn.ReLU(True),bias=True,res_scale=1,branched=False,branches=2):
        super(ResBlock,self).__init__()
        
        self.branched=branched
        self.branches=branches
        self.prebody=nn.Sequential(
                        nn.Conv3d(con_feats,n_feats,1),
                        nn.PReLU(n_feats))
        if not self.branched:
            self.body=nn.Sequential(
                            conv3D(n_feats,n_feats,kernel_size,bias=bias),
                            act,
                            conv3D(n_feats,n_feats,kernel_size,bias=bias)
                            )
        else:
            self.body=nn.ModuleList()
            for i in range(self.branches):
                self.body.append(nn.Sequential(
                            conv3D(n_feats,n_feats//self.branches,kernel_size,bias=bias),
                            act,
                            conv3D(n_feats//self.branches,n_feats//self.branches,kernel_size,bias=bias)
                            ))
            
        self.res_scale=res_scale
    
    def forward(self,x):
        
        x=self.prebody(x)
        
        if not self.branched:
            res=self.body(x).mul(self.res_scale)
        else:
            res_list=[]
            for i,p in enumerate(self.body):
                res_list.append(self.body[i](x))
            res=torch.cat(res_list,dim=1).mul(self.res_scale)
            del res_list
        res+=x
        del x
        return res
        

class DenseBlock(nn.Module):
    def __init__(self,conv3D,con_feats,n_feats,kernel_size,act=nn.ELU(),bias=True):
        super(DenseBlock,self).__init__()
        
        
        

        
        self.body=nn.Sequential(
                        nn.BatchNorm3d(con_feats,eps=0.001,momentum=0.99),
                        act,
                        conv3D(con_feats,n_feats,kernel_size,bias=bias)
                        )
        
        
    
    def forward(self,x):
        
        res=self.body(x)
        
        del x
        return res

        
class DResBlock(nn.Module):
    def __init__(self,conv3D,dcon_feats,dn_feats,kernel_size,n_resblocks,act=nn.ReLU(True),bias=True,res_scale=1,branched=False,branches=2):
        super(DResBlock,self).__init__()

        self.dprebody=nn.Sequential(
                        nn.Conv3d(dcon_feats,dn_feats,1),
                        nn.PReLU(dn_feats))
        dn_feats_m=dn_feats
        self.resmodules=nn.ModuleList()
        for i in range(n_resblocks):
            self.resmodules.append(ResBlock(conv3D,dn_feats_m,dn_feats,kernel_size,act=act,bias=bias,res_scale=res_scale,branched=branched,branches=branches))
            dn_feats_m+=dn_feats
        self.body=nn.Sequential(
                        conv3D(dn_feats_m,dn_feats,1),
                        act
 #                       conv3D(dn_feats,dn_feats,kernel_size)
                        )
        
    def forward(self,x):
       x=self.dprebody(x)
       
       res_list=[x]
       for i,p in enumerate(self.resmodules):
           
           if i==0:
               res_list.append(self.resmodules[i](x))
           else:
               
               res_list.append(self.resmodules[i](torch.cat(res_list,dim=1)))
       
          
       res=self.body(torch.cat(res_list,dim=1))
       del res_list
#       res+=x
       del x
       return res


class Upsampler(nn.Sequential):
    def __init__(self,conv3D,scale,n_feats,bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv3D(n_feats, 8 * n_feats, 1, bias))
                m.append(PixelShuffle3D(2))
#                if bn:
#                    m.append(nn.BatchNorm2d(n_feats))
#                if act == 'relu':
#                    m.append(nn.ReLU(True))
#                elif act == 'prelu':
#                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv3D(n_feats, 27 * n_feats, 1, bias))
            m.append(PixelShuffle3D(3))
#            if bn:
#                m.append(nn.BatchNorm2d(n_feats))
#            if act == 'relu':
#                m.append(nn.ReLU(True))
#            elif act == 'prelu':
#                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        del m
class PixelShuffle3D(nn.Module):
    def __init__(self,scale):
        super(PixelShuffle3D,self).__init__()
        self.scale=scale
    def forward(self,x):
        batch_size,channels,in_depth,in_height,in_width=x.size()
        channels//=self.scale**3
        out_depth=in_depth*self.scale
        out_height=in_height*self.scale
        out_width=in_width*self.scale
        x_view=x.contiguous().view(batch_size,channels,self.scale,self.scale,self.scale,in_depth,in_height,in_width)
        out=x_view.permute(0,1,5,2,6,3,7,4).contiguous().view(batch_size,channels,out_depth,out_height,out_width)
        del x_view
        return out
        
#        input_size = list(input.size())
#        dimensionality = len(input_size) - 2
#    
#        input_size[1] //= (upscale_factor ** dimensionality)
#        output_size = [dim * upscale_factor for dim in input_size[2:]]
#    
#        input_view = input.contiguous().view(
#            input_size[0], input_size[1],
#            *(([upscale_factor] * dimensionality) + input_size[2:])
#        )
#    
#        indicies = list(range(2, 2 + 2 * dimensionality))
#        indicies = indicies[1::2] + indicies[0::2]
#    
#        shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
#        return shuffle_out.view(input_size[0], input_size[1], *output_size)