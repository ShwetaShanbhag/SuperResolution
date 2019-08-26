# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:05:38 2019

@author: shanbhsa
"""
import numpy as np
import torch.nn as nn
import common
def make_model(args):
    return NDFSRCNN(args)
class NDFSRCNN(nn.Module):
    def __init__(self,args=None,conv2D=common.default_conv2D):
        super(NDFSRCNN,self).__init__()
        self.scale=2#args.scale[0]
        self.head=nn.Sequential(nn.Conv2d(1,56,kernel_size=5,stride=1,padding=2),
                                nn.PReLU(56),
                                nn.Conv2d(56,12,kernel_size=1,stride=1,padding=0),
                                nn.PReLU(12))
        body=[nn.Conv2d(12,12,kernel_size=3,stride=1,padding=1) for i in range(4)]
        body.append(nn.PReLU(12))
        body.append(nn.Conv2d(12,56,kernel_size=1,stride=1,padding=0))
        body.append(nn.PReLU(56))
        self.body=nn.Sequential(*body)
        kernel_size=9
        self.tail=nn.ConvTranspose2d(56,1,kernel_size=9,stride=self.scale,padding=kernel_size//2,dilation=1,output_padding =0)
                                
    def forward(self,x):
        x=self.head(x)
        
        x=self.body(x)
          
        x=self.tail(x)
        
        return x  
#if __name__ == '__main__':    
#    m=FSRCNN() 
#    m.apply(weight_init)
#    for param in m.tail.parameters():
#        print ('tail',param.size())       
#    for param in m.body.parameters():
#        print ('body',param.size())     
#    for param in m.head.parameters():
#        print ('head',param.size()) 
#deconv=list(filter(lambda x:x[0] =='tail.weight',m.named_parameters()))
#others=filter(lambda x:x[0] !='tail.weight',m.named_parameters())
#print(deconv)
#print(others)
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))