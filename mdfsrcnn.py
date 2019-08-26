# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:05:38 2019

@author: shanbhsa
"""
import numpy as np
import torch.nn as nn
import common

    

def make_model(args):
    return MDFSRCNN(args)
    
#def weight_init(m):   
#    if isinstance(m,nn.Conv3d):
#        std_dev=np.sqrt(2.0/(m.weight.size()[-1]**3*m.out_channels*1.0625))        
#        m.weight.data.normal_(0,std_dev) 
#    elif isinstance(m,nn.ConvTranspose3d):
#        m.weight.data.normal_(0,0.001)     
class MDFSRCNN(nn.Module):
    def __init__(self,args=None,conv3D=common.default_conv3D):
        super(MDFSRCNN,self).__init__()
        
        self.scale=2#args.scale[0]
        self.head1=nn.Sequential(nn.Conv3d(1,56,kernel_size=5,stride=1,padding=2),
                                nn.PReLU(56),
                                nn.Conv3d(56,12,kernel_size=1,stride=1,padding=0),
                                nn.PReLU(12))
        body1=[nn.Conv3d(12,12,kernel_size=3,stride=1,padding=1) for i in range(4)]
        body1.append(nn.PReLU(12))
        body1.append(nn.Conv3d(12,56,kernel_size=1,stride=1,padding=0))
        body1.append(nn.PReLU(56))
        self.body1=nn.Sequential(*body1)
        kernel_size=9
        self.tail1=nn.ConvTranspose3d(56,1,kernel_size=9,stride=self.scale,padding=kernel_size//2,dilation=1,output_padding =0)
                                
    def forward(self,x):
        x=self.head1(x)
        
        x=self.body1(x)
         
        x=self.tail1(x)
        
        return x 
#if __name__ == '__main__':    
#    m=MDFSRCNN() 
##    m.apply(weight_init)
##    for param in m.tail.parameters():
##        print ('tail',param.size())       
##    for param in m.body.parameters():
##        print ('body',param.size())     
##    for name, param in m.body1.0.named_parameters():
##        print (name) 
##    deconv=filter(lambda x:x[0] in ['tail1.weight','tail1.bias'],m.named_parameters())
##    params=['head1.0.weight','head1.0.bias','head1.1.weight','head1.2.weight','head1.2.bias','head1.3.weight','body1.0.weight','body1.0.bias','body1.1.weight','body1.1.bias','body1.2.weight','body1.2.bias','body1.3.weight','body1.3.bias','body1.4.weight','body1.5.weight','body1.5.bias','body1.6.weight']
##    others=filter(lambda x:x[0] in params,m.named_parameters())
##    print(deconv)
##    print('others \n')
##    print(others)
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