# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 00:39:59 2019

@author: shanbhsa
"""

import os
from importlib import import_module
import apex.fp16_utils as fp16
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
from torch.autograd import Variable
def weight_init(m):   
    if isinstance(m,nn.Conv3d):
        std_dev=np.sqrt(2.0/(m.weight.size()[-1]**3*m.out_channels*1.0625))        
        m.weight.data.normal_(0,std_dev) 
    elif isinstance(m,nn.ConvTranspose3d):
        m.weight.data.normal_(0,0.001) 
def dcsrn_weight_init(m): 
    growth_rate=24  
    if isinstance(m,nn.Conv3d):
        std_dev=np.sqrt(2.0/(m.weight.size()[-1]**3*growth_rate))        
        m.weight.data.normal_(0,std_dev) 
    
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        
        self.input_large = args.input_large 
        
        
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module(args.model.lower())
        self.model = module.make_model(args).to(self.device)
        
        if args.model=='DCSRN':
            self.model.apply(dcsrn_weight_init)
        elif args.model=='DCRSR' :
            self.model.apply(weight_init)
        if args.precision == 'half':
            self.model.half()
            self.model_params, self.master_params = fp16.prep_param_lists(
                self.model)
            

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    
        
        
    def forward(self, x):
        
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
           
            forward_function = self.model.forward


            return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)
