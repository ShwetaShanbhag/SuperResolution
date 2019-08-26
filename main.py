# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:15:11 2019

@author: shanbhsa
"""
#import torch
import lidc_idri
from option import args
import os
import math
from decimal import Decimal
import apex.fp16_utils as fp16
import utility
from importlib import import_module
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.utils as utils
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from trainer import Trainer
from torch.utils.data import  DataLoader


#os.environ['CUDA_VISIBLE_DEVICES'] = '7'         

    
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader[1] if len(loader)>1 else None
        self.loader_test = loader[0]
        self.model = my_model
        self.loss = my_loss
        if args.precision=='half':
            trainable = filter(lambda x: x.requires_grad,self.model.master_params)
        else:
            trainable = filter(lambda x: x.requires_grad,self.model.parameters())
        if self.args.model in ['MDFSRCNN','NDFSRCNN']:
#             deconv=filter(lambda x:x[0] in ['tail1.weight','tail1.bias'],self.model.named_parameters())
#             params=['head1.0.weight','head1.0.bias','head1.1.weight','head1.2.weight','head1.2.bias','head1.3.weight','body1.0.weight','body1.0.bias','body1.1.weight','body1.1.bias','body1.2.weight','body1.2.bias','body1.3.weight','body1.3.bias','body1.4.weight','body1.5.weight','body1.5.bias','body1.6.weight']
#             others=filter(lambda x:x[0] in params,self.model.named_parameters())
#             self.optimizer = optim.SGD([    {'params': others},
#                                            {'params':deconv,'lr':0.0001}
#                                       ],lr=0.001,momentum=0.9)
             self.optimizer = optim.Adam(trainable,lr=args.lr,betas=args.betas,eps=args.epsilon,weight_decay=args.weight_decay)
#        if self.args.model in ['3DFSRCNN','2DFSRCNN']:
#        
#            self.optimizer = optim.SGD([
#                                            {'params': model.head.parameters()},
#                                            {'params': model.body.parameters()},
#                                            {'params':model.tail.parameters(),'lr':0.0001}
#                                       ],lr=0.001)
        elif self.args.model=='DCSRN':
            self.optimizer=optim.Adam(trainable,lr=args.lr,betas=args.betas,eps=args.epsilon,weight_decay=args.weight_decay)
        elif self.args.model in ['DCRSR','EDSR']:
            self.optimizer = optim.Adam(trainable,lr=args.lr,betas=args.betas,eps=args.epsilon,weight_decay=args.weight_decay)
            self.scheduler=scheduler.StepLR(self.optimizer ,step_size=args.decay,gamma=args.gamma)
        if self.args.load != '':
            self.optimizer.load_state_dict(os.path.join(ckp.dir,'optimizer.pt'))
            epoch=len(ckp.log)
            if epoch>1 and self.args.model in ['DCRSR','EDSR']:
                for _ in range(epoch): self.scheduler.step()
        self.error_last = 1e8

    def train(self,epoch):
        if self.args.model in ['DCRSR','EDSR']:
            self.scheduler.step()
            self.loss.step()
        #epoch = self.optimizer.get_last_epoch() + 1
        #lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(self.args.lr))
        )
        self.loss.start_log()
        self.model.train()
        self.ckp.add_log(
            torch.zeros( 1, len(self.scale)),True
            )
        timer_data, timer_model = utility.timer(), utility.timer()
        idx_scale=0
        for batch in range(self.loader_train.train_batches):
            #print(batch)
            lr,hr=self.loader_train._getitem_and_make_batch(batch,self.args.batch_size)
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            sr = self.model(lr)
#            print(sr.shape)
#            print(hr.shape)
            loss = self.loss(sr.float(), hr.float())
            
            self.model.zero_grad()
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.model_params,
                    self.args.gclip
                )
            if self.args.precision == 'half':
                # Now we move the calculated gradients to the master params
                # so that we can apply the gradient update in FP32.
                fp16.model_grads_to_master_grads(self.model.model_params,self.model.master_params)
                if self.loss.loss[0]['weight']>1:
                    # If we scaled our losses now is a good time to scale it
                    # back since our gradients are in FP32.
                    for params in self.model.master_params:
                        if  params.grad is not None:
                             params.grad.data.mul_(1./self.loss.loss[0]['weight'])
                # Apply weight update in FP32.
                self.optimizer.step()
                # Copy the updated weights back FP16 model weights.
                fp16.master_params_to_model_params(self.model.model_params,self.model.master_params)
            else:
                self.optimizer.step()
            
            timer_model.hold()
            
            self.ckp.log2[-1, idx_scale] += utility.calc_psnr(
                        sr.float(), hr.float() 
                    )
            
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    self.loader_train.train_batches* self.args.batch_size,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        for idx_scale, scale in enumerate(self.scale):
            self.ckp.log2[-1, idx_scale] /= (batch+1)
            best = self.ckp.log2.max(0)
            self.ckp.write_log(
                '[x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(                   
                    scale,
                    self.ckp.log2[-1,idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )
        self.loss.end_log(self.loader_train.train_batches,'train')
        self.error_last = self.loss.log[-1, -1]
        #print(self.ckp.log2.shape)
    def test(self,epoch):
        torch.set_grad_enabled(False)
        if not self.args.test_only:
            self.loss.start_log('test')
        with torch.no_grad():
            #epoch = self.optimizer.get_last_epoch() + 1
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
                torch.zeros(1, len(self.scale))
            )
            self.model.eval()
            index=0
            timer_test = utility.timer()
            if self.args.save_results: self.ckp.begin_background()
            if self.args.test_only: 
                patch_list_sr=[] 
                patch_list_hr=[] 
            for idx_scale, scale in enumerate(self.scale):    
                for batch in range(self.loader_test.test_batches) :
                    #print(batch)
                    lr,hr=self.loader_test._getitem_and_make_batch(batch,1)
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr)
                    if not self.args.test_only:
                        loss = self.loss(sr.float(), hr.float(),'test')
#                    #sr = utility.quantize(sr, 32767)
#                    print(sr.shape)
#                    print(hr.shape)
                    
                    #save_list = [sr]
                    if self.args.test_only:
                        if self.args.model in ['DCRSR','DCSRN','EDSR','MDFSRCNN']:
                            patch_list_sr.append(sr.squeeze_().cpu().numpy().transpose(1,2,0))
                            patch_list_hr.append(hr.squeeze_().cpu().numpy().transpose(1,2,0))
                        else:
                            patch_list_sr.append(sr.squeeze_().cpu().numpy())
                            patch_list_hr.append(hr.squeeze_().cpu().numpy())
                        if len(patch_list_sr) ==self.loader_test.n_patches[index]:
                            #print(len(patch_list_sr))  
                            
                            n_slices=self.loader_test.test_slices[index]
                            info=self.loader_test.info[index]
                            filename=os.path.join(self.loader_test.dir_hr,self.loader_test.subject_list[index])
                            [psnr,ssim]=self.ckp.aggregateandcalcpsnr(patch_list_sr,patch_list_hr,n_slices,filename,info)
                            self.ckp.log[-1, idx_scale] += psnr
                            self.ckp.ssimlog[-1, idx_scale] += ssim
                            self.ckp.write_log(
                            '{} \tPSNR: {:.3f} \tSSIM: {:.4f} '.format(
                            filename,
                            psnr,
                            ssim
                            )
                            )
                            
                            patch_list_sr.clear()
                            patch_list_hr.clear()
                            index+=1
                            
                    else:
                        
                        self.ckp.log[-1, idx_scale] += utility.calc_psnr(
                            sr.float(), hr.float()
                            )
#                    if self.args.save_gt:
#                        save_list.extend([lr, hr])
    
                    if self.args.save_results:
                        self.ckp.save_results(self.args.data_test, filename[0], save_list, self.scale)
                if self.args.test_only:
                    self.ckp.log[-1, idx_scale] /= len(self.loader_test.subject_list)
                    self.ckp.ssimlog[-1, idx_scale] /= len(self.loader_test.subject_list)
                    self.ckp.write_log(
                    '[{} x{}]\tSSIM: {:.4f} '.format(
                        self.args.data_test,
                        self.scale,
                        self.ckp.ssimlog[-1, idx_scale]
                        
                        )
                    )
                else:
                    self.ckp.write_log('Validation Loss:\t{}'.format(
                        self.loss.display_loss(self.loader_test.test_batches,'test'),
                        ))                   
                    self.ckp.log[-1, idx_scale] /= self.loader_test.test_batches
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        self.scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][ idx_scale],
                        best[1][ idx_scale] + 1
                    )
                )
    
            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')
    
            if self.args.save_results:
                self.ckp.end_background()
    
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
    
            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )
        self.loss.end_log(self.loader_test.test_batches,'test')
        torch.set_grad_enabled(True)

    def  prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]
        
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            
            

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()
        self.test_log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr,mode='train'):
        if mode=='test':
            losses = []
            for i, l in enumerate(self.loss):
                if l['function'] is not None:
                    loss = l['function'](sr, hr)
                    effective_loss = l['weight'] * loss.float()
                    losses.append(effective_loss)
                    self.test_log[-1, i] += effective_loss.item()
                elif l['type'] == 'DIS':
                    self.test_log[-1, i] += self.loss[i - 1]['function'].loss

            loss_sum = sum(losses)
            if len(self.loss) > 1:
                self.log[-1, -1] += loss_sum.item()
        else:
            losses = []
            for i, l in enumerate(self.loss):
                if l['function'] is not None:
                    loss = l['function'](sr, hr)
                    effective_loss = l['weight'] * loss.float()
                    losses.append(effective_loss)
                    self.log[-1, i] += effective_loss.item()
                elif l['type'] == 'DIS':
                    self.log[-1, i] += self.loss[i - 1]['function'].loss
    
            loss_sum = sum(losses)
            if len(self.loss) > 1:
                self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self,mode='train'):
        if mode=='train':
            self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))
        else:
            self.test_log=torch.cat((self.test_log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches,mode='train'):
        if mode=='train':
            self.log[-1].div_(n_batches)
        else:
            self.test_log[-1].div_(n_batches)

    def display_loss(self, batch,mode='train'):
        if mode=='test':
            n_samples = batch + 1
            log = []
            for l, c in zip(self.loss, self.test_log[-1]):
                log.append('[{}: {:.4f}]'.format(l['type'], (c/l['weight']) / n_samples))

            return ''.join(log) 
        else:

            n_samples = batch + 1
            log = []
            for l, c in zip(self.loss, self.log[-1]):
                log.append('[{}: {:.4f}]'.format(l['type'], (c/l['weight']) / n_samples))

            return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.text(0.75, 0.75, 'Scale {}'.format(self.args.scale[0]),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes)
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label='Training')
            plt.plot(axis, self.test_log[:, i].numpy(), label='Validation')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
        torch.save(self.test_log, os.path.join(apath, 'Validation_loss_log.pt'))
    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        self.test_log = torch.load(os.path.join(apath, 'Validation_loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()


checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint)
loss = Loss(args, checkpoint) if not args.test_only else None
if not args.test_only:
    trainset = lidc_idri.CTImageDataset(args,name=args.data_train,train=True)
    
    #train_loader = DataLoader(trainset, batch_size=args.batch_size,
    #                        shuffle=True)#, num_workers=4)
    evalset = lidc_idri.CTImageDataset(args,name=args.data_train,train=False)
    #eval_loader = DataLoader(evalset, batch_size=1,
    #                        shuffle=True)# ,num_workers=4)
    
    t=Trainer(args,[evalset,trainset],model,loss,checkpoint) 
    for epoch in range(args.epochs):
        
        t.train(epoch+1)
        t.test(epoch+1)
else:
    testset = lidc_idri.CTImageDataset(args,name=args.data_test,train=False)
    #test_loader = DataLoader(testset, batch_size=1,
    #                       shuffle=True, num_workers=args.n_threads,pin_memory=not args.cpu)
    
    t=Trainer(args,[testset],model,loss,checkpoint) 
    t.test(1)      
