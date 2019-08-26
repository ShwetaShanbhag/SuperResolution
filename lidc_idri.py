# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:34:41 2019

@author: shanbhsa
"""


import random
import torch
import os
import pydicom
import numpy as np
from matplotlib import pyplot, cm
import glob
from skimage.util import view_as_windows
import torch.utils.data as data
import nibabel
import re
import os.path
from option import args
class CTImageDataset(object):
    def __init__(self,args,name='LIDC-IDRI',train=True):
        self.args=args
        self.scale=args.scale
        if not self.args.test_only: 
            self.name=args.data_train 
        else: 
            self.name=args.data_test
        self.input_large=args.input_large
        self._set_filesystem(args.dir_data)
        self.width_patch=args.patch_size
        self.height_patch=args.patch_size
        self.depth_patch=args.patch_size
        self.overlapping=args.overlapping
        self.test_only=args.test_only
        self.subject_list=os.listdir(self.dir_lr)
        self.train=train
        from itertools import chain
        
        
        
        #self.subject_list.remove('.DS_Store') 
        
               

        self.images_hr=[]
        self.images_lr=[]
        if self.train:
            
            self.train_subject_list=self.subject_list[:int(args.partition*len(self.subject_list))]
            #print(self.train_subject_list)
            if args.prepare_data:
                for subject in self.train_subject_list:
                
                    subject_path_hr=os.path.join(self.dir_hr,subject)
                    subject_path_lr=os.path.join(self.dir_lr,subject)
                    print('File ', subject_path_hr )
                    id=int(subject.split('-')[-1])
                    
                    self._scan_and_stack(subject_path_hr,subject_path_lr,id)
                   # self.get_patch_and_save(subject_path_hr,subject_path_lr,id)
            
            if self.args.model in ['DCRSR','MDFSRCNN','DCSRN','EDSR']:
                self.images_hr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_hr,sub,'patch*.nii')),self.train_subject_list))))
                self.images_lr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_lr,sub,'patch*.nii')),self.train_subject_list))))
                self.images_hr = sorted(self.images_hr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                self.images_lr= sorted(self.images_lr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
#            print(self.images_hr[:15])
            
                random.seed(4)
                random.shuffle(self.images_hr)
                random.seed(4)
                random.shuffle(self.images_lr)
            #self.images_lr=glob.glob(os.path.join(self.dir_lr,'*','*_*.npy'))
            
                self.train_batches=len(self.images_hr)//self.args.batch_size
            else:
                self.images_hr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_hr,sub,'*.dcm')),self.train_subject_list))))
                self.images_lr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_lr,sub,'*.dcm')),self.train_subject_list))))
                self.images_hr = sorted(self.images_hr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                self.images_lr= sorted(self.images_lr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                self.train_batches=(len(self.images_hr))//self.args.batch_size
                print(len(self.images_hr),len(self.images_lr))
            
            #print(self.train_subject_list,len(self.train_subject_list),len(self.images_hr),self.train_batches)
#            n_patches = args.batch_size * args.test_every
#            n_images = len(args.data_train) * len(self.images_hr)
#            if n_images == 0:
#                self.repeat = 0
#            else:
#                self.repeat = max(n_patches // n_images, 1)
        else:
            if not self.args.test_only:
                self.test_subject_list=self.subject_list[int(args.partition*len(self.subject_list)):len(self.subject_list)] 
                
                if args.prepare_data:
                    for subject in self.test_subject_list:
                    
                        subject_path_hr=os.path.join(self.dir_hr,subject)
                        subject_path_lr=os.path.join(self.dir_lr,subject)
                        print('File ', subject_path_hr )
                        id=int(subject.split('-')[-1])
                        
                        self._scan_and_stack(subject_path_hr,subject_path_lr,id)
                        #self.get_patch_and_save(subject_path_hr,subject_path_lr,id)
            
                if self.args.model in ['DCRSR','MDFSRCNN','DCSRN','EDSR']:
                    self.images_hr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_hr,sub,'patch*.nii')),self.test_subject_list))))
                    self.images_lr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_lr,sub,'patch*.nii')),self.test_subject_list)))) 
                    self.images_hr = sorted(self.images_hr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                    self.images_lr= sorted(self.images_lr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                    print(len(self.images_hr),len(self.images_lr))
                else:
                    self.images_hr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_hr,sub,'*.dcm')),self.test_subject_list))))
                    self.images_lr=list(chain.from_iterable(list(map(lambda sub: glob.glob(os.path.join(self.dir_lr,sub,'*.dcm')),self.test_subject_list))))
                    self.images_hr = sorted(self.images_hr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                    self.images_lr= sorted(self.images_lr, key=lambda x:float(re.findall("(\d+)",os.path.splitext(os.path.basename(x))[0])[0]))
                    print(len(self.images_hr),len(self.images_lr))
#                random.seed(4)
#                random.shuffle(self.images_hr)
#                random.seed(4)
#                random.shuffle(self.images_lr)          
#            self.images_hr=glob.glob(os.path.join(self.dir_hr,'*','*_*.npy'))
#            self.images_lr=glob.glob(os.path.join(self.dir_lr,'*','*_*.npy'))
            else:
                self.subject_list=os.listdir(self.dir_lr)
                self.test_slices=[]
                self.info=[]
                self.n_patches=[]
                #print(self.subject_list)
                if self.args.model in ['DCRSR','MDFSRCNN','DCSRN','EDSR']:
                    for subject in self.subject_list:
                        self.test_slices.append(len(glob.glob(os.path.join(self.dir_hr,subject,'*.dcm'))))
                        self.images_hr.extend( sorted(glob.glob(os.path.join(self.dir_hr,subject,'patch*.nii')),key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0])))
                        self.images_lr.extend( sorted(glob.glob(os.path.join(self.dir_lr,subject,'patch*.nii')),key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0])))
                        self.info.append(pydicom.dcmread(os.path.join(self.dir_hr,subject,'000001.dcm')))
                        self.n_patches.append(len(glob.glob(os.path.join(self.dir_lr,subject,'patch*.nii'))))
                else:
                    for subject in self.subject_list:
                        self.test_slices.append(len(glob.glob(os.path.join(self.dir_hr,subject,'*.dcm'))))
                        self.images_hr.extend( sorted(glob.glob(os.path.join(self.dir_hr,subject,'*.dcm')),key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0])))
                        self.images_lr.extend( sorted(glob.glob(os.path.join(self.dir_lr,subject,'*.dcm')),key=lambda x:float(re.findall("(\d+)",os.path.basename(x))[0])))
                        self.info.append(pydicom.dcmread(os.path.join(self.dir_hr,subject,'000001.dcm')))
                    self.n_patches=self.test_slices
                        
                #print(self.images_hr[:30],self.images_lr[:30])
                #print(self.n_patches)
                #print(self.test_slices)
#            if self.args.model in ['DCRSR','3DFSRCNN','DCSRN']:
            self.test_batches=len(self.images_lr)
#            else:
#            
    def _set_filesystem(self,dir_data):
        self.dir_hr=os.path.join(dir_data,self.name)
        if self.args.input_large:
            self.dir_lr=os.path.join(dir_data,self.name +'_lanczos2_big','X2')
        elif self.args.model == 'MDFSRCNN':
            self.dir_lr=os.path.join(dir_data,self.name +'_cubic','X2')
        elif self.args.model == 'NDFSRCNN':
            self.dir_lr=os.path.join(dir_data,self.name +'_cubic2D','X2')
        else:
            self.dir_lr=os.path.join(dir_data,self.name +'_lanczos2','X2')
        self.ext=['.nii','.dcm']
        if self.args.test_only:
           self.dir_hr=os.path.join(dir_data,self.name+'_HR')
           self.dir_lr=os.path.join(dir_data,self.name +'_LR') 
        
    def _scan_and_stack(self,subject_path_hr,subject_path_lr,id):
        slices_lr=[]
        for s in os.listdir(subject_path_lr):
            if s.find('.dcm')>= 0:
                slices_lr.append(pydicom.dcmread(subject_path_lr+'/'+s)) 
        slices_lr.sort(key=lambda x: int(x.InstanceNumber))
        images_lr=np.stack([s.pixel_array for s in slices_lr])
        images_lr=images_lr.astype(np.int16)
        step=(self.depth_patch//2,self.height_patch//2,self.width_patch//2)
        window_shape=(self.depth_patch//2,self.height_patch//2,self.width_patch//2)
        patches=view_as_windows(images_lr,window_shape,step)
        idd=1
        for d in range(patches.shape[0]):
                for v in range(patches.shape[1]):
                    for h in range(patches.shape[2]):
                        p = patches[d, v, h, :]
                        p = p[:, np.newaxis]
                        
                        p = p.transpose((0, 2, 3, 1))
                        
                        np.save(subject_path_lr+'/'+str(id)+"_%d.npy" %(idd),p)
                        idd+=1
        
        del patches
        #np.save(subject_path_lr+'/'+"fullimages%d.npy" %(id),images_lr)
        
        (slices,a,aa)=images_lr.shape
        slices_hr=[]
        for s in os.listdir(subject_path_hr):
            if s.find('.dcm')>= 0:
                slices_hr.append(pydicom.dcmread(subject_path_hr+'/'+s)) 
        
        slices_hr.sort(key=lambda x: int(x.InstanceNumber))
        images_hr=np.stack([s.pixel_array for s in slices_hr])
        images_hr=images_hr.astype(np.int16)
        patches=view_as_windows(images_hr[:slices*2,:,:],(self.depth_patch,self.height_patch,self.width_patch),step=(self.depth_patch,self.height_patch,self.width_patch))
        idd=1
        for d in range(patches.shape[0]):
                for v in range(patches.shape[1]):
                    for h in range(patches.shape[2]):
                        p = patches[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        np.save(subject_path_hr+'/'+str(id)+"_%d.npy" %(idd),p)
                        idd+=1
        del patches
        #np.save(subject_path_hr+'/'+"fullimages%d.npy" %(id),images_hr[:slices*2,:,:])
        
        
#    def get_patch_and_save(self,subject_path_hr,subject_path_lr,id):
#        print('Processing ',subject_path_hr)
#        hr_file=np.ascontiguousarray(np.array(np.load(glob.glob(subject_path_hr+'/full*.npy')[0])))#.transpose(1,2,0))
#        lr_file=np.ascontiguousarray(np.array(np.load(glob.glob(subject_path_lr+'/full*.npy')[0])))#.transpose(1,2,0))
#        
#        print('  Data shape is ' + str(hr_file.shape) + ' .')
#        patches=view_as_windows(hr_file,(self.depth_patch,self.height_patch,self.width_patch),step=(self.depth_patch,self.height_patch,self.width_patch))
#        idd=1
#        for d in range(patches.shape[0]):
#                for v in range(patches.shape[1]):
#                    for h in range(patches.shape[2]):
#                        p = patches[d, v, h, :]
#                        p = p[:, np.newaxis]
#                        p = p.transpose((0, 2, 3, 1))
#                        np.save(subject_path_hr+'/'+str(id)+"_%d.npy" %(idd),p)
#                        idd+=1
#        del patches
#        
#        print('  Data shape is ' + str(lr_file.shape) + ' .')
#        step=(self.depth_patch//2,self.height_patch//2,self.width_patch//2)
#        window_shape=(self.depth_patch//2,self.height_patch//2,self.width_patch//2)
#        
#        patches=view_as_windows(lr_file,window_shape,step)
#        idd=1
#        for d in range(patches.shape[0]):
#                for v in range(patches.shape[1]):
#                    for h in range(patches.shape[2]):
#                        p = patches[d, v, h, :]
#                        p = p[:, np.newaxis]
#                        
#                        p = p.transpose((0, 2, 3, 1))
#                        
#                        np.save(subject_path_lr+'/'+str(id)+"_%d.npy" %(idd),p)
#                        idd+=1
#        
#        del patches
        
    def _getitem_and_make_batch(self,batch_idx,batch_size):
        lr_batch=torch.Tensor()
        hr_batch=torch.Tensor()
        for img_idx in range(batch_size):
            lr, hr, _ = self._load_file(self._get_index(img_idx,batch_idx,batch_size))
            
            #print('load_file',hr.dtype)
            #pair_t = self._np2Tensor(lr, hr)
            lr_batch=torch.cat((lr_batch,lr.unsqueeze(0)),0)
            hr_batch=torch.cat((hr_batch,hr.unsqueeze(0)),0)
            #print('batchsize',hr_batch.size())
        return lr_batch, hr_batch

#    def __len__(self):
#        if self.train:
#            return len(self.images_hr) * self.repeat
#        else:
#            return len(self.images_hr)

    def _get_index(self, img_idx,batch_idx,batch_size):
        if self.train:
            return batch_idx*batch_size+img_idx
        else:
            return batch_idx*batch_size+img_idx

    def _load_file(self, idx):
        #idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        
        #hr = imageio.imread(f_hr)
        #lr = imageio.imread(f_lr)
        if self.args.model in ['NDFSRCNN']:
            hr=pydicom.dcmread(f_hr)
            lr=pydicom.dcmread(f_lr)
            hr=hr.pixel_array
            lr=lr.pixel_array
            hr=hr[:-1,:-1]
            hr=hr[np.newaxis,:]
            lr=lr[np.newaxis,:]
            
            np_transpose = np.ascontiguousarray(hr.transpose((0, 1, 2)))
            hr=torch.from_numpy(np_transpose.astype('int16') ).float()
            np_transpose1 = np.ascontiguousarray(lr.transpose((0, 1, 2)))
            lr=torch.from_numpy(np_transpose1.astype('int16') ).float()
            
        else:
                
            hr=nibabel.load(f_hr).get_data()
            lr=nibabel.load(f_lr).get_data()
            #print('Data Shape from nii file', hr.shape, f_hr)
            if self.args.model in ['MDFSRCNN']:
                hr=hr[:-1,:-1,:-1]
                
            hr=hr[np.newaxis,:]
            lr=lr[np.newaxis,:]
    
            np_transpose = np.ascontiguousarray(hr.transpose((0, 3, 1, 2)))
            hr=torch.from_numpy(np_transpose.astype('int16') ).float()
            np_transpose1 = np.ascontiguousarray(lr.transpose((0, 3, 1, 2)))
            lr=torch.from_numpy(np_transpose1.astype('int16') ).float()
        
        #print('before batch',hr.shape)
        return lr, hr, filename

               
        

        
