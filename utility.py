import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
from skimage.measure import compare_ssim as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import nibabel
import glob
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.ssimlog = torch.Tensor()
        self.log = torch.Tensor()
        self.log2 = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
#            self.dir = os.path.join('..', 'experiment', args.save)
            self.dir = os.path.join('.', 'experiment', args.save)
        else:
            self.dir = os.path.join('.', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                self.log2 = torch.load(self.get_path('train_psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        #for d in args.data_test:
        os.makedirs(self.get_path('results-{}'.format(args.data_test)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self,trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        #self.plot_psnr(epoch,'test')
        #trainer.optimizer.save(self.dir)
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        torch.save(self.log, self.get_path('psnr_log.pt'))
        torch.save(self.log2, self.get_path('train_psnr_log.pt'))

    def add_log(self, log, train=False):
        if train:
            self.log2 = torch.cat([self.log2, log])
        else:
            self.log = torch.cat([self.log, log])
            self.ssimlog=torch.cat([self.ssimlog, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        
        label = 'SR on {}'.format(self.args.data_train)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.text(0.75, 0.25, 'Scale {}'.format(self.args.scale[0]),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
        plt.title(label)
            
        plt.plot(	axis,
                    self.log2[:, idx_scale].numpy(),
                    label='Training'
                    )
        plt.plot(
                    axis,
                    self.log[:, idx_scale].numpy(),
                    label='Validation'
                    )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)

        plt.savefig(self.get_path('PSNR.pdf'))
        plt.close(fig)
        

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset),
                '{}'.format(filename)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}.png'.format(filename), tensor_cpu))
    def aggregateandcalcpsnr(self,patches_sr,patches_hr,n_slices,subject_path_hr,ds):
        i=0
        patch=self.args.patch_size
        overlapping=patch//2
        filename = self.get_path(
                    'results-{}'.format(self.args.data_test),
                    '{}'.format(subject_path_hr.split('/')[-1])
                )
        os.makedirs(filename, exist_ok=True)
        whole_sr=torch.zeros(512,512,n_slices,device=torch.device('cpu'))
        whole_hr=torch.zeros(512,512,n_slices,device=torch.device('cpu'))
        if self.args.model== 'NDFSRCNN':
            for n_slice in range(n_slices):
                whole_sr[:-1,:-1,n_slice]=torch.from_numpy(patches_sr[n_slice])
                whole_hr[:-1,:-1,n_slice]=torch.from_numpy(patches_hr[n_slice])
        elif self.args.model== 'MDFSRCNN':
#            patch_r=torch.zeros(128,128,device=torch.device('cpu'))
            overlapping=256
            patch=256
            for depth in range(0,n_slices,64):
                for height in range(0,512,overlapping):
                    for width in range(0,512,overlapping):
                        start_x,start_y,start_z=width,height,depth
                        end_x,end_y,end_z=width+patch-1,height+patch-1,depth+64-1
                        if (end_x<=512) and (end_y<=512) and (end_z<=n_slices) :
    #                        whole_sr[start_x:end_x,start_y:end_y,start_z:end_z]+=torch.from_numpy(patches_sr[i])
                            
                            
                            
                            whole_hr[start_x:end_x,start_y:end_y,start_z:end_z]=torch.from_numpy(patches_hr[i])
                            whole_sr[start_x:end_x,start_y:end_y,start_z:end_z]=torch.from_numpy(patches_sr[i])
                            i+=1
                            
        else:
              
            for depth in range(0,n_slices,overlapping):
                for height in range(0,512,overlapping):
                    for width in range(0,512,overlapping):
                        start_x,start_y,start_z=width,height,depth
                        end_x,end_y,end_z=width+patch,height+patch,depth+patch
                        if (end_x<=512) and (end_y<=512) and (end_z<=n_slices) :
    #                        whole_sr[start_x:end_x,start_y:end_y,start_z:end_z]+=torch.from_numpy(patches_sr[i])
                            
                            
                            
                            whole_hr[start_x:end_x,start_y:end_y,start_z:end_z]=torch.from_numpy(patches_hr[i])
                            whole_sr[start_x:end_x,start_y:end_y,start_z:end_z]=torch.from_numpy(patches_sr[i])
    #                        if (not(start_x==0) and not(start_y==0)and not(start_z==0)):
    #                            whole_sr[start_x:end_x-32,start_y:end_y-32,start_z:end_z-32]/=2
    #                        elif (not(start_x==0) and not(start_y==0)):
    #                            whole_sr[start_x:end_x-32,start_y:end_y-32,start_z:end_z]/=2
    #                        elif (not(start_x==0) and not(start_z==0)):
    #                            whole_sr[start_x:end_x-32,start_y:end_y,start_z:end_z-32]/=2
    #                        elif (not(start_y==0) and not(start_z==0)):
    #                            whole_sr[start_x:end_x,start_y:end_y-32,start_z:end_z-32]/=2
    #                        elif not(start_x==0):
    #                            whole_sr[start_x:end_x-32,start_y:end_y,start_z:end_z]/=2
    #                        elif not(start_y==0):
    #                            whole_sr[start_x:end_x,start_y:end_y-32,start_z:end_z]/=2
    #                        elif not(start_z==0):
    #                            whole_sr[start_x:end_x,start_y:end_y,start_z:end_z-32]/=2
                            i+=1
        #whole_sr = quantize(whole_sr, 32767)
        #whole_hr = quantize(whole_hr, 32767)
        #whole_hr=torch.from_numpy(np.ascontiguousarray(np.array(np.load(glob.glob(subject_path_hr+'/full*.npy')[0]))).transpose(1,2,0)).float().cpu()
        psnr=calc_psnr(whole_sr,whole_hr)
        ssim=calc_ssim(whole_sr.numpy(),whole_hr.numpy())
        #print(whole_hr.shape)
        for i in range(n_slices):
            ds.PixelData=np.ascontiguousarray(np.around(whole_sr[:,:,i].numpy()),dtype=np.int16)
            ds.save_as(filename+'/'+str(i)+'_sr.dcm')
            ds.PixelData=np.ascontiguousarray(np.around(whole_hr[:,:,i].numpy()),dtype=np.int16)
            ds.save_as(filename+'/'+str(i)+'_hr.dcm')
        #nibabel.save(nibabel.Nifti1Image(whole_sr.numpy(),np.diag([1, 2, 3, 1])),filename+'_sr.nii')
        #nibabel.save(nibabel.Nifti1Image(whole_hr.numpy(),np.diag([1, 2, 3, 1])),filename+'_hr.nii')
        #print(whole_hr[:,:,i])
        return psnr, ssim
        
def quantize(img, rgb_range):
    
    pixel_range = 32767 / rgb_range
    return img.mul(pixel_range).clamp(-2048, 6916).round().div(pixel_range)
def calc_psnr(sr, hr, dataset=None):
    if hr.nelement() == 1: return 0
#    hr_=hr-torch.min(hr)
#    imax=torch.max(hr_)
#    hr_grey=(hr_/imax)*255
#    sr_=sr-torch.min(sr)
#    imax=torch.max(sr_)
#    sr_grey=(sr_/imax)*255
#    squared_error = (hr_grey- sr_grey).pow(2)
#    mse = squared_error.mean()
    squared_error = (hr- sr).pow(2)
    mse = squared_error.mean()
    amax=torch.max(hr)
    psnr = 10 * math.log10(amax**2/mse)
    
   
    
    
    
    return psnr

def calc_ssim(sr, hr, dataset=None):
    #if hr.nelement() == 1: return 0
        
    ssim_=ssim(hr, sr, gradient=False, data_range=None, multichannel=False, gaussian_weights=True, full=False)
    
   
    
    
    
    return ssim_
