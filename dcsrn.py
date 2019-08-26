import torch 
import torch.nn as nn
import common


def make_model(args):
    return DCSRN(args)
 
class DCSRN(nn.Module):
    def __init__(self,args,conv3D=common.default_conv3D):
        super(DCSRN,self).__init__()
        kernel_size=3
        growth_rate=24
        n_feats=2*growth_rate
        act=nn.ELU()
        n_resblocks=4
        self.head=conv3D(1,n_feats,kernel_size,bias=True)
                        
        
        self.resmodules=nn.ModuleList()
        for i in range(n_resblocks):
            n_feats_m=growth_rate*(2+i)
            self.resmodules.append(common.DenseBlock(conv3D,n_feats_m,growth_rate,kernel_size,act=act,bias=True))
        n_feats_m=growth_rate*(2+n_resblocks)    
        self.tail=conv3D(n_feats_m,1,kernel_size,bias=True)
                        
        
    def forward(self,x):
       x=self.head(x)
       
       res_list=[x]
       for i,p in enumerate(self.resmodules):
           
           if i==0:
               res_list.append(self.resmodules[i](x))
           else:
               
               res_list.append(self.resmodules[i](torch.cat(res_list,dim=1)))
           
       res=self.tail(torch.cat(res_list,dim=1))
       del res_list
       
       del x
       
       return res