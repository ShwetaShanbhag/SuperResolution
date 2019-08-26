import torch
import torch.nn as nn
import common


def make_model(args):
    return DCRSR(args)
 
class DCRSR(nn.Module):
    def __init__(self,args,conv3D=common.default_conv3D):
        super(DCRSR,self).__init__()
        n_dresblocks=args.n_dresblocks
        n_feats=args.n_feats
        kernel_size=3
        res_scale=args.res_scale
        n_resblocks=args.n_resblocks
        act=nn.ReLU(True)
        branched=args.branched
        branches=args.branches
        self.scale = args.scale[0]
        
        self.head=conv3D(1,n_feats,kernel_size)
        
        con_feats=n_feats
        self.dresmodules=nn.ModuleList()        
        for i in range(n_dresblocks):
            self.dresmodules.append(
                common.DResBlock(conv3D,con_feats,n_feats,kernel_size,n_resblocks,act=act,res_scale=res_scale,branched=branched,branches=branches)
                )
            con_feats=n_feats*2

            
        self.body=nn.Sequential(
                        conv3D(con_feats,n_feats,1),
                        act
  #                      conv3D(n_feats,n_feats,kernel_size)
                        )
        self.tail=nn.Sequential(
                        common.Upsampler(conv3D,self.scale,n_feats),
                        #nn.Upsample(scale_factor=scale),
                        
                        conv3D(n_feats,1,kernel_size)
                        )
        self.res_scale=res_scale
    def forward(self,x):
        x=self.head(x)
        cur_input=x
        prev_output=x
        for i,p in enumerate(self.dresmodules):
            cur_output=self.dresmodules[i](cur_input)
            cur_input=torch.cat((cur_output,prev_output),dim=1)
            prev_output=cur_output
        del prev_output, cur_output
        res=self.body(cur_input).mul(self.res_scale)
        del cur_input
        res += x
        #res=nn.functional.interpolate(res, scale_factor=self.scale, mode='trilinear', align_corners=True)
        x = self.tail(res)
        del res
        return x    
        
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
