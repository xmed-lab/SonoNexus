import torch
import torch.nn as nn
import torch.nn.functional as F
from .vqvae import Codebook


class MedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=2, 
                kernel_size:int=7, 
                do_res:int=True,
                norm_type:str = 'group',
                dim = '3d',
                grn = True
                ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
            
        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            print("grn")
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

 
    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1  
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=True, norm_type = 'group', dim='2d', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size, 
                        do_res = False, norm_type = norm_type, dim=dim,
                        grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1

    
    
class Encoder_MedNext(nn.Module):

    def __init__(self, 
        in_channels: int = 3, 
        n_channels: int = 32,
        kernel_size: int = 5,                      # Ofcourse can test kernel_size
        norm_type = 'group',
        dim = '2d',                                # 2d or 3d
        grn = False,
        mode = "M",
        patch_size = 16,
        img_size = 304,
    ):

        super().__init__()
        print("mednext {} with channels {}".format(mode, n_channels))
        assert dim in ['2d', '3d']
        
        if mode == "S":
            exp_r=[2,2,2,2,2]     
            block_counts = [2,2,2,2,2]
        elif mode == "B":
            exp_r=[2,3,4,4,4]     
            block_counts = [2,2,2,2,2]
        elif mode == "M":
            exp_r = [2,3,4,4,4]
            block_counts= [3,4,4,4,4]
        elif mode == "L":
            #exp_r=[3,4,8,8,8]
            exp_r=[1,4,8,8,8]
            block_counts = [3,4,8,8,8]

    
        enc_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        self.stem = conv(in_channels, n_channels, kernel_size=1)
        
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
    
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )            
            for i in range(block_counts[3])]
        )

        # self.down_3 = MedNeXtDownBlock(
        #     in_channels=8*n_channels,
        #     out_channels=16*n_channels,
        #     exp_r=exp_r[4],
        #     kernel_size=enc_kernel_size,
        #     norm_type=norm_type,
        #     dim=dim,
        #     grn=grn
        # )

        # self.enc_block_4 = nn.Sequential(*[
        #     MedNeXtBlock(
        #         in_channels=n_channels*16,
        #         out_channels=n_channels*16,
        #         exp_r=exp_r[4],
        #         kernel_size=enc_kernel_size,
        #         norm_type=norm_type,
        #         dim=dim,
        #         grn=grn
        #         )            
        #     for i in range(block_counts[4])]
        # )
        

        out_channels = in_channels * patch_size ** 2
        self.out = nn.Sequential(
            MedNeXtBlock(in_channels=n_channels*8, out_channels=n_channels*8, exp_r=2, kernel_size=enc_kernel_size, norm_type=norm_type, dim=dim, grn=grn),
            MedNeXtBlock(in_channels=n_channels*8, out_channels=n_channels*8, exp_r=2, kernel_size=enc_kernel_size, norm_type=norm_type, dim=dim, grn=grn),
            conv(n_channels*8, out_channels, 3, padding=1),
        )
        self.patch_size = patch_size

        #self.codebook = Codebook(latent_dim=n_channels*8, num_codebook_vectors=2048*2)

    def forward_feats(self, x, mask):
        
        x = self.stem(x)
        x_res_0 = self.enc_block_0(x)
        x_res_0 = x_res_0 * (1-mask)

        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)

        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)

        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)

        # x = self.down_3(x_res_3)
        # x_res_4 = self.enc_block_4(x)

        #mask_down = F.avg_pool2d(mask, self.patch_size)

        #x_res_3, loss_code = self.codebook(x_res_3, mask_down)
        
        x_out = self.out(x_res_3)
        return x_out#, loss_code

    def forward(self, x, mask):
        #x_patch = self.patchify(x)

        x_out = self.forward_feats(x, mask)
        x_out = self.unpatchify(x_out)
        loss_mask, loss_unmask = self.forward_loss(x, x_out, mask)
        return x_out, loss_mask, loss_unmask #, 0#, loss_code

    def forward_loss(self, x_patch, x_out, mask):
        loss = torch.abs(x_patch - x_out) #** 2
        loss = loss.mean(1)

        loss_mask = (loss * mask[:,0,:,:]).sum() / mask[:,0,:,:].sum()
        loss_unmask = (loss * (1-mask[:,0,:,:])).sum() / (1-mask[:,0,:,:]).sum()

        return loss_mask, loss_unmask

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h, w, p**2 * 3)).permute(0, 3, 1, 2)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h, w = x.shape[2], x.shape[3]
        assert h == w
        
        x = x.reshape(shape=(x.shape[0], p, p, 3, h, w))
        x = torch.einsum('npqchw->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    

if __name__ == '__main__':
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    encoder = Encoder_MedNext(
        mode = "M",
        kernel_size = 5,
        img_size = 304,
    ).cuda()

    print(encoder)
    bs = 1
    H = 304
    a = torch.randn(bs,3,H,H).cuda()
    mask = torch.randn(bs,1,H,H).cuda()
    t_b = time.time()
    loss_1, loss_2 = encoder(a, mask)
    t_e = time.time()
    print("runing time is {}".format(t_e - t_b))
    print("loss_mask {}; loss_unmask {}".format(loss_1, loss_2))
    
    time.sleep(30)
    