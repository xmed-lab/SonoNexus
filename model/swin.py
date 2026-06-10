import timm
from timm.models.vision_transformer import Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

def info_nce_loss(feature1, feature2, temperature=0.07):
    """
    计算 InfoNCE Loss
    feature1: 原始特征，形状为 (batch_size, feature_dim)
    feature2: 增强特征，形状为 (batch_size, feature_dim)
    temperature: 温度参数，控制对比学习的难度
    """
    # 将特征进行 L2 归一化
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(feature1, feature2.T) / temperature  # 形状为 (batch_size, batch_size)

    # 构造标签，正样本的位置
    batch_size = feature1.size(0)
    labels = torch.arange(batch_size).to(similarity_matrix.device)  # 形状为 (batch_size)

    # 计算 InfoNCE Loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


class VisionUlt(nn.Module):

    def __init__(self, 
        in_channels: int = 3, 
        img_size: int = 224,
        patch_size: int = 8,
    ):

        super().__init__()
        self.patch_size = patch_size
        print("swin")
        self.model = timm.create_model(
            "swin_base_patch4_window7_224.ms_in22k",
            pretrained=False,
            depths=(2, 2, 18, 10),
            features_only=True,
        )
        self.model.patch_embed.proj = nn.Conv2d(3, 128, kernel_size=2, stride=2)

        out_channels = in_channels * patch_size ** 2
        decoder_depth = 4
        self.decoder_blocks = nn.Sequential(*[
            Block(512, 8, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        
        self.decoder_fc = nn.Linear(512, out_channels, bias=True) 

    def forward(self, x, mask):
        x_mask = x * (1 - mask)

        f = self.model(x)
        f3 = f[3].permute(0, 3, 1, 2)

        f_mask = self.model(x_mask)
        f2_mask, f3_mask = f_mask[2], f_mask[3].permute(0, 3, 1, 2)

        ### MAE recon
        batch_size, self.h, self.w, dim = f2_mask.size()
        f_up_mask = self.decoder_fc(self.decoder_blocks(f2_mask.reshape(batch_size, self.h*self.w, dim)))
        x_recon = self.unpatchify(f_up_mask)
        #print(x_recon.size())
        loss_mask, loss_unmask = self.forward_reconloss(x, x_recon, mask)
        #print(loss_mask, loss_unmask)

        ### Contrastive Learning
        f3 = F.avg_pool2d(f3, kernel_size=(14, 14))
        f3 = f3.squeeze(-1).squeeze(-1)

        f3_mask = F.avg_pool2d(f3_mask, kernel_size=(14, 14))
        f3_mask = f3_mask.squeeze(-1).squeeze(-1)

        loss_cl = info_nce_loss(f3, f3_mask)
        #print(loss_cl)
        return x_recon, loss_mask, loss_unmask, loss_cl

    def forward_reconloss(self, x, x_out, mask):
        loss = torch.abs(x - x_out) #** 2
        loss = loss.mean(1)

        loss_mask = (loss * mask[:,0,:,:]).sum() / mask[:,0,:,:].sum()
        loss_unmask = (loss * (1-mask[:,0,:,:])).sum() / (1-mask[:,0,:,:]).sum()

        return loss_mask, loss_unmask

    def unpatchify(self, x):
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, self.h*self.patch_size, self.w*self.patch_size))
        return imgs

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

    x = torch.randn(3, 3, 224, 224)
    mask = torch.randint(0, 2, (3, 1, 224, 224)).float()

    model = VisionUlt()
    x_recon, loss1, loss_2, loss_3 = model(x, mask)
    print(loss1, loss_2, loss_3)
    print(x_recon.size())
