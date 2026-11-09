import logging
import os
import sys
import wandb
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import shutil
import matplotlib
matplotlib.use("Agg")  # Training saves previews without opening native GUI windows.
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        config = {key: str(value) if isinstance(value, torch.device) else value
                  for key, value in vars(args).items()}
        yaml.safe_dump(config, outfile)

def psnr(img1, img2):
    data_range = 255
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    err = np.mean((img1 - img2) ** 2, dtype=np.float64)
    if err == 0:
        return float('inf')
    return 10 * np.log10((data_range**2) / err)

def visualize(x, x_out, masks, ratios, it, dir):
    N = min(10, x.shape[0])
    imgs = []
    mks = []
    outs = []
    ratios = np.round(ratios.numpy(), 2)

    std = np.array([[[0.229, 0.224, 0.225]]])
    mean = np.array([[[0.485, 0.456, 0.406]]])
    for i in range(N):
        img = x[i].permute(1,2,0).detach().cpu().numpy() * std * 255 + mean * 255
        mask = masks[i].permute(1,2,0).detach().cpu().numpy() 
        out = x_out[i].permute(1,2,0).detach().cpu().numpy() * std * 255 + mean * 255
        img_mask = img * (1-mask)

        img = np.clip(img, 0, 255).astype(int)
        img_mask = np.clip(img_mask, 0, 255).astype(int)
        out = np.clip(out, 0, 255).astype(int)

        imgs.append(img)
        mks.append(img_mask)
        outs.append(out)

    fig, axs = plt.subplots(3, N, figsize=(2.5 * N, 10), squeeze=False)
    for i, ax in enumerate(axs[0]):
        ax.imshow(imgs[i])
        ax.axis('off')  # 隐藏坐标轴
        ax.set_title(f"Original {i+1}")
    for i, ax in enumerate(axs[1]):
        ax.imshow(mks[i])
        ax.axis('off')  # 隐藏坐标轴
        ax.set_title("ratio {}".format(ratios[i]))
    for i, ax in enumerate(axs[2]):
        ps = psnr(imgs[i], outs[i])
        ps = round(ps, 3)
        ax.imshow(outs[i])
        ax.axis('off')  # 隐藏坐标轴
        ax.set_title("PSNR {}".format(ps))

    plt.savefig(os.path.join(dir, "test_{}.png".format(it)))
    plt.close(fig)


class Loss_Rec():
    def __init__(self):
        self.avg_loss = 0
        self.cnt = 0
    def __call__(self, loss):
        self.avg_loss = (loss + self.avg_loss * self.cnt) / (self.cnt + 1)
        self.cnt += 1

class Train_MAE(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        self.init_wandb()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.mkdirs()

    def init_wandb(self):
        self.wandb = wandb.init(
            project="SonoNexus",
            mode=self.args.wandb_mode,
            config={key: str(value) if isinstance(value, torch.device) else value
                    for key, value in vars(self.args).items()},
        )

    def mkdirs(self):
        save_dir = self.writer.log_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_image_dir = os.path.join(save_dir, "images")
        os.makedirs(self.save_image_dir, exist_ok=True)

    def train(self, train_loader):
        scaler = GradScaler("cuda", enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SonoNexus pretraining for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        print("begin training")
        for epoch_counter in range(self.args.epochs):
            self.model.train()
            print("current epoch: {} with lr {}".format(epoch_counter, self.scheduler.get_last_lr()[0]))
            pbar = tqdm(train_loader)
            loss_mask_rec = Loss_Rec()
            loss_unmask_rec = Loss_Rec()
            loss_cl_rec = Loss_Rec()
            itera = 0
            for data_batch in pbar:
                views, masks, ratios = data_batch
                if isinstance(views, (tuple, list)):
                    images, masked_view = (view.to(self.args.device) for view in views)
                else:
                    images = views.to(self.args.device)
                    masked_view = images
                masks = masks.to(self.args.device)
                with autocast(device_type=self.args.device.type, dtype=torch.float16,
                              enabled=self.args.fp16_precision):
                    x_out, loss_mask, loss_unmask, loss_cl = self.model(images, masks, masked_view)

                    loss_mask = loss_mask.mean()
                    loss_unmask = loss_unmask.mean() 
                    loss_cl = loss_cl.mean()

                    loss = loss_mask + loss_unmask + self.args.cl_weight * loss_cl
                    #print(loss)
                    

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_mask_rec(loss_mask.item())
                loss_unmask_rec(loss_unmask.item())
                loss_cl_rec(loss_cl.item())

                dic = {}
                dic["loss_mask"] = loss_mask_rec.avg_loss
                dic["loss_unmask"] = loss_unmask_rec.avg_loss
                dic["loss_cl"] = loss_cl_rec.avg_loss
                dic["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                #dic["loss_code"] = loss_code_rec.avg_loss
                self.wandb.log(dic)
                pbar.set_description('Loss_mask: {:.4f}; Loss_unmask: {:.4f};  Loss_cl: {:.4f} in Epoch: {}'.format(loss_mask_rec.avg_loss, loss_unmask_rec.avg_loss, loss_cl_rec.avg_loss, epoch_counter))
                n_iter += 1

                if self.args.visualize_every > 0 and itera % self.args.visualize_every == 0:
                    visualize(masked_view, x_out, masks, ratios, n_iter, self.save_image_dir)
                itera += 1
            if epoch_counter % 1 == 0:
                print("save the {}-th model".format(epoch_counter))
                torch.save(self.model.state_dict(), os.path.join(self.writer.log_dir, 'timm_model_{}.pth'.format(epoch_counter)))
            self.scheduler.step()
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'args': {key: str(value) if isinstance(value, torch.device) else value
                     for key, value in vars(self.args).items()},
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        torch.save(self.model.state_dict(),os.path.join(self.writer.log_dir, 'timm_model.pth'))
        #model.load_state_dict(torch.load('./checkpoint/timm_model.pth'))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        self.writer.close()
        self.wandb.finish()
