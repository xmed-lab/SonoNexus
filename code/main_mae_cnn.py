import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from timm.models import create_model
from dataset_mae_cnn import get_data
from train_mae_cnn import Train_MAE
from model.mednext import Encoder_MedNext
from model.swin import VisionUlt
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MAE MedNext')
parser.add_argument('--model', type=str, default='mednext_m')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mednext_m',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--mask', default=0.75, type=float, help='mask ratio.')
parser.add_argument('--imgsize', default=304, type=int, help='image size')
parser.add_argument('--gpus', type=str, default='0,1,2,3')
import torch.nn as nn
import timm
import os


def main():
    args = parser.parse_args()
    args.fp16_precision = True
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    train_loader = get_data(batch_size=args.batch_size, num_workers=args.workers, img_size=args.imgsize, patch_size=8)
    print("load {}".format(args.model))

    #model = Encoder_MedNext(mode="M", kernel_size=5, img_size=args.imgsize, patch_size=8) 
    model = VisionUlt()
    model = torch.nn.DataParallel(model).cuda()
    #{"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, betas=(0.9, 0.98), eps=1e-06)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5.0e-6)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        trainer = Train_MAE(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        trainer.train(train_loader)


if __name__ == "__main__":
    main()
