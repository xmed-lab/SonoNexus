import argparse
import os
import random

import numpy as np
import torch

from dataset_mae_cnn import get_data
from model.swin import VisionUlt
from train_mae_cnn import Train_MAE
from training_utils import warmup_cosine_scheduler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SonoNexus self-supervised pretraining")
    parser.add_argument("-data", "--data", default="./datasets", help="Pretraining image folder")
    parser.add_argument("-j", "--workers", default=32, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("-b", "--batch_size", "--batch-size", default=512, type=int)
    parser.add_argument("--base-lr", default=1.5e-4, type=float, help="LR at total batch size 256")
    parser.add_argument("--lr", "--learning-rate", default=None, type=float, help="Optional absolute LR override")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", default=0.05, type=float)
    parser.add_argument("--warmup-epochs", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--fp16-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--cl-weight", default=1.0, type=float)
    parser.add_argument("--mask-min", default=0.1, type=float)
    parser.add_argument("--mask-max", default=0.8, type=float)
    parser.add_argument("--imgsize", default=224, type=int)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--visualize-every", default=200, type=int, help="0 disables reconstruction previews")
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    args.fp16_precision = args.fp16_precision and use_cuda
    args.arch = "swin_base_patch2_window7_224"
    args.lr = args.lr if args.lr is not None else args.base_lr * args.batch_size / 256

    train_loader = get_data(
        data_root=args.data, batch_size=args.batch_size, num_workers=args.workers,
        img_size=args.imgsize, patch_size=8, lower_ratio=args.mask_min, upper_ratio=args.mask_max,
        two_views=True,
    )
    model = VisionUlt(img_size=args.imgsize, temperature=args.temperature).to(args.device)
    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scheduler = warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs)
    Train_MAE(model=model, optimizer=optimizer, scheduler=scheduler, args=args).train(train_loader)


if __name__ == "__main__":
    main()
