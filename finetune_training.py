"""Paper fine-tuning settings shared by the three downstream demos.

Dataset hooks remain in each demo. Validation AUC selects the checkpoint and
controls early stopping (pixel-wise macro AUC for segmentation).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision import transforms

from training_utils import warmup_cosine_scheduler


def add_training_arguments(parser, default_output):
    parser.add_argument('--checkpoint', default='', help='Full pretraining or encoder-only checkpoint')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', '--lora-lr', dest='lr', type=float, default=2e-4)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--dummy-samples', type=int, default=64)
    parser.add_argument('--dummy-val-samples', type=int, default=64)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--output', default=default_output, help='Best validation-AUC checkpoint')
    parser.add_argument('--seed', type=int, default=2026)


def get_finetune_transform(img_size=224, training=True):
    """Use in real classification/diagnosis dataset hooks; no jitter or blur."""
    steps = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()] if training else [
        transforms.Resize((img_size, img_size))
    ]
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_optimizer(model, args):
    lora = [param for param in model.encoder.parameters() if param.requires_grad]
    head = [param for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith('encoder.')]
    return torch.optim.AdamW([
        {'params': lora, 'lr': args.lr, 'name': 'lora'},
        {'params': head, 'lr': args.head_lr, 'name': 'task_head'},
    ], betas=(0.9, 0.999), weight_decay=args.weight_decay)


def task_loss(logits, targets, task, label_smoothing=0.1):
    if task == 'classification' or (task == 'segmentation' and logits.shape[1] > 1):
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    # Binary/multilabel analogue of smoothing toward the uniform distribution.
    smoothed = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    loss = F.binary_cross_entropy_with_logits(logits, smoothed)
    if task == 'segmentation':
        probs = logits.sigmoid()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        loss = loss + 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return loss


def auc_inputs(logits, targets, task):
    categorical = task == 'classification' or (task == 'segmentation' and logits.shape[1] > 1)
    if categorical:
        probs = logits.softmax(dim=1)
        classes = logits.shape[1]
        if task == 'segmentation':
            probs = probs.movedim(1, -1).reshape(-1, classes)
            targets = targets.flatten()
        truth = F.one_hot(targets.long(), classes)
    else:
        probs = logits.sigmoid()
        truth = targets
        if task == 'segmentation':
            probs = probs.movedim(1, -1).reshape(-1, logits.shape[1])
            truth = truth.movedim(1, -1).reshape_as(probs)
    return probs.detach().cpu(), truth.detach().cpu()


def validation_auc(probabilities, targets):
    probs = torch.cat(probabilities).numpy()
    truth = torch.cat(targets).numpy()
    scores = []
    for column in range(truth.shape[1]):
        if len(np.unique(truth[:, column])) != 2:
            raise ValueError('Validation AUC requires positive and negative examples for every class/label.')
        scores.append(roc_auc_score(truth[:, column], probs[:, column]))
    return float(np.mean(scores))


def run_epoch(model, loader, device, task, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, samples = 0.0, 0
    predictions, labels = [], []
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, desc='train' if training else 'validation', ncols=100):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = task_loss(logits, targets, task, args.label_smoothing)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad), args.grad_clip,
                )
                optimizer.step()
            else:
                probs, truth = auc_inputs(logits, targets, task)
                predictions.append(probs)
                labels.append(truth)
            total_loss += loss.item() * len(images)
            samples += len(images)
    if not samples:
        raise ValueError('Training and validation loaders must both be nonempty.')
    result = {'loss': total_loss / samples}
    if not training:
        result['auc'] = validation_auc(predictions, labels)
    return result


def fit_downstream(model, train_loader, val_loader, device, args, task):
    optimizer = build_optimizer(model, args)
    scheduler = warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_auc, stale_epochs = -float('inf'), 0
    for epoch in range(args.epochs):
        train = run_epoch(model, train_loader, device, task, args, optimizer)
        validation = run_epoch(model, val_loader, device, task, args)
        print(f"Epoch {epoch + 1}: train_loss={train['loss']:.4f}, "
              f"val_loss={validation['loss']:.4f}, val_auc={validation['auc']:.4f}")
        scheduler.step()
        if validation['auc'] > best_auc:
            best_auc, stale_epochs = validation['auc'], 0
            torch.save({
                'model': model.state_dict(), 'args': vars(args), 'epoch': epoch + 1,
                'validation_auc': best_auc, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'Early stopping after {stale_epochs} epochs without validation-AUC improvement.')
                break
    print(f'Saved best {task} checkpoint to {output} (validation AUC={best_auc:.4f})')
