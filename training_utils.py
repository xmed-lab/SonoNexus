import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_scheduler(optimizer, epochs, warmup_epochs):
    """Epoch-based linear warm-up followed by cosine decay to zero."""
    if epochs <= 0 or warmup_epochs < 0:
        raise ValueError("epochs must be positive and warmup_epochs nonnegative")
    warmup_epochs = min(warmup_epochs, epochs)

    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)
