import torch
from src.unet2d.model import UNet


def save_model_checkpoint(model: UNet, optimizer, path, metadata=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_params": {
            "image_channels": model.image_channels,
            "n_channels": model.n_channels,
            "ch_mults": model.ch_mults,
            "is_attn": model.is_attn,
            "n_blocks": model.n_blocks
        }
    }

    if metadata is not None:
        for k, v in metadata.items():
            if k not in checkpoint:
                checkpoint[k] = v

    torch.save(checkpoint, path)


def load_model_checkpoint(path, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = UNet(**checkpoint['model_params'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    metadata = {k: v for k, v in checkpoint.items() if k not in
                ['model_state_dict', 'optimizer_state_dict', 'model_params']}

    return model, metadata
