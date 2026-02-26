from src.cnn.model import *

def save_cnn_fc_checkpoint(model: CNNForecaster, optimizer, path, metadata=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_params": {
            "entry_channels": model.entry_channels,
            "output_horizon": model.output_horizon,
            "n_channels": model.n_channels,
            "ch_mults": model.ch_mults,
            "is_attn": model.is_attn,
            "n_blocks": model.n_blocks
        }
    }
    if metadata: 
        checkpoint.update(metadata)
    torch.save(checkpoint, path)

def load_cnn_fc_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = CNNForecaster(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), checkpoint

