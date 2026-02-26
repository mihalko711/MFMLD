from src.tcn.model import *

def save_tcn_checkpoint(model: TCNForecaster, optimizer, path, metadata=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_params": {
            "input_dim": model.input_dim,
            "output_horizon": model.output_horizon,
            "channels": model.channels,
            "kernel_size": model.kernel_size,
            "dropout": model.dropout
        }
    }

    if metadata is not None:
        checkpoint.update({k: v for k, v in metadata.items() if k not in checkpoint})

    torch.save(checkpoint, path)

def load_tcn_checkpoint(path, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    
    # Создаем модель, используя сохраненные параметры
    model = TCNForecaster(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Извлекаем все остальное (epoch, loss и т.д.)
    metadata = {k: v for k, v in checkpoint.items() if k not in 
                ['model_state_dict', 'optimizer_state_dict', 'model_params']}

    return model, metadata