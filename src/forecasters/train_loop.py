import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.unet1d.load_save import save_model_checkpoint


def forecast_train_loop(model, train_ds, test_ds, optimizer, epochs, device, batch_size,
                        info_interval=5, path="../ae_checkpoints/", name="forecast_model"):
    """
    Training loop for forecasting models.
    
    The key difference from ae_train_loop is that the dataset returns pairs:
    (input_signal, target_signal), where input_signal is fed to the model
    and target_signal is used as the ground truth for loss calculation.
    
    Args:
        model: Forecasting model (e.g., UNet1D, MLP, etc.)
        train_ds: Training dataset returning (input, target) pairs
        test_ds: Validation dataset returning (input, target) pairs
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        batch_size: Batch size for DataLoader
        info_interval: Interval (in epochs) for saving checkpoints
        path: Directory to save model checkpoints
        name: Base name for saved model files
    """
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)
    history = {"train_loss": [], "val_loss": []}

    criterion = nn.MSELoss()
    epoch_bar = tqdm(range(epochs), leave=False)
    
    for epoch in epoch_bar:
        epoch_bar.set_description()
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        train_bar = tqdm(train_loader, desc="Training")
        for batch in train_bar:
            input_data, target_data = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            prediction = model(input_data)

            loss = criterion(prediction, target_data)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix({"train_loss": loss.item()})

        model.eval()
        val_bar = tqdm(val_loader, desc="Validation")
        for batch in val_bar:
            input_data, target_data = batch[0].to(device), batch[1].to(device)

            prediction = model(input_data)
            loss = criterion(prediction, target_data)

            total_val_loss += loss.item()
            val_bar.set_postfix({"val_loss": loss.item()})

        epoch_bar.set_postfix({
            "train_loss": total_train_loss / len(train_loader),
            "val_loss": total_val_loss / len(val_loader)
        })
        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(total_val_loss / len(val_loader))

        if (epoch + 1) % info_interval == 0:
            metadata = {"epochs": epoch + 1}
            save_model_checkpoint(model, optimizer, f"{path}/{name}_epoch{epoch + 1}.pth", metadata)

    save_model_checkpoint(model, optimizer, f"{path}/{name}_epoch{epochs}.pth", {"epochs": epochs})
    
    return history
