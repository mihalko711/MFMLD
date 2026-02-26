import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

def forecast_train_loop(model, train_ds, test_ds, optimizer, epochs, device, batch_size,
                        save_func, info_interval=5, path="../fc_checkpoints/", name="forecast_model"):
    """
    Универсальный цикл обучения для моделей прогнозирования (TCN, UNet и др.)
    с детальной индикацией прогресса.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # num_workers=0 для стабильности в ноутбуках, если нужно ускорить - ставь 2 или 4
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=0)
    
    history = {"train_loss": [], "val_loss": []}
    criterion = nn.MSELoss()
    
    device = torch.device(device)
    model.to(device)

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    
    for epoch in epoch_bar:
        # --- ТРЕНИРОВКА ---
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for input_data, target_data in train_bar:
            input_data = input_data.to(device)
            target_data = target_data.to(device).squeeze(1) # [B, 1, L] -> [B, L]

            optimizer.zero_grad()
            prediction = model(input_data)

            if prediction.shape != target_data.shape:
                prediction = prediction.squeeze(1)

            loss = criterion(prediction, target_data)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_train_loss += current_loss
            train_bar.set_postfix({"loss": f"{current_loss:.6f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- ВАЛИДАЦИЯ ---
        model.eval()
        total_val_loss = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad(): 
            for input_data, target_data in val_bar:
                input_data = input_data.to(device)
                target_data = target_data.to(device).squeeze(1)

                prediction = model(input_data)
                
                if prediction.shape != target_data.shape:
                    prediction = prediction.squeeze(1)
                
                loss = criterion(prediction, target_data)
                
                current_loss = loss.item()
                total_val_loss += current_loss
                val_bar.set_postfix({"loss": f"{current_loss:.6f}"})

        avg_val_loss = total_val_loss / len(val_loader)

        # Обновляем историю и главный прогресс-бар
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        epoch_bar.set_postfix({
            "tr_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{avg_val_loss:.6f}"
        })

        # Сохранение чекпоинта
        if (epoch + 1) % info_interval == 0:
            metadata = {
                "epoch": epoch + 1, 
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss
            }
            save_func(model, optimizer, f"{path}/{name}_epoch{epoch + 1}.pth", metadata)

    # Финальное сохранение
    save_func(model, optimizer, f"{path}/{name}_final.pth", {"epoch": epochs, "val_loss": avg_val_loss})
    
    return model, history