from src.unet2d.load_save import save_model_checkpoint
from src.unet2d.model import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def ae_train_loop(ae_model, train_ds, test_ds, optimizer, epochs, device, batch_size, info_interval=5, path="../ae_checkpoints/", name="ae_model"):
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size) 
    val_loader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)
    history = {"train_loss": [], "val_loss": []}
    
    criterion = nn.MSELoss()
    epoch_bar = tqdm(range(epochs), leave=False)
    for epoch in epoch_bar:
        epoch_bar.set_description()
        total_train_loss = 0
        total_val_loss = 0

        ae_model.train()
        train_bar = tqdm(train_loader, desc="Training")
        for batch in train_bar:
            data = batch.to(device)
            
            optimizer.zero_grad()
            
            rec_data = ae_model(data)

            loss =  criterion(rec_data, data)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix({"train_loss":loss.item()})
            
            
        ae_model.eval()
        val_bar = tqdm(val_loader, desc="Validation")
        for batch in val_bar:
            data = batch.to(device)
            
            rec_data = ae_model(data)
            loss =  criterion(rec_data, data)
            
            total_val_loss += loss.item()
            val_bar.set_postfix({"val_loss":loss.item()})
            
        epoch_bar.set_postfix({"train_loss" : total_train_loss / len(train_loader), "val_loss" : total_val_loss / len(val_loader)})
        history["train_loss"].append(total_train_loss / len(train_loader))
        history["val_loss"].append(total_val_loss / len(val_loader))
    
        if (epoch + 1) % info_interval == 0:
            metadata = {"epochs": epoch + 1}
            save_model_checkpoint(ae_model, optimizer, f"{path}/{name}_epoch{epoch + 1}.pth")

    save_model_checkpoint(ae_model, optimizer, f"{path}/{name}_epoch{epochs}.pth")
    return ae_model, history