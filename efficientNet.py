import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

def train():
    # --- configurações ---
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dir  = 'data/train'      
    valid_dir  = 'data/valid'      
    model_path = 'eff3vs8.pth'
    num_epochs = 10
    batch_size = 32
    lr         = 1e-4

    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    valid_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # Datasets e DataLoaders
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_tfm)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    valid_ld = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)

    # Modelo EfficientNet-B0 (API weights em vez de pretrained)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    def train_epoch():
        model.train()
        running_loss = 0.0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        return running_loss / len(train_ds)

    def validate():
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in valid_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        return correct / len(valid_ds)

    # Loop de treino
    for epoch in range(1, num_epochs+1):
        train_loss = train_epoch()
        val_acc    = validate()
        print(f'Epoch {epoch}/{num_epochs} — Loss: {train_loss:.4f} — Val Acc: {val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'  ➤ Novo best: {best_acc:.4f}, salvo em {model_path}')

if __name__ == '__main__':
    train()
