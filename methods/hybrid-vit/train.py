import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from former import HybridVisionFormer
from PIL import Image

class RAFDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                img_name, label = line.split()

                # RAF label: 1~7
                label = int(label) - 1

                self.samples.append(
                    (img_name, label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]

        img_path = os.path.join(
            self.root_dir,
            img_name
        )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
# --- IMPLEMENT FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_ROOT = os.environ["TRAIN_ROOT"]
    
    BATCH_SIZE = 64
    EPOCHS = 40 # Tăng số epoch lên vì ResNet50 cần nhiều thời gian để hội tụ hơn
    MAX_LR = 3e-4 

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=3, magnitude=9), # Tăng ops từ 2 lên 3
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    LABEL_FILE = os.environ["LABEL_FILE"]

    full_dataset = RAFDataset(
        root_dir=TRAIN_ROOT,
        label_file=LABEL_FILE
    )

    targets = [label for _, label in full_dataset.samples]

    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=0.15,
        random_state=42,
        stratify=targets
    )

    train_dataset = Subset(
        RAFDataset(
            root_dir=TRAIN_ROOT,
            label_file=LABEL_FILE,
            transform=train_transform
        ),
        train_idx
    )

    val_dataset = Subset(
        RAFDataset(
            root_dir=TRAIN_ROOT,
            label_file=LABEL_FILE,
            transform=val_transform
        ),
        val_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = HybridVisionFormer(num_classes=7).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # DÙNG FOCAL LOSS THAY CHO CROSS ENTROPY
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1 
    )
    
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_correct, total_train = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # --- ÁP DỤNG GRADIENT CLIPPING ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=running_correct/total_train)

        epoch_train_loss = running_loss / len(train_idx)
        epoch_train_acc = running_correct / len(train_idx)

        # Validation Phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_idx)
        epoch_val_acc = val_correct / len(val_idx)
        
        print(f"Summary -> Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(">>> Saved best model!")

    # Plotting...
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('training_plots.png')
    plt.show()

if __name__ == "__main__":
    train_model()