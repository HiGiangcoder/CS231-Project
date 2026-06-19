import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

DATA_ROOT = os.environ["DATA_ROOT"]
LABEL_FILE = os.environ["LABEL_FILE"]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants for RAF-DB (7 emotions)
EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
NUM_CLASSES = len(EMOTIONS)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class RAFDataset(Dataset):

    def __init__(self,
                 root_dir,
                 label_file,
                 indices=None,
                 transform=None):

        self.root_dir = root_dir
        self.transform = transform
        samples = []

        with open(label_file) as f:
            for line in f:
                img_name, label = line.strip().split()
                samples.append(
                    (
                        img_name,
                        int(label) - 1
                    )
                )
        if indices is not None:
            samples = [samples[i] for i in indices]
        self.samples = samples
        print(f"Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(
            self.root_dir,
            img_name
        )
        image = Image.open(
            img_path
        ).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading RAF-DB datasets...")
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split

all_labels = []

with open(LABEL_FILE) as f:
    for line in f:
        _, label = line.strip().split()
        all_labels.append(int(label) - 1)

all_labels = np.array(all_labels)

indices = np.arange(len(all_labels))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    stratify=all_labels,
    random_state=42
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.15,
    stratify=all_labels[train_idx],
    random_state=42
)

train_dataset = RAFDataset(
    DATA_ROOT,
    LABEL_FILE,
    train_idx,
    train_transform
)

val_dataset = RAFDataset(
    DATA_ROOT,
    LABEL_FILE,
    val_idx,
    test_transform
)

test_dataset = RAFDataset(
    DATA_ROOT,
    LABEL_FILE,
    test_idx,
    test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Initialize EfficientNetV2 model
print("\nInitializing EfficientNetV2 model...")
efficientnet = models.efficientnet_v2_s(pretrained=True)
num_features = efficientnet.classifier[1].in_features
efficientnet.classifier[1] = nn.Linear(num_features, NUM_CLASSES)
efficientnet = efficientnet.to(device)

# show the model architecture
print(efficientnet)

# Training function with early stopping
def train_efficientnet(model, model_name, train_loader, val_loader, test_loader, epochs=EPOCHS, patience=5):
    print(f"\n{'='*50}")
    print(f"Training {model_name} on RAF-DB with Early Stopping (patience={patience})")
    print('='*50)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    stopped_epoch = epochs

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_pbar.set_postfix({'loss': running_loss/(len(train_pbar)), 
                                   'acc': 100.*correct/total})

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({'loss': val_loss/(len(val_pbar)), 
                                     'acc': 100.*correct/total})

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}_rafdb.pth')
            print(f"✓ New best model saved with accuracy: {val_acc:.2f}%")

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            stopped_epoch = epoch + 1
            break

    # Final evaluation with best model
    print(f"\n📊 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f'best_{model_name}_rafdb.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'predictions': all_preds,
        'true_labels': all_labels,
        'stopped_epoch': stopped_epoch
    }

# Train EfficientNetV2 on RAF-DB
print("\n" + "="*60)
print("STARTING EFFICIENTNETV2 TRAINING ON RAF-DB")
print("="*60)

efficientnet_results = train_efficientnet(efficientnet, 'EfficientNetV2_RAFDB', train_loader, val_loader, test_loader, patience=5)

# Plotting results for EfficientNetV2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = range(1, len(efficientnet_results['train_losses']) + 1)

# Plot losses
axes[0].plot(epochs_range, efficientnet_results['train_losses'], label='Train Loss', marker='o', color='green')
axes[0].plot(epochs_range, efficientnet_results['val_losses'], label='Val Loss', marker='s', color='red')
axes[0].axvline(x=efficientnet_results['stopped_epoch'], color='red', linestyle='--', alpha=0.7, label='Early Stop')
axes[0].set_title(f'EfficientNetV2 on RAF-DB - Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot accuracies
axes[1].plot(epochs_range, efficientnet_results['train_accs'], label='Train Acc', marker='o', color='green')
axes[1].plot(epochs_range, efficientnet_results['val_accs'], label='Val Acc', marker='s', color='red')
axes[1].axvline(x=efficientnet_results['stopped_epoch'], color='red', linestyle='--', alpha=0.7, label='Early Stop')
axes[1].set_title('EfficientNetV2 on RAF-DB - Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('efficientnet_rafdb_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


cm = confusion_matrix(efficientnet_results['true_labels'], efficientnet_results['predictions'])
cm_norm = confusion_matrix(efficientnet_results['true_labels'], efficientnet_results['predictions'], normalize='true')

fig, ax = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax[0])
ax[0].set_title('Raw confusion matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1, xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax[1])
ax[1].set_title('Normalized confusion matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')

plt.tight_layout()
plt.show()

# Classification report for EfficientNetV2
print("\n" + "="*60)
print("EFFICIENTNETV2 ON RAF-DB - CLASSIFICATION REPORT")
print("="*60)
print(classification_report(efficientnet_results['true_labels'], efficientnet_results['predictions'], 
                           target_names=EMOTIONS, zero_division=0, digits=4))

# Per-class performance analysis
print(f"\n📈 Detailed Analysis for EfficientNetV2 on RAF-DB:")
cm = confusion_matrix(efficientnet_results['true_labels'], efficientnet_results['predictions'])

# Calculate per-class accuracy
class_acc = cm.diagonal() / cm.sum(axis=1)
print(f"\n{'Emotion':<15} {'Accuracy':<15}")
print("-"*30)
for emotion, acc in zip(EMOTIONS, class_acc):
    print(f"{emotion:<15} {acc*100:.2f}%")

# Find easiest and hardest emotions
best_class_idx = np.argmax(class_acc)
worst_class_idx = np.argmin(class_acc)
print(f"\n✅ Best performing emotion: {EMOTIONS[best_class_idx]} ({class_acc[best_class_idx]*100:.2f}%)")
print(f"❌ Worst performing emotion: {EMOTIONS[worst_class_idx]} ({class_acc[worst_class_idx]*100:.2f}%)")

# Save results
print(f"\n📁 EfficientNetV2 on RAF-DB results saved")
print(f"   Best validation accuracy: {efficientnet_results['best_val_acc']:.2f}%")
print(f"   Stopped at epoch: {efficientnet_results['stopped_epoch']}")
print(f"   Total epochs trained: {len(efficientnet_results['train_accs'])}")

# With A, B is the label index of the predicted emotion and the true emotion, respectively. 
# For example, if the model predicted 'Happy' (index 3) but the true label is 'Sad' (index 4), then A=3 and B=4.

# Plot the image, which the predict the model got wrong, and the predicted label is A but the true label is B

# with each image, the different failure cases are
def plot_misclassified_images(model, dataset, num_images=100):
    model.eval()
    misclassified = []
    pairs = []
    skip = 20
    labels = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]

            if label in labels:
                continue 
            input_img = image.unsqueeze(0).to(device)
            output = model(input_img)
            _, predicted = output.max(1)

            if predicted.item() != label:
                if (predicted.item(), label) in pairs:
                    continue
                if skip > 0:
                    skip -= 1
                    continue
                labels.append(label)
                pairs.append((predicted.item(), label))  
                misclassified.append((image.cpu(), predicted.item(), label))
                if len(misclassified) >= num_images:
                    break

    # Plot misclassified images
    plt.figure(figsize=(15, 5))
    for i, (img, pred, true) in enumerate(misclassified):
        plt.subplot(1, num_images, i+1)
        img = img.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Unnormalize
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Pred: {EMOTIONS[pred]}\nTrue: {EMOTIONS[true]}")
        plt.axis('off')
    plt.suptitle('Misclassified Images by EfficientNetV2 on RAF-DB', fontsize=16)
    plt.tight_layout()
    plt.savefig('efficientnet_rafdb_misclassified.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot misclassified images for EfficientNetV2
plot_misclassified_images(efficientnet, test_dataset, num_images=5)

