import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from former import HybridVisionFormer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEST_ROOT = '/kaggle/input/datasets/phamthinhzoro/dataset-origin/dataset/test'
    MODEL_PATH = 'best_model.pth'
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    class_names = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    # --- LOAD MODEL ---
    model = HybridVisionFormer(num_classes=7).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}")
        return

    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    all_preds = []
    all_labels = []

    # --- INFERENCE ---
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- BÁO CÁO ---
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Hybrid Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    test_model()