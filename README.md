# CS231 – Facial Expression Recognition Benchmark

## 1. Mục tiêu

Thực hiện benchmark các phương pháp nhận diện cảm xúc khuôn mặt (Facial Expression Recognition – FER) trên cùng tập dữ liệu RAF-DB.

Các phương pháp được đánh giá:

1. Advanced-FER-CNN (Phong)
2. EfficientNetV2-S (Trường)
3. Hybrid Vision Transformer (Hybrid-ViT) (Thịnh)
4. LNSU (Giang)
5. DAN (ResNet18 + Discriminative Loss) (Thiện)

Các checkpoint trong tất cả các thực nghiệm trên đều được lưu ở [đây](https://huggingface.co/yangtzentg/LNSU/tree/main), và các hướng dẫn lưu checkpoint ở [đây](https://huggingface.co/yangtzentg/LNSU/blob/main/README.md)

---

# 2. Cấu trúc thư mục

```text
```text
CS231-Project/
├── dataset/
│   ├── RAF/
│   └── list_patition_label.txt
│
├── methods/
│   ├── Advanced-FER-CNN/
│   │   ├── train.py
│   │   └── train.sh
│   │
│   ├── EfficientNetV2-S/
│   │   ├── train.py
│   │   └── train.sh
│   │
│   ├── hybrid-vit/
│   │   ├── train.py
│   │   ├── train.sh
│   │   └── former.py
│   │
│   ├── LNSU/
│   │   ├── configs/
│   │   ├── utils/
│   │   ├── train_exp.py
│   │   └── train_exp.sh
│   │
│   └── ResNet18-Discriminative-loss/
│       ├── models/
│       │   └── resnet18_msceleb.pth
│       ├── MLP_Affinity_RAF_DB_Training.ipynb
│       ├── rafdb_resnet18_mlp_pretrained_ce_affinity.pth
│       ├── rafdb_resnet18_mlp_pretrained_ce.pth
│       ├── rafdb_resnet18_mlp_random_ce_affinity.pth
│       └── rafdb_resnet18_mlp_random_ce.pth
│
├── weights/
│   ├── best_cnn_model.h5
│   ├── best_EfficientNetV2_RAFDB_rafdb.pth
│   ├── best_model.pth
│   ├── checkpoint_step_59999_gpu_0.pt
│   ├── start_0.pt
│   ├── resnet18_msceleb.pth
│   └── datasets.zip
│
└── README.md
```

---

# 3. Dataset

## RAF-DB

Sử dụng RAF-DB với 7 lớp cảm xúc:

| Label | Emotion  |
| ----- | -------- |
| 1     | Surprise |
| 2     | Fear     |
| 3     | Disgust  |
| 4     | Happy    |
| 5     | Sad      |
| 6     | Angry    |
| 7     | Neutral  |

File nhãn:

```text
train_00001.jpg 5
train_00002.jpg 5
train_00003.jpg 4
...
```

---

# 4. Môi trường thực nghiệm

## Hardware

* GPU: NVIDIA RTX 3080 Ti
* CUDA: 12.x
* RAM: 16GB

## Software

* Python 3.12
* PyTorch 2.x
* TorchVision
* timm
* scikit-learn
* matplotlib
* numpy

Cài đặt:

```bash
pip install torch torchvision timm scikit-learn matplotlib tqdm pillow
```

---

# 5. Method 1 – Advanced-FER-CNN

Thư mục:

```text
methods/Advanced-FER-CNN
```

## Chạy huấn luyện

```bash
cd methods/Advanced-FER-CNN

./train.sh
```

Hoặc:

```bash
export DATA_ROOT=../../dataset/RAF
export LABEL_FILE=../../dataset/list_patition_label.txt

python train.py
```

## Output

```text
best_cnn_model.h5
training_curves.png
```

---

# 6. Method 2 – EfficientNetV2-S

Thư mục:

```text
methods/EfficientNetV2-S
```

## Chạy huấn luyện

```bash
cd methods/EfficientNetV2-S

./train.sh
```

Hoặc:

```bash
export DATA_ROOT=../../dataset/RAF
export LABEL_FILE=../../dataset/list_patition_label.txt

python train.py
```

## Output

```text
best_EfficientNetV2_RAF.pth
efficientnet_rafdb_training.png
```

---

# 7. Method 3 – Hybrid Vision Transformer

Thư mục:

```text
methods/hybrid-vit
```

## Chạy huấn luyện

```bash
cd methods/hybrid-vit

./train.sh
```

Hoặc:

```bash
export TRAIN_ROOT=../../dataset/RAF
export LABEL_FILE=../../dataset/list_patition_label.txt

python train.py
```

## Output

```text
best_model.pth
training_plots.png
```

---

# 8. Method 4 – LNSU
Thư mục:

```text
methods/LNSU
```

## Pretrained Model

LNSU sử dụng checkpoint khởi tạo `start_0.pt`.

Nếu chưa có file này, có thể tải trực tiếp từ Hugging Face [start_0.pt](https://huggingface.co/yangtzentg/LNSU/tree/main)

Sau khi tải, đặt file vào thư mục:

```text
methods/LNSU/
└── start_0.pt
```

## Chạy huấn luyện

```bash
cd methods/LNSU

./train_exp.sh
```

Hoặc:

```bash
python train_exp.py configs/base.py
```

## Output

```text
results/
checkpoints/
tensorboard/
```

---
# 9. Method 5 – DAN (ResNet18 + Discriminative Loss)

Paper:

**Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition**

Thư mục:

```text
methods/ResNet18-Discriminative-loss
```

## Pretrained Backbone

DAN sử dụng backbone ResNet18 pretrained trên MSCeleb.

```text
methods/ResNet18-Discriminative-loss/
└── models/
    └── resnet18_msceleb.pth
```

## Experimental Notebook

Toàn bộ pipeline huấn luyện và đánh giá được thực hiện trong:

```text
MLP_Affinity_RAF_DB_Training.ipynb
```

Mở notebook:

```bash
jupyter notebook
```

hoặc

```bash
jupyter lab
```

sau đó chạy toàn bộ notebook theo thứ tự.

## Experimental Variants

| Model                                     | Initialization        | Affinity Loss |
| ----------------------------------------- | --------------------- | ------------- |
| rafdb_resnet18_mlp_pretrained_ce_affinity | MSCeleb pretrained    | ✓             |
| rafdb_resnet18_mlp_pretrained_ce          | MSCeleb pretrained    | ✗             |
| rafdb_resnet18_mlp_random_ce_affinity     | Random initialization | ✓             |
| rafdb_resnet18_mlp_random_ce              | Random initialization | ✗             |

## Output Checkpoints

```text
rafdb_resnet18_mlp_pretrained_ce_affinity.pth
rafdb_resnet18_mlp_pretrained_ce.pth
rafdb_resnet18_mlp_random_ce_affinity.pth
rafdb_resnet18_mlp_random_ce.pth
```
