# CS231 – Facial Expression Recognition Benchmark

## 1. Mục tiêu

Thực hiện benchmark các phương pháp nhận diện cảm xúc khuôn mặt (Facial Expression Recognition – FER) trên cùng tập dữ liệu RAF-DB.

Các phương pháp được đánh giá:

1. Advanced-FER-CNN
2. EfficientNetV2-S
3. Hybrid Vision Transformer (Hybrid-ViT)
4. LNSU
5. DAN (ResNet18 + Discriminative Loss)

---

# 2. Cấu trúc thư mục

```text
CS231/
├── dataset/
│   ├── RAF/
│   │   ├── train_00001.jpg
│   │   ├── train_00002.jpg
│   │   └── ...
│   └── list_patition_label.txt
│
├── methods/
│   ├── Advanced-FER-CNN/
│   │   ├── train.py
│   │   └── train.sh
│   ├── EfficientNetV2-S/
│   │   ├── train.py
│   │   └── train.sh
│   ├── hybrid-vit/
│   │   ├── train.py
│   │   └── former.py
│   ├── LNSU/
│   │   ├── configs/
│   │   ├── utils/
│   │   └── train_exp.py
│   └── ResNet18-Discriminative-loss/
│       ├── rafdb.py
│       ├── affectnet.py
│       └── networks/
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

LNSU sử dụng checkpoint khởi tạo `star_0.pt`.

Nếu chưa có file này, có thể tải trực tiếp từ Hugging Face:

:contentReference[oaicite:0]{index=0}

Sau khi tải, đặt file vào thư mục:

```text
methods/LNSU/
└── star_0.pt
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

Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition

Thư mục:

```text
methods/ResNet18-Discriminative-loss
```

## Chạy huấn luyện

```bash
cd methods/ResNet18-Discriminative-loss

CUDA_VISIBLE_DEVICES=0 python rafdb.py
```

## Output

```text
checkpoints/
best_model.pth
```
