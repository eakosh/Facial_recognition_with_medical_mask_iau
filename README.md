# Facial recognition with medical mask iau
The activity in a pair from the IAU course at FIIT. Classification of the images (with mask/without mask)
=======
# Face Mask Classification

Binary image classification using convolutional neural networks. Predicting whether a face has a medical mask or not

Built as the activity in a pair from the IAU course at FIIT

*By [Olena Kosharova](https://github.com/eakosh) and [Ivan Oliinyk](https://github.com/frtndmd)*

## Dataset

Each image shows a single face, labeled with mask or without mask. Most people appear in the dataset twice, once masked and once not. All images share the same resolution, which simplifies preprocessing

Source: [Kaggle - Periocular Detection](https://www.kaggle.com/datasets/ruchi798/periocular-detection/data)

## Preventing data leakage

The dataset stores two photos of the same person: one with a mask, one without. To prevent data leakage of the same person in training and test sets the split is done by person's id, at the person level. This guarantees the reported test metrics reflect generalisation to new people, not memorised faces

## Pipeline

| Step | Description |
|---|---|
| EDA | Class balance, image dimensions, per-class average images and their difference map |
| Split | Person-level, stratified, 60 / 20 / 20|
| Preprocessing | Resize to 224x224, augmentation: flip, rotation, zoom, contrast, translation, brightness, noise |
| Models | Custom CNN, LeNet-5, MobileNetV2 |
| Training | Adam, binary cross-entropy, batch size 32 |
| Evaluation | Accuracy, precision, recall, F1, confusion matrices, train-validation gap check |

## Results

All metrics are computed on a held-out test set

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom CNN | 0.925 | 0.887 | 0.969 | 0.926 |
| LeNet-5 | 0.910 | 0.891 | 0.928 | 0.909 |
| **MobileNetV2** | **0.980** | **0.960** | **1.000** | **0.980** |

The two simpler models trained from scratch stayed close to each other, while MobileNetV2 pulled ahead with ImageNet transfer learning

## Usage

Download the [dataset](https://www.kaggle.com/datasets/ruchi798/periocular-detection/data) and place it at `data/`, then:

```bash
pip install -r requirements.txt
jupyter notebook face_mask_classification.ipynb
```
