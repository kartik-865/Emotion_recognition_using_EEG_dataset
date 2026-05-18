# Emotion Recognition using EEG and MRI Analysis

A collection of deep learning models and experiments for neurological signal processing and medical image analysis. This repository covers three interconnected problem domains: emotion recognition from EEG signals, brain tumor classification from MRI scans, and tumor segmentation using volumetric MRI data. Each model is implemented as a self-contained Jupyter Notebook and is designed to be easily adapted into larger pipelines or production systems.

---

## Overview

This repository brings together machine learning research at the intersection of neuroscience and medical imaging. It contains four primary notebooks addressing distinct but related tasks:

- Classifying human emotional states (valence and arousal) from raw EEG signals using the DEAP dataset
- Detecting and classifying brain tumors from MRI images with a convolutional neural network achieving 99% accuracy
- Performing pixel-level tumor segmentation on multimodal MRI scans using the U-Net architecture and the BraTS20 benchmark dataset
- Exploring a hybrid CNN-SVM approach for MRI classification that combines deep feature extraction with classical decision boundaries

All notebooks are kept lean, containing only the essential model code, making them straightforward to integrate into backend services or automated training pipelines.

---

## Repository Structure

```
Emotion_recognition_using_EEG_dataset/
|
|-- MRI/                                              # Core notebooks directory
|   |-- EEG_Emotion_Recognition_DEAP.ipynb            # Emotion recognition from EEG signals
|   |-- brain-tumor-mri-accuracy-99.ipynb             # MRI brain tumor classification (99% accuracy)
|   |-- brats20-unet-tumorsegmentation.ipynb          # Tumor segmentation with U-Net on BraTS20
|   |-- mri-classification-using-hybrid-conv-svm-99-accu.ipynb   # Hybrid CNN-SVM classification
|   |-- notebook8492bae121.ipynb                      # Additional architecture experiments
|
|-- LICENSE                                           # MIT License
|-- README.md                                         # Project documentation
```

All core work resides in the `MRI/` directory. Notebooks have been streamlined to retain only executable model logic, removing markdown commentary and non-essential cells, making them suitable for direct integration or script conversion.

---

## Notebooks

### 1. EEG Emotion Recognition — DEAP Dataset

**File**: `EEG_Emotion_Recognition_DEAP.ipynb`

This notebook implements an emotion recognition system using EEG (Electroencephalography) signals from the DEAP dataset. EEG captures the brain's electrical activity across multiple channels over time, producing a high-dimensional time-series signal that can be correlated with emotional states.

The pipeline covers:

- Loading and preprocessing the DEAP dataset (32 participants, 40 trials, 32 EEG channels)
- Signal segmentation and band-power feature extraction across standard frequency bands (delta, theta, alpha, beta, gamma)
- Binary classification of **valence** (positive vs. negative) and **arousal** (high vs. low) dimensions of emotion
- Model training using deep learning or classical machine learning approaches
- Evaluation using accuracy, F1 score, and confusion matrices

The valence-arousal model is one of the most widely used frameworks in affective computing, and the DEAP dataset is a standard benchmark in the EEG emotion recognition literature.

---

### 2. Brain Tumor MRI Classification

**File**: `brain-tumor-mri-accuracy-99.ipynb`

This notebook trains a convolutional neural network to classify MRI brain scans into tumor and non-tumor categories, achieving 99% accuracy on the test set.

The pipeline covers:

- Loading and augmenting a labeled MRI image dataset
- Building and training a CNN with convolutional, pooling, batch normalization, and dropout layers
- Fine-tuning using transfer learning from a pretrained backbone (e.g., VGG16 or ResNet)
- Evaluation with accuracy, precision, recall, and F1 score
- Visualization of model predictions and misclassifications

This model demonstrates that deep learning can reliably detect the presence of brain tumors from MRI images with accuracy comparable to reported benchmarks in the literature.

---

### 3. Brain Tumor Segmentation — BraTS20

**File**: `brats20-unet-tumorsegmentation.ipynb`

This notebook addresses tumor segmentation — identifying the precise pixel-level boundaries of a tumor within an MRI scan — using the **U-Net** architecture on the **BraTS 2020** (Brain Tumor Segmentation) benchmark dataset.

The pipeline covers:

- Loading multimodal MRI data (T1, T1ce, T2, FLAIR modalities) from the BraTS20 dataset
- Preprocessing volumetric 3D scans: normalization, slicing, and mask preparation
- Building a U-Net model with encoder-decoder architecture and skip connections
- Training with binary cross-entropy or Dice loss to handle class imbalance between tumor and background pixels
- Evaluation using the Dice Similarity Coefficient (DSC) and Intersection over Union (IoU)
- Visualization of ground truth vs. predicted segmentation masks

Tumor segmentation is a significantly harder task than classification, as the model must learn to localize and delineate tumor regions rather than simply detect their presence.

---

### 4. Hybrid CNN-SVM MRI Classification

**File**: `mri-classification-using-hybrid-conv-svm-99-accu.ipynb`

This notebook explores a hybrid approach that combines the feature extraction power of a convolutional neural network with the decision boundary strength of a Support Vector Machine (SVM), achieving 99% accuracy on MRI classification.

The pipeline covers:

- Training a CNN up to (but not including) the final classification layer
- Extracting the deep feature vectors from the penultimate layer for each image
- Feeding these CNN-derived feature representations into an SVM classifier
- Comparing performance against a standalone CNN softmax classifier
- Hyperparameter tuning for the SVM kernel (RBF, linear) and regularization parameter C

This hybrid approach is particularly effective in low-data regimes, where a fully trained deep classifier may overfit but CNN features still provide strong representational power.

---

## Datasets

| Dataset | Task | Source |
|---|---|---|
| DEAP | EEG-based emotion recognition | [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| Brain Tumor MRI Dataset | Tumor classification | [Kaggle - Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |
| BraTS 2020 | Tumor segmentation | [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/) |

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras, PyTorch |
| Classical ML | Scikit-learn (SVM, preprocessing) |
| Signal Processing | NumPy, SciPy |
| Data Handling | Pandas, NumPy |
| Image Processing | OpenCV, Pillow |
| Visualization | Matplotlib, Seaborn |
| Notebook Environment | Jupyter Notebook, JupyterLab, Google Colab |

---

## Results

| Notebook | Task | Metric | Score |
|---|---|---|---|
| EEG Emotion Recognition | Valence / Arousal Classification | Accuracy | Benchmark-competitive |
| Brain Tumor Classification | Binary MRI Classification | Accuracy | ~99% |
| BraTS20 Segmentation | Pixel-level Tumor Segmentation | Dice Score | Competitive with published U-Net baselines |
| Hybrid CNN-SVM | MRI Classification | Accuracy | ~99% |

Detailed per-epoch training curves, confusion matrices, and metric outputs are generated within each notebook upon execution.

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code with attribution to the original author.

See the [LICENSE](./LICENSE) file for the full license text.

---

*Developed by Kartik Gahlot. For questions or collaboration, please open an issue on this repository.*
