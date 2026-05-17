# Emotion Recognition using EEG & MRI Analysis

This repository contains machine learning models and experiments related to Medical Imaging (MRI) and Electroencephalography (EEG) data. The project involves state-of-the-art deep learning techniques for tasks such as brain tumor classification, segmentation, and emotion recognition.

## 📂 Repository Structure

All the core models are located in the `MRI/` directory. The notebooks have been optimized by removing markdown and comments to retain only the essential code logic.

- **`EEG_Emotion_Recognition_DEAP.ipynb`**
  Emotion recognition using EEG signals based on the DEAP dataset.

- **`brain-tumor-mri-accuracy-99.ipynb`**
  Brain tumor classification using MRI images, achieving 99% accuracy.

- **`brats20-unet-tumorsegmentation.ipynb`**
  Tumor segmentation using the U-Net architecture on the BraTS20 dataset.

- **`mri-classification-using-hybrid-conv-svm-99-accu.ipynb`**
  A hybrid Convolutional Neural Network and Support Vector Machine (CNN-SVM) approach for MRI classification, achieving 99% accuracy.

- **`notebook8492bae121.ipynb`**
  Additional model architecture and experimentation.

## 🚀 Getting Started

To use these models, you will need a standard Python data science environment with the following libraries:
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- NumPy & Pandas
- Matplotlib / Seaborn

## 🧠 Usage

Because these notebooks contain purely the model code, they can be easily integrated into a larger backend service (such as FastAPI) or converted into python scripts (`.py`) for automated training pipelines and production deployments.

## 📄 License

This project is open-source and available under the standard MIT License.
