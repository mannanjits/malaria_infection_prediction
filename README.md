# Malaria Detection Using Convolutional Neural Networks

### Automated Classification of Malaria-Infected and Healthy Red Blood Cell Images Using Deep Learning

**Project:** Malaria Detection Using CNN  
**Task:** Binary Image Classification  
**Framework:** TensorFlow / Keras  
**Environment:** Google Colab  
**GitHub:** [mannanjits](https://github.com/mannanjits)

---

## 1. Project Overview

Malaria is a mosquito-borne disease caused by *Plasmodium* parasites. Microscopic examination of blood samples is commonly used to identify malaria infection, but manual examination can be time-consuming and requires trained personnel.

This project explores a deep learning-based approach for classifying microscopic red blood cell images as either malaria-infected or healthy.

A Convolutional Neural Network (CNN) is trained to learn visual features from labeled cell images and perform binary classification. The project also evaluates transfer learning using MobileNetV2 and explores Grad-CAM visualizations for model interpretability.

The objective is to investigate how deep learning-based image classification can support research into faster and more consistent malaria screening.

## 2. Objectives

- Perform exploratory data analysis (EDA) on the image dataset.
- Analyze class distributions and image metadata.
- Preprocess and normalize microscopic cell images.
- Develop and train a custom CNN classifier.
- Evaluate performance using standard classification metrics.
- Compare the custom CNN with a MobileNetV2 transfer-learning model.
- Investigate model predictions using Grad-CAM visualizations.
- Audit training, validation, and test splits for exact duplicate images.
- Identify limitations and opportunities for future improvement.

## 3. Dataset

The project uses a labeled microscopic red blood cell image dataset containing two classes.

| Property | Value |
|---|---|
| Total images | 27,558 |
| Parasitized images | 13,779 |
| Uninfected images | 13,779 |
| Number of classes | 2 |
| Input image size | 128 × 128 pixels |
| Image channels | RGB |
| Classification task | Binary |

### Dataset classes

- **Parasitized:** Red blood cell images labeled as malaria-infected.
- **Uninfected:** Red blood cell images labeled as healthy.

The dataset is balanced, with an equal number of images in each class.

### Dataset split

| Split | Images | Percentage |
|---|---:|---:|
| Training | 19,290 | 70% |
| Validation | 4,134 | 15% |
| Testing | 4,134 | 15% |
| **Total** | **27,558** | **100%** |

A fixed random seed of 42 is used for reproducibility.

**Dataset note:** Obtain the images from the authorized dataset source and follow its applicable licensing and usage terms. The dataset itself does not need to be committed to GitHub.

## 4. Data Preprocessing

The model receives an RGB image and predicts the probability that it belongs to the parasitized class.

The preprocessing pipeline includes:

1. Loading images from the dataset directories.
2. Decoding images into RGB format.
3. Resizing images to 128 × 128 pixels.
4. Converting pixel values to floating-point numbers.
5. Normalizing pixel values to the range [0, 1] for the custom CNN.
6. Batching and prefetching images to support efficient training.

For MobileNetV2, the corresponding TensorFlow Keras preprocessing function is used instead of the custom CNN's normalization procedure.

### Class encoding

| Class | Label |
|---|---:|
| Uninfected | 0 |
| Parasitized | 1 |

## 5. Project Workflow

1. **Exploratory Data Analysis:** Examine image counts, class balance, dimensions, formats, and metadata.
2. **Dataset Preparation:** Organize the dataset and create training, validation, and test splits.
3. **Image Preprocessing:** Resize, decode, normalize, and batch the images.
4. **CNN Development:** Design a convolutional neural network for binary classification.
5. **Model Training:** Train the CNN using binary cross-entropy and the Adam optimizer.
6. **Training Optimization:** Apply early stopping, learning-rate reduction, and model checkpointing.
7. **Performance Evaluation:** Calculate accuracy, precision, recall, F1-score, and ROC-AUC.
8. **Transfer Learning:** Evaluate a pretrained MobileNetV2 model.
9. **Model Comparison:** Compare both architectures on the held-out test set.
10. **Data Integrity Audit:** Check for exact duplicate image files across dataset splits.
11. **Interpretability:** Explore Grad-CAM visualizations of selected predictions.

## 6. Model Architectures

### 6.1 Custom CNN

The custom CNN learns visual features through convolutional and pooling layers.

The architecture consists of:

- Convolutional layers with 32, 64, 128, and 128 filters.
- ReLU activation functions.
- Max-pooling layers for spatial downsampling.
- Global average pooling.
- A dense layer with 128 units and L2 regularization.
- Dropout regularization with a rate of 0.4.
- A sigmoid output neuron for binary classification.

### Training configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Binary cross-entropy |
| Batch size | 32 |
| Maximum epochs | 20 |
| Early stopping patience | 4 epochs |
| Learning-rate reduction factor | 0.5 |
| Random seed | 42 |

Model checkpointing selects the best model based on validation loss. Early stopping and learning-rate reduction help manage training and reduce unnecessary computation.

### 6.2 MobileNetV2 Transfer Learning

MobileNetV2 is evaluated as an alternative architecture using pretrained ImageNet weights.

The convolutional backbone is frozen for the reported experiment. A classification head consisting of global average pooling, dropout, and a sigmoid output layer is trained for the binary classification task.

This experiment compares a custom CNN with a pretrained feature extractor.

## 7. Evaluation Metrics

The models are evaluated using five metrics:

- **Accuracy:** Proportion of correctly classified images.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the predicted scores distinguish between the two classes.

Precision and recall provide additional information beyond accuracy, particularly when examining false-positive and false-negative predictions.

## 8. Experimental Results

Both models were evaluated on the same held-out test set containing 4,134 images.

### Custom CNN vs. MobileNetV2

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test accuracy | **96.15%** | 92.96% |
| Precision | **96.31%** | 94.14% |
| Recall | **95.98%** | 91.63% |
| F1-score | **96.15%** | 92.87% |
| ROC-AUC | **0.9904** | 0.9789 |
| Test loss | 0.1192 | 0.1863 |

### Best-performing model: Custom CNN

The custom CNN achieved the strongest performance among the two evaluated configurations.

- **Test accuracy:** 96.15%
- **Precision:** 96.31%
- **Recall:** 95.98%
- **F1-score:** 96.15%
- **ROC-AUC:** 0.9904

The results indicate that the custom CNN performed better than the tested frozen-backbone MobileNetV2 configuration across all five reported classification metrics.

These findings are specific to the reported experiment and test split. They do not establish performance on independent clinical datasets.

## 9. Error Analysis

The custom CNN incorrectly classified 159 of the 4,134 test images.

These errors consist of false positives and false negatives. Analyzing misclassified images can help identify challenging visual patterns and potential weaknesses in the model.

Error analysis can investigate:

- Visually ambiguous cell images.
- Image artifacts and unusual image characteristics.
- Differences between the two classes.
- Potential weaknesses in learned image features.
- Opportunities for improving preprocessing and generalization.

## 10. Data Leakage Audit

A data-integrity audit was performed by calculating SHA-256 hashes of image files and checking for exact duplicate contents across the training, validation, and test sets.

| Comparison | Exact duplicate matches |
|---|---:|
| Training vs. validation | 0 |
| Training vs. testing | 0 |
| Validation vs. testing | 0 |

No exact duplicate image files were detected across the three splits during this audit.

However, this check does not detect near-duplicates, visually similar images with different file contents, or possible overlap between images originating from the same source slide.

The audit therefore supports basic split integrity but does not prove complete independence of the data.

## 11. Model Interpretability with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is explored to visualize image regions that contribute to a model's prediction.

Grad-CAM uses gradients associated with the selected class to generate a heatmap over the input image. The heatmap can be overlaid on the original image to inspect which regions influence the prediction.

This analysis can help explore model behavior and investigate potentially suspicious prediction patterns.

Grad-CAM is an exploratory interpretability technique. Its visualizations do not prove that the model relies on medically valid features or that a prediction is clinically correct.

## 12. Key Findings

- The custom CNN achieved 96.15% test accuracy.
- The custom CNN outperformed the evaluated MobileNetV2 configuration on all five reported classification metrics.
- The test ROC-AUC of 0.9904 indicates strong discrimination on this dataset.
- The model incorrectly classified 159 test images, demonstrating that errors remain despite high overall accuracy.
- No exact duplicate image files were detected across the dataset splits.
- Grad-CAM provides a way to explore visual regions associated with selected predictions.
- Independent evaluation is needed to establish how well the model generalizes to new data.

## 13. Limitations and Future Work

### Limitations

- The model has not been clinically validated.
- Performance on one dataset does not guarantee equivalent performance on images from different laboratories or microscopes.
- Exact duplicate checking does not rule out near-duplicates or source-slide overlap.
- Confidence scores have not been established as calibrated probabilities.
- Grad-CAM visualizations do not establish clinical interpretability.
- End-to-end inference speed and suitability for clinical workflows have not been established.

### Future improvements

- Evaluate the model on independent datasets from different sources.
- Investigate additional data augmentation and regularization strategies.
- Explore fine-tuning pretrained CNN architectures.
- Analyze false positives and false negatives in greater detail.
- Evaluate probability calibration and prediction uncertainty.
- Measure inference latency and resource requirements.
- Develop a user-friendly prediction interface.
- Conduct appropriate external and clinical validation before considering diagnostic deployment.

## 14. Repository Structure

The following is a suggested structure. Keep only files that actually exist in your GitHub repository.
```
malaria-detection-cnn/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_malaria_eda.ipynb
│   ├── 02_malaria_preprocessing.ipynb
│   └── 03_malaria_training_evaluation.ipynb
└── outputs/
    ├── dataset_eda.png
    ├── sample_images.png
    ├── training_curves.png
    ├── model_comparison.png
    └── misclassified_examples.png
```
Depending on the experiments completed, additional outputs may include training history files, test metrics, predictions, trained model checkpoints, Grad-CAM visualizations, and a final project report.

Large datasets and trained models can be stored externally when appropriate. Ensure external links have the correct access permissions.

## 15. Technologies Used

- **Python:** Main programming language.
- **TensorFlow / Keras:** CNN implementation and model training.
- **NumPy:** Numerical operations.
- **Pandas:** Dataset metadata and evaluation tables.
- **Matplotlib:** Training curves and image visualizations.
- **Scikit-learn:** Evaluation metrics and classification analysis.
- **Google Colab:** Notebook execution and model experimentation.
- **Google Drive:** Dataset and artifact storage.

## 16. Installation and Usage

### Step 1: Clone the repository

```
git clone https://github.com/mannanjits/malaria_infection_prediction
cd malaria-detection-cnn
```

This command assumes the repository is named `malaria-detection-cnn`. Update the URL if your actual repository name differs.

### Step 2: Install dependencies

If a `requirements.txt` file is included, install its dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Use package versions compatible with the environment in which the notebooks were developed.

### Step 3: Prepare the dataset

Obtain the dataset from its authorized source and arrange it as follows:

```text
cell_images/
├── Parasitized/
└── Uninfected/
```

Update the dataset path in the notebooks to match your environment.

### Step 4: Run the notebooks

Start Jupyter Notebook:

```
jupyter notebook
```

Run the notebooks in this order:

1. `01_malaria_eda.ipynb`
2. `02_malaria_preprocessing.ipynb`
3. `03_malaria_training_evaluation.ipynb`

The notebooks cover dataset analysis, preprocessing, training, and evaluation. Some experiments may require significant memory and computational resources.

## 17. References

1. TensorFlow documentation: https://www.tensorflow.org/
2. Keras documentation: https://keras.io/
3. Scikit-learn documentation: https://scikit-learn.org/stable/
4. Selvaraju, R. R. et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017.
5. Dataset-https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria?utm_s

## Acknowledgement

This project explores the application of convolutional neural networks to microscopic image classification and demonstrates a machine learning workflow involving exploratory data analysis, preprocessing, model development, evaluation, model comparison, and interpretability.

## Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated medical diagnostic system and must not be used as a substitute for professional medical assessment, laboratory testing, or qualified healthcare decisions.
