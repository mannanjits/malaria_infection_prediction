# Malaria Cell Classification Using CNNs

## 1. Objective
Build and evaluate a deep-learning classifier to distinguish
Parasitized from Uninfected malaria cell images.

## 2. Dataset and preprocessing
- Training images: 19,290
- Validation images: 4,134
- Test images: 4,134
- Classes: Parasitized and Uninfected
- Input size: 128 × 128 RGB
- Pixel scaling: [0, 255] to [-1, 1] for MobileNetV2
- Split manifests and image metadata are retained in the outputs folder.

## 3. Custom CNN
A custom convolutional neural network was trained using Adam,
binary cross-entropy, regularization, dropout, early stopping,
and learning-rate reduction.

Held-out test results:
- Accuracy: 96.15%
- Precision: 96.31%
- Recall: 95.98%
- F1-score: 96.15%
- ROC-AUC: 0.9904

## 4. Transfer learning: MobileNetV2
A pretrained MobileNetV2 feature extractor was frozen and a binary
classification head was trained. The best checkpoint was selected
using validation loss.

Held-out test results:
- Accuracy: 92.96%
- Precision: 94.14%
- Recall: 91.63%
- F1-score: 92.87%
- ROC-AUC: 0.9789
- Training time: approximately 178 seconds
- Parameters: approximately 2.26 million

## 5. Model comparison
The custom CNN outperformed this frozen-backbone MobileNetV2
experiment on all reported test metrics. This conclusion applies
to these specific model configurations and this dataset split;
it does not prove custom CNNs are generally superior to transfer
learning.

## 6. Explainability
Grad-CAM visualizations were generated to explore image regions
that contribute to selected predictions. These are exploratory
visual explanations, not proof that the model relies on medically
meaningful features.

## 7. Data leakage audit
An exact SHA-256 image-content comparison found zero exact duplicate
matches between train/validation, train/test, and validation/test.

This does not rule out near-duplicates, related images from the same
slide, or other forms of leakage. Source-group independence has not
been established by this check alone.

## 8. Limitations
- Results are specific to this dataset and split.
- Independent clinical validation has not been performed.
- Model confidence may not be calibrated.
- Grad-CAM maps are not causal explanations.
- MobileNetV2 was tested with a frozen backbone and limited training.
- Further checks for near-duplicates and source-slide overlap are
  recommended if source identifiers are available.

## 9. Conclusion
The custom CNN achieved 96.15% test accuracy and 0.9904 ROC-AUC,
outperforming the tested MobileNetV2 configuration. The project
includes a baseline model, an alternative model experiment,
test predictions, and exploratory Grad-CAM explanations.

This is an educational research prototype, not a clinically
validated diagnostic system.
