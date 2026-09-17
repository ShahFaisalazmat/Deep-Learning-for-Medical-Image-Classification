# Deep Learning for Medical Image Classification

## Chest X-Ray Pneumonia Classification

A reproducible deep learning project for binary classification of
pediatric chest X-ray images into **Normal** and **Pneumonia** classes.

The project develops a lightweight convolutional neural network (CNN)
from scratch and compares it with ImageNet-pretrained **VGG16** and
**ResNet50** models under frozen and fine-tuned settings. The study also
investigates augmentation, hyperparameter configurations, training-data
volume, model performance, and classification errors.

> **Course:** Generative Artificial Intelligence --- Assignment #1\
> **Department:** Software Engineering, FAST --- National University of
> Computer and Emerging Sciences (NUCES), Islamabad\
> **Authors:** Shah Faisal (23I-0058), Hamad Khan (23I-3095)

------------------------------------------------------------------------

## Overview

Medical image classification is an important application of deep
learning. This project investigates automated classification of
pediatric chest X-rays as either **Normal** or **Pneumonia**.

The complete pipeline covers:

-   Dataset cleaning and duplicate detection
-   Reproducible train/validation/test splitting
-   Class-imbalance handling
-   Training-set data augmentation
-   Custom CNN development
-   Transfer learning using VGG16 and ResNet50
-   Frozen and fine-tuned model comparisons
-   Hyperparameter experimentation
-   Training-data-volume analysis
-   Confusion matrix and ROC/AUC evaluation
-   Per-class precision and recall
-   Error analysis and discussion of model failure patterns

All experiments were executed on a **Kaggle Notebook using a single
Tesla T4 GPU**, with a fixed random seed for reproducibility.

------------------------------------------------------------------------

## Dataset

The project uses the Kaggle **Chest X-Ray Images (Pneumonia)** dataset.

The original dataset contains pediatric chest X-rays labeled:

-   `NORMAL`
-   `PNEUMONIA`

### Data Cleaning

Each image was checked for corruption using `PIL.Image.verify()` and
exact duplicates were identified using MD5 hashes.

After cleaning:

-   **5,824 images** remained
-   **1,579 Normal**
-   **4,245 Pneumonia**
-   **32 exact duplicates** were removed
-   No corrupted files were found

### Reproducible Data Split

The cleaned dataset was pooled and split using a stratified **80/10/10**
split with `random_state=42`.

  Split          Normal   Pneumonia   Total
  ------------ -------- ----------- -------
  Train           1,263       3,396   4,659
  Validation        158         424     582
  Test              158         425     583

An explicit filepath-overlap check was used to verify that the splits
did not contain overlapping images.

------------------------------------------------------------------------

## Preprocessing

The image preprocessing pipeline includes:

1.  Resize images to **128 × 128**
2.  Convert grayscale images to three-channel RGB
3.  Rescale pixel values to `[0, 1]`
4.  Apply class weights during training to address class imbalance

The approximate Pneumonia-to-Normal ratio after cleaning is **2.7:1**.

Class weights used during model training:

-   Normal: **1.844**
-   Pneumonia: **0.686**

------------------------------------------------------------------------

## Data Augmentation

Augmentation was applied **only to the training set**.

The pipeline includes:

-   Rotation: ±15°
-   Width/height shift: ±10%
-   Zoom: ±15%
-   Brightness adjustment: ×\[0.85, 1.15\]
-   Horizontal flipping

Validation and test data were not augmented.

------------------------------------------------------------------------

## Model Architectures

### 1. Custom CNN

A lightweight CNN was developed from scratch using four convolutional
blocks:

``` text
Input: 128 × 128 × 3

Conv2D (32)
Batch Normalization
Max Pooling

Conv2D (64)
Batch Normalization
Max Pooling

Conv2D (128)
Batch Normalization
Max Pooling

Conv2D (128)
Batch Normalization
Max Pooling

Global Average Pooling
Dense (128, ReLU)
Dropout (0.4)
Dense (1, Sigmoid)
```

The model contains approximately **0.26 million trainable parameters**.

Global Average Pooling was used instead of Flatten to reduce the number
of parameters and help control overfitting.

------------------------------------------------------------------------

### 2. VGG16 Transfer Learning

ImageNet-pretrained VGG16 was evaluated in two configurations:

-   **Frozen:** pretrained backbone weights remain fixed while the
    classification head is trained
-   **Fine-tuned:** the backbone is unfrozen and trained with a lower
    learning rate

------------------------------------------------------------------------

### 3. ResNet50 Transfer Learning

ImageNet-pretrained ResNet50 was also evaluated in:

-   **Frozen**
-   **Fine-tuned**

This resulted in five evaluated models in total:

``` text
1. primary_cnn
2. vgg16_frozen
3. vgg16_finetuned
4. resnet50_frozen
5. resnet50_finetuned
```

------------------------------------------------------------------------

## Training

The models use:

-   **Optimizer:** Adam
-   **CNN loss:** Binary Cross-Entropy
-   **Monitoring metric:** Validation AUC
-   **ModelCheckpoint:** Save best-performing weights
-   **EarlyStopping:** Restore best weights

Training budgets were adjusted according to the model configuration.

The custom CNN was trained for up to 8 epochs, while frozen and
fine-tuned transfer-learning models used shorter training schedules.

------------------------------------------------------------------------

## Experimental Studies

### Hyperparameter Search

A scripted Cartesian grid was defined across eight dimensions:

-   Batch size
-   Learning rate
-   Epochs
-   Dropout
-   Early-stopping patience
-   L1 regularization
-   L2 regularization
-   Normalization
-   Augmentation policy

A deterministic **12-trial subset** was executed on a fixed stratified
tuning subset because of the available single-GPU session budget.

The best observed configuration reached:

**Validation AUC = 0.890**

with:

-   Batch size: 64
-   Learning rate: 10⁻³
-   Epochs: 5
-   Dropout: 0.3
-   Patience: 2
-   L1: 0
-   L2: 10⁻⁴
-   Standardization
-   Light augmentation

------------------------------------------------------------------------

### Training-Data Volume Study

The primary CNN was retrained using:

-   25%
-   50%
-   75%
-   100%

of the training data.

The purpose was to investigate how training-data volume affected test
performance.

------------------------------------------------------------------------

## Results

The final test-set comparison was:

  Model                    Accuracy    Macro-F1     AUC-ROC
  --------------------- ----------- ----------- -----------
  Primary CNN                 0.856       0.777       0.973
  VGG16 Frozen                0.822       0.804       0.964
  VGG16 Fine-tuned        **0.873**   **0.857**   **0.990**
  ResNet50 Frozen             0.794       0.765       0.878
  ResNet50 Fine-tuned         0.604       0.602       0.913

The fine-tuned VGG16 achieved the highest reported accuracy, macro-F1,
and AUC-ROC among the five evaluated models.

The custom CNN achieved an **AUC-ROC of 0.973** while using fewer than
0.26 million parameters.

------------------------------------------------------------------------

## Primary CNN Error Analysis

The primary CNN achieved:

-   Test accuracy: **0.856**
-   AUC-ROC: **0.973**
-   Pneumonia precision: **0.838**
-   Pneumonia recall: **0.995**
-   Normal recall: **0.481**

Its confusion matrix contained:

-   **76/158 Normal** images correctly classified
-   **423/425 Pneumonia** images correctly classified

The error analysis showed a strong tendency toward predicting Pneumonia.
Among the 84 test errors, 82 were False Positives and 2 were False
Negatives.

This highlights the importance of decision-threshold calibration and
class-balance considerations in medical image classification.

------------------------------------------------------------------------

## Reproducibility

The experiments use:

``` text
SEED = 42
```

The seed is applied across the relevant Python, NumPy, and TensorFlow
components.

The pipeline also includes explicit leakage checks:

-   No overlapping image filepaths between Q1 train/validation/test sets
-   Reproducible stratified splitting
-   Training-only augmentation

Because GPU operations can introduce some non-determinism, exact
numerical reproduction may vary slightly across environments.

------------------------------------------------------------------------

## Technology Stack

-   **Python**
-   **TensorFlow / Keras**
-   **NumPy**
-   **Pandas**
-   **Matplotlib**
-   **Seaborn**
-   **Scikit-learn**
-   **Pillow**
-   **KaggleHub**
-   **Kaggle Notebooks**
-   **Tesla T4 GPU**

------------------------------------------------------------------------

## Repository Structure

``` text
Deep-Learning-Medical-Image-Classification/
│
├── Compiled_Question01.ipynb
├── main.tex
├── figs/
└── README.md
```

### Main Notebook

`Compiled_Question01.ipynb`

The notebook executes the complete chest X-ray classification pipeline,
including preprocessing, augmentation, model training, evaluation,
hyperparameter experimentation, training-data-volume analysis, and error
analysis.

------------------------------------------------------------------------

## How to Run

The notebook was designed and executed in **Kaggle Notebooks**.

### Requirements

A GPU is strongly recommended because the pipeline trains multiple CNN
and transfer-learning models.

The environment uses:

-   Python 3.12
-   TensorFlow / Keras
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Pillow
-   KaggleHub

### Dataset Access

The notebook downloads the dataset through KaggleHub. When running
outside Kaggle, a valid Kaggle API configuration is required.

------------------------------------------------------------------------

## Important Note

This project is an academic deep-learning study and **is not a clinical
diagnostic system**. The reported results are based on the specified
pediatric chest X-ray dataset and experimental setup and should not be
interpreted as medical validation or deployment evidence.

------------------------------------------------------------------------

## Future Work

Potential extensions identified in the study include:

-   Grad-CAM visual interpretability maps
-   Multi-class classification
-   Model ensembling
-   Decision-threshold calibration
-   External validation using data from a different hospital
-   Additional evaluation of model generalization

------------------------------------------------------------------------

## Authors

**Shah Faisal** --- 23I-0058\
**Hamad Khan** --- 23I-3095

Department of Software Engineering\
FAST --- National University of Computer and Emerging Sciences (NUCES),
Islamabad

------------------------------------------------------------------------

## Academic Context

**Generative Artificial Intelligence --- Assignment #1**

The broader assignment also includes a separate English→Urdu neural
machine translation study using a vanilla RNN encoder--decoder. This
repository section focuses specifically on the **medical image
classification / chest X-ray pneumonia classification** component.
