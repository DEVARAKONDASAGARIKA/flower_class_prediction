#  Flower Classification Project – Indian Flower Dataset

##  Project Overview
This project implements a comprehensive **flower classification system** using traditional **computer vision and machine learning techniques**.  
It extracts **texture, color, and shape-based features** from flower images and compares multiple ML classifiers to accurately identify **Indian flower species**.

The project focuses on **feature engineering + classical ML models**

## Project Goals
- Develop a robust flower classification pipeline using **Computer Vision + ML**
- Extract meaningful features from images:
  - Texture (Gabor Filters, LBP)
  - Color (HSV histograms, RGB statistics, color moments)
- Implement and compare multiple machine learning models
- Provide a **reproducible and extensible** classification framework

---

##  Dataset
- **Source:** Indian Flower Dataset (custom / local)
- **Location:**  
C:\Users\DELL\Desktop\indianflower

- **Structure:**  
Each flower species is stored in a **separate folder**
- **Images:** Color images of Indian flowers
- **Classes:** Automatically extracted from folder names

---

##  Project Structure

flower-classification/
│
├── flower_classification.ipynb          # Main implementation script
│
├── results/                          # Generated plots and visualizations
│
└── README.md                         # Project documentation

# Technical Implementation
1️⃣ Image Preprocessing

->Resize images to 128 × 128
->Convert images to grayscale for texture analysis
->Apply Gaussian filtering (σ = 1) for noise reduction
->Perform histogram equalization for contrast enhancement
->Preserve color images for color feature extraction

2️⃣ Feature Extraction

Each image is represented using ~560 handcrafted features.

#Texture Features
->Gabor Filter Features
->Kernel size: 15 × 15
->Sigma values: [2, 3]
->Orientations: 0°, 45°, 90°, 135°
->Features extracted: Mean & Variance
->Local Binary Pattern (LBP)
->Neighbors (P): 24
->Radius (R): 3
->Method: uniform
->Histogram bins: 26

#Color Features

->HSV Color Histogram
->Bins: 8 × 8 × 8 (512 features)
->Captures hue, saturation, and brightness distribution
->RGB Statistics
->Mean and standard deviation of R, G, B channels (6 features)
->Color Moments
->Mean, Variance, Skewness per channel (9 features)

🔢 Total Features per Image

≈ 560 features
Implemented Machine Learning Models
Four classifiers were implemented and evaluated:

🔹 Support Vector Machine (SVM)

->Kernel: RBF
->Hyperparameter tuning using GridSearchCV
->5-fold cross-validation

🔹 Random Forest Classifier

->Number of trees: 300
->Max depth: Unlimited
->Random state: 42

🔹 K-Nearest Neighbors (KNN)

->Neighbors: k = 5
->Distance metric: Euclidean

🔹 Logistic Regression

->Max iterations: 500
->Multi-class strategy: One-vs-Rest

#Model Training & Evaluation

->Train-Test Split: 80% / 20% (Stratified)
->Feature Scaling: StandardScaler (except Random Forest)
->Cross-Validation: 10-fold CV

#Evaluation Metrics

->Accuracy
->Precision, Recall, F1-score
->Confusion Matrix

#Results & Performance
🔹 Model Comparison (Test Accuracy)

SVM: Highest accuracy (dataset-dependent)
Random Forest: Robust and interpretable
KNN: Simple and effective for smaller datasets
Logistic Regression: Fast training and interpretable

#Visualizations Generated

The script automatically generates:
Sample training images
Confusion matrices for all models
Model comparison bar chart
10-fold cross-validation accuracy plots
Correct vs incorrect prediction samples

#How to Run
Prerequisites
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn scikit-image joblib

#Execution Steps

Update dataset path inside the script:
data_path = r"C:\Users\DELL\Desktop\indianflower"

Run the script:
python flower_classification.py

#Output

Loads and preprocesses images
Extracts handcrafted features
Trains and evaluates all models
Generates visualizations
Saves trained models to:
C:\Users\DELL\Desktop\models

# Customization Options

1️⃣ Change Image Size

img_size = 128  # Try 64 or 224

2️⃣ Modify Feature Extraction

Tune Gabor filter parameters
Adjust LBP (P, R)
Change HSV histogram bins

3️⃣ Add New Models

Add classifiers from scikit-learn to the models dictionary.

4️⃣ Modify Evaluation Strategy

test_size = 0.2
cv = 10

#Performance Optimization Tips
🔹 For Larger Datasets

Apply PCA for dimensionality reduction
Use feature selection techniques
Use incremental learning

🔹 For Higher Accuracy

Add HOG / SIFT features
Apply data augmentation
Use ensemble methods

🔹 For Faster Training

Reduce image size
Reduce feature dimensions
Use Linear SVM instead of RBF


#Future Enhancements

->CNN-based deep learning model
->Transfer learning (VGG, ResNet)
->Real-time flower classification
->Web interface using Flask / Django
