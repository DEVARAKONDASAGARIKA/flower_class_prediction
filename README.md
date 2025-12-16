# flower_class_prediction
# 🌸 Indian Flower Classification using Machine Learning

This repository contains a **complete end-to-end Machine Learning pipeline** for classifying **Indian origin flowers** using image data. The project demonstrates **image preprocessing, feature extraction, model training, evaluation, and result analysis**, making it suitable for **academic projects, hackathons, interviews, and viva presentations**.

---

## 📌 Project Overview

The goal of this project is to **automatically classify flower images into their respective categories** using traditional Machine Learning techniques (not deep learning). The pipeline is designed to be **interpretable, modular, and easy to explain**, which is ideal for **undergraduate ML coursework and interviews**.

Key highlights:

* Uses **classical ML models** (KNN, SVM)
* Employs **feature extraction techniques** like texture and color descriptors
* Works on **colored images**
* Includes **performance evaluation metrics**

---

## 📂 Dataset Structure

The dataset directory should follow this structure:

```
IndianFlowerDataset/
│
├── Rose/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── Lotus/
│   ├── img1.jpg
│
├── Sunflower/
│   ├── img1.jpg
│
└── ...
```

* Each **subfolder name represents the class label**
* Images can be in `.jpg`, `.png`, or `.jpeg` format

---

## ⚙️ Technologies Used

* **Python 3.x**
* **OpenCV** – image loading & resizing
* **NumPy & Pandas** – numerical operations
* **Scikit-learn** – ML models & evaluation
* **Matplotlib & Seaborn** – visualization
* **SciPy / skimage** – feature extraction

---

## 🔄 Pipeline Architecture

```
Image Loading
     ↓
Image Resizing (128×128)
     ↓
Feature Extraction
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Evaluation & Metrics
```

---

## 🧠 Feature Extraction Techniques

The following features are extracted from **colored images**:

### 1️⃣ Color Features

* Mean and standard deviation of RGB channels
* Captures color distribution of flowers

### 2️⃣ Texture Features

* Gray-Level Co-occurrence Matrix (GLCM)
* Haralick texture properties

### 3️⃣ Edge Features

* Histogram of Oriented Gradients (HOG)
* Captures shape and petal structure

All extracted features are concatenated into a **single feature vector**.

---

## 🤖 Machine Learning Models Used

The project implements **four different Machine Learning models** to compare performance and understand their strengths on image-based classification tasks.

### 🔹 Support Vector Machine (SVM)

* Constructs an optimal separating hyperplane
* Effective in high-dimensional feature spaces
* Works well with extracted image features
* Uses kernel trick for non-linear separation

### 🔹 K-Nearest Neighbors (KNN)

* Distance-based, instance-based learning algorithm
* Simple and intuitive to understand
* Performance depends on choice of *k* and distance metric
* Suitable for small to medium-sized datasets

### 🔹 Random Forest Classifier

* Ensemble learning method using multiple decision trees
* Reduces overfitting compared to single decision trees
* Handles non-linear relationships effectively
* Provides feature importance insights

### 🔹 Logistic Regression

* Linear classification algorithm
* Uses sigmoid function to estimate class probabilities
* Fast, interpretable, and easy to explain in interviews
* Serves as a strong baseline model

---

## 📊 Model Evaluation Metrics

The performance of the models is evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report**

  * Precision
  * Recall
  * F1-score

Additionally, predicted labels are displayed **along with true labels** for better interpretability.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/indian-flower-classification.git
cd indian-flower-classification
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Update Dataset Path

In the notebook or script, update:

```python
data_path = "path/to/IndianFlowerDataset"
```

### 4️⃣ Run the Notebook

Open and execute:

```
Indian_Flower_Classification.ipynb
```

---

## 📈 Results

* Achieved **high classification accuracy** on test data
* KNN performs well for smaller datasets
* SVM provides more stable and generalized results

Exact accuracy may vary depending on:

* Dataset size
* Number of flower classes
* Feature combinations

---

## 🎯 Why This Project is Interview-Friendly

✔ Uses **classical ML (easy to explain)**
✔ Clear **problem → solution mapping**
✔ Modular and clean code
✔ Covers **end-to-end ML workflow**
✔ Ideal for **AI Engineer / ML Engineer roles**

---

## 🔮 Future Enhancements

* Add Deep Learning (CNN) for comparison
* Perform feature selection / PCA
* Deploy model using Flask or FastAPI
* Add real-time flower prediction

---

## 👤 Author

**Undergraduate CSE Student**
Indian Institute of Technology (IIT)

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ *If you find this project useful, feel free to star the repository!*
