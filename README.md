# Breast-Cancer-Classification-Project-using-ML-and-DL
Binary classification of breast cancer tumors using Machine Learning (Scikit-learn) and Deep Learning (Tensorflow) models with performance comparison.

🩺 Breast Cancer Classification using Machine Learning & Deep Learning
<p align="center"> <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-red?style=for-the-badge&logo=tensorflow"/> <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge"/> </p>
📌 Project Overview

Breast cancer is one of the most common cancers worldwide.
This project focuses on building a binary classification system to predict whether a breast tumor is Malignant or Benign using Machine Learning and Deep Learning techniques.

The goal is to compare traditional ML models with a Neural Network and analyze their performance on medical data.

📊 Dataset Information

Dataset Name: Breast Cancer Wisconsin Dataset

Source: Kaggle / Scikit-learn

Total Features: 30 numerical features

Target Variable:

0 → Benign

1 → Malignant

Sample Features:

Mean Radius

Mean Texture

Mean Smoothness

Mean Concavity

🔄 Project Workflow

Data Loading & Exploration

Data Cleaning & Preprocessing

Feature Scaling using StandardScaler

Train-Test Split

Model Training

Model Evaluation & Comparison

🧠 Models Implemented
🔹 Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

🔹 Deep Learning Model

Fully Connected Neural Network

ReLU activation (Hidden Layers)

Softmax activation (Output Layer)

Adam Optimizer

Sparse Categorical Crossentropy Loss

📈 Results & Performance
Model	Accuracy
Logistic Regression	~52%
SVM	~53%
Random Forest	~51%
Neural Network	~95%

✅ The Neural Network outperformed traditional ML models, demonstrating the effectiveness of deep learning for complex medical datasets.

⚙️ Technologies & Tools

Programming Language: Python

Libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras

Platform: Google Colab

▶️ How to Run the Project
# Clone the repository
git clone https://github.com/your-username/Breast-Cancer-Classification-ML-DL.git

# Navigate to the project folder
cd Breast-Cancer-Classification-ML-DL

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

🎯 Key Learnings

Importance of feature scaling in ML models

Performance comparison between ML and DL

Handling medical datasets responsibly

Building end-to-end ML pipelines

🚀 Future Improvements

Hyperparameter tuning (GridSearchCV)

Cross-validation

ROC-AUC and Confusion Matrix visualization

Model deployment using Flask or Streamlit

👨‍💻 Author

Aadarsh Shukla
Aspiring AI/ML Engineer 🚀
Passionate about Machine Learning, Deep Learning & Real-world problem solving
