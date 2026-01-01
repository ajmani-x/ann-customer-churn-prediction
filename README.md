# Customer Churn Prediction using Artificial Neural Network (ANN)

The model is trained using TensorFlow/Keras and includes multiple layers, dropout regularization, and feature scaling as part of hands-on learning of ANN fundamentals.

## 📝 Project Description
This project implements an **Artificial Neural Network (ANN)** to predict customer churn based on a dataset containing customer details from a bank. The goal is to identify customers who are likely to leave, helping businesses make data-driven retention decisions.

The model is trained using **TensorFlow/Keras** and includes multiple layers, dropout regularization, and feature scaling to improve performance.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Matplotlib, scikit-learn  
- **Tools:** Jupyter Notebook / Google Colab  

---

## 🔍 Dataset
- **File:** `Churn_Modelling.csv`  
- **Dependent Variable:** `Exited` (binary classification: 0 = stayed, 1 = left)  
- **Independent Variables:** Customer demographics, account information, credit score, etc.  

---

## ⚙️ Features & Preprocessing
- Handled **categorical variables** using one-hot encoding (`Geography`, `Gender`)  
- Split the dataset into **training (80%)** and **testing (20%)** sets  
- Applied **feature scaling** using `StandardScaler`  

---

## 🧠 Model Architecture
- **Input Layer:** 11 units, ReLU activation  
- **Hidden Layer 1:** 7 units, ReLU + Dropout (0.2)  
- **Hidden Layer 2:** 6 units, ReLU + Dropout (0.3)  
- **Output Layer:** 1 unit, Sigmoid activation (binary classification)  
- **Optimizer:** Adam with learning rate 0.01  
- **Loss Function:** Binary Crossentropy  
- **Early Stopping:** Monitored validation loss to prevent overfitting  

---

## 📈 Model Evaluation
- Evaluated using **accuracy** and **confusion matrix**  
- Achieved an **accuracy of ~86.35%** on the test set  
- Plotted **training vs validation accuracy** to monitor performance  

---

## 📌 Key Learnings
- Data preprocessing and handling categorical variables  
- Building and tuning **ANNs for binary classification**  
- Using **dropout layers** to prevent overfitting  
- Implementing **early stopping** for efficient training  
- Evaluating model performance with accuracy and confusion matrix  

---

## 🔗 References
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Keras Guide](https://keras.io/)  
- Dataset Source: [Kaggle - Churn Modelling](https://www.kaggle.com/datasets)  
