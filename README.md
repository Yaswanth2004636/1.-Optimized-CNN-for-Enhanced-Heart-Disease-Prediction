# 🫀 Optimized CNN with Grasshopper Optimization for Heart Disease Prediction

This project introduces a **hybrid deep learning model** that combines the power of **Convolutional Neural Networks (CNNs)** with the **Grasshopper Optimization Algorithm (GHO)** to predict heart disease with high accuracy. It addresses the limitations of conventional models by automating the hyperparameter tuning process and significantly boosting model performance.

---

## 📌 Overview

Heart disease remains a leading cause of death worldwide. Early detection is crucial, and machine learning has proven to be a powerful tool in this domain. Our approach leverages the **CNN-GHO hybrid model** to outperform traditional prediction techniques by:

- Automating hyperparameter tuning
- Reducing overfitting through optimized dropout rates
- Increasing robustness and generalization of the model

---

## 🧠 Methodology

We trained and evaluated the model on the **Cleveland Heart Disease Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

### 🔧 Key Features:

- **CNN Architecture**: Learns complex patterns from clinical data.
- **Grasshopper Optimization Algorithm (GHO)**: Simulates grasshopper swarm behavior to find optimal values for:
  - Learning rate
  - Dropout rate
  - Number of neurons in dense layers

This automated hyperparameter tuning helps the CNN reach peak performance without manual intervention.

---

## 📈 Performance Metrics

| Metric     | Score    |
|------------|----------|
| Accuracy   | **88.52%** |
| Precision  | **87.87%** |
| Recall     | **90.62%** |
| F1-Score   | **89.23%** |

These results demonstrate the superior capability of the CNN-GHO model in comparison to baseline CNNs and traditional ML models.

---

## 🛠️ Technologies Used

- 🐍 Python  
- 🧠 TensorFlow & Keras  
- 📘 Scikit-learn  
- 🐼 Pandas  
- 🔢 NumPy  
- 📊 Matplotlib & Seaborn

---

## 📂 Project Structure

heart-disease-cnn-gho/
│
├── data/
│ └── heart_disease.csv # Cleaned Cleveland dataset
│
├── models/
│ ├── cnn_model.py # CNN architecture implementation
│ └── cnn_gho_model.py # CNN + GHO hybrid model
│
├── optimization/
│ └── grasshopper_optimizer.py # Grasshopper Optimization Algorithm
│
├── results/
│ ├── accuracy_plot.png # Accuracy graph
│ ├── loss_plot.png # Loss curve
│ └── classification_report.txt # Precision, Recall, F1
│
├── utils/
│ └── data_preprocessing.py # Data loading and preprocessing
│
├── main.py # Main script to run the model
├── requirements.txt # Required Python packages
└── README.md # Project documentation
---

