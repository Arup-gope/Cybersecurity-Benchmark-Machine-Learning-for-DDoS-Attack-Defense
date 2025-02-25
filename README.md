# DDoS Attack Detection using Machine Learning
A machine learning-based system for detecting DDoS attacks using network traffic data. This project compares three models (Random Forest, Logistic Regression, and Neural Network) to classify malicious traffic and provides detailed performance metrics and visualizations.

---

## Features
- **Three ML Models**: Random Forest, Logistic Regression, and Neural Network (MLP).
- **Data Preprocessing**: Robust handling of missing values, outlier scaling, and feature engineering.
- **Performance Metrics**: Accuracy, F1 Score, Precision, Recall, and ROC-AUC analysis.
- **Visualizations**: Confusion matrices and ROC curves for model comparison.
- **Dataset**: Uses the [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) from Kaggle.

---

## Installation

### Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Arup-gope/Cybersecurity-Benchmark-Machine-Learning-for-DDoS-Attack-Defense.git
   cd Cybersecurity-Benchmark-Machine-Learning-for-DDoS-Attack-Defense
## How to Run

### Prerequisites
- Python 3.8+ installed
- [Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) downloaded

### Quick Start
1. Clone repository and install dependencies:
```bash
git clone https://github.com/Arup-gope/Cybersecurity-Benchmark-Machine-Learning-for-DDoS-Attack-Defense.git
cd Cybersecurity-Benchmark-Machine-Learning-for-DDoS-Attack-Defense
pip install -r requirements.txt
```
2. Set up dataset:
```
mkdir -p Data  # Place downloaded "DDos2.csv" file in the Data folder
```

3. Run the analysis:
```
python project.py
```

## Expected Output
```
Train shape: (735000, 82) Test shape: (315000, 82)

Training Random Forest...
[===] 100% | 100 trees built
Random Forest Metrics:
Accuracy: 0.9999 | F1: 0.9999 | Precision: 0.9999 | Recall: 0.9999

Training Logistic Regression...
Logistic Regression Metrics:
Accuracy: 0.9954 | F1: 0.9960 | Precision: 0.9950 | Recall: 0.9970

Training Neural Network...
Epoch 500/500 - loss: 0.0012
Neural Network Metrics:
Accuracy: 0.9992 | F1: 0.9993 | Precision: 0.9995 | Recall: 0.9991
```
## Project Results and Visualizations
This section documents the key visual results from our machine learning models trained to detect DDoS attacks. The results include an ROC curve comparison and confusion matrices for Random Forest, Logistic Regression, and Neural Network models.
### Confusion Matrix: Random Forest

The Random Forest model demonstrated the best overall performance with minimal errors:

True Positives: 38,403
True Negatives: 29,304
False Positives: 2
False Negatives: 5
![RandomForest](https://github.com/user-attachments/assets/a01f9e6e-a5e8-41cb-92e7-cfc60889b75e)

### Confusion Matrix: Logistic Regression

The Logistic Regression model also performed well, though with slightly higher misclassification:

True Positives: 38,291
True Negatives: 29,112
False Positives: 194
False Negatives: 117
![LogisticRegression](https://github.com/user-attachments/assets/33be36a2-ae59-407c-b3d7-3d9b4fe5a456)

###  Confusion Matrix: Neural Network

The Neural Network achieved high accuracy with very few misclassifications:

True Positives (DDoS correctly detected): 38,375
True Negatives (Benign correctly identified): 29,286
False Positives: 20
False Negatives: 33
![NeuralNetwork](https://github.com/user-attachments/assets/5efbbcaf-820c-483b-91db-0888426f55a4)

### ROC Curve Comparison

The ROC (Receiver Operating Characteristic) curve illustrates the performance of our three models:
Random Forest (AUC = 1.00)
Logistic Regression (AUC = 1.00)
Neural Network (AUC = 1.00)

All models achieved an ideal AUC score of 1.00, indicating excellent discrimination capability.
![ROC_Curve](https://github.com/user-attachments/assets/ed55ea3f-0676-4489-ae5a-9a009100dfef)

## Summary of Results

Overall, the Random Forest classifier emerged as the top-performing model, providing exceptional accuracy and the fewest misclassifications, followed closely by the Neural Network and Logistic Regression models


















