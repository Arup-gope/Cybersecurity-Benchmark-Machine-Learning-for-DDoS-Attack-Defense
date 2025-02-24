import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import RobustScaler
from itertools import cycle

def load_data():
    df = pd.read_csv("Data/DDos2.csv")
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().copy()
    return df_clean

def preprocess_data(df):
    # Convert text labels to numerical values (0: BENIGN, 1: DDoS)

    df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
    # Separate features (X) and target variable (y)
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Convert to float32 to reduce memory usage
    X = X.astype(np.float32)

    # Scale features using RobustScaler (resistant to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def plot_distributions(df):
    plt.figure(figsize=(8, 4))
    sns.countplot(x='Label', data=df)
    plt.title('Class Distribution')
    plt.show()

# Load and clean data
if __name__ == "__main__":
    # Original Random Forest Implementation
    df = load_data()
    df_clean = clean_data(df)
    X, y = preprocess_data(df_clean)
    # Split data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate performance metrics
    print("\nRandom Forest Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, rf_pred):.4f}")
    print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
    print(f"Recall: {recall_score(y_test, rf_pred):.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, rf_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'])
    plt.title('Random Forest Confusion Matrix')
    plt.show()

    # Added Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Calculate performance metrics
    print("\nLogistic Regression Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, lr_pred):.4f}")
    print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
    print(f"Recall: {recall_score(y_test, lr_pred):.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, lr_pred), 
                annot=True, fmt='d', cmap='Greens',
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'])
    plt.title('Logistic Regression Confusion Matrix')
    plt.show()

    # Added Neural Network
    print("\nTraining Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)
    
    # Calculate performance metrics
    print("\nNeural Network Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, nn_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, nn_pred):.4f}")
    print(f"Precision: {precision_score(y_test, nn_pred):.4f}")
    print(f"Recall: {recall_score(y_test, nn_pred):.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, nn_pred), 
                annot=True, fmt='d', cmap='Reds',
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'])
    plt.title('Neural Network Confusion Matrix')
    plt.show()

    # Model Comparison
    models = {
        'Random Forest': rf_pred,
        'Logistic Regression': lr_pred,
        'Neural Network': nn_pred
    }
    
    # ROC Curve Comparison
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'green', 'red'])
    
    for name, pred in models.items():
        if name == 'Random Forest':
            proba = rf_model.predict_proba(X_test)[:, 1]
        elif name == 'Logistic Regression':
            proba = lr_model.predict_proba(X_test)[:, 1]
        else:
            proba = nn_model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=next(colors),
                 label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Metrics Table
    metrics_data = []
    for name, pred in models.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, pred),
            'F1 Score': f1_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred)
        })
    
    metrics_df = pd.DataFrame(metrics_data).set_index('Model')
    print("\nModel Comparison Table:")
    print(metrics_df.round(4))