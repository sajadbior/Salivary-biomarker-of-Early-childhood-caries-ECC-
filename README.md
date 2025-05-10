import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import catboost
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Upload file
uploaded = files.upload()

# Loading data after upload
file_path = next(iter(uploaded))  
data = pd.read_csv(file_path)

# (target variable)
features = ['SEX', 'Mother age', 'M-EDU', 'father age', 'F-EDU', 'parent or child', 'age brush', 'Vitamin',
            'NO. Sweet', 'Floride toothpathe', 'Floride therapy', 'Dentist refere', 'Night Milk', 'Weight 1',
            'Weight 2', 'Age', 'STATH', '2BDF2']
X = data[features]
y = data['ECC']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model lists
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": catboost.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0),
    "Neural Network": Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

# Evaluation of models
results = {}

# Training and evaluating each model
for name, model in models.items():
    print(f"Training {name}...")
    # Model training
    if name == "Neural Network":
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        y_pred_prob = model.predict(X_test)[:, 0]
    else:
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Forecasting and evaluation
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert prediction to class 0 or 1
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # TPR
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TNR
    auc = roc_auc_score(y_test, y_pred_prob)

    # Save results
    results[name] = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': auc
    }

    # Drawing an AUC graph
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Show results
results_df = pd.DataFrame(results).T
print(results_df)

