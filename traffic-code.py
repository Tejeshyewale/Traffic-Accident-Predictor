import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('dataset_traffic_accident_prediction1.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Identify categorical and numerical columns
categorical_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 
                    'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
numerical_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 
                  'Driver_Alcohol', 'Driver_Age', 'Driver_Experience']

# Handle missing values
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Separate features (X) and target (y)
X = data.drop('Accident', axis=1)
y = data['Accident']

# Handle missing values in the target variable
y.fillna(y.mode()[0], inplace=True)

# Drop any rows where y is still NaN (if any exist)
X = X[~y.isnull()]
y = y[~y.isnull()]

# Identify ordinal and nominal categorical features for encoding
ordinal_features = ['Accident_Severity']
nominal_features = [col for col in categorical_cols if col != 'Accident_Severity']

# First, apply one-hot encoding to nominal features and keep the rest.
X_encoded = pd.get_dummies(X, columns=nominal_features, drop_first=True)

# Apply Label Encoding to the ordinal feature `Accident_Severity`
severity_le = LabelEncoder()
X_encoded['Accident_Severity_encoded'] = severity_le.fit_transform(X_encoded['Accident_Severity'])

# Drop the original categorical columns (use errors='ignore' to avoid KeyError if some cols are already gone)
X_encoded = X_encoded.drop(columns=categorical_cols, errors='ignore')

# Ensure target y is numeric for stratify/metrics
y_le = LabelEncoder()
y = pd.Series(y_le.fit_transform(y), index=y.index)

# Data Splitting: 70% train, 15% validation, 15% test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X_encoded, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=(0.15/0.85), random_state=42, stratify=y_train_temp)

print("Data shapes after splitting:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

## 2. Model Training and Evaluation

# Initialize and train the XGBoost model
# We'll use a fixed set of optimized parameters for this example.
xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train the model with the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set for tuning (if needed) and demonstration
y_val_pred = xgb_model.predict(X_val)
print("\nValidation Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")

# Make final predictions on the unseen test set
y_test_pred = xgb_model.predict(X_test)
print("\nTest Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")

# Display feature importance
feature_importances = pd.DataFrame(
    {'feature': X_encoded.columns, 'importance': xgb_model.feature_importances_}
).sort_values('importance', ascending=False)
print("\nFeature Importances (Top 10):")
print(feature_importances.head(10))

## 3. Deployment and Monitoring Preparation

# Save the trained model for later deployment
import joblib
joblib.dump(xgb_model, 'xgboost_traffic_model.pkl')
print("\nModel saved as xgboost_traffic_model.pkl")