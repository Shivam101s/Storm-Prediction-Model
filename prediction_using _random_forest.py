# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

# Load the dataset
# Replace 'weather_data.csv' with your dataset file
# Dataset should include columns: 'wind_speed', 'precipitation', 'humidity', 'high_intensity_storm'
data = pd.read_csv('weather_data.csv')

# Handle missing values (if any)
data.dropna(inplace=True)

# Split data into features and target
X = data[['wind_speed', 'precipitation', 'humidity']]
y = data['high_intensity_storm']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and tune individual models
def train_random_forest(X, y):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Train individual feature models
wind_speed_model = train_random_forest(X_train[['wind_speed']], y_train)
precipitation_model = train_random_forest(X_train[['precipitation']], y_train)
humidity_model = train_random_forest(X_train[['humidity']], y_train)

# Feature importance visualization
def plot_feature_importance(model, feature_name):
    plt.figure(figsize=(8, 6))
    importance = model.feature_importances_
    sns.barplot(x=importance, y=[feature_name])
    plt.title(f'Feature Importance for {feature_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(wind_speed_model, 'wind_speed')
plot_feature_importance(precipitation_model, 'precipitation')
plot_feature_importance(humidity_model, 'humidity')

# Generate predictions for training data from feature models
train_wind_preds = wind_speed_model.predict_proba(X_train[['wind_speed']])[:, 1]
train_precip_preds = precipitation_model.predict_proba(X_train[['precipitation']])[:, 1]
train_humidity_preds = humidity_model.predict_proba(X_train[['humidity']])[:, 1]

# Combine feature model predictions into a new dataset for meta-model
meta_train_X = np.column_stack((train_wind_preds, train_precip_preds, train_humidity_preds))
meta_train_y = y_train

# Scale data for meta-model
scaler = StandardScaler()
meta_train_X_scaled = scaler.fit_transform(meta_train_X)

# Train the meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_train_X_scaled, meta_train_y)

# Generate predictions for testing data from feature models
test_wind_preds = wind_speed_model.predict_proba(X_test[['wind_speed']])[:, 1]
test_precip_preds = precipitation_model.predict_proba(X_test[['precipitation']])[:, 1]
test_humidity_preds = humidity_model.predict_proba(X_test[['humidity']])[:, 1]

# Combine feature model predictions into a new dataset for meta-model inference
meta_test_X = np.column_stack((test_wind_preds, test_precip_preds, test_humidity_preds))
meta_test_X_scaled = scaler.transform(meta_test_X)

# Meta-model inference
final_predictions = meta_model.predict(meta_test_X_scaled)

# Evaluate the final model
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("\nClassification Report:\n", classification_report(y_test, final_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, meta_model.predict_proba(meta_test_X_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Example: Inference on new data
new_data = pd.DataFrame({
    'wind_speed': [20, 15, 30],  # Replace with new data
    'precipitation': [5, 10, 15],
    'humidity': [70, 65, 80]
})

# Generate predictions for new data
new_wind_preds = wind_speed_model.predict_proba(new_data[['wind_speed']])[:, 1]
new_precip_preds = precipitation_model.predict_proba(new_data[['precipitation']])[:, 1]
new_humidity_preds = humidity_model.predict_proba(new_data[['humidity']])[:, 1]

# Combine predictions for meta-model inference
new_meta_X = np.column_stack((new_wind_preds, new_precip_preds, new_humidity_preds))
new_meta_X_scaled = scaler.transform(new_meta_X)
new_final_predictions = meta_model.predict(new_meta_X_scaled)

# Output the predictions
print("Predictions for New Data:", new_final_predictions)

