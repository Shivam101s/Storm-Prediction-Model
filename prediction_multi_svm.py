import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Step 1: Load and preprocess the dataset
df = pd.read_csv('processed_my.csv')

# Preprocessing the dataset
def process_latitude(lat):
    return float(lat[:-1]) * (1 if lat[-1] == 'N' else -1)

def process_longitude(lon):
    return float(lon[:-1]) * (1 if lon[-1] == 'E' else -1)

df['Latitude_NS'] = df['Latitude_NS'].apply(process_latitude)
df['Longitude_EW'] = df['Longitude_EW'].apply(process_longitude)

df['Storm_Intensity'] = df['Storm_Intensity'].astype('category')

# Step 2: Define features and target variable
X = df[['Latitude_NS', 'Longitude_EW', 'Min_Pressure']]
y = df['Storm_Intensity']
y = y.cat.codes

# Standardize features for SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train separate models for each feature using SVM
feature_models = []  # List to store models for each feature
for i, feature in enumerate(X_train.T):
    model = SVC(kernel='rbf', probability=True, random_state=42)  # SVM with RBF kernel
    model.fit(X_train[:, [i]], y_train)  # Train model on one feature at a time
    feature_models.append(model)

# Step 4: Function to get predictions from feature-specific models
def get_feature_predictions(models, X):
    feature_preds = []
    for i, model in enumerate(models):
        feature_pred = model.predict_proba(X[:, [i]])  # Predict probabilities
        feature_preds.append(feature_pred)
    return np.hstack(feature_preds)  # Combine predictions

# Get combined predictions for training and test datasets
train_feature_preds = get_feature_predictions(feature_models, X_train)
test_feature_preds = get_feature_predictions(feature_models, X_test)

# Step 5: Train a meta-model on feature-specific predictions using SVM
meta_model = SVC(kernel='rbf', probability=False, random_state=42)  # Meta-model with SVM
meta_model.fit(train_feature_preds, y_train)

# Step 6: Evaluate the ensemble model
meta_predictions = meta_model.predict(test_feature_preds)

# Calculate total accuracy
accuracy = accuracy_score(y_test, meta_predictions)
print(f"Total Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, meta_predictions))

# Step 7: Visualizations
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, meta_predictions)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=df['Storm_Intensity'].cat.categories)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix for Meta-Model")
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Feature Predictions Visualization (Optional)
# Predicted vs Actual for the Meta-Model
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7, color='blue')
plt.scatter(range(len(meta_predictions)), meta_predictions, label='Predicted', alpha=0.7, color='red')
plt.title("Actual vs Predicted Storm Intensity")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Storm Intensity (Encoded)")
plt.savefig('plots/actual_vs_predicted.png')
plt.close()
