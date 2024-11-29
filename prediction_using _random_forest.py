# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'weather_data.csv' with your dataset file)
# Dataset should include columns: 'wind_speed', 'precipitation', 'humidity', 'high_intensity_storm'
data = pd.read_csv('weather_data.csv')

# Split data into features and target
X = data[['wind_speed', 'precipitation', 'humidity']]
y = data['high_intensity_storm']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize individual models
wind_speed_model = RandomForestClassifier(random_state=42)
precipitation_model = RandomForestClassifier(random_state=42)
humidity_model = RandomForestClassifier(random_state=42)

# Train individual feature models
wind_speed_model.fit(X_train[['wind_speed']], y_train)
precipitation_model.fit(X_train[['precipitation']], y_train)
humidity_model.fit(X_train[['humidity']], y_train)

# Generate predictions for training data from feature models
train_wind_preds = wind_speed_model.predict_proba(X_train[['wind_speed']])[:, 1]
train_precip_preds = precipitation_model.predict_proba(X_train[['precipitation']])[:, 1]
train_humidity_preds = humidity_model.predict_proba(X_train[['humidity']])[:, 1]

# Combine feature model predictions into a new dataset for meta-model
meta_train_X = np.column_stack((train_wind_preds, train_precip_preds, train_humidity_preds))
meta_train_y = y_train

# Train the meta-model
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_train_X, meta_train_y)

# Generate predictions for testing data from feature models
test_wind_preds = wind_speed_model.predict_proba(X_test[['wind_speed']])[:, 1]
test_precip_preds = precipitation_model.predict_proba(X_test[['precipitation']])[:, 1]
test_humidity_preds = humidity_model.predict_proba(X_test[['humidity']])[:, 1]

# Combine feature model predictions into a new dataset for meta-model inference
meta_test_X = np.column_stack((test_wind_preds, test_precip_preds, test_humidity_preds))

# Meta-model inference
final_predictions = meta_model.predict(meta_test_X)

# Evaluate the final model
print("Final Model Accuracy:", accuracy_score(y_test, final_predictions))
print("\nClassification Report:\n", classification_report(y_test, final_predictions))

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
new_final_predictions = meta_model.predict(new_meta_X)

# Output the predictions
print("Predictions for New Data:", new_final_predictions)
