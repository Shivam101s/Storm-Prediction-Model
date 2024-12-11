import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

if not os.path.exists('multi_nn_plots'):
    os.makedirs('multi_nn_plots')

# Step 1: Load and preprocess the dataset
df = pd.read_csv('processed_my.csv')
df = df.rename(columns={"Min_Pressure": "Max_Sustained_Wind"})


# Preprocessing the dataset
def process_latitude(lat):
    return float(lat[:-1]) * (1 if lat[-1] == 'N' else -1)

def process_longitude(lon):
    return float(lon[:-1]) * (1 if lon[-1] == 'E' else -1)

df['Latitude_NS'] = df['Latitude_NS'].apply(process_latitude)
df['Longitude_EW'] = df['Longitude_EW'].apply(process_longitude)

# Encode target variable
df['Storm_Intensity'] = df['Storm_Intensity'].astype('category')
label_encoder = LabelEncoder()
df['Storm_Intensity'] = label_encoder.fit_transform(df['Storm_Intensity'])

# Features and Target
X = df[['Latitude_NS', 'Longitude_EW', 'Max_Sustained_Wind']]
y = df['Storm_Intensity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scalers = {}
for feature in X_train.columns:
    scalers[feature] = StandardScaler()
    X_train[feature] = scalers[feature].fit_transform(X_train[[feature]])
    X_test[feature] = scalers[feature].transform(X_test[[feature]])

# Step 2: Define individual ANN models for each feature
def create_individual_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train individual models for each feature
feature_models = {}
for feature in X_train.columns:
    print(f"Training model for feature: {feature}")
    model = create_individual_model(input_dim=1, num_classes=len(np.unique(y_train)))
    model.fit(X_train[[feature]], y_train, epochs=20, batch_size=32, verbose=0)
    feature_models[feature] = model

# Step 3: Get predictions from feature-specific models
def get_feature_predictions(models, X):
    feature_preds = []
    for feature, model in models.items():
        pred = model.predict(X[[feature]], verbose=0)  # Predict probabilities
        feature_preds.append(pred)
    return np.hstack(feature_preds)  # Combine predictions

train_feature_preds = get_feature_predictions(feature_models, X_train)
test_feature_preds = get_feature_predictions(feature_models, X_test)

# Step 4: Train a meta-model on feature-specific predictions
meta_model = create_individual_model(input_dim=train_feature_preds.shape[1], num_classes=len(np.unique(y_train)))
print("Training meta-model...")
meta_model.fit(train_feature_preds, y_train, epochs=30, batch_size=32, verbose=0)

# Step 5: Evaluate the meta-model
meta_predictions = meta_model.predict(test_feature_preds)
meta_predictions = np.argmax(meta_predictions, axis=1)

# Step 6: Calculate performance metrics
accuracy = accuracy_score(y_test, meta_predictions)
print(f"Total Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, meta_predictions, target_names=label_encoder.classes_))

# Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, meta_predictions)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix for Meta-Model")
plt.savefig('multi_nn_plots/confusion_matrix_meta_model.png')
plt.close()

# Step 8: Predicted vs Actual Visualization
plt.figure(figsize=(8, 6))

# Scatter plot for Actual vs Predicted
plt.scatter(y_test, meta_predictions, alpha=0.7, color='blue', label='Predicted', s=50)

# Add a diagonal line to represent perfect predictions (y = x line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

# Add title and labels
plt.title("Actual vs Predicted Storm Intensity", fontsize=14)
plt.xlabel("Actual Storm Intensity (Encoded)", fontsize=12)
plt.ylabel("Predicted Storm Intensity (Encoded)", fontsize=12)
plt.legend()

# Save the plot
plt.savefig('multi_nn_plots/actual_vs_predicted_meta_model_improved.png')
plt.close()


# Step 9: K-Means Clustering Visualization
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X[['Latitude_NS', 'Longitude_EW']])
df['Cluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(df['Latitude_NS'], df['Longitude_EW'], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering of Storm Data")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.colorbar(label='Cluster')
plt.savefig('multi_nn_plots/kmeans_clustering.png')
plt.close()

# Step 10: Feature Importance Visualization
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Plotting the feature importance
plt.figure(figsize=(8, 5))
plt.bar(X.columns, importances)
plt.title("Feature Importance for Storm Intensity Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig('multi_nn_plots/feature_importance.png')
plt.close()

# Step 11: Geospatial Clustering Visualization (Geopandas)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude_EW'], df['Latitude_NS']))
gdf.plot(column='Storm_Intensity', cmap='coolwarm', legend=True, figsize=(8, 6))
plt.title("Geospatial Distribution of Storm Intensity")
plt.savefig('multi_nn_plots/geospatial_clustering.png')
plt.close()


