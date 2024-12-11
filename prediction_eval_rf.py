import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Step 1: Create a folder to save plots
if not os.path.exists('plots1'):
    os.makedirs('plots1')

# Step 2: Load and preprocess the dataset
df = pd.read_csv('processed_my.csv')
df = df.rename(columns={"Min_Pressure": "Max_Sustained_Wind"})


# Preprocess Latitude and Longitude into numeric values
def process_latitude(lat):
    return float(lat[:-1]) * (1 if lat[-1] == 'N' else -1)

def process_longitude(lon):
    return float(lon[:-1]) * (1 if lon[-1] == 'E' else -1)

df['Latitude_NS'] = df['Latitude_NS'].apply(process_latitude)
df['Longitude_EW'] = df['Longitude_EW'].apply(process_longitude)
df['Storm_Intensity'] = df['Storm_Intensity'].astype('category')

# Step 3: Define features and target variable
X = df[['Latitude_NS', 'Longitude_EW', 'Max_Sustained_Wind']]
y = df['Storm_Intensity'].cat.codes

# Step 4: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
rf_predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# Step 6: Generate Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=df['Storm_Intensity'].cat.categories, 
            yticklabels=df['Storm_Intensity'].cat.categories)
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('plots1/confusion_matrix_rf.png')
plt.close()

# Step 7: Feature Importance Visualization
plt.figure(figsize=(8, 6))
feature_importances = rf_model.feature_importances_
sns.barplot(x=feature_importances, y=X.columns, palette='viridis')
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig('plots1/feature_importance_rf.png')
plt.close()

# Step 8: K-Means Clustering for Spatial Analysis
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude_NS', 'Longitude_EW']])

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Longitude_EW', y='Latitude_NS', hue='Cluster', data=df, palette='deep')
plt.title("K-Means Clustering of Storm Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('plots1/kmeans_clustering.png')
plt.close()

# Step 9: Additional Evaluation
# Predicted vs Actual Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7, color='blue')
plt.scatter(range(len(rf_predictions)), rf_predictions, label='Predicted', alpha=0.7, color='red')
plt.title("Actual vs Predicted Storm Intensity")
plt.xlabel("Sample Index")
plt.ylabel("Storm Intensity (Encoded)")
plt.legend()
plt.savefig('plots1/actual_vs_predicted.png')
plt.close()

print("\nAll plots saved in the 'plots1' folder.")
