import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Create a directory for plots if it doesn't exist
if not os.path.exists('plots2'):
    os.makedirs('plots2')

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

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert target variable to categorical
num_classes = len(y.unique())
y = to_categorical(y, num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the ANN model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),  # Input layer with 64 neurons
    Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    Dense(num_classes, activation='softmax')  # Output layer (number of classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the ANN model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Step 5: Evaluate the ANN model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoding back to labels

# Calculate total accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Total Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred))

# Step 6: Visualizations
# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=df['Storm_Intensity'].cat.categories)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix for ANN")
plt.savefig('plots2/confusion_matrix.png')
plt.close()

# Training history visualization
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('plots2/training_accuracy.png')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('plots2/training_loss.png')
plt.close()

# Predicted vs Actual
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test_labels)), y_test_labels, label='Actual', alpha=0.7, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, color='red')
plt.title("Actual vs Predicted Storm Intensity")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Storm Intensity (Encoded)")
plt.savefig('plots2/actual_vs_predicted.png')
plt.close()
