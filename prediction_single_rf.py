import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the processed dataset
file_path = 'processed_my.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Rename columns for clarity
df.columns = ['Storm_Intensity', 'Latitude_NS', 'Longitude_EW', 'Min_Pressure']

# Step 1: Data Preprocessing
# Extract numeric values from Latitude and Longitude while encoding direction
def process_latitude(lat):
    return float(lat[:-1]) * (1 if lat[-1] == 'N' else -1)

def process_longitude(lon):
    return float(lon[:-1]) * (1 if lon[-1] == 'E' else -1)

df['Latitude_NS'] = df['Latitude_NS'].apply(process_latitude)
df['Longitude_EW'] = df['Longitude_EW'].apply(process_longitude)

# Encode Storm Intensity (categorical) into numerical values
label_encoder_storm = LabelEncoder()
df['Storm_Intensity'] = label_encoder_storm.fit_transform(df['Storm_Intensity'])

# Handle missing or invalid values (if any)
df = df.dropna()

# Step 2: Define Features and Target
X = df[['Latitude_NS', 'Longitude_EW', 'Min_Pressure']]  # Predictors
y = df['Storm_Intensity']  # Target

# Standardize features (Latitude, Longitude, Min Pressure)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Classification Model
# Use Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder_storm.classes_))

# Step 6: Predict Storm Intensity for New Data
# Example: Predict for a new sample
new_sample = [[28.5, -95.0, 950]]  # Latitude = 28.5N, Longitude = 95.0W, Min Pressure = 950
new_sample_scaled = scaler.transform(new_sample)
predicted_class = model.predict(new_sample_scaled)
predicted_label = label_encoder_storm.inverse_transform(predicted_class)

print("Predicted Storm Intensity:", predicted_label[0])
