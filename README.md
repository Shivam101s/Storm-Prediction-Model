# Storm-Prediction-Model
Key Steps in the "Prediction using random forest" Code:

Feature Models:
Three separate RandomForestClassifier models for wind speed, precipitation, and humidity.
Each model predicts the probability of a high-intensity storm based on its feature.

Meta-Model:
A LogisticRegression model combines predictions from feature models to classify storm intensity.


Training pipeline:
Feature models are trained independently.
Their predictions on training data are used as input features for the meta-model.

Inference Pipeline:
For new data, predictions from feature models are combined into the meta-model for final classification.

In "prediction_using_simple_neural_networks.py" file, we have used one artificial neural network that takes all the features and predict the storm intensity.
