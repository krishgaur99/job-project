# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv('personality_data.csv')  # Replace with your dataset path

# Preview the data
print(df.head())

# Step 2: Data Preprocessing
# Handle missing values if any
df = df.dropna()

# Text data vectorization (if responses are textual)
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust as necessary
X = vectorizer.fit_transform(df['responses']).toarray()

# Encode personality labels (assuming categorical personality type)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['personality_type'])

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a Model
# Initialize RandomForest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Step 5: Model Prediction
# Predict on test set
y_pred = classifier.predict(X_test)

# Step 6: Model Evaluation
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 7: Save the Model for Future Use
# Save the model
joblib.dump(classifier, 'personality_prediction_model.pkl')

# Load the model (for future use)
loaded_model = joblib.load('personality_prediction_model.pkl')

# Example prediction
sample_response = ["I prefer working alone and enjoy analyzing complex problems."]  # Example response
sample_response_vectorized = vectorizer.transform(sample_response).toarray()
personality_pred = loaded_model.predict(sample_response_vectorized)
print(f'Predicted Personality: {label_encoder.inverse_transform(personality_pred)[0]}')

# Optional: Hyperparameter Tuning (if required)
from sklearn.model_selection import GridSearchCV

# Define parameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print(f'Best Parameters: {grid_search.best_params_}')
