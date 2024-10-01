# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Assuming dataset is in a CSV file with columns: 'text' and 'label'
df = pd.read_csv('news_data.csv')

# Step 2: Explore the dataset
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Step 3: Fill missing values (if any)
# Fill missing text with an empty string
df['text'] = df['text'].fillna('')

# Step 4: Prepare features and target
X = df['text']  # Text data (news articles)
y = df['label']  # Labels (1 for fake news, 0 for real news)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Feature extraction using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # Limit to 5000 most frequent words
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 7: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
