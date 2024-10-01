# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
ds = pd.read_csv('loan_data.csv')

# Step 2: Handling missing values
imputer = SimpleImputer(strategy='mean')
ds['LoanAmount'] = imputer.fit_transform(ds[['LoanAmount']])
ds['Loan_Amount_Term'] = imputer.fit_transform(ds[['Loan_Amount_Term']])
ds['Credit_History'] = imputer.fit_transform(ds[['Credit_History']])

# Fill categorical missing values with mode
ds['Gender'].fillna(ds['Gender'].mode()[0], inplace=True)
ds['Married'].fillna(ds['Married'].mode()[0], inplace=True)
ds['Dependents'].fillna(ds['Dependents'].mode()[0], inplace=True)
ds['Self_Employed'].fillna(ds['Self_Employed'].mode()[0], inplace=True)

# Step 3: Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']

for column in categorical_columns:
    ds[column] = label_encoder.fit_transform(ds[column])

# Step 4: Splitting the dsset into features and target
X = ds.drop(columns=['Loan_Status'])
y = ds['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Converting target variable into binary

# Step 5: Splitting the ds into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Build the Decision Tree Model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = classifier.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
