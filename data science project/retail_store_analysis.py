# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'your_file.csv' with your actual file path
data = pd.read_csv('your_file.csv')

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Data cleaning
# Fill missing values or drop them based on your analysis
data.fillna(0, inplace=True)  # Example: filling NaNs with 0

# Exploratory Data Analysis (EDA)

# Summary statistics
summary_statistics = data.describe()
print("Summary statistics:\n", summary_statistics)

# Visualizing sales over time (assuming there is a 'Date' and 'Sales' column)
data['Date'] = pd.to_datetime(data['Date'])
daily_sales = data.groupby('Date')['Sales'].sum().reset_index()

plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Sales', data=daily_sales)
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sales distribution
plt.figure(figsize=(14, 7))
sns.histplot(data['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Analyzing sales by category (assuming there is a 'Category' column)
category_sales = data.groupby('Category')['Sales'].sum().reset_index()

plt.figure(figsize=(14, 7))
sns.barplot(x='Sales', y='Category', data=category_sales)
plt.title('Sales by Category')
plt.xlabel('Sales')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

# Sales by region (assuming there is a 'Region' column)
region_sales = data.groupby('Region')['Sales'].sum().reset_index()

plt.figure(figsize=(14, 7))
sns.barplot(x='Sales', y='Region', data=region_sales)
plt.title('Sales by Region')
plt.xlabel('Sales')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Save cleaned data
data.to_csv('cleaned_retail_data.csv', index=False)
