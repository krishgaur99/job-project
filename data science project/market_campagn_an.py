import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset
num_customers = 1000
data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, size=num_customers),
    'Income': np.random.randint(20000, 100000, size=num_customers),
    'Campaign_Response': np.random.choice(['Yes', 'No'], size=num_customers, p=[0.3, 0.7]),
    'Campaign_Type': np.random.choice(['Email', 'SMS', 'Social Media'], size=num_customers)
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Campaign Response Analysis
response_counts = df['Campaign_Response'].value_counts()
print("\nCampaign Response Counts:\n", response_counts)

# Plotting the campaign response
plt.figure(figsize=(8, 5))
sns.countplot(x='Campaign_Response', data=df, palette='Set2')
plt.title('Campaign Response Counts')
plt.xlabel('Response')
plt.ylabel('Count')
plt.show()

# Analyzing response by age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Campaign_Response', y='Age', data=df, palette='Set1')
plt.title('Age Distribution by Campaign Response')
plt.xlabel('Response')
plt.ylabel('Age')
plt.show()

# Analyzing response by income
plt.figure(figsize=(10, 6))
sns.boxplot(x='Campaign_Response', y='Income', data=df, palette='Set1')
plt.title('Income Distribution by Campaign Response')
plt.xlabel('Response')
plt.ylabel('Income')
plt.show()

# Campaign Type Analysis
campaign_response_type = df.groupby(['Campaign_Type', 'Campaign_Response']).size().unstack()
print("\nCampaign Response by Type:\n", campaign_response_type)

# Plotting the campaign type response
campaign_response_type.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'], figsize=(10, 6))
plt.title('Campaign Response by Type')
plt.xlabel('Campaign Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Response')
plt.show()

# Analyzing response rates
response_rates = campaign_response_type.div(campaign_response_type.sum(axis=1), axis=0)
print("\nResponse Rates by Campaign Type:\n", response_rates)

# Plotting response rates
response_rates.plot(kind='bar', figsize=(10, 6))
plt.title('Response Rates by Campaign Type')
plt.xlabel('Campaign Type')
plt.ylabel('Response Rate')
plt.xticks(rotation=0)
plt.legend(title='Response', labels=['No', 'Yes'])
plt.show()
