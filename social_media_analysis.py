# Import necessary libraries
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Set up Twitter API credentials
# Replace these with your actual credentials
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Function to fetch tweets
def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search, q=query, lang='en').items(count)
    data = [{'Tweet': tweet.text, 'Created_at': tweet.created_at} for tweet in tweets]
    return pd.DataFrame(data)

# Fetch tweets related to a topic (e.g., "Python")
tweets_df = fetch_tweets("Python", count=200)

# Display the first few tweets
print(tweets_df.head())

# Data Cleaning
# Remove duplicates
tweets_df.drop_duplicates(subset=['Tweet'], inplace=True)

# Sentiment Analysis
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

# Apply sentiment analysis
tweets_df['Sentiment'] = tweets_df['Tweet'].apply(get_sentiment)

# EDA: Summary statistics
print(tweets_df.describe())

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))
sns.histplot(tweets_df['Sentiment'], bins=30, kde=True)
plt.title('Sentiment Distribution of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--')  # Show neutral line
plt.show()

# Time series analysis of tweets
tweets_df['Created_at'] = pd.to_datetime(tweets_df['Created_at'])
tweets_df.set_index('Created_at', inplace=True)
tweets_per_day = tweets_df.resample('D').size()

plt.figure(figsize=(12, 6))
sns.lineplot(x=tweets_per_day.index, y=tweets_per_day.values)
plt.title('Number of Tweets Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.show()

# Save the cleaned data to a CSV file
tweets_df.to_csv('cleaned_tweets_data.csv', index=False)
